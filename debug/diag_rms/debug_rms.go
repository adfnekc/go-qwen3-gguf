package main

import (
	"fmt"
	"math"
	"os"

	"gguf/gguf"
	"gguf/model"
	llmmath "gguf/math"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run debug_rms.go <model_path>")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(modelPath)
	if err != nil {
		fmt.Printf("Failed to load: %v\n", err)
		os.Exit(1)
	}
	defer mmapFile.Close()

	config, err := model.LoadQwen3Config(reader)
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil {
		fmt.Printf("Failed to load weights: %v\n", err)
		os.Exit(1)
	}

	// Output norm stats
	fmt.Printf("OutputNorm: len=%d, min=%.6f, max=%.6f, mean=%.6f, rms=%.6f\n",
		len(weights.OutputNorm), min32(weights.OutputNorm), max32(weights.OutputNorm),
		mean32(weights.OutputNorm), rms32(weights.OutputNorm))

	// For now, just use token 6023 "hi" as a simple test
	tokenID := 6023 // "hi"
	embedding := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, config.EmbeddingLength, tokenID)

	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	headDim := config.AttentionKeyLength
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	embeddingDim := config.EmbeddingLength
	ffnDim := config.FeedForwardLength

	hiddenStates := make([]float32, embeddingDim)
	copy(hiddenStates, embedding)

	fmt.Printf("\nInitial embedding RMS: %.6f\n\n", rms32(hiddenStates))

	// Run full forward pass through all layers
	for layer := 0; layer < config.BlockCount; layer++ {
		bw := weights.BlockWeights[layer]

		// Attention
		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normed := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
		q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kvDim, embeddingDim)

		if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
			for h := 0; h < nHeads; h++ {
				qStart := h * headDim
				qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
				copy(q[qStart:], qNormed)
			}
			for h := 0; h < nKvHeads; h++ {
				kStart := h * headDim
				kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
				copy(k[kStart:], kNormed)
			}
		}

		// RoPE at pos 0 (single token)
		for h := 0; h < nKvHeads; h++ {
			kStart := h * headDim
			_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], 0, headDim, config.RopeFreqBase)
			copy(k[kStart:], kRope)
		}
		for h := 0; h < nHeads; h++ {
			qStart := h * headDim
			qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, 0, headDim, config.RopeFreqBase)
			copy(q[qStart:], qRope)
		}

		vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)
		attnOut := llmmath.MatMulTransposed(vRepeated, bw.AttentionOutput, 1, embeddingDim, qDim)
		hiddenStates = llmmath.VectorAdd(residual, attnOut)

		// FFN
		residual = make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normedFFN := llmmath.RMSNorm(hiddenStates, bw.FeedForwardNorm, config.LayerNormRmsEps)
		ffnGate := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardGate, 1, ffnDim, embeddingDim)
		ffnUp := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardUp, 1, ffnDim, embeddingDim)
		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnOut := llmmath.MatMulTransposed(swigluOutput, bw.FeedForwardDown, 1, embeddingDim, ffnDim)
		hiddenStates = llmmath.VectorAdd(residual, ffnOut)

		rms := rms32(hiddenStates)
		fmt.Printf("Layer %2d: RMS=%.6f", layer, rms)
		if layer%5 == 0 || layer == config.BlockCount-1 {
			// Check top token at this layer
			normedOutput := llmmath.RMSNorm(hiddenStates, weights.OutputNorm, config.LayerNormRmsEps)
			var logits []float32
			if weights.Output != nil {
				logits = llmmath.MatMulTransposed(normedOutput, weights.Output, 1, config.VocabSize, embeddingDim)
			} else {
				logits = llmmath.MatMulTransposed(normedOutput, weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
			}
			topIdx := llmmath.Argmax(logits)
			topVal := logits[topIdx]
			hasNaN := false
			for _, v := range logits {
				if math.IsNaN(float64(v)) {
					hasNaN = true
					break
				}
			}
			fmt.Printf("  top=#%d(%.2f) NaN=%v", topIdx, topVal, hasNaN)
		}
		fmt.Println()
	}

	// Final output
	normedOutput := llmmath.RMSNorm(hiddenStates, weights.OutputNorm, config.LayerNormRmsEps)
	fmt.Printf("\nFinal normed output RMS: %.6f\n", rms32(normedOutput))

	var logits []float32
	if weights.Output != nil {
		logits = llmmath.MatMulTransposed(normedOutput, weights.Output, 1, config.VocabSize, embeddingDim)
	} else {
		logits = llmmath.MatMulTransposed(normedOutput, weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
	}

	// Top tokens
	type scored struct {
		id    int
		score float32
	}
	var top10 []scored
	for i := 0; i < 10; i++ {
		best := -1
		bestVal := float32(math.Inf(-1))
		for j, v := range logits {
			skip := false
			for _, t := range top10 {
				if t.id == j {
					skip = true
					break
				}
			}
			if !skip && v > bestVal {
				bestVal = v
				best = j
			}
		}
		if best >= 0 {
			top10 = append(top10, scored{best, bestVal})
		}
	}
	fmt.Println("\nTop 10 tokens:")
	for _, t := range top10 {
		fmt.Printf("  #%d: %.4f\n", t.id, t.score)
	}

	// Also check: what if we only use layers 0+1?
	fmt.Println("\n=== Test: layers 0-1 only ===")
	hiddenStates2 := make([]float32, embeddingDim)
	copy(hiddenStates2, embedding)
	for layer := 0; layer < 2; layer++ {
		bw := weights.BlockWeights[layer]
		residual := make([]float32, len(hiddenStates2))
		copy(residual, hiddenStates2)
		normed := llmmath.RMSNorm(hiddenStates2, bw.AttentionNorm, config.LayerNormRmsEps)
		q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kvDim, embeddingDim)
		if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
			for h := 0; h < nHeads; h++ {
				qStart := h * headDim
				qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
				copy(q[qStart:], qNormed)
			}
			for h := 0; h < nKvHeads; h++ {
				kStart := h * headDim
				kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
				copy(k[kStart:], kNormed)
			}
		}
		for h := 0; h < nKvHeads; h++ {
			kStart := h * headDim
			_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], 0, headDim, config.RopeFreqBase)
			copy(k[kStart:], kRope)
		}
		for h := 0; h < nHeads; h++ {
			qStart := h * headDim
			qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, 0, headDim, config.RopeFreqBase)
			copy(q[qStart:], qRope)
		}
		vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)
		attnOut := llmmath.MatMulTransposed(vRepeated, bw.AttentionOutput, 1, embeddingDim, qDim)
		hiddenStates2 = llmmath.VectorAdd(residual, attnOut)

		residual = make([]float32, len(hiddenStates2))
		copy(residual, hiddenStates2)
		normedFFN := llmmath.RMSNorm(hiddenStates2, bw.FeedForwardNorm, config.LayerNormRmsEps)
		ffnGate := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardGate, 1, ffnDim, embeddingDim)
		ffnUp := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardUp, 1, ffnDim, embeddingDim)
		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnOut := llmmath.MatMulTransposed(swigluOutput, bw.FeedForwardDown, 1, embeddingDim, ffnDim)
		hiddenStates2 = llmmath.VectorAdd(residual, ffnOut)
	}
	normedOutput2 := llmmath.RMSNorm(hiddenStates2, weights.OutputNorm, config.LayerNormRmsEps)
	var logits2 []float32
	if weights.Output != nil {
		logits2 = llmmath.MatMulTransposed(normedOutput2, weights.Output, 1, config.VocabSize, embeddingDim)
	} else {
		logits2 = llmmath.MatMulTransposed(normedOutput2, weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
	}
	topIdx2 := llmmath.Argmax(logits2)
	fmt.Printf("Top token after 2 layers: #%d (%.4f)\n", topIdx2, logits2[topIdx2])

	var top5_2 []scored
	for i := 0; i < 5; i++ {
		best := -1
		bestVal := float32(math.Inf(-1))
		for j, v := range logits2 {
			skip := false
			for _, t := range top5_2 {
				if t.id == j {
					skip = true
					break
				}
			}
			if !skip && v > bestVal {
				bestVal = v
				best = j
			}
		}
		if best >= 0 {
			top5_2 = append(top5_2, scored{best, bestVal})
		}
	}
	fmt.Printf("Top 5: ")
	for _, t := range top5_2 {
		fmt.Printf("#%d(%.2f) ", t.id, t.score)
	}
	fmt.Println()

	// Also test with N tokens (prefill-like)
	fmt.Println("\n=== Test: 2-token prefill (tokens 6023 and 4999) ===")
	tokens := []int{6023, 4999} // "hi" and "hello"
	embeddings := make([]float32, 2*embeddingDim)
	for idx, tok := range tokens {
		emb := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, tok)
		copy(embeddings[idx*embeddingDim:], emb)
	}
	hiddenStates3 := make([]float32, 2*embeddingDim)
	copy(hiddenStates3, embeddings)

	// Just run layer 0 with 2 tokens
	bw0 := weights.BlockWeights[0]
	for layer := 0; layer < 1; layer++ {
		bw := weights.BlockWeights[layer]
		seqLen := 2

		residual := make([]float32, len(hiddenStates3))
		copy(residual, hiddenStates3)

		normedHidden := make([]float32, seqLen*embeddingDim)
		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates3[i*embeddingDim:(i+1)*embeddingDim], bw.AttentionNorm, config.LayerNormRmsEps)
			copy(normedHidden[i*embeddingDim:], normed)
		}

		q := llmmath.MatMulTransposed(normedHidden, bw.AttentionQ, seqLen, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normedHidden, bw.AttentionK, seqLen, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normedHidden, bw.AttentionV, seqLen, kvDim, embeddingDim)

		if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
			for i := 0; i < seqLen; i++ {
				for h := 0; h < nHeads; h++ {
					qStart := i*qDim + h*headDim
					qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
					copy(q[qStart:], qNormed)
				}
				for h := 0; h < nKvHeads; h++ {
					kStart := i*kvDim + h*headDim
					kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
					copy(k[kStart:], kNormed)
				}
			}
		}

		// RoPE for each position
		for i := 0; i < seqLen; i++ {
			for h := 0; h < nKvHeads; h++ {
				kStart := i*kvDim + h*headDim
				_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], i, headDim, config.RopeFreqBase)
				copy(k[kStart:], kRope)
			}
			for h := 0; h < nHeads; h++ {
				qStart := i*qDim + h*headDim
				qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, i, headDim, config.RopeFreqBase)
				copy(q[qStart:], qRope)
			}
		}

		// Cache simulation for multi-token
		// Manually prepare pastK, pastV (like KV cache)
		pastK := make([]float32, seqLen*nKvHeads*headDim)
		pastV := make([]float32, seqLen*nKvHeads*headDim)
		for i := 0; i < seqLen; i++ {
			copy(pastK[i*kvDim:], k[i*kvDim:(i+1)*kvDim])
			copy(pastV[i*kvDim:], v[i*kvDim:(i+1)*kvDim])
		}

		// GQA repeat
		pastK = llmmath.RepeatKV(pastK, nKvHeads, nHeads, headDim, seqLen)
		pastV = llmmath.RepeatKV(pastV, nKvHeads, nHeads, headDim, seqLen)
		_ = llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, seqLen)
		_ = llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, seqLen)

		// Attention
		attentionOutput := make([]float32, seqLen*qDim)
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				qHead := q[s*qDim+h*headDim : s*qDim+(h+1)*headDim]

				scores := make([]float32, seqLen)
				for t := 0; t < seqLen; t++ {
					kHead := pastK[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]
					var score float32
					for d := 0; d < headDim; d++ {
						score += qHead[d] * kHead[d]
					}
					scores[t] = score * scale
				}

				// Causal mask
				for t := 0; t < seqLen; t++ {
					if t > s {
						scores[t] = float32(math.Inf(-1))
					}
				}

				probs := llmmath.VectorSoftmax(scores)

				outputHead := make([]float32, headDim)
				for t := 0; t < seqLen; t++ {
					vHead := pastV[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]
					for d := 0; d < headDim; d++ {
						outputHead[d] += probs[t] * vHead[d]
					}
				}

				copy(attentionOutput[s*qDim+h*headDim:], outputHead)
			}
		}

		attnOut := llmmath.MatMulTransposed(attentionOutput, bw.AttentionOutput, seqLen, embeddingDim, qDim)
		hiddenStates3 = llmmath.VectorAdd(residual, attnOut)

		// FFN
		residual = make([]float32, len(hiddenStates3))
		copy(residual, hiddenStates3)

		normedFFN := make([]float32, seqLen*embeddingDim)
		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates3[i*embeddingDim:(i+1)*embeddingDim], bw.FeedForwardNorm, config.LayerNormRmsEps)
			copy(normedFFN[i*embeddingDim:], normed)
		}

		ffnGate := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardGate, seqLen, ffnDim, embeddingDim)
		ffnUp := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardUp, seqLen, ffnDim, embeddingDim)
		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnOut := llmmath.MatMulTransposed(swigluOutput, bw.FeedForwardDown, seqLen, embeddingDim, ffnDim)
		hiddenStates3 = llmmath.VectorAdd(residual, ffnOut)
		_ = bw0
	}

	// Get last token's hidden state
	lastHidden := hiddenStates3[(2-1)*embeddingDim : 2*embeddingDim]
	fmt.Printf("Last token RMS after layer 0: %.6f\n", rms32(lastHidden))

	normedLast := llmmath.RMSNorm(lastHidden, weights.OutputNorm, config.LayerNormRmsEps)
	var logits3 []float32
	if weights.Output != nil {
		logits3 = llmmath.MatMulTransposed(normedLast, weights.Output, 1, config.VocabSize, embeddingDim)
	} else {
		logits3 = llmmath.MatMulTransposed(normedLast, weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
	}

	topIdx3 := llmmath.Argmax(logits3)
	fmt.Printf("Top token after layer 0 (2 tokens): #%d (%.4f)\n", topIdx3, logits3[topIdx3])
	var top5_3 []scored
	for i := 0; i < 5; i++ {
		best := -1
		bestVal := float32(math.Inf(-1))
		for j, v := range logits3 {
			skip := false
			for _, t := range top5_3 {
				if t.id == j {
					skip = true
					break
				}
			}
			if !skip && v > bestVal {
				bestVal = v
				best = j
			}
		}
		if best >= 0 {
			top5_3 = append(top5_3, scored{best, bestVal})
		}
	}
	fmt.Printf("Top 5: ")
	for _, t := range top5_3 {
		fmt.Printf("#%d(%.2f) ", t.id, t.score)
	}
	fmt.Println()
}

func min32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	m := v[0]
	for _, x := range v[1:] {
		if x < m { m = x }
	}
	return m
}

func max32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	m := v[0]
	for _, x := range v[1:] {
		if x > m { m = x }
	}
	return m
}

func mean32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	var sum float32
	for _, x := range v { sum += x }
	return sum / float32(len(v))
}

func rms32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	var sum float32
	for _, x := range v { sum += x * x }
	return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
