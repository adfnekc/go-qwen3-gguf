package main

import (
	"fmt"
	"math"
	"os"

	"gguf/gguf"
	"gguf/model"
	llmmath "gguf/math"
	"gguf/tokenizer"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(modelPath)
	if err != nil {
		panic(err)
	}
	defer mmapFile.Close()

	config, err := model.LoadQwen3Config(reader)
	if err != nil {
		panic(err)
	}

	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil {
		panic(err)
	}

	tok, err := tokenizer.NewTokenizerFromGGUF(reader)
	if err != nil {
		panic(err)
	}

	ed := config.EmbeddingLength
	hd := config.AttentionKeyLength
	nh := config.AttentionHeadCount
	nk := config.AttentionHeadCountKv
	qd := nh * hd
	kd := nk * hd

	// Build prompt
	imStart, imEnd := 151644, 151645
	prompt := []int{imStart}
	prompt = append(prompt, tok.Encode("user\n")...)
	prompt = append(prompt, tok.Encode("hi")...)
	prompt = append(prompt, imEnd)
	prompt = append(prompt, 198)
	prompt = append(prompt, imStart)
	prompt = append(prompt, tok.Encode("assistant\n")...)

	// APPROACH 1: Run model.Forward with prompt + <think> together (one-shot)
	m1 := model.NewQwen3Model(config, weights)
	combined := append([]int{}, prompt...)
	combined = append(combined, 151667) // <think>
	logits1, err := m1.Forward(combined, 0)
	if err != nil {
		panic(err)
	}
	_ = logits1

	// APPROACH 2: Run step-by-step with cache, then inspect hidden states
	m2 := model.NewQwen3Model(config, weights)
	// Step 1: prompt
	logitsP, err := m2.Forward(prompt, 0)
	if err != nil {
		panic(err)
	}
	_ = logitsP
	// Step 2: <think>
	logits2, err := m2.Forward([]int{151667}, len(prompt))
	if err != nil {
		panic(err)
	}
	_ = logits2

	fmt.Println("Comparison done. Logits from one-shot vs cache:")
	fmt.Printf("  One-shot top1: %d (%.4f)\n", argmax(logits1), logits1[argmax(logits1)])
	fmt.Printf("  Cache top1: %d (%.4f)\n", argmax(logits2), logits2[argmax(logits2)])
	fmt.Printf("  Max diff: %.6f\n", maxAbsDiff(logits1, logits2))

	// Now, let's manually trace through each layer and compare hidden states
	fmt.Println("\n=== Manual layer-by-layer comparison ===")

	// Build embeddings for prompt (9 tokens) + <think> (1 token)
	embeddings := make([]float32, 10*ed)
	for i, tid := range combined {
		emb := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, ed, tid)
		copy(embeddings[i*ed:], emb)
	}

	// Cache approach: store per-layer hidden states for position 9
	// We need to trace the single-token forward for <think>
	cacheHidden := make([]float32, ed)
	copy(cacheHidden, embeddings[9*ed:]) // start from embedding of <think>

	// One-shot approach: store per-layer hidden states for position 9
	oneShotHidden := make([]float32, ed)
	copy(oneShotHidden, embeddings[9*ed:])

	// For one-shot, we need ALL positions' hidden states through each layer
	allHidden := make([]float32, 10*ed)
	copy(allHidden, embeddings)

	// Manual KV cache for cache approach (store per-layer)
	cacheK := make([][]float32, config.BlockCount)
	cacheV := make([][]float32, config.BlockCount)
	cacheSize := make([]int, config.BlockCount)

	for l := 0; l < config.BlockCount; l++ {
		cacheK[l] = make([]float32, 0)
		cacheV[l] = make([]float32, 0)
	}

	// First, process the 9 prompt tokens through ALL layers (one token at a time, caching)
	for pos := 0; pos < 9; pos++ {
		tokenHidden := make([]float32, ed)
		copy(tokenHidden, embeddings[pos*ed:(pos+1)*ed])

		for l := 0; l < config.BlockCount; l++ {
			bw := weights.BlockWeights[l]

			// --- Attention ---
			normed := llmmath.RMSNorm(tokenHidden, bw.AttentionNorm, config.LayerNormRmsEps)

			q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qd, ed)
			k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kd, ed)
			v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kd, ed)

			applyQKRoqu(q, k, nk, nh, hd, bw, config, 1, pos)

			// Cache K,V
			cacheK[l] = append(cacheK[l], k...)
			cacheV[l] = append(cacheV[l], v...)
			cacheSize[l] = pos + 1

			// Self-attention with cache
			pastSeqLen := cacheSize[l]
			kRep := llmmath.RepeatKV(cacheK[l], nk, nh, hd, pastSeqLen)
			vRep := llmmath.RepeatKV(cacheV[l], nk, nh, hd, pastSeqLen)
			kCurRep := llmmath.RepeatKV(k, nk, nh, hd, 1)
			vCurRep := llmmath.RepeatKV(v, nk, nh, hd, 1)

			allK := append(kRep, kCurRep...)
			allV := append(vRep, vCurRep...)

			// Only attend to positions <= pos
			scale := 1.0 / float32(math.Sqrt(float64(hd)))
			attnOut := make([]float32, qd)

			for h := 0; h < nh; h++ {
				qHead := q[h*hd : (h+1)*hd]
				var maxScore float32
				scores := make([]float32, pos+1)
				for t := 0; t <= pos; t++ {
					var sum float32
					kHead := allK[t*nh*hd+h*hd : t*nh*hd+(h+1)*hd]
					for d := 0; d < hd; d++ {
						sum += qHead[d] * kHead[d]
					}
					scores[t] = sum * scale
					if t == 0 || scores[t] > maxScore {
						maxScore = scores[t]
					}
				}
				var sumExp float32
				for t := 0; t <= pos; t++ {
					scores[t] = float32(math.Exp(float64(scores[t] - maxScore)))
					sumExp += scores[t]
				}
				for t := 0; t <= pos; t++ {
					scores[t] /= sumExp
				}
				for d := 0; d < hd; d++ {
					for t := 0; t <= pos; t++ {
						vHead := allV[t*nh*hd+h*hd : t*nh*hd+(h+1)*hd]
						attnOut[h*hd+d] += scores[t] * vHead[d]
					}
				}
			}

			// Output projection
			attnProj := llmmath.MatMulTransposed(attnOut, bw.AttentionOutput, 1, ed, qd)

			// Add residual
			residual := make([]float32, ed)
			copy(residual, tokenHidden)
			tokenHidden = llmmath.VectorAdd(tokenHidden, attnProj)

			// --- FFN ---
			residual2 := make([]float32, ed)
			copy(residual2, tokenHidden)
			ffnNormed := llmmath.RMSNorm(tokenHidden, bw.FeedForwardNorm, config.LayerNormRmsEps)
			ffnGate := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardGate, 1, config.FeedForwardLength, ed)
			ffnUp := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardUp, 1, config.FeedForwardLength, ed)
			swiglu := llmmath.SwiGLU(ffnGate, ffnUp)
			ffnDown := llmmath.MatMulTransposed(swiglu, bw.FeedForwardDown, 1, ed, config.FeedForwardLength)
			tokenHidden = llmmath.VectorAdd(residual2, ffnDown)
		}
		// After all layers, tokenHidden is the output for this position
		// But we don't use it for logits — we only cache K,V for future positions
	}

	// Now process the <think> token (position 9) through the cache
	cacheThinkHidden := make([]float32, ed)
	copy(cacheThinkHidden, embeddings[9*ed:])

	for l := 0; l < config.BlockCount; l++ {
		bw := weights.BlockWeights[l]
		pos := 9

		// Attention norm
		normed := llmmath.RMSNorm(cacheThinkHidden, bw.AttentionNorm, config.LayerNormRmsEps)
		q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qd, ed)
		k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kd, ed)
		v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kd, ed)

		// QK norm + RoPE
		applyQKRoqu(q, k, nk, nh, hd, bw, config, 1, pos)

		// Cache K,V for position 9
		cacheK[l] = append(cacheK[l], k...)
		cacheV[l] = append(cacheV[l], v...)
		cacheSize[l] = pos + 1

		// Self-attention
		pastSeqLen := cacheSize[l]
		kRep := llmmath.RepeatKV(cacheK[l], nk, nh, hd, pastSeqLen)
		vRep := llmmath.RepeatKV(cacheV[l], nk, nh, hd, pastSeqLen)

		scale := 1.0 / float32(math.Sqrt(float64(hd)))
		attnOut := make([]float32, qd)

		for h := 0; h < nh; h++ {
			qHead := q[h*hd : (h+1)*hd]
			scores := make([]float32, pastSeqLen)
			for t := 0; t < pastSeqLen; t++ {
				var sum float32
				kHead := kRep[t*nh*hd+h*hd : t*nh*hd+(h+1)*hd]
				for d := 0; d < hd; d++ {
					sum += qHead[d] * kHead[d]
				}
				scores[t] = sum * scale
			}

			// Causal mask
			for t := 0; t < pastSeqLen; t++ {
				if t > pastSeqLen-1 {
					scores[t] = float32(math.Inf(-1))
				}
			}

			maxScore := scores[0]
			for _, s := range scores[1:] {
				if s > maxScore {
					maxScore = s
				}
			}
			var sumExp float32
			for _, s := range scores {
				sumExp += float32(math.Exp(float64(s - maxScore)))
			}
			probs := make([]float32, pastSeqLen)
			for t, s := range scores {
				probs[t] = float32(math.Exp(float64(s-maxScore))) / sumExp
			}

			for d := 0; d < hd; d++ {
				for t := 0; t < pastSeqLen; t++ {
					vHead := vRep[t*nh*hd+h*hd : t*nh*hd+(h+1)*hd]
					attnOut[h*hd+d] += probs[t] * vHead[d]
				}
			}
		}

		attnProj := llmmath.MatMulTransposed(attnOut, bw.AttentionOutput, 1, ed, qd)
		residual := make([]float32, ed)
		copy(residual, cacheThinkHidden)
		cacheThinkHidden = llmmath.VectorAdd(cacheThinkHidden, attnProj)

		// FFN
		residual2 := make([]float32, ed)
		copy(residual2, cacheThinkHidden)
		ffnNormed := llmmath.RMSNorm(cacheThinkHidden, bw.FeedForwardNorm, config.LayerNormRmsEps)
		ffnGate := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardGate, 1, config.FeedForwardLength, ed)
		ffnUp := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardUp, 1, config.FeedForwardLength, ed)
		swiglu := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnDown := llmmath.MatMulTransposed(swiglu, bw.FeedForwardDown, 1, ed, config.FeedForwardLength)
		cacheThinkHidden = llmmath.VectorAdd(residual2, ffnDown)

		// Track diff from one-shot
		diff := maxAbsDiff(cacheThinkHidden, oneShotHidden)
		if diff > 0.001 {
			fmt.Printf("Layer %2d: cache RMS=%.4f  one-shot RMS=%.4f  diff=%.6f\n",
				l, rms32(cacheThinkHidden), rms32(oneShotHidden), diff)
		}
	}

	// Compute logits from hidden
	cacheNormed := llmmath.RMSNorm(cacheThinkHidden, weights.OutputNorm, config.LayerNormRmsEps)
	var cacheLogits []float32
	if weights.Output != nil {
		cacheLogits = llmmath.MatMulTransposed(cacheNormed, weights.Output, 1, config.VocabSize, ed)
	} else {
		cacheLogits = llmmath.MatMulTransposed(cacheNormed, weights.TokenEmbedding, 1, config.VocabSize, ed)
	}
	fmt.Printf("\nCache logits top1: %d (%.4f)\n", argmax(cacheLogits), cacheLogits[argmax(cacheLogits)])
}

func applyQKRoqu(q, k []float32, nKvHeads, nHeads, headDim int, bw *model.Qwen3BlockWeights, config *model.Qwen3Config, seqLen, startPos int) {
	qd := nHeads * headDim
	kd := nKvHeads * headDim
	if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
		for i := 0; i < seqLen; i++ {
			for h := 0; h < nHeads; h++ {
				qStart := i*qd + h*headDim
				qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
				copy(q[qStart:], qNormed)
			}
			for h := 0; h < nKvHeads; h++ {
				kStart := i*kd + h*headDim
				kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
				copy(k[kStart:], kNormed)
			}
		}
	}
	for i := 0; i < seqLen; i++ {
		pos := startPos + i
		for h := 0; h < nKvHeads; h++ {
			kStart := i*kd + h*headDim
			_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], pos, headDim, config.RopeFreqBase, llmmath.RoPE_NEOX)
			copy(k[kStart:], kRope)
		}
		for h := 0; h < nHeads; h++ {
			qStart := i*qd + h*headDim
			qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, pos, headDim, config.RopeFreqBase, llmmath.RoPE_NEOX)
			copy(q[qStart:], qRope)
		}
	}
}

func rms32(v []float32) float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return float32(math.Sqrt(sum / float64(len(v))))
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return -1
	}
	var maxDiff float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func argmax(v []float32) int {
	maxIdx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}
