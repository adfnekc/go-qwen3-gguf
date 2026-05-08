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
		fmt.Println("Usage: go run debug_compare.go <model_path>")
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

	fmt.Printf("Config: embeddingDim=%d, nHeads=%d, nKvHeads=%d, headDim=%d, ffnDim=%d, layers=%d\n",
		config.EmbeddingLength, config.AttentionHeadCount, config.AttentionHeadCountKv,
		config.AttentionKeyLength, config.FeedForwardLength, config.BlockCount)

	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	headDim := config.AttentionKeyLength
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	embeddingDim := config.EmbeddingLength
	ffnDim := config.FeedForwardLength

	fmt.Printf("\nComputed dimensions: qDim=%d, kvDim=%d, embed=%d, ffn=%d\n", qDim, kvDim, embeddingDim, ffnDim)

	// List all tensor names with shapes
	fmt.Println("\n=== All tensor names, shapes, types ===")
	for _, t := range reader.Tensors {
		fmt.Printf("  %-50s shape=%-20v type=%s offset=%d\n", t.Name, t.Shape, t.Type, t.Offset)
	}

	// Load weights
	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil {
		fmt.Printf("Failed to load weights: %v\n", err)
		os.Exit(1)
	}

	// Compare key tensors for layers 0 and 1
	for layer := 0; layer <= 1; layer++ {
		bw := weights.BlockWeights[layer]
		fmt.Printf("\n=== Layer %d weights ===\n", layer)
		fmt.Printf("  AttentionNorm: len=%d, min=%.6f, max=%.6f, mean=%.6f\n",
			len(bw.AttentionNorm), min32(bw.AttentionNorm), max32(bw.AttentionNorm), mean32(bw.AttentionNorm))
		fmt.Printf("  AttentionQ: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.AttentionQ), min32(bw.AttentionQ), max32(bw.AttentionQ), mean32(bw.AttentionQ))
		fmt.Printf("  AttentionK: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.AttentionK), min32(bw.AttentionK), max32(bw.AttentionK), mean32(bw.AttentionK))
		fmt.Printf("  AttentionV: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.AttentionV), min32(bw.AttentionV), max32(bw.AttentionV), mean32(bw.AttentionV))
		fmt.Printf("  AttentionOutput: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.AttentionOutput), min32(bw.AttentionOutput), max32(bw.AttentionOutput), mean32(bw.AttentionOutput))
		fmt.Printf("  FeedForwardGate: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.FeedForwardGate), min32(bw.FeedForwardGate), max32(bw.FeedForwardGate), mean32(bw.FeedForwardGate))
		fmt.Printf("  FeedForwardUp: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.FeedForwardUp), min32(bw.FeedForwardUp), max32(bw.FeedForwardUp), mean32(bw.FeedForwardUp))
		fmt.Printf("  FeedForwardDown: len=%d, min=%.4f, max=%.4f, mean=%.4f\n",
			len(bw.FeedForwardDown), min32(bw.FeedForwardDown), max32(bw.FeedForwardDown), mean32(bw.FeedForwardDown))
		if bw.AttentionQNorm != nil {
			fmt.Printf("  AttentionQNorm: len=%d, min=%.6f, max=%.6f, mean=%.6f\n",
				len(bw.AttentionQNorm), min32(bw.AttentionQNorm), max32(bw.AttentionQNorm), mean32(bw.AttentionQNorm))
		}
		if bw.AttentionKNorm != nil {
			fmt.Printf("  AttentionKNorm: len=%d, min=%.6f, max=%.6f, mean=%.6f\n",
				len(bw.AttentionKNorm), min32(bw.AttentionKNorm), max32(bw.AttentionKNorm), mean32(bw.AttentionKNorm))
		}
		fmt.Printf("  FeedForwardNorm: len=%d, min=%.6f, max=%.6f, mean=%.6f\n",
			len(bw.FeedForwardNorm), min32(bw.FeedForwardNorm), max32(bw.FeedForwardNorm), mean32(bw.FeedForwardNorm))

		// Verify expected lengths
		fmt.Printf("  Expected: Q=[%d,%d]=%d K=[%d,%d]=%d V=[%d,%d]=%d Output=[%d,%d]=%d Gate=[%d,%d]=%d Up=[%d,%d]=%d Down=[%d,%d]=%d\n",
			embeddingDim, qDim, embeddingDim*qDim,
			embeddingDim, kvDim, embeddingDim*kvDim,
			embeddingDim, kvDim, embeddingDim*kvDim,
			qDim, embeddingDim, qDim*embeddingDim,
			embeddingDim, ffnDim, embeddingDim*ffnDim,
			embeddingDim, ffnDim, embeddingDim*ffnDim,
			ffnDim, embeddingDim, ffnDim*embeddingDim)
	}

	// Get the embedding for "hi" token (ID 6023)
	fmt.Println("\n=== Single-layer forward pass trace ===")
	tokenID := 6023
	embedding := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, tokenID)
	fmt.Printf("Embedding for token %d: len=%d, min=%.6f, max=%.6f, rms=%.6f\n",
		tokenID, len(embedding), min32(embedding), max32(embedding), rms32(embedding))

	hiddenStates := make([]float32, embeddingDim)
	copy(hiddenStates, embedding)

	// Test each layer 0 through 3
	for layer := 0; layer <= 3; layer++ {
		bw := weights.BlockWeights[layer]

		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		// Attention norm
		normed := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
		fmt.Printf("\nLayer %d:\n", layer)
		fmt.Printf("  After attn_norm: rms=%.6f, min=%.6f, max=%.6f\n", rms32(normed), min32(normed), max32(normed))

		// QKV projections
		q := llmmath.MatMul(normed, bw.AttentionQ, 1, qDim, embeddingDim)
		k := llmmath.MatMul(normed, bw.AttentionK, 1, kvDim, embeddingDim)
		v := llmmath.MatMul(normed, bw.AttentionV, 1, kvDim, embeddingDim)

		fmt.Printf("  Q: rms=%.6f, K: rms=%.6f, V: rms=%.6f\n", rms32(q), rms32(k), rms32(v))

		// QK norm
		if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
			for h := 0; h < nHeads; h++ {
				qStart := h * headDim
				qVec := q[qStart : qStart+headDim]
				qNormed := llmmath.RMSNorm(qVec, bw.AttentionQNorm, config.LayerNormRmsEps)
				copy(q[qStart:], qNormed)
			}
			for h := 0; h < nKvHeads; h++ {
				kStart := h * headDim
				kVec := k[kStart : kStart+headDim]
				kNormed := llmmath.RMSNorm(kVec, bw.AttentionKNorm, config.LayerNormRmsEps)
				copy(k[kStart:], kNormed)
			}
			fmt.Printf("  After QK norm: Q rms=%.6f, K rms=%.6f\n", rms32(q), rms32(k))
		}

		// RoPE
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
		fmt.Printf("  After RoPE (pos=0): Q rms=%.6f, K rms=%.6f\n", rms32(q), rms32(k))

		// Simulate single-token attention
		// GQA repeat
		vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)

		// Single-token attention: just passes through V after GQA
		attentionOutput := make([]float32, qDim)
		copy(attentionOutput, vRepeated)

		// Output projection
		attnOut := llmmath.MatMul(attentionOutput, bw.AttentionOutput, 1, embeddingDim, qDim)
		fmt.Printf("  Attention output: rms=%.6f, min=%.6f, max=%.6f\n", rms32(attnOut), min32(attnOut), max32(attnOut))

		// Residual
		hiddenStates = llmmath.VectorAdd(residual, attnOut)
		fmt.Printf("  After attn residual: rms=%.6f, min=%.6f, max=%.6f\n", rms32(hiddenStates), min32(hiddenStates), max32(hiddenStates))

		// FFN
		residual2 := make([]float32, len(hiddenStates))
		copy(residual2, hiddenStates)

		normedFFN := llmmath.RMSNorm(hiddenStates, bw.FeedForwardNorm, config.LayerNormRmsEps)
		fmt.Printf("  After ffn_norm: rms=%.6f\n", rms32(normedFFN))

		ffnGate := llmmath.MatMul(normedFFN, bw.FeedForwardGate, 1, ffnDim, embeddingDim)
		ffnUp := llmmath.MatMul(normedFFN, bw.FeedForwardUp, 1, ffnDim, embeddingDim)
		fmt.Printf("  FFN gate: rms=%.6f, FFN up: rms=%.6f\n", rms32(ffnGate), rms32(ffnUp))

		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)
		fmt.Printf("  SwiGLU: rms=%.6f, min=%.6f, max=%.6f\n", rms32(swigluOutput), min32(swigluOutput), max32(swigluOutput))

		ffnOut := llmmath.MatMul(swigluOutput, bw.FeedForwardDown, 1, embeddingDim, ffnDim)
		fmt.Printf("  FFN output: rms=%.6f, min=%.6f, max=%.6f\n", rms32(ffnOut), min32(ffnOut), max32(ffnOut))

		hiddenStates = llmmath.VectorAdd(residual2, ffnOut)
		fmt.Printf("  After FFN residual: rms=%.6f, min=%.6f, max=%.6f\n", rms32(hiddenStates), min32(hiddenStates), max32(hiddenStates))

		// Compute output logits
		var logits []float32
		normedOutput := llmmath.RMSNorm(hiddenStates, weights.OutputNorm, config.LayerNormRmsEps)
		if weights.Output != nil {
			logits = llmmath.MatMul(normedOutput, weights.Output, 1, config.VocabSize, embeddingDim)
		} else {
			logits = llmmath.MatMul(normedOutput, weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
		}

		topIdx := llmmath.Argmax(logits)
		topVal := logits[topIdx]
		fmt.Printf("  Top token: ID=%d, logit=%.4f\n", topIdx, topVal)

		// Check for NaN/Inf
		hasNaN := false
		hasInf := false
		for _, v := range logits {
			if math.IsNaN(float64(v)) {
				hasNaN = true
			}
			if math.IsInf(float64(v), 0) {
				hasInf = true
			}
		}
		if hasNaN || hasInf {
			fmt.Printf("  WARNING: NaN=%v, Inf=%v in logits!\n", hasNaN, hasInf)
		}

		// Show top 5
		type scored struct {
			id    int
			score float32
		}
		var top5 []scored
		for i := 0; i < 5; i++ {
			best := -1
			bestVal := float32(math.Inf(-1))
			for j, v := range logits {
				skip := false
				for _, t := range top5 {
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
				top5 = append(top5, scored{best, bestVal})
			}
		}
		fmt.Printf("  Top 5 tokens: ")
		for _, t := range top5 {
			fmt.Printf("#%d(%.2f) ", t.id, t.score)
		}
		fmt.Println()
	}
}

func min32(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	m := v[0]
	for _, x := range v[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func max32(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func mean32(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	var sum float32
	for _, x := range v {
		sum += x
	}
	return sum / float32(len(v))
}

func rms32(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
