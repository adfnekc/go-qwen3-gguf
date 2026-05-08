package main

import (
	"fmt"
	"math"
	"os"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

func main() {
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

	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	headDim := config.AttentionKeyLength
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	embeddingDim := config.EmbeddingLength
	ffnDim := config.FeedForwardLength

	tokenID := 6023 // "hi"
	embedding := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, tokenID)
	hiddenStates := make([]float32, embeddingDim)
	copy(hiddenStates, embedding)

	fmt.Printf("Initial embedding: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(hiddenStates), min32(hiddenStates), max32(hiddenStates))

	for layer := 0; layer < 3; layer++ {
		bw := weights.BlockWeights[layer]

		// Check weight stats
		fmt.Printf("\n=== LAYER %d WEIGHT STATS ===\n", layer)
		fmt.Printf("  attnNorm: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionNorm), min32(bw.AttentionNorm), max32(bw.AttentionNorm), rms32(bw.AttentionNorm))
		fmt.Printf("  attnQ: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionQ), min32(bw.AttentionQ), max32(bw.AttentionQ), rms32(bw.AttentionQ))
		fmt.Printf("  attnK: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionK), min32(bw.AttentionK), max32(bw.AttentionK), rms32(bw.AttentionK))
		fmt.Printf("  attnV: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionV), min32(bw.AttentionV), max32(bw.AttentionV), rms32(bw.AttentionV))
		fmt.Printf("  attnOutput: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionOutput), min32(bw.AttentionOutput), max32(bw.AttentionOutput), rms32(bw.AttentionOutput))
		fmt.Printf("  ffnNorm: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.FeedForwardNorm), min32(bw.FeedForwardNorm), max32(bw.FeedForwardNorm), rms32(bw.FeedForwardNorm))
		fmt.Printf("  ffnGate: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.FeedForwardGate), min32(bw.FeedForwardGate), max32(bw.FeedForwardGate), rms32(bw.FeedForwardGate))
		fmt.Printf("  ffnUp: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.FeedForwardUp), min32(bw.FeedForwardUp), max32(bw.FeedForwardUp), rms32(bw.FeedForwardUp))
		fmt.Printf("  ffnDown: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.FeedForwardDown), min32(bw.FeedForwardDown), max32(bw.FeedForwardDown), rms32(bw.FeedForwardDown))

		fmt.Printf("\n=== LAYER %d FORWARD TRACE ===\n", layer)
		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)
		fmt.Printf("  residual (input): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(residual), min32(residual), max32(residual))

		normed := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
		fmt.Printf("  attn_norm output: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(normed), min32(normed), max32(normed))

		q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kvDim, embeddingDim)
		fmt.Printf("  Q (pre-norm): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(q), min32(q), max32(q))
		fmt.Printf("  K (pre-norm): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(k), min32(k), max32(k))
		fmt.Printf("  V: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(v), min32(v), max32(v))

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
		fmt.Printf("  Q (post-norm): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(q), min32(q), max32(q))
		fmt.Printf("  K (post-norm): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(k), min32(k), max32(k))

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
		fmt.Printf("  Q (post-RoPE): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(q), min32(q), max32(q))
		fmt.Printf("  K (post-RoPE): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(k), min32(k), max32(k))

		// Single token attention - with self, prob = 1.0, output = V
		vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)
		fmt.Printf("  V (repeated): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(vRepeated), min32(vRepeated), max32(vRepeated))

		attnOut := llmmath.MatMulTransposed(vRepeated, bw.AttentionOutput, 1, embeddingDim, qDim)
		fmt.Printf("  attn_out (pre-residual): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(attnOut), min32(attnOut), max32(attnOut))

		hiddenStates = llmmath.VectorAdd(residual, attnOut)
		fmt.Printf("  after attention residual: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(hiddenStates), min32(hiddenStates), max32(hiddenStates))

		// FFN
		residual = make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normedFFN := llmmath.RMSNorm(hiddenStates, bw.FeedForwardNorm, config.LayerNormRmsEps)
		fmt.Printf("  ffn_norm output: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(normedFFN), min32(normedFFN), max32(normedFFN))

		ffnGate := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardGate, 1, ffnDim, embeddingDim)
		ffnUp := llmmath.MatMulTransposed(normedFFN, bw.FeedForwardUp, 1, ffnDim, embeddingDim)
		if ffnGate == nil || ffnUp == nil {
			fmt.Printf("  ERROR: ffnGate or ffnUp is nil!\n")
			fmt.Printf("    len(normedFFN)=%d, len(gate)=%d, ffnDim=%d, embeddingDim=%d\n", len(normedFFN), len(bw.FeedForwardGate), ffnDim, embeddingDim)
			os.Exit(1)
		}
		fmt.Printf("  ffn_gate: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(ffnGate), min32(ffnGate), max32(ffnGate))
		fmt.Printf("  ffn_up: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(ffnUp), min32(ffnUp), max32(ffnUp))

		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)
		fmt.Printf("  swiglu_output: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(swigluOutput), min32(swigluOutput), max32(swigluOutput))

		ffnOut := llmmath.MatMulTransposed(swigluOutput, bw.FeedForwardDown, 1, embeddingDim, ffnDim)
		if ffnOut == nil {
			fmt.Printf("  ERROR: ffnOut is nil!\n")
			fmt.Printf("    len(swiglu)=%d, len(down)=%d, embeddingDim=%d, ffnDim=%d\n", len(swigluOutput), len(bw.FeedForwardDown), embeddingDim, ffnDim)
			os.Exit(1)
		}
		fmt.Printf("  ffn_out (pre-residual): RMS=%.6f, min=%.6f, max=%.6f\n", rms32(ffnOut), min32(ffnOut), max32(ffnOut))

		hiddenStates = llmmath.VectorAdd(residual, ffnOut)
		fmt.Printf("  after FFN residual: RMS=%.6f, min=%.6f, max=%.6f\n", rms32(hiddenStates), min32(hiddenStates), max32(hiddenStates))
	}
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

func rms32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	var sum float32
	for _, x := range v { sum += x * x }
	return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
