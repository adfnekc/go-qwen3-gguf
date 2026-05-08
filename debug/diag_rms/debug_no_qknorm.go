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

	tokenID := 6023
	embedding := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, tokenID)
	hiddenStates := make([]float32, embeddingDim)
	copy(hiddenStates, embedding)
	fmt.Printf("Initial embedding RMS: %.6f\n", rms32(hiddenStates))

	// Replace QK norm weights with ones
	ones := make([]float32, headDim)
	for i := range ones {
		ones[i] = 1.0
	}

	for layer := 0; layer < 3; layer++ {
		bw := weights.BlockWeights[layer]
		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normed := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
		q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normed, bw.AttentionV, 1, kvDim, embeddingDim)

		// Use ones instead of actual QK norm weights
		for h := 0; h < nHeads; h++ {
			qStart := h * headDim
			qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], ones, config.LayerNormRmsEps)
			copy(q[qStart:], qNormed)
		}
		for h := 0; h < nKvHeads; h++ {
			kStart := h * headDim
			kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], ones, config.LayerNormRmsEps)
			copy(k[kStart:], kNormed)
		}

		// RoPE at pos 0
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

		fmt.Printf("Layer %d: RMS=%.6f\n", layer, rms32(hiddenStates))
	}
}

func rms32(v []float32) float32 {
	if len(v) == 0 { return 0 }
	var sum float32
	for _, x := range v { sum += x * x }
	return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
