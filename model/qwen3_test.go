package model

import (
	"fmt"
	"testing"

	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

func TestGenerateWithMockWeights(t *testing.T) {
	vocabSize := 100
	embeddingDim := 64
	contextLength := 512
	blockCount := 2
	headCount := 4
	headCountKv := 2
	headDim := 16
	ffnDim := 128

	config := &Qwen3Config{
		VocabSize:           vocabSize,
		EmbeddingLength:     embeddingDim,
		ContextLength:       contextLength,
		BlockCount:          blockCount,
		AttentionHeadCount:  headCount,
		AttentionHeadCountKv: headCountKv,
		AttentionKeyLength:  headDim,
		AttentionValueLength: headDim,
		FeedForwardLength:   ffnDim,
		RopeFreqBase:        10000.0,
		LayerNormRmsEps:     1e-6,
		QKVRMSNorm:          false,
	}

	weights := createMockWeights(config)

	model := NewQwen3Model(config, weights)

	inputTokens := []int{10, 20, 30}
	maxNewTokens := 5
	temperature := float32(0.0)

	generated, err := model.Generate(inputTokens, maxNewTokens, temperature)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if len(generated) <= len(inputTokens) {
		t.Errorf("Expected generated tokens > input tokens (%d), got %d", len(inputTokens), len(generated))
	}

	fmt.Printf("Input tokens: %v\n", inputTokens)
	fmt.Printf("Generated tokens: %v\n", generated)
	fmt.Printf("New tokens: %v\n", generated[len(inputTokens):])

	for i, token := range generated {
		if token < 0 || token >= vocabSize {
			t.Errorf("Token %d at position %d is out of range [0, %d)", token, i, vocabSize)
		}
	}

	allZero := true
	for _, token := range generated[len(inputTokens):] {
		if token != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Errorf("All generated tokens are zero, which indicates a bug")
	}
}

func TestGenerateWithTemperature(t *testing.T) {
	vocabSize := 100
	embeddingDim := 64
	contextLength := 512
	blockCount := 2
	headCount := 4
	headCountKv := 2
	headDim := 16
	ffnDim := 128

	config := &Qwen3Config{
		VocabSize:           vocabSize,
		EmbeddingLength:     embeddingDim,
		ContextLength:       contextLength,
		BlockCount:          blockCount,
		AttentionHeadCount:  headCount,
		AttentionHeadCountKv: headCountKv,
		AttentionKeyLength:  headDim,
		AttentionValueLength: headDim,
		FeedForwardLength:   ffnDim,
		RopeFreqBase:        10000.0,
		LayerNormRmsEps:     1e-6,
		QKVRMSNorm:          false,
	}

	weights := createMockWeights(config)
	model := NewQwen3Model(config, weights)

	inputTokens := []int{10, 20, 30}
	maxNewTokens := 10
	temperature := float32(0.8)

	generated, err := model.Generate(inputTokens, maxNewTokens, temperature)
	if err != nil {
		t.Fatalf("Generate with temperature failed: %v", err)
	}

	if len(generated) <= len(inputTokens) {
		t.Errorf("Expected generated tokens > input tokens (%d), got %d", len(inputTokens), len(generated))
	}

	fmt.Printf("Temperature sampling test:\n")
	fmt.Printf("  Input tokens: %v\n", inputTokens)
	fmt.Printf("  Generated tokens: %v\n", generated)
	fmt.Printf("  New tokens: %v\n", generated[len(inputTokens):])

	for i, token := range generated {
		if token < 0 || token >= vocabSize {
			t.Errorf("Token %d at position %d is out of range [0, %d)", token, i, vocabSize)
		}
	}

	allZero := true
	for _, token := range generated[len(inputTokens):] {
		if token != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Errorf("All generated tokens are zero with temperature sampling, which indicates a bug")
	}
}

func TestArgmax(t *testing.T) {
	logits := []float32{1.0, 3.0, 2.0, 5.0, 4.0}
	idx := llmmath.Argmax(logits)
	if idx != 3 {
		t.Errorf("Argmax returned %d, expected 3", idx)
	}

	allZeros := []float32{0.0, 0.0, 0.0, 0.0}
	idx2 := llmmath.Argmax(allZeros)
	if idx2 != 0 {
		t.Errorf("Argmax with all zeros returned %d, expected 0", idx2)
	}
}

func TestSampleCategorical(t *testing.T) {
	probs := []float32{0.1, 0.2, 0.3, 0.4}
	
	samples := make(map[int]int)
	for i := 0; i < 1000; i++ {
		idx := llmmath.SampleCategorical(probs)
		if idx < 0 || idx >= len(probs) {
			t.Errorf("SampleCategorical returned %d, expected in range [0, %d)", idx, len(probs))
		}
		samples[idx]++
	}

	expected := []float32{100, 200, 300, 400}
	tolerance := 50

	for i, exp := range expected {
		count := float32(samples[i])
		if count < exp-float32(tolerance) || count > exp+float32(tolerance) {
			t.Logf("Warning: Category %d sampled %d times, expected ~%d (within ±%d)", i, samples[i], int(exp), tolerance)
		}
	}
}

func createMockWeights(config *Qwen3Config) *Qwen3Weights {
	weights := &Qwen3Weights{
		TokenEmbedding: make([]float32, config.EmbeddingLength*config.VocabSize),
		OutputNorm:     make([]float32, config.EmbeddingLength),
		Output:         make([]float32, config.EmbeddingLength*config.VocabSize),
		BlockWeights:   make([]*Qwen3BlockWeights, config.BlockCount),
	}

	for i := range weights.TokenEmbedding {
		weights.TokenEmbedding[i] = float32(i%100) / 100.0
	}

	for i := range weights.OutputNorm {
		weights.OutputNorm[i] = 1.0
	}

	for i := range weights.Output {
		weights.Output[i] = float32(i%100) / 100.0
	}

	for b := 0; b < config.BlockCount; b++ {
		block := &Qwen3BlockWeights{
			AttentionNorm:    make([]float32, config.EmbeddingLength),
			AttentionQ:       make([]float32, config.EmbeddingLength*config.AttentionHeadCount*config.AttentionKeyLength),
			AttentionK:       make([]float32, config.EmbeddingLength*config.AttentionHeadCountKv*config.AttentionKeyLength),
			AttentionV:       make([]float32, config.EmbeddingLength*config.AttentionHeadCountKv*config.AttentionValueLength),
			AttentionOutput:  make([]float32, config.AttentionHeadCount*config.AttentionKeyLength*config.EmbeddingLength),
			FeedForwardNorm:  make([]float32, config.EmbeddingLength),
			FeedForwardGate:  make([]float32, config.EmbeddingLength*config.FeedForwardLength),
			FeedForwardUp:    make([]float32, config.EmbeddingLength*config.FeedForwardLength),
			FeedForwardDown:  make([]float32, config.FeedForwardLength*config.EmbeddingLength),
		}

		for i := range block.AttentionNorm {
			block.AttentionNorm[i] = 1.0
		}
		for i := range block.AttentionQ {
			block.AttentionQ[i] = float32(i%50) / 50.0
		}
		for i := range block.AttentionK {
			block.AttentionK[i] = float32(i%50) / 50.0
		}
		for i := range block.AttentionV {
			block.AttentionV[i] = float32(i%50) / 50.0
		}
		for i := range block.AttentionOutput {
			block.AttentionOutput[i] = float32(i%50) / 50.0
		}
		for i := range block.FeedForwardNorm {
			block.FeedForwardNorm[i] = 1.0
		}
		for i := range block.FeedForwardGate {
			block.FeedForwardGate[i] = float32(i%50) / 50.0
		}
		for i := range block.FeedForwardUp {
			block.FeedForwardUp[i] = float32(i%50) / 50.0
		}
		for i := range block.FeedForwardDown {
			block.FeedForwardDown[i] = float32(i%50) / 50.0
		}

		weights.BlockWeights[b] = block
	}

	return weights
}
