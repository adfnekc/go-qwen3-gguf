package main

import (
	"fmt"
	"math"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
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

	// Print first 10 values of token_embd for first 3 tokens
	fmt.Println("=== Token Embedding Values ===")
	for tok := 0; tok < 3; tok++ {
		rms := rms32(weights.TokenEmbedding[tok*config.EmbeddingLength : (tok+1)*config.EmbeddingLength])
		fmt.Printf("Token %d embedding (first 10 of %d, RMS=%.6f):\n", tok, config.EmbeddingLength, rms)
		for i := 0; i < 10; i++ {
			fmt.Printf("  [%d] = %.6f\n", i, weights.TokenEmbedding[tok*config.EmbeddingLength+i])
		}
	}

	// Check token 6023 ("hi")
	fmt.Printf("\nToken 6023 'hi' embedding (first 10, RMS=%.6f):\n", rms32(weights.TokenEmbedding[6023*config.EmbeddingLength:6023*config.EmbeddingLength+config.EmbeddingLength]))
	for i := 0; i < 10; i++ {
		fmt.Printf("  [%d] = %.6f\n", i, weights.TokenEmbedding[6023*config.EmbeddingLength+i])
	}

	// Check output_norm
	fmt.Println("\n=== Output Norm ===")
	fmt.Printf("Output norm (first 10 of %d):\n", len(weights.OutputNorm))
	for i := 0; i < 10; i++ {
		fmt.Printf("  [%d] = %.6f\n", i, weights.OutputNorm[i])
	}
	fmt.Printf("RMS = %.6f\n", rms32(weights.OutputNorm))

	// Check if embeddings look reasonable (not all zeros, not NaN)
	fmt.Println("\n=== Sanity Checks ===")
	emb := weights.TokenEmbedding[6023*config.EmbeddingLength : 6023*config.EmbeddingLength+config.EmbeddingLength]
	hasNaN := false
	hasInf := false
	allZero := true
	for _, v := range emb {
		if math.IsNaN(float64(v)) {
			hasNaN = true
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
		}
		if v != 0 {
			allZero = false
		}
	}
	fmt.Printf("Token 6023 embedding: hasNaN=%v, hasInf=%v, allZero=%v, RMS=%.6f\n", hasNaN, hasInf, allZero, rms32(emb))

	// Check emb[0:5] RMS across all tokens
	fmt.Println("\n=== Token Embedding RMS Across Vocab ===")
	minRMS := float32(math.MaxFloat32)
	maxRMS := float32(-math.MaxFloat32)
	totalRMS := float32(0)
	for tok := 0; tok < config.VocabSize; tok++ {
		start := tok * config.EmbeddingLength
		r := rms32(weights.TokenEmbedding[start : start+config.EmbeddingLength])
		if r < minRMS {
			minRMS = r
		}
		if r > maxRMS {
			maxRMS = r
		}
		totalRMS += r
	}
	avgRMS := totalRMS / float32(config.VocabSize)
	fmt.Printf("Embedding RMS across %d tokens: min=%.6f, max=%.6f, avg=%.6f\n", config.VocabSize, minRMS, maxRMS, avgRMS)

	// Check token ID 0's embedding shape
	fmt.Printf("\nTokenEmbedding length: %d (expect %d)\n", len(weights.TokenEmbedding), config.VocabSize*config.EmbeddingLength)
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
