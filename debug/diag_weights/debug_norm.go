package main

import (
	"fmt"
	"math"
	"os"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
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

	_ = config.AttentionKeyLength

	// Check QK norm weights for the first few layers
	for layer := 0; layer < 4; layer++ {
		bw := weights.BlockWeights[layer]
		fmt.Printf("Layer %d:\n", layer)
		fmt.Printf("  attn_q_norm: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionQNorm), min32(bw.AttentionQNorm), max32(bw.AttentionQNorm), rms32(bw.AttentionQNorm))
		fmt.Printf("  attn_k_norm: len=%d, min=%.6f, max=%.6f, rms=%.6f\n", len(bw.AttentionKNorm), min32(bw.AttentionKNorm), max32(bw.AttentionKNorm), rms32(bw.AttentionKNorm))
		if len(bw.AttentionQNorm) >= 4 {
			fmt.Printf("    First 4 q_norm values: [%.4f, %.4f, %.4f, %.4f]\n", bw.AttentionQNorm[0], bw.AttentionQNorm[1], bw.AttentionQNorm[2], bw.AttentionQNorm[3])
		}
		if len(bw.AttentionKNorm) >= 4 {
			fmt.Printf("    First 4 k_norm values: [%.4f, %.4f, %.4f, %.4f]\n", bw.AttentionKNorm[0], bw.AttentionKNorm[1], bw.AttentionKNorm[2], bw.AttentionKNorm[3])
		}
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
