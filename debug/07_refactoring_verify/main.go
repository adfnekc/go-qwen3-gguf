package main

import (
	"fmt"
	"os"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run debug/07_refactoring_verify/main.go <model.gguf>")
		os.Exit(1)
	}
	modelPath := os.Args[1]

	reader := gguf.NewGGUFReader()
	if err := reader.LoadFromFile(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load: %v\n", err)
		os.Exit(1)
	}

	config, err := model.LoadQwen3Config(reader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config: %v\n", err)
		os.Exit(1)
	}

	weights, err := model.LoadQwen3Weights(reader, config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load weights: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Config:\n")
	fmt.Printf("  ContextLength: %d\n", config.ContextLength)
	fmt.Printf("  EmbeddingLength: %d\n", config.EmbeddingLength)
	fmt.Printf("  BlockCount: %d\n", config.BlockCount)
	fmt.Printf("  FeedForwardLength: %d\n", config.FeedForwardLength)
	fmt.Printf("  AttentionHeadCount: %d\n", config.AttentionHeadCount)
	fmt.Printf("  AttentionHeadCountKv: %d\n", config.AttentionHeadCountKv)
	fmt.Printf("  AttentionKeyLength: %d\n", config.AttentionKeyLength)
	fmt.Printf("  RopeFreqBase: %g\n", config.RopeFreqBase)
	fmt.Printf("  LayerNormRmsEps: %g\n", config.LayerNormRmsEps)
	fmt.Printf("  VocabSize: %d\n", config.VocabSize)
	fmt.Printf("  QKVRMSNorm: %v\n", config.QKVRMSNorm)
	fmt.Println()

	mod := model.NewQwen3Model(config, weights)
	fmt.Printf("Model created: OK\n")

	fmt.Printf("\n--- Forward pass test (token IDs [44, 402]) ---\n")
	logits, err := mod.Forward([]int{44, 402}, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Forward failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Logits length: %d\n", len(logits))

	topK := 10
	fmt.Printf("Top-%d logits:\n", topK)
	indices := make([]int, len(logits))
	for i := range logits {
		indices[i] = i
	}
	for i := 0; i < len(logits)-1; i++ {
		for j := i + 1; j < len(logits); j++ {
			if logits[indices[j]] > logits[indices[i]] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}
	for i := 0; i < topK && i < len(indices); i++ {
		fmt.Printf("  [%5d] %8.4f\n", indices[i], logits[indices[i]])
	}

	fmt.Printf("\n--- Generate test (\"hi\", temp=0, tokens=5) ---\n")
	generated, err := mod.Generate([]int{44, 402}, 5, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Generate failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Generated tokens: %v\n", generated)
	fmt.Printf("Input: [44, 402], Output: %v\n", generated[2:])
	fmt.Println("\nAll checks passed!")
}
