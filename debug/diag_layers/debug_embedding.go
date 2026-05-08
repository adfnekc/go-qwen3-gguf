package main

import (
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
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
	
	// Test: compute output projection WITHOUT any layers
	// Just embed → norm → project
	
	// "hi" = token 6023 (from GPT-2 BPE)
	emb := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, config.EmbeddingLength, 6023)
	
	// RMSNorm
	normed := llmmath.RMSNorm(emb, weights.OutputNorm, config.LayerNormRmsEps)
	
	// Output projection
	var logits []float32
	if weights.Output != nil {
		logits = llmmath.MatMul(normed, weights.Output, 1, config.VocabSize, config.EmbeddingLength)
	} else {
		logits = llmmath.MatMul(normed, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	}
	
	// Top 10
	type TokenScore struct {
		idx int
		val float32
	}
	top := make([]TokenScore, 10)
	for i := range top { top[i].val = -math.MaxFloat32 }
	
	for i, v := range logits {
		for j := 0; j < 10; j++ {
			if v > top[j].val {
				for k := 9; k > j; k-- {
					top[k] = top[k-1]
				}
				top[j] = TokenScore{i, v}
				break
			}
		}
	}
	
	fmt.Println("Direct embedding → norm → output projection (NO model layers):")
	fmt.Println("Top 10 tokens:")
	tok, _ := reader.GetMetadata("tokenizer.ggml.tokens")
	vocab := tok.([]string)
	for i, ts := range top {
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, ts.idx, vocab[ts.idx], ts.val)
	}
	
	// Also do this for the actual model forward to compare
	mod := model.NewQwen3Model(config, weights)
	fullLogits, err := mod.Forward([]int{6023}, 0)
	if err != nil {
		panic(err)
	}
	
	top2 := make([]TokenScore, 10)
	for i := range top2 { top2[i].val = -math.MaxFloat32 }
	for i, v := range fullLogits {
		for j := 0; j < 10; j++ {
			if v > top2[j].val {
				for k := 9; k > j; k-- {
					top2[k] = top2[k-1]
				}
				top2[j] = TokenScore{i, v}
				break
			}
		}
	}
	
	fmt.Println("\nFull model forward (28 layers):")
	fmt.Println("Top 10 tokens:")
	for i, ts := range top2 {
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, ts.idx, vocab[ts.idx], ts.val)
	}
}
