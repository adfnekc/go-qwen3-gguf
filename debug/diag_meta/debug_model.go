package main

import (
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/tokenizer"
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
	
	// Check output_norm weight stats
	var sum, min, max float32 = 0, math.MaxFloat32, -math.MaxFloat32
	for i, v := range weights.OutputNorm {
		sum += v
		if v < min { min = v }
		if v > max { max = v }
		if i < 5 {
			fmt.Printf("output_norm[%d] = %f\n", i, v)
		}
	}
	fmt.Printf("output_norm: mean=%f, min=%f, max=%f\n", sum/float32(len(weights.OutputNorm)), min, max)
	
	// Check token_embd stats (first few dims for first few tokens)
	fmt.Println("\ntoken_embd.weight first few elements:")
	for i := 0; i < 5; i++ {
		fmt.Printf("  embd[%d][0] = %f\n", i, weights.TokenEmbedding[i*config.VocabSize])
	}
	
	// Check layer 0 attn_norm
	b0 := weights.BlockWeights[0]
	sum, min, max = 0, math.MaxFloat32, -math.MaxFloat32
	for i, v := range b0.AttentionNorm {
		sum += v
		if v < min { min = v }
		if v > max { max = v }
		if i < 5 {
			fmt.Printf("blk.0.attn_norm[%d] = %f\n", i, v)
		}
	}
	fmt.Printf("blk.0.attn_norm: mean=%f, min=%f, max=%f\n", sum/float32(len(b0.AttentionNorm)), min, max)
	
	// Check layer 0 attn_q first few dims
	fmt.Println("\nblk.0.attn_q.weight first few elements:")
	qDim := config.AttentionHeadCount * config.AttentionKeyLength
	for i := 0; i < 5; i++ {
		fmt.Printf("  attn_q[%d][0] = %f\n", i, b0.AttentionQ[i*qDim])
	}
	
	// Create model and run forward on "hi"
	mod := model.NewQwen3Model(config, weights)
	tok, _ := tokenizer.NewTokenizerFromGGUF(reader)
	tokens := tok.Encode("hi")
	
	fmt.Printf("\nRunning forward on 'hi' (token %d)...\n", tokens[0])
	logits, err := mod.Forward(tokens, 0)
	if err != nil {
		panic(err)
	}
	
	// Check logit stats
	var logitSum, logitMin, logitMax float32 = 0, math.MaxFloat32, -math.MaxFloat32
	topIdx := make([]int, 5)
	topVals := make([]float32, 5)
	for i := range topVals { topVals[i] = -math.MaxFloat32 }
	
	for i, v := range logits {
		logitSum += v
		if v < logitMin { logitMin = v }
		if v > logitMax { logitMax = v }
		
		// Track top-5
		for j := 0; j < 5; j++ {
			if v > topVals[j] {
				// Shift down
				for k := 4; k > j; k-- {
					topVals[k] = topVals[k-1]
					topIdx[k] = topIdx[k-1]
				}
				topVals[j] = v
				topIdx[j] = i
				break
			}
		}
	}
	
	fmt.Printf("Logits stats: mean=%f, min=%f, max=%f\n", logitSum/float32(len(logits)), logitMin, logitMax)
	fmt.Println("Top 5 logits:")
	for i := 0; i < 5; i++ {
		tokenStr, _ := tok.GetToken(topIdx[i])
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, topIdx[i], tokenStr, topVals[i])
	}
	
	// Check for NaN/Inf
	nanCount := 0
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nanCount++
		}
	}
	fmt.Printf("NaN/Inf count in logits: %d\n", nanCount)
	
	fmt.Println("\n----------------------------------------")
	fmt.Println("Now compare with 'Hello':")
	
	tokens2 := tok.Encode("Hello")
	fmt.Printf("Running forward on 'Hello' (token %d)...\n", tokens2[0])
	logits2, err := mod.Forward(tokens2, 0)
	if err != nil {
		panic(err)
	}
	
	// Check top-5
	topIdx2 := make([]int, 5)
	topVals2 := make([]float32, 5)
	for i := range topVals2 { topVals2[i] = -math.MaxFloat32 }
	
	for i, v := range logits2 {
		for j := 0; j < 5; j++ {
			if v > topVals2[j] {
				for k := 4; k > j; k-- {
					topVals2[k] = topVals2[k-1]
					topIdx2[k] = topIdx2[k-1]
				}
				topVals2[j] = v
				topIdx2[j] = i
				break
			}
		}
	}
	
	fmt.Println("Top 5 logits for 'Hello':")
	for i := 0; i < 5; i++ {
		tokenStr, _ := tok.GetToken(topIdx2[i])
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, topIdx2[i], tokenStr, topVals2[i])
	}
	
	// Check if logits differ
	same := true
	for i := range logits {
		if logits[i] != logits2[i] {
			same = false
			break
		}
	}
	fmt.Printf("Logits are identical between 'hi' and 'Hello': %v\n", same)
}
