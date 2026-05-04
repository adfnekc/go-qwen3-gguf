package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"gguf/gguf"
	"gguf/model"
	"gguf/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run infer.go <model_path> [input_text]")
		fmt.Println("Example: go run infer.go /mnt/d/model/Qwen3-0.6B-Q8_0.gguf \"Hello\"")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	inputText := "Hello, world!"

	if len(os.Args) > 2 {
		inputText = os.Args[2]
	}

	maxNewTokens := 10
	temperature := float32(0.0)

	fmt.Println("========================================")
	fmt.Println("GGUF Model Inference")
	fmt.Println("========================================")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Input: %s\n", inputText)
	fmt.Printf("Max New Tokens: %d\n", maxNewTokens)
	fmt.Println()

	var reader *gguf.GGUFReader
	var mmapFile *gguf.MMapFile
	var config *model.Qwen3Config
	var weights *model.Qwen3Weights
	var err error
	var usingMMap bool

	fmt.Println("Loading GGUF file with mmap (default)...")
	startTime := time.Now()

	reader = gguf.NewGGUFReader()
	mmapFile, err = reader.LoadFromFileMMap(modelPath)
	if err != nil {
		log.Printf("Warning: Failed to load with mmap: %v", err)
		fmt.Println("Falling back to regular file loading...")
		
		reader = gguf.NewGGUFReader()
		if err := reader.LoadFromFile(modelPath); err != nil {
			log.Fatalf("Failed to load GGUF file: %v", err)
		}
		usingMMap = false
	} else {
		usingMMap = true
		defer mmapFile.Close()
	}

	loadTime := time.Since(startTime)
	fmt.Printf("GGUF file loaded successfully in %v!\n", loadTime)
	fmt.Printf("Loading method: %s\n", map[bool]string{true: "mmap", false: "regular file"}[usingMMap])
	fmt.Println()

	reader.PrintInfo()

	fmt.Println("\n========================================")
	fmt.Println("Loading Model Configuration")
	fmt.Println("========================================")

	config, err = model.LoadQwen3Config(reader)
	if err != nil {
		log.Fatalf("Failed to load model config: %v", err)
	}

	fmt.Printf("Model Config:\n")
	fmt.Printf("  Context Length: %d\n", config.ContextLength)
	fmt.Printf("  Embedding Length: %d\n", config.EmbeddingLength)
	fmt.Printf("  Block Count: %d\n", config.BlockCount)
	fmt.Printf("  Feed Forward Length: %d\n", config.FeedForwardLength)
	fmt.Printf("  Attention Heads: %d\n", config.AttentionHeadCount)
	fmt.Printf("  Attention Heads (KV): %d\n", config.AttentionHeadCountKv)
	fmt.Printf("  Attention Key Length: %d\n", config.AttentionKeyLength)
	fmt.Printf("  Attention Value Length: %d\n", config.AttentionValueLength)
	fmt.Printf("  RoPE Freq Base: %g\n", config.RopeFreqBase)
	fmt.Printf("  RMS Eps: %g\n", config.LayerNormRmsEps)
	fmt.Printf("  Vocab Size: %d\n", config.VocabSize)
	fmt.Printf("  QKV RMSNorm: %v\n", config.QKVRMSNorm)

	fmt.Println("\n========================================")
	fmt.Println("Loading Model Weights")
	fmt.Println("========================================")
	fmt.Println("This may take a moment for large models...")

	weightsStartTime := time.Now()

	if usingMMap && mmapFile != nil {
		weights, err = model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	} else {
		weights, err = model.LoadQwen3Weights(reader, config)
	}

	if err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	weightsLoadTime := time.Since(weightsStartTime)
	fmt.Printf("Weights loaded successfully in %v!\n", weightsLoadTime)

	mod := model.NewQwen3Model(config, weights)
	fmt.Println("Model created successfully!")

	fmt.Println("\n========================================")
	fmt.Println("Loading Tokenizer")
	fmt.Println("========================================")

	tok, err := tokenizer.NewTokenizerFromGGUF(reader)
	if err != nil {
		log.Printf("Warning: Failed to load BPE tokenizer: %v", err)
		fmt.Println("Falling back to simple byte tokenizer...")
		simpleTok := tokenizer.NewSimpleTokenizer()
		tokens := simpleTok.Encode(inputText)
		runInference(mod, simpleTok, tokens, maxNewTokens, temperature)
		return
	}

	fmt.Printf("Tokenizer loaded successfully!\n")
	fmt.Printf("Vocab Size: %d\n", tok.GetVocabSize())

	fmt.Println("\n========================================")
	fmt.Println("Encoding Input Text")
	fmt.Println("========================================")

	tokens := tok.Encode(inputText)
	fmt.Printf("Input text: %s\n", inputText)
	fmt.Printf("Encoded tokens: %v\n", tokens)
	fmt.Printf("Token count: %d\n", len(tokens))

	if len(tokens) > 0 {
		fmt.Printf("\nFirst few tokens decoded:\n")
		for i := 0; i < len(tokens) && i < 5; i++ {
			tokenStr, ok := tok.GetToken(tokens[i])
			if ok {
				fmt.Printf("  Token %d: ID=%d, Text=%q\n", i, tokens[i], tokenStr)
			} else {
				fmt.Printf("  Token %d: ID=%d, Text=<unknown>\n", i, tokens[i])
			}
		}
	}

	runInference(mod, tok, tokens, maxNewTokens, temperature)
}

func runInference(mod *model.Qwen3Model, tok interface{}, tokens []int, maxNewTokens int, temperature float32) {
	fmt.Println("\n========================================")
	fmt.Println("Running Inference")
	fmt.Println("========================================")
	fmt.Printf("Generating %d new tokens...\n", maxNewTokens)

	inferenceStartTime := time.Now()

	generatedTokens, err := mod.Generate(tokens, maxNewTokens, temperature)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	inferenceTime := time.Since(inferenceStartTime)
	fmt.Printf("Inference completed in %v\n", inferenceTime)

	fmt.Println("\n========================================")
	fmt.Println("Inference Results")
	fmt.Println("========================================")
	fmt.Printf("Total tokens generated: %d\n", len(generatedTokens))
	fmt.Printf("Input tokens: %d\n", len(tokens))
	fmt.Printf("New tokens: %d\n", len(generatedTokens)-len(tokens))
	fmt.Printf("Generated token IDs: %v\n", generatedTokens)

	var decodedText string
	switch t := tok.(type) {
	case *tokenizer.Tokenizer:
		decodedText = t.Decode(generatedTokens)
	case *tokenizer.SimpleTokenizer:
		decodedText = t.Decode(generatedTokens)
	}

	fmt.Printf("\nDecoded text: %s\n", decodedText)

	if len(generatedTokens) > len(tokens) {
		newTokens := generatedTokens[len(tokens):]
		var newDecoded string
		switch t := tok.(type) {
		case *tokenizer.Tokenizer:
			newDecoded = t.Decode(newTokens)
		case *tokenizer.SimpleTokenizer:
			newDecoded = t.Decode(newTokens)
		}
		fmt.Printf("\nNewly generated tokens (%d):\n", len(newTokens))
		fmt.Printf("  Token IDs: %v\n", newTokens)
		fmt.Printf("  Decoded: %s\n", newDecoded)
	}

	fmt.Println("\n========================================")
	fmt.Println("Inference Complete")
	fmt.Println("========================================")
}
