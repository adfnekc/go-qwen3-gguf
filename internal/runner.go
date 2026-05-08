package internal

import (
	"fmt"
	"time"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/tokenizer"
)

type RunOptions struct {
	ForceNoMMap bool
	UseChat     bool
	Debug       bool
	Temperature float32
	MaxTokens   int
}

type Runner struct{}

func NewRunner() *Runner { return &Runner{} }

func (r *Runner) Run(modelPath, inputText string, opts RunOptions) error {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 10
	}

	fmt.Println("========================================")
	fmt.Println("GGUF Model Inference")
	fmt.Println("========================================")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Input: %s\n", inputText)
	fmt.Printf("Max New Tokens: %d\n", opts.MaxTokens)
	fmt.Println()

	var reader *gguf.GGUFReader
	var mmapFile *gguf.MMapFile
	var config *model.Qwen3Config
	var weights *model.Qwen3Weights
	var err error
	var usingMMap bool

	fmt.Println("Loading GGUF file...")
	startTime := time.Now()

	reader = gguf.NewGGUFReader()
	if opts.ForceNoMMap {
		fmt.Println("  Using regular file loading (--no-mmap)")
		if err := reader.LoadFromFile(modelPath); err != nil {
			return fmt.Errorf("failed to load GGUF file: %w", err)
		}
		usingMMap = false
	} else {
		fmt.Println("  Trying mmap loading...")
		mmapFile, err = reader.LoadFromFileMMap(modelPath)
		if err != nil {
			fmt.Printf("Warning: Failed to load with mmap: %v\n", err)
			fmt.Println("Falling back to regular file loading...")
			reader = gguf.NewGGUFReader()
			if err := reader.LoadFromFile(modelPath); err != nil {
				return fmt.Errorf("failed to load GGUF file: %w", err)
			}
			usingMMap = false
		} else {
			usingMMap = true
			defer mmapFile.Close()
		}
	}

	loadTime := time.Since(startTime)
	fmt.Printf("GGUF file loaded successfully in %v!\n", loadTime)
	method := "regular file"
	if usingMMap {
		method = "mmap"
	}
	fmt.Printf("Loading method: %s\n", method)
	fmt.Println()

	reader.PrintInfo()

	fmt.Println("\n========================================")
	fmt.Println("Loading Model Configuration")
	fmt.Println("========================================")

	config, err = model.LoadQwen3Config(reader)
	if err != nil {
		return fmt.Errorf("failed to load model config: %w", err)
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
		return fmt.Errorf("failed to load weights: %w", err)
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
		fmt.Printf("Warning: Failed to load BPE tokenizer: %v\n", err)
		fmt.Println("Falling back to simple byte tokenizer...")
		simpleTok := tokenizer.NewSimpleTokenizer()
		tokens := simpleTok.Encode(inputText)
		runInference(mod, simpleTok, tokens, opts)
		return nil
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

	if opts.UseChat {
		chatTokens := model.ApplyQwen3ChatTemplate(tok, inputText)
		fmt.Printf("\nChat template applied:\n")
		fmt.Printf("  Encoded tokens: %v\n", chatTokens)
		fmt.Printf("  Token count: %d\n", len(chatTokens))
		tokens = chatTokens
	}

	if len(tokens) > 0 && !opts.Debug {
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

	runInference(mod, tok, tokens, opts)
	return nil
}

func runInference(mod *model.Qwen3Model, tok tokenizer.Tokenizer, tokens []int, opts RunOptions) {
	fmt.Println("\n========================================")
	fmt.Println("Running Inference")
	fmt.Println("========================================")
	fmt.Printf("Generating %d new tokens...\n", opts.MaxTokens)

	inferenceStartTime := time.Now()

	generatedTokens, err := mod.Generate(tokens, opts.MaxTokens, opts.Temperature)
	if err != nil {
		fmt.Printf("Inference failed: %v\n", err)
		return
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

	fmt.Printf("\nToken-by-token decoding:\n")
	for i, id := range generatedTokens {
		if tokenStr, ok := tok.GetToken(id); ok {
			fmt.Printf("  Token %d: ID=%d, Text=%q\n", i, id, tokenStr)
		}
	}

	decodedText := tok.Decode(generatedTokens)
	fmt.Printf("\nDecoded text: %s\n", decodedText)

	if len(generatedTokens) > len(tokens) {
		newTokens := generatedTokens[len(tokens):]
		newDecoded := tok.Decode(newTokens)
		fmt.Printf("\nNewly generated tokens (%d):\n", len(newTokens))
		fmt.Printf("  Token IDs: %v\n", newTokens)
		fmt.Printf("  Decoded: %s\n", newDecoded)
	}

	fmt.Println("\n========================================")
	fmt.Println("Inference Complete")
	fmt.Println("========================================")
}
