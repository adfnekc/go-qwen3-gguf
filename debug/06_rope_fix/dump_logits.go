package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/tokenizer"
)

type kv struct {
	Key   int
	Value float32
}

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

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

	tok, err := tokenizer.NewTokenizerFromGGUF(reader)
	if err != nil {
		panic(err)
	}

	imStart, imEnd := 151644, 151645
	input := "hi"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		input = os.Args[2]
	}

	prompt := []int{imStart}
	prompt = append(prompt, tok.Encode("user\n")...)
	prompt = append(prompt, tok.Encode(input)...)
	prompt = append(prompt, imEnd)
	prompt = append(prompt, 198)
	prompt = append(prompt, imStart)
	prompt = append(prompt, tok.Encode("assistant\n")...)

	fmt.Printf("Prompt tokens: %v\n", prompt)
	fmt.Printf("Prompt length: %d\n", len(prompt))

	m := model.NewQwen3Model(config, weights)

	// Step 1: prompt only
	logits, err := m.Forward(prompt, 0)
	if err != nil {
		panic(err)
	}
	dumpTopK("After prompt (step 1)", logits, tok, 20)

	// Step 2: prompt + <think>
	firstTok := 151667 // <think>
	logits2, err := m.Forward([]int{firstTok}, len(prompt))
	if err != nil {
		panic(err)
	}
	dumpTopK("After <think> (step 2)", logits2, tok, 20)
}

func dumpTopK(label string, logits []float32, tok *tokenizer.Tokenizer, k int) {
	pairs := make([]kv, len(logits))
	for i, v := range logits {
		pairs[i] = kv{i, v}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Value > pairs[j].Value
	})

	fmt.Printf("\n=== %s ===\n", label)
	for i := 0; i < k && i < len(pairs); i++ {
		id := pairs[i].Key
		s, _ := tok.GetToken(id)
		fmt.Printf("  %5d %-25q %.6f\n", id, s, pairs[i].Value)
	}
}
