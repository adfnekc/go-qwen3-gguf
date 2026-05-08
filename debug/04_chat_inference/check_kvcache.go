package main

import (
	"fmt"
	"os"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/tokenizer"
)

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
	prompt := []int{imStart}
	prompt = append(prompt, tok.Encode("user\n")...)
	prompt = append(prompt, tok.Encode("hi")...)
	prompt = append(prompt, imEnd)
	prompt = append(prompt, 198)
	prompt = append(prompt, imStart)
	prompt = append(prompt, tok.Encode("assistant\n")...)

	// Generate step-by-step with KV cache, storing logits at each step
	m := model.NewQwen3Model(config, weights)
	cacheLogits := make([][]float32, 0)
	generated := []int{}

	// Step 0: prompt → predict <think>
	logits, err := m.Forward(prompt, 0)
	if err != nil {
		panic(err)
	}
	cacheLogits = append(cacheLogits, logits)
	tok0 := argmax(logits)
	generated = append(generated, tok0)

	// Step 1: prompt + <think>
	logits, err = m.Forward([]int{tok0}, len(prompt))
	if err != nil {
		panic(err)
	}
	cacheLogits = append(cacheLogits, logits)
	tok1 := argmax(logits)
	generated = append(generated, tok1)

	// Step 2: prompt + <think> + \n
	logits, err = m.Forward([]int{tok1}, len(prompt)+1)
	if err != nil {
		panic(err)
	}
	cacheLogits = append(cacheLogits, logits)
	tok2 := argmax(logits)
	generated = append(generated, tok2)

	// Step 3: prompt + <think> + \n + \n
	logits, err = m.Forward([]int{tok2}, len(prompt)+2)
	if err != nil {
		panic(err)
	}
	cacheLogits = append(cacheLogits, logits)
	tok3 := argmax(logits)
	generated = append(generated, tok3)

	fmt.Printf("Generated via cache: %v\n", generated)
	for i, t := range generated {
		fmt.Printf("  Step %d: %d (%q)\n", i, t, vocab(t, tok))
	}

	// Compare each step with one-shot forward
	// cacheLogits[step] predicts what comes AFTER seeing prompt + generated[0..step-1]
	// one-shot: Forward(prompt + generated[0..step-1], 0) should give the same
	fmt.Println("\n=== One-shot comparison ===")

	for step := 0; step < len(generated); step++ {
		// Build input: prompt + all previously generated tokens
		combined := []int{}
		combined = append(combined, prompt...)
		for g := 0; g < step; g++ {
			combined = append(combined, generated[g])
		}

		m2 := model.NewQwen3Model(config, weights)
		logitsOS, err := m2.Forward(combined, 0)
		if err != nil {
			panic(err)
		}

		osTok := argmax(logitsOS)
		cacheTop := argmax(cacheLogits[step])
		diff := maxAbsDiff(cacheLogits[step], logitsOS)

		agree := top5agree(cacheLogits[step], logitsOS)

		fmt.Printf("Step %d (seqLen=%d):\n", step, len(combined))
		fmt.Printf("  Cache top1: %d (%q, %.4f)\n", cacheTop, vocab(cacheTop, tok), cacheLogits[step][cacheTop])
		fmt.Printf("  One-shot top1: %d (%q, %.4f)\n", osTok, vocab(osTok, tok), logitsOS[osTok])
		fmt.Printf("  Max diff: %.6f  Top-5 agree: %v\n", diff, agree)
		if diff > 0.001 || !agree {
			printTop5("  Cache top-5", cacheLogits[step], tok)
			printTop5("  One-shot top-5", logitsOS, tok)
		}
		fmt.Println()
	}
}

func printTop5(label string, logits []float32, tok *tokenizer.Tokenizer) {
	type pair struct{ id int; score float32 }
	top := make([]pair, 5)
	for i, v := range logits {
		for j := 0; j < 5; j++ {
			if v > top[j].score {
				copy(top[j+1:], top[j:])
				top[j] = pair{i, v}
				break
			}
		}
	}
	fmt.Printf("%s:", label)
	for _, t := range top {
		fmt.Printf(" %d(%q,%.2f)", t.id, vocab(t.id, tok), t.score)
	}
	fmt.Println()
}

func top5agree(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	topA := topK(a, 5)
	topB := topK(b, 5)
	for i := 0; i < 5; i++ {
		if topA[i] != topB[i] {
			return false
		}
	}
	return true
}

func topK(v []float32, k int) []int {
	type pair struct{ id int; score float32 }
	top := make([]pair, k)
	for i, s := range v {
		for j := 0; j < k; j++ {
			if s > top[j].score {
				copy(top[j+1:], top[j:])
				top[j] = pair{i, s}
				break
			}
		}
	}
	res := make([]int, k)
	for i := range top {
		res[i] = top[i].id
	}
	return res
}

func vocab(id int, tok *tokenizer.Tokenizer) string {
	if id < 0 || id > 151936 {
		return "<OOB>"
	}
	if s, ok := tok.GetToken(id); ok {
		return s
	}
	return "<unknown>"
}

func argmax(v []float32) int {
	maxIdx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return -1
	}
	var maxDiff float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}
