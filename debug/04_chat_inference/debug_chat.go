package main

import (
	"fmt"
	"os"

	"gguf/gguf"
	"gguf/model"
	"gguf/tokenizer"
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

	vocabMeta, ok := reader.GetMetadata("tokenizer.ggml.tokens")
	if !ok {
		panic("no vocab")
	}
	vocab := vocabMeta.([]string)

	// Build chat-formatted prompt
	input := "hi"
	imStart := 151644
	imEnd := 151645
	chatTokens := []int{imStart}
	chatTokens = append(chatTokens, tok.Encode("user\n")...)
	chatTokens = append(chatTokens, tok.Encode(input)...)
	chatTokens = append(chatTokens, imEnd)
	chatTokens = append(chatTokens, 198) // "\n"
	chatTokens = append(chatTokens, imStart)
	chatTokens = append(chatTokens, tok.Encode("assistant\n")...)

	fmt.Printf("Chat prompt tokens: %v\n", chatTokens)
	fmt.Printf("Token count: %d\n", len(chatTokens))

	// Step 1: Forward with full prompt (no generation)
	m := model.NewQwen3Model(config, weights)
	logits, err := m.Forward(chatTokens, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println("\n=== Step 1 (prompt): top-10 predictions ===")
	printTopK("Logits", logits, vocab, 10)

	// Check if <think> (151667) is the top prediction
	thinkIdx := 151667
	if int(thinkIdx) < len(logits) {
		fmt.Printf("  Token <think> (151667) logit: %.4f\n", logits[thinkIdx])
	}
	hiIdx := 6023
	if hiIdx < len(logits) {
		fmt.Printf("  Token 'hi' (6023) logit: %.4f\n", logits[hiIdx])
	}

	// Step 2: Full generation with debug
	fmt.Println("\n=== Full Generation ===")
	generateDebug(m, tok, chatTokens, 10, vocab)
}

func generateDebug(m *model.Qwen3Model, tok *tokenizer.Tokenizer, tokens []int, maxNew int, vocab []string) {
	m.ResetCache()
	generated := make([]int, len(tokens))
	copy(generated, tokens)

	startPos := 0
	eosId := 151645

	for step := 0; step < maxNew; step++ {
		var inputTokens []int
		if step == 0 {
			inputTokens = generated
		} else {
			inputTokens = generated[len(generated)-1:]
		}

		logits, err := m.Forward(inputTokens, startPos)
		if err != nil {
			panic(err)
		}

		startPos += len(inputTokens)

		// Print top-5 predictions
		topN := 5
		type scored struct {
			id    int
			score float32
			text  string
		}
		top := make([]scored, 0, topN)
		for i, v := range logits {
			if len(top) < topN {
				top = append(top, scored{i, v, vocab[i]})
			} else {
				minIdx := 0
				for j := 1; j < topN; j++ {
					if top[j].score < top[minIdx].score {
						minIdx = j
					}
				}
				if v > top[minIdx].score {
					top[minIdx] = scored{i, v, vocab[i]}
				}
			}
		}
		// Sort descending
		for i := 0; i < len(top); i++ {
			for j := i + 1; j < len(top); j++ {
				if top[j].score > top[i].score {
					top[i], top[j] = top[j], top[i]
				}
			}
		}

		nextToken := top[0].id
		generated = append(generated, nextToken)

		fmt.Printf("Step %d (input=%d, startPos=%d): top1=%d(%q, %.2f)  top5=%d(%q, %.2f)\n",
			step, len(inputTokens), startPos-len(inputTokens),
			top[0].id, top[0].text, top[0].score,
			top[4].id, top[4].text, top[4].score)

		if nextToken == eosId {
			fmt.Println("  → EOS token reached, stopping")
			break
		}
	}

	// Decode full output
	var decoded string
	if t, ok := interface{}(tok).(*tokenizer.Tokenizer); ok {
		decoded = t.Decode(generated)
	}
	fmt.Printf("\nFull decoded output:\n%s\n", decoded)

	// Decode just the new tokens
	newTokens := generated[len(tokens):]
	var newDecoded string
	if t, ok := interface{}(tok).(*tokenizer.Tokenizer); ok {
		newDecoded = t.Decode(newTokens)
	}
	fmt.Printf("\nNewly generated tokens (%d): %v\n", len(newTokens), newTokens)
	fmt.Printf("Decoded: %s\n", newDecoded)
}

func printTopK(label string, logits []float32, vocab []string, k int) {
	type scored struct {
		id    int
		score float32
		text  string
	}
	top := make([]scored, 0, k)
	for i, v := range logits {
		if len(top) < k {
			top = append(top, scored{i, v, vocab[i]})
		} else {
			minIdx := 0
			for j := 1; j < k; j++ {
				if top[j].score < top[minIdx].score {
					minIdx = j
				}
			}
			if v > top[minIdx].score {
				top[minIdx] = scored{i, v, vocab[i]}
			}
		}
	}
	for i := 0; i < len(top); i++ {
		for j := i + 1; j < len(top); j++ {
			if top[j].score > top[i].score {
				top[i], top[j] = top[j], top[i]
			}
		}
	}
	fmt.Printf("%s:\n", label)
	for _, t := range top {
		fmt.Printf("  #%d %6.2f  %q\n", t.id, t.score, t.text)
	}
}

