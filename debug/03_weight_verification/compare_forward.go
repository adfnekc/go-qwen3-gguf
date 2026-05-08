package main

import (
	"fmt"
	"math"
	"os"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
	"github.com/adfnekc/go-qwen3-gguf/model"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
	"github.com/adfnekc/go-qwen3-gguf/tokenizer"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	// 1. Load
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

	// 2. Encode "hi"
	input := "hi"
	tokens := tok.Encode(input)
	fmt.Printf("Input: %q → tokens: %v\n", input, tokens)

	// 3. Run model.Forward
	fmt.Println("\n=== model.Forward ===")
	m := model.NewQwen3Model(config, weights)
	logits, err := m.Forward(tokens, 0)
	if err != nil {
		panic(err)
	}
	printTopK("Forward logits", logits, vocab, 10)

	// 4. Run manual forward (same as qwen3.go but printed step by step)
	fmt.Println("\n=== Manual Forward (single token) ===")
	seqLen := 1
	embeddingDim := config.EmbeddingLength
	headDim := config.AttentionKeyLength
	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim

	// Embedding
	hiddenStates := make([]float32, seqLen*embeddingDim)
	for i, token := range tokens {
		emb := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, token)
		copy(hiddenStates[i*embeddingDim:], emb)
	}
	fmt.Printf("Initial hidden RMS: %.6f\n", rms32(hiddenStates))

	for layer := 0; layer < config.BlockCount; layer++ {
		bw := weights.BlockWeights[layer]

		// Attention norm
		normedHidden := make([]float32, seqLen*embeddingDim)
		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates[i*embeddingDim:(i+1)*embeddingDim], bw.AttentionNorm, config.LayerNormRmsEps)
			copy(normedHidden[i*embeddingDim:], normed)
		}

		// QKV
		q := llmmath.MatMulTransposed(normedHidden, bw.AttentionQ, seqLen, qDim, embeddingDim)
		k := llmmath.MatMulTransposed(normedHidden, bw.AttentionK, seqLen, kvDim, embeddingDim)
		v := llmmath.MatMulTransposed(normedHidden, bw.AttentionV, seqLen, kvDim, embeddingDim)

		// QK norm
		if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
			for i := 0; i < seqLen; i++ {
				for h := 0; h < nHeads; h++ {
					qStart := i*qDim + h*headDim
					qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
					copy(q[qStart:], qNormed)
				}
				for h := 0; h < nKvHeads; h++ {
					kStart := i*kvDim + h*headDim
					kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
					copy(k[kStart:], kNormed)
				}
			}
		}

		// RoPE
		for i := 0; i < seqLen; i++ {
			for h := 0; h < nKvHeads; h++ {
				kStart := i*kvDim + h*headDim
				_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], i, headDim, config.RopeFreqBase)
				copy(k[kStart:], kRope)
			}
			for h := 0; h < nHeads; h++ {
				qStart := i*qDim + h*headDim
				qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, i, headDim, config.RopeFreqBase)
				copy(q[qStart:], qRope)
			}
		}

		// Self-attention (no cache, single token)
		kRep := llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, seqLen)
		vRep := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, seqLen)

		scale := 1.0 / float32(math.Sqrt(float64(headDim)))
		attentionOutput := make([]float32, seqLen*qDim)

		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				qHead := q[s*qDim+h*headDim : s*qDim+(h+1)*headDim]

				// Only attend to positions <= s (causal)
				var score float32 = 0
				kHead := kRep[s*nHeads*headDim+h*headDim : s*nHeads*headDim+(h+1)*headDim]
				for d := 0; d < headDim; d++ {
					score += qHead[d] * kHead[d]
				}
				score *= scale
				prob := softmax1(score) // single value: softmax([score, -inf...]) = [1]

				vHead := vRep[s*nHeads*headDim+h*headDim : s*nHeads*headDim+(h+1)*headDim]
				for d := 0; d < headDim; d++ {
					attentionOutput[s*qDim+h*headDim+d] = prob * vHead[d]
				}
			}
		}

		// Output projection
		attnProj := llmmath.MatMulTransposed(attentionOutput, bw.AttentionOutput, seqLen, embeddingDim, qDim)
		hiddenStates = llmmath.VectorAdd(hiddenStates, attnProj)

		// FFN
		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates[i*embeddingDim:(i+1)*embeddingDim], bw.FeedForwardNorm, config.LayerNormRmsEps)
			copy(normedHidden[i*embeddingDim:], normed)
		}

		ffnGate := llmmath.MatMulTransposed(normedHidden, bw.FeedForwardGate, seqLen, config.FeedForwardLength, embeddingDim)
		ffnUp := llmmath.MatMulTransposed(normedHidden, bw.FeedForwardUp, seqLen, config.FeedForwardLength, embeddingDim)
		swigluOut := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnOut := llmmath.MatMulTransposed(swigluOut, bw.FeedForwardDown, seqLen, embeddingDim, config.FeedForwardLength)

		hiddenStates = llmmath.VectorAdd(residual, ffnOut)

		if layer < 3 || layer%5 == 4 {
			fmt.Printf("Layer %2d: hidden RMS=%.6f\n", layer, rms32(hiddenStates))
		}

		if layer <= 2 {
			// Detailed trace for layer 2
			preFFNrms := rms32(residual)
			ffnNormInput := make([]float32, embeddingDim)
			copy(ffnNormInput, residual)
			ffnNormed := llmmath.RMSNorm(ffnNormInput, bw.FeedForwardNorm, config.LayerNormRmsEps)
			ffnNormedRMS := rms32(ffnNormed)
			gateOut := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardGate, seqLen, config.FeedForwardLength, embeddingDim)
			upOut := llmmath.MatMulTransposed(ffnNormed, bw.FeedForwardUp, seqLen, config.FeedForwardLength, embeddingDim)
			gateRMS := rms32(gateOut)
			upRMS := rms32(upOut)

			if layer == 2 {
				fmt.Printf("  Layer 2 gate[0:10]:")
				for i := 0; i < 10 && i < len(gateOut); i++ {
					fmt.Printf(" %.2f", gateOut[i])
				}
				fmt.Println()
				fmt.Printf("  Layer 2 up[0:10]:   ")
				for i := 0; i < 10 && i < len(upOut); i++ {
					fmt.Printf(" %.2f", upOut[i])
				}
				fmt.Println()
				negCount, posCount := 0, 0
				for i := 0; i < len(gateOut); i++ {
					if gateOut[i] > 0 {
						posCount++
					} else {
						negCount++
					}
				}
				fmt.Printf("  Layer 2 gate signs: %d neg, %d pos\n", negCount, posCount)
				negCount, posCount = 0, 0
				for i := 0; i < len(upOut); i++ {
					if upOut[i] > 0 {
						posCount++
					} else {
						negCount++
					}
				}
				fmt.Printf("  Layer 2 up signs: %d neg, %d pos\n", negCount, posCount)
			}

			swiglu := llmmath.SwiGLU(gateOut, upOut)
			swigluRMS := rms32(swiglu)
			if layer == 2 {
				fmt.Printf("  Layer 2 swiglu[0:10]:")
				for i := 0; i < 10 && i < len(swiglu); i++ {
					fmt.Printf(" %.1f", swiglu[i])
				}
				fmt.Println()
			}
			downOut := llmmath.MatMulTransposed(swiglu, bw.FeedForwardDown, seqLen, embeddingDim, config.FeedForwardLength)
			downRMS := rms32(downOut)
			fmt.Printf("  Layer %d FFN detail: preNormRMS=%.4f  normedRMS=%.4f  gate=%.2f  up=%.2f  swiglu=%.1f  down=%.1f\n",
				layer, preFFNrms, ffnNormedRMS, gateRMS, upRMS, swigluRMS, downRMS)
		}
	}

	// Output projection
	lastHidden := hiddenStates[(seqLen-1)*embeddingDim : seqLen*embeddingDim]
	normedOutput := llmmath.RMSNorm(lastHidden, weights.OutputNorm, config.LayerNormRmsEps)
	fmt.Printf("Normed output RMS: %.6f\n", rms32(normedOutput))

	var logits2 []float32
	if weights.Output != nil {
		logits2 = llmmath.MatMulTransposed([]float32(normedOutput), weights.Output, 1, config.VocabSize, embeddingDim)
	} else {
		logits2 = llmmath.MatMulTransposed([]float32(normedOutput), weights.TokenEmbedding, 1, config.VocabSize, embeddingDim)
	}
	printTopK("Manual logits", logits2, vocab, 10)

	// Compare
	if len(logits) == len(logits2) {
		diff := float32(0)
		for i := range logits {
			d := logits[i] - logits2[i]
			if d > diff {
				diff = d
			}
			if -d > diff {
				diff = -d
			}
		}
		fmt.Printf("\nMax logit diff: %.6f\n", diff)
		if diff < 0.001 {
			fmt.Println("✓ model.Forward matches manual forward!")
		} else {
			fmt.Println("✗ model.Forward DOES NOT MATCH manual forward!")
		}
	}
}

func softmax1(x float32) float32 {
	return 1.0 // single element softmax, always 1.0
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
	// Sort descending
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
