package main

import (
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

func computeTop10(logits []float32, vocab []string) {
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
	for i, ts := range top {
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, ts.idx, vocab[ts.idx], ts.val)
	}
}

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
	
	tok, _ := reader.GetMetadata("tokenizer.ggml.tokens")
	vocab := tok.([]string)
	
	// Forward with just 0 layers (direct embedding + output projection)
	emb := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, config.EmbeddingLength, 6023)
	normed := llmmath.RMSNorm(emb, weights.OutputNorm, config.LayerNormRmsEps)
	logits0 := llmmath.MatMul(normed, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	fmt.Println("After 0 layers:")
	computeTop10(logits0, vocab)
	
	// Forward with just 1 layer (manually)
	hiddenStates := emb
	seqLen := 1
	embeddingDim := config.EmbeddingLength
	headDim := config.AttentionKeyLength
	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	
	for layer := 0; layer < 1; layer++ {
		bw := weights.BlockWeights[layer]
		
		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)
		
		normedHidden := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
		
		q := llmmath.MatMul(normedHidden, bw.AttentionQ, seqLen, qDim, embeddingDim)
		k := llmmath.MatMul(normedHidden, bw.AttentionK, seqLen, kvDim, embeddingDim)
		v := llmmath.MatMul(normedHidden, bw.AttentionV, seqLen, kvDim, embeddingDim)
		
		// QK RMSNorm
		for h := 0; h < nHeads; h++ {
			qStart := h * headDim
			qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
			copy(q[qStart:], qNormed)
		}
		for h := 0; h < nKvHeads; h++ {
			kStart := h * headDim
			kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
			copy(k[kStart:], kNormed)
		}
		
		// RoPE (pos=0, no rotation)
		
		// Attention for single token
		scale := 1.0 / float32(math.Sqrt(float64(headDim)))
		attentionOutput := make([]float32, qDim)
		
		// Repeat KV for GQA
		// repeat := nHeads / nKvHeads
		kRepeated := llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, 1)
		vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)
		
		for h := 0; h < nHeads; h++ {
			qHead := q[h*headDim : (h+1)*headDim]
			kHead := kRepeated[h*headDim : (h+1)*headDim]
			
			var score float32 = 0
			for d := 0; d < headDim; d++ {
				score += qHead[d] * kHead[d]
			}
			score *= scale
			
			// Single token: softmax = 1.0
			outputHead := attentionOutput[h*headDim : (h+1)*headDim]
			vHead := vRepeated[h*headDim : (h+1)*headDim]
			for d := 0; d < headDim; d++ {
				outputHead[d] = vHead[d] // prob=1.0 for single token
			}
		}
		
		// Output projection
		attnOut := llmmath.MatMul(attentionOutput, bw.AttentionOutput, seqLen, embeddingDim, qDim)
		
		hiddenStates = llmmath.VectorAdd(residual, attnOut)
		
		// FFN
		residual = make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)
		
		normedHidden = llmmath.RMSNorm(hiddenStates, bw.FeedForwardNorm, config.LayerNormRmsEps)
		
		ffnGate := llmmath.MatMul(normedHidden, bw.FeedForwardGate, seqLen, config.FeedForwardLength, embeddingDim)
		ffnUp := llmmath.MatMul(normedHidden, bw.FeedForwardUp, seqLen, config.FeedForwardLength, embeddingDim)
		swigluOut := llmmath.SwiGLU(ffnGate, ffnUp)
		ffnOut := llmmath.MatMul(swigluOut, bw.FeedForwardDown, seqLen, embeddingDim, config.FeedForwardLength)
		
		hiddenStates = llmmath.VectorAdd(residual, ffnOut)
	}
	
	normed = llmmath.RMSNorm(hiddenStates, weights.OutputNorm, config.LayerNormRmsEps)
	logits1 := llmmath.MatMul(normed, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	fmt.Println("\nAfter 1 layer:")
	computeTop10(logits1, vocab)
	
	// Compare with full model
	mod := model.NewQwen3Model(config, weights)
	fullLogits, _ := mod.Forward([]int{6023}, 0)
	fmt.Println("\nFull model (28 layers):")
	computeTop10(fullLogits, vocab)
}
