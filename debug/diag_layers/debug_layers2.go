package main

import (
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

func computeTop10(logits []float32, vocab []string) {
	type TS struct { idx int; val float32 }
	top := make([]TS, 10)
	for i := range top { top[i].val = -math.MaxFloat32 }
	for i, v := range logits {
		for j := 0; j < 10; j++ {
			if v > top[j].val {
				for k := 9; k > j; k-- { top[k] = top[k-1] }
				top[j] = TS{i, v}
				break
			}
		}
	}
	for i, ts := range top {
		fmt.Printf("  #%d: token %d (%q) = %f\n", i+1, ts.idx, vocab[ts.idx], ts.val)
	}
}

func forwardOneLayer(hiddenStates []float32, bw *model.Qwen3BlockWeights, config *model.Qwen3Config, pos int) []float32 {
	seqLen := 1
	embeddingDim := config.EmbeddingLength
	headDim := config.AttentionKeyLength
	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	
	residual := make([]float32, len(hiddenStates))
	copy(residual, hiddenStates)
	
	normedHidden := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
	
	q := llmmath.MatMul(normedHidden, bw.AttentionQ, seqLen, qDim, embeddingDim)
	k := llmmath.MatMul(normedHidden, bw.AttentionK, seqLen, kvDim, embeddingDim)
	v := llmmath.MatMul(normedHidden, bw.AttentionV, seqLen, kvDim, embeddingDim)
	
	// QK RMSNorm
	if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
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
	}
	
	// RoPE
	for h := 0; h < nKvHeads; h++ {
		kStart := h * headDim
		_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], pos, headDim, config.RopeFreqBase)
		copy(k[kStart:], kRope)
	}
	for h := 0; h < nHeads; h++ {
		qStart := h * headDim
		qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, pos, headDim, config.RopeFreqBase)
		copy(q[qStart:], qRope)
	}
	
	// Attention (single token)
	scale := 1.0 / float32(math.Sqrt(float64(headDim)))
	attentionOutput := make([]float32, qDim)
	
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
		
		outputHead := attentionOutput[h*headDim : (h+1)*headDim]
		vHead := vRepeated[h*headDim : (h+1)*headDim]
		for d := 0; d < headDim; d++ {
			outputHead[d] = vHead[d]
		}
	}
	
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
	
	return hiddenStates
}

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	
	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(modelPath)
	if err != nil { panic(err) }
	defer mmapFile.Close()
	
	config, err := model.LoadQwen3Config(reader)
	if err != nil { panic(err) }
	
	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil { panic(err) }
	
	tokMeta, _ := reader.GetMetadata("tokenizer.ggml.tokens")
	vocab := tokMeta.([]string)
	
	emb := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, config.EmbeddingLength, 6023)
	
	for _, nLayers := range []int{0, 1, 2, 4, 8, 16, 28} {
		hs := make([]float32, len(emb))
		copy(hs, emb)
		
		for l := 0; l < nLayers && l < config.BlockCount; l++ {
			hs = forwardOneLayer(hs, weights.BlockWeights[l], config, 0)
		}
		
		normed := llmmath.RMSNorm(hs, weights.OutputNorm, config.LayerNormRmsEps)
		var logits []float32
		if weights.Output != nil {
			logits = llmmath.MatMul(normed, weights.Output, 1, config.VocabSize, config.EmbeddingLength)
		} else {
			logits = llmmath.MatMul(normed, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
		}
		
		fmt.Printf("\nAfter %d layers:\n", nLayers)
		computeTop10(logits, vocab)
	}
}
