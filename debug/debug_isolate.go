package main

import (
	"fmt"
	"math"
	
	"gguf/model"
	"gguf/gguf"
	llmmath "gguf/math"
)

func computeTop(logits []float32) (int, float32) {
	topVal := float32(-math.MaxFloat32)
	topIdx := -1
	for i, v := range logits {
		if v > topVal { topVal = v; topIdx = i }
	}
	return topIdx, topVal
}

func forwardOneLayer(hs []float32, bw *model.Qwen3BlockWeights, config *model.Qwen3Config, pos int) []float32 {
	seqLen := 1
	embeddingDim := config.EmbeddingLength
	headDim := config.AttentionKeyLength
	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	
	residual := make([]float32, len(hs))
	copy(residual, hs)
	
	normedHidden := llmmath.RMSNorm(hs, bw.AttentionNorm, config.LayerNormRmsEps)
	q := llmmath.MatMul(normedHidden, bw.AttentionQ, seqLen, qDim, embeddingDim)
	k := llmmath.MatMul(normedHidden, bw.AttentionK, seqLen, kvDim, embeddingDim)
	v := llmmath.MatMul(normedHidden, bw.AttentionV, seqLen, kvDim, embeddingDim)
	
	if config.QKVRMSNorm && bw.AttentionQNorm != nil {
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
	
	scale := 1.0 / float32(math.Sqrt(float64(headDim)))
	attentionOutput := make([]float32, qDim)
	kRepeated := llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, 1)
	vRepeated := llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, 1)
	
	for h := 0; h < nHeads; h++ {
		qHead := q[h*headDim : (h+1)*headDim]
		kHead := kRepeated[h*headDim : (h+1)*headDim]
		var score float32 = 0
		for d := 0; d < headDim; d++ { score += qHead[d] * kHead[d] }
		score *= scale
		outputHead := attentionOutput[h*headDim : (h+1)*headDim]
		vHead := vRepeated[h*headDim : (h+1)*headDim]
		for d := 0; d < headDim; d++ { outputHead[d] = vHead[d] }
	}
	
	attnOut := llmmath.MatMul(attentionOutput, bw.AttentionOutput, seqLen, embeddingDim, qDim)
	hs = llmmath.VectorAdd(residual, attnOut)
	
	residual = make([]float32, len(hs))
	copy(residual, hs)
	
	normedHidden = llmmath.RMSNorm(hs, bw.FeedForwardNorm, config.LayerNormRmsEps)
	ffnGate := llmmath.MatMul(normedHidden, bw.FeedForwardGate, seqLen, config.FeedForwardLength, embeddingDim)
	ffnUp := llmmath.MatMul(normedHidden, bw.FeedForwardUp, seqLen, config.FeedForwardLength, embeddingDim)
	swigluOut := llmmath.SwiGLU(ffnGate, ffnUp)
	ffnOut := llmmath.MatMul(swigluOut, bw.FeedForwardDown, seqLen, embeddingDim, config.FeedForwardLength)
	hs = llmmath.VectorAdd(residual, ffnOut)
	
	return hs
}

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	reader := gguf.NewGGUFReader()
	mmapFile, _ := reader.LoadFromFileMMap(modelPath)
	defer mmapFile.Close()
	config, _ := model.LoadQwen3Config(reader)
	weights, _ := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	tokMeta, _ := reader.GetMetadata("tokenizer.ggml.tokens")
	vocab := tokMeta.([]string)
	
	emb := llmmath.EmbeddingLookupDimFirst(weights.TokenEmbedding, config.VocabSize, config.EmbeddingLength, 6023)
	
	fmt.Println("=== Test A: Embedding through LAYER 1 ONLY (skipping layer 0) ===")
	hsA := make([]float32, len(emb))
	copy(hsA, emb)
	hsA = forwardOneLayer(hsA, weights.BlockWeights[1], config, 0)
	normedA := llmmath.RMSNorm(hsA, weights.OutputNorm, config.LayerNormRmsEps)
	logitsA := llmmath.MatMul(normedA, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	idxA, valA := computeTop(logitsA)
	fmt.Printf("  Top: #%d %q (%.2f)\n", idxA, vocab[idxA], valA)
	
	fmt.Println("\n=== Test B: Embedding through LAYER 0 ONLY ===")
	hsB := make([]float32, len(emb))
	copy(hsB, emb)
	hsB = forwardOneLayer(hsB, weights.BlockWeights[0], config, 0)
	normedB := llmmath.RMSNorm(hsB, weights.OutputNorm, config.LayerNormRmsEps)
	logitsB := llmmath.MatMul(normedB, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	idxB, valB := computeTop(logitsB)
	fmt.Printf("  Top: #%d %q (%.2f)\n", idxB, vocab[idxB], valB)
	
	fmt.Println("\n=== Test C: Layer 0 output through LAYER 0 AGAIN ===")
	hsC := make([]float32, len(hsB))
	copy(hsC, hsB)
	hsC = forwardOneLayer(hsC, weights.BlockWeights[0], config, 0)
	normedC := llmmath.RMSNorm(hsC, weights.OutputNorm, config.LayerNormRmsEps)
	logitsC := llmmath.MatMul(normedC, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	idxC, valC := computeTop(logitsC)
	fmt.Printf("  Top: #%d %q (%.2f)\n", idxC, vocab[idxC], valC)
	
	fmt.Println("\n=== Test D: Layer 0 output through LAYER 1 (actual 2-layer path) ===")
	hsD := make([]float32, len(hsB))
	copy(hsD, hsB)
	hsD = forwardOneLayer(hsD, weights.BlockWeights[1], config, 0)
	normedD := llmmath.RMSNorm(hsD, weights.OutputNorm, config.LayerNormRmsEps)
	logitsD := llmmath.MatMul(normedD, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
	idxD, valD := computeTop(logitsD)
	fmt.Printf("  Top: #%d %q (%.2f)\n", idxD, vocab[idxD], valD)
	
	fmt.Println("\n=== Test E: Compare full 2-layer with full model ===")
	mod := model.NewQwen3Model(config, weights)
	fullLogits, _ := mod.Forward([]int{6023}, 0)
	idxE, valE := computeTop(fullLogits)
	fmt.Printf("  Full model top: #%d %q (%.2f)\n", idxE, vocab[idxE], valE)
}
