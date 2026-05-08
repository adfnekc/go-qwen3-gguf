package main

import (
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

func computeRMS(hs []float32) float32 {
	var sum float32 = 0
	for _, v := range hs { sum += v * v }
	return float32(math.Sqrt(float64(sum / float32(len(hs)))))
}

func forwardOneLayer(hs []float32, bw *model.Qwen3BlockWeights, config *model.Qwen3Config, pos int) ([]float32, float32, float32, float32, float32, float32) {
	seqLen := 1
	embeddingDim := config.EmbeddingLength
	headDim := config.AttentionKeyLength
	nHeads := config.AttentionHeadCount
	nKvHeads := config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	
	residual := make([]float32, len(hs))
	copy(residual, hs)
	rmsResidual := computeRMS(residual)
	
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
	rmsAttnOut := computeRMS(attnOut)
	
	hs = llmmath.VectorAdd(residual, attnOut)
	rmsPostAttn := computeRMS(hs)
	
	// FFN
	residual = make([]float32, len(hs))
	copy(residual, hs)
	
	normedHidden = llmmath.RMSNorm(hs, bw.FeedForwardNorm, config.LayerNormRmsEps)
	ffnGate := llmmath.MatMul(normedHidden, bw.FeedForwardGate, seqLen, config.FeedForwardLength, embeddingDim)
	ffnUp := llmmath.MatMul(normedHidden, bw.FeedForwardUp, seqLen, config.FeedForwardLength, embeddingDim)
	swigluOut := llmmath.SwiGLU(ffnGate, ffnUp)
	ffnOut := llmmath.MatMul(swigluOut, bw.FeedForwardDown, seqLen, embeddingDim, config.FeedForwardLength)
	rmsFFNOut := computeRMS(ffnOut)
	
	hs = llmmath.VectorAdd(residual, ffnOut)
	rmsFinal := computeRMS(hs)
	
	return hs, rmsResidual, rmsAttnOut, rmsFFNOut, rmsPostAttn, rmsFinal
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
	embRMS := computeRMS(emb)
	fmt.Printf("Embedding RMS: %f\n", embRMS)
	
	hs := make([]float32, len(emb))
	copy(hs, emb)
	
	for l := 0; l < 28; l++ {
		var rmsRes, rmsAttn, rmsFFN, rmsPostAttn, rmsFinal float32
		hs, rmsRes, rmsAttn, rmsFFN, rmsPostAttn, rmsFinal = forwardOneLayer(hs, weights.BlockWeights[l], config, 0)
		
		normed := llmmath.RMSNorm(hs, weights.OutputNorm, config.LayerNormRmsEps)
		logits := llmmath.MatMul(normed, weights.TokenEmbedding, 1, config.VocabSize, config.EmbeddingLength)
		
		// Find top token
		topVal := float32(-math.MaxFloat32)
		topIdx := -1
		for i, v := range logits {
			if v > topVal { topVal = v; topIdx = i }
		}
		
		fmt.Printf("Layer %2d: RMS(res=%.4f, attn=%.4f, postAttn=%.4f, ffn=%.4f, final=%.4f) Top: #%d %q (%.2f)\n",
			l, rmsRes, rmsAttn, rmsPostAttn, rmsFFN, rmsFinal, topIdx, vocab[topIdx], topVal)
	}
}
