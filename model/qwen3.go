package model

import (
	"fmt"
	stdmath "math"

	llmmath "github.com/adfnekc/go-qwen3-gguf/math"
)

type Qwen3Model struct {
	Config  *Qwen3Config
	Weights *Qwen3Weights
	Cache   *KVCache

	// reusable buffers to reduce allocations in hot loops
	scoreBuf  []float32
	headBuf   []float32
}

func NewQwen3Model(config *Qwen3Config, weights *Qwen3Weights) *Qwen3Model {
	headDim := config.AttentionKeyLength
	return &Qwen3Model{
		Config:   config,
		Weights:  weights,
		Cache:    NewKVCache(config.ContextLength, config.BlockCount, config.AttentionHeadCountKv, headDim),
		scoreBuf: make([]float32, config.ContextLength),
		headBuf:  make([]float32, headDim),
	}
}

func (m *Qwen3Model) ResetCache() {
	headDim := m.Config.AttentionKeyLength
	m.Cache = NewKVCache(m.Config.ContextLength, m.Config.BlockCount, m.Config.AttentionHeadCountKv, headDim)
}

func (m *Qwen3Model) embeddingLookup(tokens []int, seqLen int) ([]float32, error) {
	vocabSize := m.Config.VocabSize
	embDim := m.Config.EmbeddingLength
	hs := make([]float32, seqLen*embDim)
	for i, token := range tokens {
		if token < 0 || token >= vocabSize {
			return nil, fmt.Errorf("token %d out of range", token)
		}
		start := token * embDim
		copy(hs[i*embDim:], m.Weights.TokenEmbedding[start:start+embDim])
	}
	return hs, nil
}

func applyQKNorm(q, k []float32, bw *Qwen3BlockWeights, config *Qwen3Config, seqLen, nHeads, nKvHeads, headDim int) {
	if !config.QKVRMSNorm || bw.AttentionQNorm == nil || bw.AttentionKNorm == nil {
		return
	}
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	for i := 0; i < seqLen; i++ {
		for h := 0; h < nHeads; h++ {
			start := i*qDim + h*headDim
			llmmath.RMSNormInPlace(q[start:start+headDim], q[start:start+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
		}
		for h := 0; h < nKvHeads; h++ {
			start := i*kvDim + h*headDim
			llmmath.RMSNormInPlace(k[start:start+headDim], k[start:start+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
		}
	}
}

func applyRoPEToHeads(q, k []float32, startPos, seqLen, nHeads, nKvHeads, headDim int, ropeFreqBase float32) {
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim
	for i := 0; i < seqLen; i++ {
		pos := startPos + i
		for h := 0; h < nKvHeads; h++ {
			start := i*kvDim + h*headDim
			_, kRope := llmmath.RoPE(nil, k[start:start+headDim], pos, headDim, ropeFreqBase, llmmath.RoPE_NEOX)
			copy(k[start:], kRope)
		}
		for h := 0; h < nHeads; h++ {
			start := i*qDim + h*headDim
			qRope, _ := llmmath.RoPE(q[start:start+headDim], nil, pos, headDim, ropeFreqBase, llmmath.RoPE_NEOX)
			copy(q[start:], qRope)
		}
	}
}

func applyAttention(q, k, v []float32, cache *KVCache, layer, seqLen, nHeads, nKvHeads, headDim int, scoreBuf, headBuf []float32) []float32 {
	qDim := nHeads * headDim

	pastK, pastV := cache.GetKV(layer, nKvHeads, headDim)
	pastSeqLen := cache.GetSize(layer)

	repeat := nHeads / nKvHeads
	if repeat > 1 {
		pastK = llmmath.RepeatKV(pastK, nKvHeads, nHeads, headDim, pastSeqLen)
		pastV = llmmath.RepeatKV(pastV, nKvHeads, nHeads, headDim, pastSeqLen)
		k = llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, seqLen)
		v = llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, seqLen)
	}

	output := make([]float32, seqLen*qDim)
	scale := 1.0 / float32(stdmath.Sqrt(float64(headDim)))

	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			qHead := q[s*qDim+h*headDim : s*qDim+(h+1)*headDim]

			scores := scoreBuf[:pastSeqLen]
			for t := 0; t < pastSeqLen; t++ {
				kHead := pastK[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]
				var score float32
				for d := 0; d < headDim; d++ {
					score += qHead[d] * kHead[d]
				}
				scores[t] = score * scale
			}

			causalOffset := pastSeqLen - seqLen
			for t := 0; t < pastSeqLen; t++ {
				if t > causalOffset+s {
					scores[t] = float32(stdmath.Inf(-1))
				}
			}

			probs := llmmath.VectorSoftmax(scores)

			for d := 0; d < headDim; d++ {
				headBuf[d] = 0
			}
			for t := 0; t < pastSeqLen; t++ {
				vHead := pastV[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]
				for d := 0; d < headDim; d++ {
					headBuf[d] += probs[t] * vHead[d]
				}
			}
			copy(output[s*qDim+h*headDim:], headBuf)
		}
	}
	return output
}

func applyFFN(hs []float32, bw *Qwen3BlockWeights, config *Qwen3Config, seqLen int) ([]float32, error) {
	gate := llmmath.MatMulTransposed(hs, bw.FeedForwardGate, seqLen, config.FeedForwardLength, config.EmbeddingLength)
	up := llmmath.MatMulTransposed(hs, bw.FeedForwardUp, seqLen, config.FeedForwardLength, config.EmbeddingLength)
	if gate == nil || up == nil {
		return nil, fmt.Errorf("FFN gate/up projection failed")
	}
	swiglu := llmmath.SwiGLU(gate, up)
	down := llmmath.MatMulTransposed(swiglu, bw.FeedForwardDown, seqLen, config.EmbeddingLength, config.FeedForwardLength)
	if down == nil {
		return nil, fmt.Errorf("FFN down projection failed")
	}
	return down, nil
}

func (m *Qwen3Model) forwardLayer(hs []float32, bw *Qwen3BlockWeights, layer, startPos, seqLen int) ([]float32, error) {
	embDim := m.Config.EmbeddingLength
	headDim := m.Config.AttentionKeyLength
	nHeads := m.Config.AttentionHeadCount
	nKvHeads := m.Config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim

	// Attention residual
	residual := make([]float32, len(hs))
	copy(residual, hs)

	// Pre-attention norm
	normed := make([]float32, seqLen*embDim)
	for i := 0; i < seqLen; i++ {
		n := llmmath.RMSNorm(hs[i*embDim:(i+1)*embDim], bw.AttentionNorm, m.Config.LayerNormRmsEps)
		copy(normed[i*embDim:], n)
	}

	// QKV projection
	q := llmmath.MatMulTransposed(normed, bw.AttentionQ, seqLen, qDim, embDim)
	k := llmmath.MatMulTransposed(normed, bw.AttentionK, seqLen, kvDim, embDim)
	v := llmmath.MatMulTransposed(normed, bw.AttentionV, seqLen, kvDim, embDim)
	if q == nil || k == nil || v == nil {
		return nil, fmt.Errorf("attention projection failed for layer %d", layer)
	}

	// QK norm + RoPE
	applyQKNorm(q, k, bw, m.Config, seqLen, nHeads, nKvHeads, headDim)
	applyRoPEToHeads(q, k, startPos, seqLen, nHeads, nKvHeads, headDim, m.Config.RopeFreqBase)

	// KV cache update
	for i := 0; i < seqLen; i++ {
		m.Cache.Update(layer, k[i*kvDim:(i+1)*kvDim], v[i*kvDim:(i+1)*kvDim])
	}

	// Attention
	attnOut := applyAttention(q, k, v, m.Cache, layer, seqLen, nHeads, nKvHeads, headDim, m.scoreBuf, m.headBuf)
	attnOut = llmmath.MatMulTransposed(attnOut, bw.AttentionOutput, seqLen, embDim, qDim)
	if attnOut == nil {
		return nil, fmt.Errorf("attention output projection failed for layer %d", layer)
	}

	// First residual add
	hs = llmmath.VectorAdd(residual, attnOut)

	// Pre-FFN norm
	residual2 := make([]float32, len(hs))
	copy(residual2, hs)
	normed = make([]float32, seqLen*embDim)
	for i := 0; i < seqLen; i++ {
		n := llmmath.RMSNorm(hs[i*embDim:(i+1)*embDim], bw.FeedForwardNorm, m.Config.LayerNormRmsEps)
		copy(normed[i*embDim:], n)
	}

	// FFN
	ffnOut, err := applyFFN(normed, bw, m.Config, seqLen)
	if err != nil {
		return nil, fmt.Errorf("layer %d FFN: %w", layer, err)
	}

	return llmmath.VectorAdd(residual2, ffnOut), nil
}

func (m *Qwen3Model) Forward(tokens []int, startPos int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokens is empty")
	}

	seqLen := len(tokens)
	embDim := m.Config.EmbeddingLength

	hiddenStates, err := m.embeddingLookup(tokens, seqLen)
	if err != nil {
		return nil, err
	}

	for layer := 0; layer < m.Config.BlockCount; layer++ {
		bw := m.Weights.BlockWeights[layer]
		hiddenStates, err = m.forwardLayer(hiddenStates, bw, layer, startPos, seqLen)
		if err != nil {
			return nil, err
		}
	}

	// Output projection
	lastHidden := hiddenStates[(seqLen-1)*embDim : seqLen*embDim]
	normedOutput := llmmath.RMSNorm(lastHidden, m.Weights.OutputNorm, m.Config.LayerNormRmsEps)

	outputWeights := m.Weights.Output
	if outputWeights == nil {
		outputWeights = m.Weights.TokenEmbedding
	}
	logits := llmmath.MatMulTransposed([]float32(normedOutput), outputWeights, 1, m.Config.VocabSize, embDim)
	if logits == nil {
		return nil, fmt.Errorf("output projection failed")
	}
	return logits, nil
}

func (m *Qwen3Model) Generate(tokens []int, maxNewTokens int, temperature float32) ([]int, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("input tokens is empty")
	}

	m.ResetCache()

	generated := make([]int, len(tokens))
	copy(generated, tokens)

	startPos := 0

	for i := 0; i < maxNewTokens; i++ {
		var inputTokens []int
		if i == 0 {
			inputTokens = generated
		} else {
			inputTokens = generated[len(generated)-1:]
		}

		logits, err := m.Forward(inputTokens, startPos)
		if err != nil {
			return nil, err
		}
		startPos += len(inputTokens)

		var nextToken int
		if temperature <= 0 {
			nextToken = llmmath.Argmax(logits)
		} else {
			scaledLogits := make([]float32, len(logits))
			for j := range logits {
				scaledLogits[j] = logits[j] / temperature
			}
			probs := llmmath.VectorSoftmax(scaledLogits)
			nextToken = llmmath.SampleCategorical(probs)
		}

		generated = append(generated, nextToken)

		if nextToken == TokenIMEnd || nextToken == 151643 {
			break
		}
	}

	return generated, nil
}
