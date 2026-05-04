package model

import (
	"fmt"
	stdmath "math"

	llmmath "gguf/math"
)

type Qwen3Model struct {
	Config  *Qwen3Config
	Weights *Qwen3Weights
	Cache   *KVCache
}

func NewQwen3Model(config *Qwen3Config, weights *Qwen3Weights) *Qwen3Model {
	headDim := config.AttentionKeyLength

	return &Qwen3Model{
		Config:  config,
		Weights: weights,
		Cache:   NewKVCache(config.ContextLength, config.BlockCount, config.AttentionHeadCountKv, headDim),
	}
}

func (model *Qwen3Model) ResetCache() {
	headDim := model.Config.AttentionKeyLength
	model.Cache = NewKVCache(model.Config.ContextLength, model.Config.BlockCount, model.Config.AttentionHeadCountKv, headDim)
}

func (model *Qwen3Model) Forward(tokens []int, startPos int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokens is empty")
	}

	seqLen := len(tokens)
	embeddingDim := model.Config.EmbeddingLength
	headDim := model.Config.AttentionKeyLength
	nHeads := model.Config.AttentionHeadCount
	nKvHeads := model.Config.AttentionHeadCountKv
	qDim := nHeads * headDim
	kvDim := nKvHeads * headDim

	hiddenStates := make([]float32, seqLen*embeddingDim)
	for i, token := range tokens {
		embedding := llmmath.EmbeddingLookupDimFirst(model.Weights.TokenEmbedding, model.Config.VocabSize, embeddingDim, token)
		if embedding == nil {
			return nil, fmt.Errorf("token %d not found in embedding", token)
		}
		copy(hiddenStates[i*embeddingDim:], embedding)
	}

	for layer := 0; layer < model.Config.BlockCount; layer++ {
		blockWeights := model.Weights.BlockWeights[layer]

		residual := make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normedHidden := make([]float32, seqLen*embeddingDim)
		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates[i*embeddingDim:(i+1)*embeddingDim], blockWeights.AttentionNorm, model.Config.LayerNormRmsEps)
			copy(normedHidden[i*embeddingDim:], normed)
		}

		q := llmmath.MatMul(normedHidden, blockWeights.AttentionQ, seqLen, qDim, embeddingDim)
		k := llmmath.MatMul(normedHidden, blockWeights.AttentionK, seqLen, kvDim, embeddingDim)
		v := llmmath.MatMul(normedHidden, blockWeights.AttentionV, seqLen, kvDim, embeddingDim)

		if q == nil || k == nil || v == nil {
			return nil, fmt.Errorf("attention projection failed for layer %d", layer)
		}

		if model.Config.QKVRMSNorm && blockWeights.AttentionQNorm != nil && blockWeights.AttentionKNorm != nil {
			for i := 0; i < seqLen; i++ {
				for h := 0; h < nHeads; h++ {
					qStart := i*qDim + h*headDim
					qVec := q[qStart : qStart+headDim]
					qNormed := llmmath.RMSNorm(qVec, blockWeights.AttentionQNorm, model.Config.LayerNormRmsEps)
					copy(q[qStart:], qNormed)
				}

				for h := 0; h < nKvHeads; h++ {
					kStart := i*kvDim + h*headDim
					kVec := k[kStart : kStart+headDim]
					kNormed := llmmath.RMSNorm(kVec, blockWeights.AttentionKNorm, model.Config.LayerNormRmsEps)
					copy(k[kStart:], kNormed)
				}
			}
		}

		for i := 0; i < seqLen; i++ {
			pos := startPos + i

			for h := 0; h < nHeads; h++ {
				qStart := i*qDim + h*headDim
				qVec := q[qStart : qStart+headDim]

				kvHeadIdx := h % nKvHeads
				kStart := i*kvDim + kvHeadIdx*headDim
				kVec := k[kStart : kStart+headDim]

				qRope, kRope := llmmath.RoPE(qVec, kVec, pos, headDim, model.Config.RopeFreqBase)
				if qRope == nil || kRope == nil {
					return nil, fmt.Errorf("RoPE failed for layer %d", layer)
				}

				copy(q[qStart:], qRope)
				copy(k[kStart:], kRope)
			}

			model.Cache.Update(layer, k[i*kvDim:], v[i*kvDim:], nKvHeads, headDim)
		}

		pastK, pastV := model.Cache.GetKV(layer, nKvHeads, headDim)
		pastSeqLen := model.Cache.Size

		repeat := nHeads / nKvHeads
		if repeat > 1 {
			pastK = llmmath.RepeatKV(pastK, nKvHeads, nHeads, headDim, pastSeqLen)
			pastV = llmmath.RepeatKV(pastV, nKvHeads, nHeads, headDim, pastSeqLen)
			k = llmmath.RepeatKV(k, nKvHeads, nHeads, headDim, seqLen)
			v = llmmath.RepeatKV(v, nKvHeads, nHeads, headDim, seqLen)
		}

		attentionOutput := make([]float32, seqLen*qDim)
		scale := 1.0 / float32(stdmath.Sqrt(float64(headDim)))

		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				qHead := q[s*qDim+h*headDim : s*qDim+(h+1)*headDim]

				attentionScores := make([]float32, pastSeqLen)
				for t := 0; t < pastSeqLen; t++ {
					kHead := pastK[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]

					var score float32 = 0.0
					for d := 0; d < headDim; d++ {
						score += qHead[d] * kHead[d]
					}
					attentionScores[t] = score * scale
				}

				causalOffset := startPos - (pastSeqLen - seqLen)
				for t := 0; t < pastSeqLen; t++ {
					if t > causalOffset+s {
						attentionScores[t] = float32(stdmath.Inf(-1))
					}
				}

				attentionProbs := llmmath.VectorSoftmax(attentionScores)

				outputHead := make([]float32, headDim)
				for t := 0; t < pastSeqLen; t++ {
					vHead := pastV[t*nHeads*headDim+h*headDim : t*nHeads*headDim+(h+1)*headDim]
					for d := 0; d < headDim; d++ {
						outputHead[d] += attentionProbs[t] * vHead[d]
					}
				}

				copy(attentionOutput[s*qDim+h*headDim:], outputHead)
			}
		}

		attentionOutput = llmmath.MatMul(attentionOutput, blockWeights.AttentionOutput, seqLen, embeddingDim, qDim)
		if attentionOutput == nil {
			return nil, fmt.Errorf("attention output projection failed for layer %d", layer)
		}

		hiddenStates = llmmath.VectorAdd(residual, attentionOutput)

		residual = make([]float32, len(hiddenStates))
		copy(residual, hiddenStates)

		normedHidden = make([]float32, seqLen*embeddingDim)
		for i := 0; i < seqLen; i++ {
			normed := llmmath.RMSNorm(hiddenStates[i*embeddingDim:(i+1)*embeddingDim], blockWeights.FeedForwardNorm, model.Config.LayerNormRmsEps)
			copy(normedHidden[i*embeddingDim:], normed)
		}

		ffnGate := llmmath.MatMul(normedHidden, blockWeights.FeedForwardGate, seqLen, model.Config.FeedForwardLength, embeddingDim)
		ffnUp := llmmath.MatMul(normedHidden, blockWeights.FeedForwardUp, seqLen, model.Config.FeedForwardLength, embeddingDim)

		if ffnGate == nil || ffnUp == nil {
			return nil, fmt.Errorf("FFN gate/up projection failed for layer %d", layer)
		}

		swigluOutput := llmmath.SwiGLU(ffnGate, ffnUp)

		ffnOutput := llmmath.MatMul(swigluOutput, blockWeights.FeedForwardDown, seqLen, embeddingDim, model.Config.FeedForwardLength)
		if ffnOutput == nil {
			return nil, fmt.Errorf("FFN down projection failed for layer %d", layer)
		}

		hiddenStates = llmmath.VectorAdd(residual, ffnOutput)
	}

	lastTokenHidden := hiddenStates[(seqLen-1)*embeddingDim : seqLen*embeddingDim]
	normedOutput := llmmath.RMSNorm(lastTokenHidden, model.Weights.OutputNorm, model.Config.LayerNormRmsEps)

	if model.Weights.Output != nil {
		logits := llmmath.MatMulTransposed([]float32(normedOutput), model.Weights.Output, 1, model.Config.VocabSize, embeddingDim)
		if logits == nil {
			return nil, fmt.Errorf("output projection failed")
		}
		return logits, nil
	} else {
		logits := llmmath.MatMul([]float32(normedOutput), model.Weights.TokenEmbedding, 1, model.Config.VocabSize, embeddingDim)
		if logits == nil {
			return nil, fmt.Errorf("output projection failed")
		}
		return logits, nil
	}
}

func (model *Qwen3Model) Generate(tokens []int, maxNewTokens int, temperature float32) ([]int, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("input tokens is empty")
	}

	model.ResetCache()

	generatedTokens := make([]int, len(tokens))
	copy(generatedTokens, tokens)

	startPos := 0

	for i := 0; i < maxNewTokens; i++ {
		var inputTokens []int
		if i == 0 {
			inputTokens = generatedTokens
		} else {
			inputTokens = generatedTokens[len(generatedTokens)-1:]
		}

		logits, err := model.Forward(inputTokens, startPos)
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
			nextToken = llmmath.Argmax(probs)
		}

		generatedTokens = append(generatedTokens, nextToken)

		if nextToken == 151645 || nextToken == 151643 {
			break
		}
	}

	return generatedTokens, nil
}
