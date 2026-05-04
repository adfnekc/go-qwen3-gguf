package model

import (
	"fmt"

	"gguf/gguf"
)

type Qwen3Weights struct {
	TokenEmbedding []float32
	OutputNorm     []float32
	Output         []float32
	BlockWeights   []*Qwen3BlockWeights
}

type Qwen3BlockWeights struct {
	AttentionNorm   []float32
	AttentionQ      []float32
	AttentionK      []float32
	AttentionV      []float32
	AttentionQNorm  []float32
	AttentionKNorm  []float32
	AttentionOutput []float32
	FeedForwardNorm []float32
	FeedForwardGate []float32
	FeedForwardUp   []float32
	FeedForwardDown []float32
}

func LoadQwen3Weights(reader GGUFReader, config *Qwen3Config) (*Qwen3Weights, error) {
	weights := &Qwen3Weights{
		BlockWeights: make([]*Qwen3BlockWeights, config.BlockCount),
	}

	tensorInfo, tensorData, ok := reader.GetTensor("token_embd.weight")
	if !ok {
		tensorInfo, tensorData, ok = reader.GetTensor("model.embed_tokens.weight")
	}
	if !ok {
		return nil, fmt.Errorf("token_embd.weight not found")
	}
	weights.TokenEmbedding = gguf.DequantizeTensor(tensorInfo, tensorData)
	if weights.TokenEmbedding == nil {
		return nil, fmt.Errorf("failed to dequantize token_embd.weight")
	}

	tensorInfo, tensorData, ok = reader.GetTensor("output_norm.weight")
	if !ok {
		tensorInfo, tensorData, ok = reader.GetTensor("model.norm.weight")
	}
	if !ok {
		return nil, fmt.Errorf("output_norm.weight not found")
	}
	weights.OutputNorm = gguf.DequantizeTensor(tensorInfo, tensorData)
	if weights.OutputNorm == nil {
		return nil, fmt.Errorf("failed to dequantize output_norm.weight")
	}

	tensorInfo, tensorData, ok = reader.GetTensor("output.weight")
	if !ok {
		tensorInfo, tensorData, ok = reader.GetTensor("lm_head.weight")
	}
	if ok {
		weights.Output = gguf.DequantizeTensor(tensorInfo, tensorData)
		if weights.Output == nil {
			return nil, fmt.Errorf("failed to dequantize output.weight")
		}
	} else {
		weights.Output = nil
	}

	for i := 0; i < config.BlockCount; i++ {
		blockWeights := &Qwen3BlockWeights{}

		prefixes := []string{
			fmt.Sprintf("blk.%d.", i),
			fmt.Sprintf("model.layers.%d.", i),
		}

		var prefix string
		for _, p := range prefixes {
			if _, _, ok := reader.GetTensor(p + "attn_norm.weight"); ok {
				prefix = p
				break
			}
			if _, _, ok := reader.GetTensor(p + "input_layernorm.weight"); ok {
				prefix = p
				break
			}
		}

		if prefix == "" {
			return nil, fmt.Errorf("block %d not found", i)
		}

		normNames := []string{"attn_norm.weight", "input_layernorm.weight"}
		for _, name := range normNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionNorm = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.AttentionNorm == nil {
			return nil, fmt.Errorf("block %d attention norm not found", i)
		}

		qNames := []string{"attn_q.weight", "self_attn.q_proj.weight"}
		for _, name := range qNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionQ = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.AttentionQ == nil {
			return nil, fmt.Errorf("block %d attention_q not found", i)
		}

		kNames := []string{"attn_k.weight", "self_attn.k_proj.weight"}
		for _, name := range kNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionK = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.AttentionK == nil {
			return nil, fmt.Errorf("block %d attention_k not found", i)
		}

		vNames := []string{"attn_v.weight", "self_attn.v_proj.weight"}
		for _, name := range vNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionV = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.AttentionV == nil {
			return nil, fmt.Errorf("block %d attention_v not found", i)
		}

		qNormNames := []string{"attn_q_norm.weight", "self_attn.q_norm.weight"}
		for _, name := range qNormNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionQNorm = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}

		kNormNames := []string{"attn_k_norm.weight", "self_attn.k_norm.weight"}
		for _, name := range kNormNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionKNorm = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}

		oNames := []string{"attn_output.weight", "self_attn.o_proj.weight"}
		for _, name := range oNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.AttentionOutput = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.AttentionOutput == nil {
			return nil, fmt.Errorf("block %d attention_output not found", i)
		}

		ffnNormNames := []string{"ffn_norm.weight", "post_attention_layernorm.weight"}
		for _, name := range ffnNormNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.FeedForwardNorm = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.FeedForwardNorm == nil {
			return nil, fmt.Errorf("block %d ffn_norm not found", i)
		}

		gateNames := []string{"ffn_gate.weight", "mlp.gate_proj.weight"}
		for _, name := range gateNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.FeedForwardGate = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.FeedForwardGate == nil {
			return nil, fmt.Errorf("block %d ffn_gate not found", i)
		}

		upNames := []string{"ffn_up.weight", "mlp.up_proj.weight"}
		for _, name := range upNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.FeedForwardUp = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.FeedForwardUp == nil {
			return nil, fmt.Errorf("block %d ffn_up not found", i)
		}

		downNames := []string{"ffn_down.weight", "mlp.down_proj.weight"}
		for _, name := range downNames {
			if tensorInfo, tensorData, ok = reader.GetTensor(prefix + name); ok {
				blockWeights.FeedForwardDown = gguf.DequantizeTensor(tensorInfo, tensorData)
				break
			}
		}
		if blockWeights.FeedForwardDown == nil {
			return nil, fmt.Errorf("block %d ffn_down not found", i)
		}

		weights.BlockWeights[i] = blockWeights
	}

	return weights, nil
}
