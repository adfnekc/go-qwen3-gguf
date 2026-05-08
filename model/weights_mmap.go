package model

import (
	"fmt"

	"gguf/gguf"
)

func calculateTensorSize(tensor *gguf.GGUFTensorInfo) uint64 {
	elementSize := tensor.Type.ElementSize()
	if elementSize == 0 {
		return 0
	}

	numElements := uint64(1)
	for _, dim := range tensor.Shape {
		numElements *= dim
	}

	switch tensor.Type {
	case gguf.GGML_TYPE_Q4_0, gguf.GGML_TYPE_Q4_1, gguf.GGML_TYPE_Q5_0, gguf.GGML_TYPE_Q5_1,
		gguf.GGML_TYPE_Q8_0, gguf.GGML_TYPE_Q8_1, gguf.GGML_TYPE_Q2_K, gguf.GGML_TYPE_Q3_K,
		gguf.GGML_TYPE_Q4_K, gguf.GGML_TYPE_Q5_K, gguf.GGML_TYPE_Q6_K, gguf.GGML_TYPE_Q8_K:
		blockSize := uint64(32)
		return (numElements + blockSize - 1) / blockSize * elementSize
	default:
		return numElements * elementSize
	}
}

func tensorAbsOffset(reader *gguf.GGUFReader, tensor *gguf.GGUFTensorInfo) int64 {
	return reader.DataStart + int64(tensor.Offset)
}

func LoadQwen3WeightsMMap(reader *gguf.GGUFReader, config *Qwen3Config, mmapFile *gguf.MMapFile) (*Qwen3Weights, error) {
	weights := &Qwen3Weights{
		BlockWeights: make([]*Qwen3BlockWeights, config.BlockCount),
	}

	tensorInfo, _, ok := reader.GetTensor("token_embd.weight")
	if !ok {
		tensorInfo, _, ok = reader.GetTensor("model.embed_tokens.weight")
	}
	if !ok {
		return nil, fmt.Errorf("token_embd.weight not found")
	}
	data, err := mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
	if err != nil {
		return nil, fmt.Errorf("failed to get token_embd.weight data: %w", err)
	}
	weights.TokenEmbedding = gguf.DequantizeTensor(tensorInfo, data)
	if weights.TokenEmbedding == nil {
		return nil, fmt.Errorf("failed to dequantize token_embd.weight")
	}

	tensorInfo, _, ok = reader.GetTensor("output_norm.weight")
	if !ok {
		tensorInfo, _, ok = reader.GetTensor("model.norm.weight")
	}
	if !ok {
		return nil, fmt.Errorf("output_norm.weight not found")
	}
	data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
	if err != nil {
		return nil, fmt.Errorf("failed to get output_norm.weight data: %w", err)
	}
	weights.OutputNorm = gguf.DequantizeTensor(tensorInfo, data)
	if weights.OutputNorm == nil {
		return nil, fmt.Errorf("failed to dequantize output_norm.weight")
	}

	tensorInfo, _, ok = reader.GetTensor("output.weight")
	if !ok {
		tensorInfo, _, ok = reader.GetTensor("lm_head.weight")
	}
	if ok {
		data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
		if err != nil {
			return nil, fmt.Errorf("failed to get output.weight data: %w", err)
		}
		weights.Output = gguf.DequantizeTensor(tensorInfo, data)
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
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionNorm = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.AttentionNorm == nil {
			return nil, fmt.Errorf("block %d attention norm not found", i)
		}

		qNames := []string{"attn_q.weight", "self_attn.q_proj.weight"}
		for _, name := range qNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionQ = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.AttentionQ == nil {
			return nil, fmt.Errorf("block %d attention_q not found", i)
		}

		kNames := []string{"attn_k.weight", "self_attn.k_proj.weight"}
		for _, name := range kNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionK = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.AttentionK == nil {
			return nil, fmt.Errorf("block %d attention_k not found", i)
		}

		vNames := []string{"attn_v.weight", "self_attn.v_proj.weight"}
		for _, name := range vNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionV = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.AttentionV == nil {
			return nil, fmt.Errorf("block %d attention_v not found", i)
		}

		qNormNames := []string{"attn_q_norm.weight", "self_attn.q_norm.weight"}
		for _, name := range qNormNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionQNorm = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}

		kNormNames := []string{"attn_k_norm.weight", "self_attn.k_norm.weight"}
		for _, name := range kNormNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionKNorm = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}

		oNames := []string{"attn_output.weight", "self_attn.o_proj.weight"}
		for _, name := range oNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.AttentionOutput = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.AttentionOutput == nil {
			return nil, fmt.Errorf("block %d attention_output not found", i)
		}

		ffnNormNames := []string{"ffn_norm.weight", "post_attention_layernorm.weight"}
		for _, name := range ffnNormNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.FeedForwardNorm = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.FeedForwardNorm == nil {
			return nil, fmt.Errorf("block %d ffn_norm not found", i)
		}

		gateNames := []string{"ffn_gate.weight", "mlp.gate_proj.weight"}
		for _, name := range gateNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.FeedForwardGate = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.FeedForwardGate == nil {
			return nil, fmt.Errorf("block %d ffn_gate not found", i)
		}

		upNames := []string{"ffn_up.weight", "mlp.up_proj.weight"}
		for _, name := range upNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.FeedForwardUp = gguf.DequantizeTensor(tensorInfo, data)
				break
			}
		}
		if blockWeights.FeedForwardUp == nil {
			return nil, fmt.Errorf("block %d ffn_up not found", i)
		}

		downNames := []string{"ffn_down.weight", "mlp.down_proj.weight"}
		for _, name := range downNames {
			if tensorInfo, _, ok = reader.GetTensor(prefix + name); ok {
				data, err = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(calculateTensorSize(tensorInfo)))
				if err != nil {
					return nil, fmt.Errorf("failed to get %s data: %w", prefix+name, err)
				}
				blockWeights.FeedForwardDown = gguf.DequantizeTensor(tensorInfo, data)
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
