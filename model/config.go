package model

import (
	"fmt"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
)

type Qwen3Config struct {
	ContextLength        int
	EmbeddingLength      int
	BlockCount           int
	FeedForwardLength    int
	AttentionHeadCount   int
	AttentionHeadCountKv int
	AttentionKeyLength   int
	AttentionValueLength int
	RopeFreqBase         float32
	LayerNormRmsEps      float32
	VocabSize            int
	QKVRMSNorm           bool
}

type GGUFReader interface {
	GetMetadata(key string) (interface{}, bool)
	GetMetadataString(key string) (string, bool)
	GetMetadataInt(key string) (int64, bool)
	GetMetadataFloat(key string) (float64, bool)
	GetTensor(name string) (*gguf.GGUFTensorInfo, []byte, bool)
}

func LoadQwen3Config(reader GGUFReader) (*Qwen3Config, error) {
	config := &Qwen3Config{}

	var ok bool
	var val int64

	if val, ok = reader.GetMetadataInt("qwen3.context_length"); !ok {
		if val, ok = reader.GetMetadataInt("llama.context_length"); !ok {
			return nil, fmt.Errorf("context_length not found")
		}
	}
	config.ContextLength = int(val)

	if val, ok = reader.GetMetadataInt("qwen3.embedding_length"); !ok {
		if val, ok = reader.GetMetadataInt("llama.embedding_length"); !ok {
			return nil, fmt.Errorf("embedding_length not found")
		}
	}
	config.EmbeddingLength = int(val)

	if val, ok = reader.GetMetadataInt("qwen3.block_count"); !ok {
		if val, ok = reader.GetMetadataInt("llama.block_count"); !ok {
			return nil, fmt.Errorf("block_count not found")
		}
	}
	config.BlockCount = int(val)

	if val, ok = reader.GetMetadataInt("qwen3.feed_forward_length"); !ok {
		if val, ok = reader.GetMetadataInt("llama.feed_forward_length"); !ok {
			return nil, fmt.Errorf("feed_forward_length not found")
		}
	}
	config.FeedForwardLength = int(val)

	if val, ok = reader.GetMetadataInt("qwen3.attention.head_count"); !ok {
		if val, ok = reader.GetMetadataInt("llama.attention.head_count"); !ok {
			return nil, fmt.Errorf("attention.head_count not found")
		}
	}
	config.AttentionHeadCount = int(val)

	if val, ok = reader.GetMetadataInt("qwen3.attention.head_count_kv"); !ok {
		if val, ok = reader.GetMetadataInt("llama.attention.head_count_kv"); !ok {
			config.AttentionHeadCountKv = config.AttentionHeadCount
		} else {
			config.AttentionHeadCountKv = int(val)
		}
	} else {
		config.AttentionHeadCountKv = int(val)
	}

	if val, ok = reader.GetMetadataInt("qwen3.attention.key_length"); ok {
		config.AttentionKeyLength = int(val)
	} else if val, ok = reader.GetMetadataInt("llama.attention.key_length"); ok {
		config.AttentionKeyLength = int(val)
	} else {
		config.AttentionKeyLength = config.EmbeddingLength / config.AttentionHeadCount
	}

	if val, ok = reader.GetMetadataInt("qwen3.attention.value_length"); ok {
		config.AttentionValueLength = int(val)
	} else if val, ok = reader.GetMetadataInt("llama.attention.value_length"); ok {
		config.AttentionValueLength = int(val)
	} else {
		config.AttentionValueLength = config.EmbeddingLength / config.AttentionHeadCount
	}

	if ropeFreqBase, ok := reader.GetMetadataFloat("qwen3.rope.freq_base"); ok {
		config.RopeFreqBase = float32(ropeFreqBase)
	} else if ropeFreqBase, ok := reader.GetMetadataFloat("llama.rope.freq_base"); ok {
		config.RopeFreqBase = float32(ropeFreqBase)
	} else {
		config.RopeFreqBase = 10000.0
	}

	if rmsEps, ok := reader.GetMetadataFloat("qwen3.attention.layer_norm_rms_epsilon"); ok {
		config.LayerNormRmsEps = float32(rmsEps)
	} else if rmsEps, ok := reader.GetMetadataFloat("llama.attention.layer_norm_rms_epsilon"); ok {
		config.LayerNormRmsEps = float32(rmsEps)
	} else {
		config.LayerNormRmsEps = 1e-5
	}

	if val, ok = reader.GetMetadataInt("tokenizer.ggml.vocab_size"); ok {
		config.VocabSize = int(val)
	} else {
		if tokensMeta, ok := reader.GetMetadata("tokenizer.ggml.tokens"); ok {
			if tokensList, ok := tokensMeta.([]string); ok {
				config.VocabSize = len(tokensList)
			} else if tokensList, ok := tokensMeta.([]interface{}); ok {
				config.VocabSize = len(tokensList)
			} else {
				return nil, fmt.Errorf("vocab_size not found and cannot infer from tokenizer.ggml.tokens")
			}
		} else {
			return nil, fmt.Errorf("vocab_size not found")
		}
	}

	if _, _, ok := reader.GetTensor("blk.0.attn_q_norm.weight"); ok {
		config.QKVRMSNorm = true
	} else if _, _, ok := reader.GetTensor("model.layers.0.self_attn.q_norm.weight"); ok {
		config.QKVRMSNorm = true
	}

	return config, nil
}
