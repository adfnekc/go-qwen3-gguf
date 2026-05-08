package model

import (
	"fmt"

	gguf "github.com/adfnekc/go-qwen3-gguf/gguf"
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

func loadTensorDequant(reader GGUFReader, name string) ([]float32, error) {
	info, data, ok := reader.GetTensor(name)
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}
	result := gguf.DequantizeTensor(info, data)
	if result == nil {
		return nil, fmt.Errorf("failed to dequantize %s", name)
	}
	return result, nil
}

func loadTensorMMap(r *gguf.GGUFReader, mmapFile *gguf.MMapFile, name string) ([]float32, error) {
	info, _, ok := r.GetTensor(name)
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}
	data, err := mmapFile.GetSlice(r.DataStart+int64(info.Offset), int(calculateTensorSize(info)))
	if err != nil {
		return nil, fmt.Errorf("failed to get %s data: %w", name, err)
	}
	result := gguf.DequantizeTensor(info, data)
	if result == nil {
		return nil, fmt.Errorf("failed to dequantize %s", name)
	}
	return result, nil
}

func resolveTensorName(reader GGUFReader, names []string) (string, bool) {
	for _, name := range names {
		if _, _, ok := reader.GetTensor(name); ok {
			return name, true
		}
	}
	return "", false
}

func resolveBlockPrefix(reader GGUFReader, blockIdx int) (string, bool) {
	prefixes := []string{
		fmt.Sprintf("blk.%d.", blockIdx),
		fmt.Sprintf("model.layers.%d.", blockIdx),
	}
	for _, p := range prefixes {
		if _, _, ok := reader.GetTensor(p + "attn_norm.weight"); ok {
			return p, true
		}
		if _, _, ok := reader.GetTensor(p + "input_layernorm.weight"); ok {
			return p, true
		}
	}
	return "", false
}

func loadFirst(loadFn func(string) ([]float32, error), names []string) ([]float32, error) {
	var lastErr error
	for _, name := range names {
		result, err := loadFn(name)
		if err == nil {
			return result, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

func loadBlockWeights(loadFn func(string) ([]float32, error), prefix string, layer int) (*Qwen3BlockWeights, error) {
	bw := &Qwen3BlockWeights{}

	var err error

	bw.AttentionNorm, err = loadFirst(loadFn, []string{prefix + "attn_norm.weight", prefix + "input_layernorm.weight"})
	if err != nil {
		return nil, fmt.Errorf("attention norm: %w", err)
	}

	bw.AttentionQ, err = loadFirst(loadFn, []string{prefix + "attn_q.weight", prefix + "self_attn.q_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("attention_q: %w", err)
	}

	bw.AttentionK, err = loadFirst(loadFn, []string{prefix + "attn_k.weight", prefix + "self_attn.k_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("attention_k: %w", err)
	}

	bw.AttentionV, err = loadFirst(loadFn, []string{prefix + "attn_v.weight", prefix + "self_attn.v_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("attention_v: %w", err)
	}

	bw.AttentionQNorm, _ = loadFirst(loadFn, []string{prefix + "attn_q_norm.weight", prefix + "self_attn.q_norm.weight"})
	bw.AttentionKNorm, _ = loadFirst(loadFn, []string{prefix + "attn_k_norm.weight", prefix + "self_attn.k_norm.weight"})

	bw.AttentionOutput, err = loadFirst(loadFn, []string{prefix + "attn_output.weight", prefix + "self_attn.o_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("attention_output: %w", err)
	}

	bw.FeedForwardNorm, err = loadFirst(loadFn, []string{prefix + "ffn_norm.weight", prefix + "post_attention_layernorm.weight"})
	if err != nil {
		return nil, fmt.Errorf("ffn_norm: %w", err)
	}

	bw.FeedForwardGate, err = loadFirst(loadFn, []string{prefix + "ffn_gate.weight", prefix + "mlp.gate_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("ffn_gate: %w", err)
	}

	bw.FeedForwardUp, err = loadFirst(loadFn, []string{prefix + "ffn_up.weight", prefix + "mlp.up_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("ffn_up: %w", err)
	}

	bw.FeedForwardDown, err = loadFirst(loadFn, []string{prefix + "ffn_down.weight", prefix + "mlp.down_proj.weight"})
	if err != nil {
		return nil, fmt.Errorf("ffn_down: %w", err)
	}

	return bw, nil
}

func LoadQwen3Weights(reader GGUFReader, config *Qwen3Config) (*Qwen3Weights, error) {
	loadFn := func(name string) ([]float32, error) {
		return loadTensorDequant(reader, name)
	}
	weights := &Qwen3Weights{
		BlockWeights: make([]*Qwen3BlockWeights, config.BlockCount),
	}

	var err error

	weights.TokenEmbedding, err = loadFirst(loadFn, []string{"token_embd.weight", "model.embed_tokens.weight"})
	if err != nil {
		return nil, err
	}

	weights.OutputNorm, err = loadFirst(loadFn, []string{"output_norm.weight", "model.norm.weight"})
	if err != nil {
		return nil, err
	}

	if output, err := loadFirst(loadFn, []string{"output.weight", "lm_head.weight"}); err == nil {
		weights.Output = output
	}

	for i := 0; i < config.BlockCount; i++ {
		prefix, ok := resolveBlockPrefix(reader, i)
		if !ok {
			return nil, fmt.Errorf("block %d not found", i)
		}
		bw, err := loadBlockWeights(loadFn, prefix, i)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
		weights.BlockWeights[i] = bw
	}

	return weights, nil
}

func LoadQwen3WeightsMMap(reader *gguf.GGUFReader, config *Qwen3Config, mmapFile *gguf.MMapFile) (*Qwen3Weights, error) {
	loadFn := func(name string) ([]float32, error) {
		return loadTensorMMap(reader, mmapFile, name)
	}
	weights := &Qwen3Weights{
		BlockWeights: make([]*Qwen3BlockWeights, config.BlockCount),
	}

	var err error

	weights.TokenEmbedding, err = loadFirst(loadFn, []string{"token_embd.weight", "model.embed_tokens.weight"})
	if err != nil {
		return nil, err
	}

	weights.OutputNorm, err = loadFirst(loadFn, []string{"output_norm.weight", "model.norm.weight"})
	if err != nil {
		return nil, err
	}

	if output, err := loadFirst(loadFn, []string{"output.weight", "lm_head.weight"}); err == nil {
		weights.Output = output
	}

	for i := 0; i < config.BlockCount; i++ {
		prefix, ok := resolveBlockPrefix(reader, i)
		if !ok {
			return nil, fmt.Errorf("block %d not found", i)
		}
		bw, err := loadBlockWeights(loadFn, prefix, i)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
		weights.BlockWeights[i] = bw
	}

	return weights, nil
}
