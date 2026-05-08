package model

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
