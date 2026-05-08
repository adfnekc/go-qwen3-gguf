package model

type KVCache struct {
	Keys     [][][]float32
	Values   [][][]float32
	Size     []int
}

func NewKVCache(maxSeqLen, nLayers, nKvHeads, headDim int) *KVCache {
	keys := make([][][]float32, nLayers)
	values := make([][][]float32, nLayers)
	size := make([]int, nLayers)
	for i := 0; i < nLayers; i++ {
		keys[i] = make([][]float32, 0, maxSeqLen)
		values[i] = make([][]float32, 0, maxSeqLen)
	}
	return &KVCache{Keys: keys, Values: values, Size: size}
}

func (c *KVCache) Update(layer int, key, value []float32, nKvHeads, headDim int) {
	k := make([]float32, len(key))
	v := make([]float32, len(value))
	copy(k, key)
	copy(v, value)
	c.Keys[layer] = append(c.Keys[layer], k)
	c.Values[layer] = append(c.Values[layer], v)
	c.Size[layer]++
}

func (c *KVCache) GetKV(layer int, nKvHeads, headDim int) ([]float32, []float32) {
	size := c.Size[layer]
	kvSize := nKvHeads * headDim
	keys := make([]float32, size*kvSize)
	values := make([]float32, size*kvSize)
	for i := 0; i < size; i++ {
		copy(keys[i*kvSize:], c.Keys[layer][i])
		copy(values[i*kvSize:], c.Values[layer][i])
	}
	return keys, values
}

func (c *KVCache) GetSize(layer int) int {
	return c.Size[layer]
}
