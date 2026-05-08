package model

type KVCache struct {
	Keys    [][][]float32
	Values  [][][]float32
	Size    []int
	MaxSize int
}

func NewKVCache(maxSeqLen, nLayers, nKvHeads, headDim int) *KVCache {
	keys := make([][][]float32, nLayers)
	values := make([][][]float32, nLayers)
	size := make([]int, nLayers)

	for i := 0; i < nLayers; i++ {
		keys[i] = make([][]float32, maxSeqLen)
		values[i] = make([][]float32, maxSeqLen)
		for j := 0; j < maxSeqLen; j++ {
			keys[i][j] = make([]float32, nKvHeads*headDim)
			values[i][j] = make([]float32, nKvHeads*headDim)
		}
	}

	return &KVCache{
		Keys:    keys,
		Values:  values,
		Size:    size,
		MaxSize: maxSeqLen,
	}
}

func (cache *KVCache) Update(layerIdx int, keys, values []float32, nKvHeads, headDim int) {
	if cache.Size[layerIdx] >= cache.MaxSize {
		return
	}

	copy(cache.Keys[layerIdx][cache.Size[layerIdx]], keys)
	copy(cache.Values[layerIdx][cache.Size[layerIdx]], values)
	cache.Size[layerIdx]++
}

func (cache *KVCache) GetKV(layerIdx int, nKvHeads, headDim int) ([]float32, []float32) {
	size := cache.Size[layerIdx]
	keys := make([]float32, size*nKvHeads*headDim)
	values := make([]float32, size*nKvHeads*headDim)

	for i := 0; i < size; i++ {
		copy(keys[i*nKvHeads*headDim:], cache.Keys[layerIdx][i])
		copy(values[i*nKvHeads*headDim:], cache.Values[layerIdx][i])
	}

	return keys, values
}

func (cache *KVCache) GetSize(layerIdx int) int {
	return cache.Size[layerIdx]
}
