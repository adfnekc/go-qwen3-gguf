package model

type KVCache struct {
	Keys    [][][]float32
	Values  [][][]float32
	Size    int
	MaxSize int
}

func NewKVCache(maxSeqLen, nLayers, nKvHeads, headDim int) *KVCache {
	keys := make([][][]float32, nLayers)
	values := make([][][]float32, nLayers)

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
		Size:    0,
		MaxSize: maxSeqLen,
	}
}

func (cache *KVCache) Update(layerIdx int, keys, values []float32, nKvHeads, headDim int) {
	if cache.Size >= cache.MaxSize {
		return
	}

	copy(cache.Keys[layerIdx][cache.Size], keys)
	copy(cache.Values[layerIdx][cache.Size], values)
	cache.Size++
}

func (cache *KVCache) GetKV(layerIdx int, nKvHeads, headDim int) ([]float32, []float32) {
	keys := make([]float32, cache.Size*nKvHeads*headDim)
	values := make([]float32, cache.Size*nKvHeads*headDim)

	for i := 0; i < cache.Size; i++ {
		copy(keys[i*nKvHeads*headDim:], cache.Keys[layerIdx][i])
		copy(values[i*nKvHeads*headDim:], cache.Values[layerIdx][i])
	}

	return keys, values
}
