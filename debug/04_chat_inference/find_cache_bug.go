package main

import (
	"fmt"
	"math"
	"os"

	"gguf/gguf"
	"gguf/model"
	llmmath "gguf/math"
	"gguf/tokenizer"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(modelPath)
	if err != nil {
		panic(err)
	}
	defer mmapFile.Close()

	config, err := model.LoadQwen3Config(reader)
	if err != nil {
		panic(err)
	}

	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil {
		panic(err)
	}

	tok, err := tokenizer.NewTokenizerFromGGUF(reader)
	if err != nil {
		panic(err)
	}

	imStart, imEnd := 151644, 151645
	prompt := []int{imStart}
	prompt = append(prompt, tok.Encode("user\n")...)
	prompt = append(prompt, tok.Encode("hi")...)
	prompt = append(prompt, imEnd)
	prompt = append(prompt, 198)
	prompt = append(prompt, imStart)
	prompt = append(prompt, tok.Encode("assistant\n")...)

	ed := config.EmbeddingLength
	hd := config.AttentionKeyLength
	nh := config.AttentionHeadCount
	nk := config.AttentionHeadCountKv
	qd := nh * hd
	kd := nk * hd

	_ = qd

	// Build embeddings for all 10 tokens
	embeddings := make([]float32, 10*ed)
	for i, tokId := range prompt {
		e := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, ed, tokId)
		copy(embeddings[i*ed:], e)
	}
	thinkEmb := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, ed, 151667)
	copy(embeddings[9*ed:], thinkEmb)

	// Layer 0 only
	l := 0
	bw := weights.BlockWeights[l]

	// Approach A (isolated): compute Q,K,V for position 9 alone
	normedA := llmmath.RMSNorm(thinkEmb, bw.AttentionNorm, config.LayerNormRmsEps)
	qA := llmmath.MatMulTransposed(normedA, bw.AttentionQ, 1, qd, ed)
	kA := llmmath.MatMulTransposed(normedA, bw.AttentionK, 1, kd, ed)
	vA := llmmath.MatMulTransposed(normedA, bw.AttentionV, 1, kd, ed)

	// QK norm + RoPE for pos=9 (single token)
	applyQKNormRoPE(qA, kA, nk, nh, hd, bw, config, 1, 9)

	// Approach B (batched): compute Q,K,V for all 10 positions together
	normedB := make([]float32, 10*ed)
	for i := 0; i < 10; i++ {
		n := llmmath.RMSNorm(embeddings[i*ed:(i+1)*ed], bw.AttentionNorm, config.LayerNormRmsEps)
		copy(normedB[i*ed:], n)
	}
	qB := llmmath.MatMulTransposed(normedB, bw.AttentionQ, 10, qd, ed)
	kB := llmmath.MatMulTransposed(normedB, bw.AttentionK, 10, kd, ed)
	vB := llmmath.MatMulTransposed(normedB, bw.AttentionV, 10, kd, ed)

	// QK norm + RoPE for positions 0-9
	applyQKNormRoPE(qB, kB, nk, nh, hd, bw, config, 10, 0)

	// Compare position 9
	p9start := 9 * kd
	qB9 := qB[9*qd : 10*qd]
	kB9 := kB[p9start : p9start+kd]
	vB9 := vB[p9start : p9start+kd]

	qDiff := maxAbsDiff(qA, qB9)
	kDiff := maxAbsDiff(kA, kB9)
	vDiff := maxAbsDiff(vA, vB9)

	fmt.Println("=== Layer 0 Q,K,V for position 9 (isolated vs batched) ===")
	fmt.Printf("Q max diff: %.12f  (A RMS=%.4f, B RMS=%.4f)\n", qDiff, rms32(qA), rms32(qB9))
	fmt.Printf("K max diff: %.12f  (A RMS=%.4f, B RMS=%.4f)\n", kDiff, rms32(kA), rms32(kB9))
	fmt.Printf("V max diff: %.12f  (A RMS=%.4f, B RMS=%.4f)\n", vDiff, rms32(vA), rms32(vB9))

	// If diff > 0, check normed hidden
	fmt.Println("\n=== Normed hidden comparison ===")
	normedDiff := maxAbsDiff(normedA, normedB[9*ed:10*ed])
	fmt.Printf("Normed hidden max diff: %.12f  (A RMS=%.4f, B RMS=%.4f)\n",
		normedDiff, rms32(normedA), rms32(normedB[9*ed:10*ed]))

	fmt.Println("\n=== Embedding comparison ===")
	embDiff := maxAbsDiff(thinkEmb, embeddings[9*ed:10*ed])
	fmt.Printf("Embedding max diff: %.12f\n", embDiff)
}

func applyQKNormRoPE(q, k []float32, nKvHeads, nHeads, headDim int, bw *model.Qwen3BlockWeights, config *model.Qwen3Config, seqLen, startPos int) {
	qd := nHeads * headDim
	kd := nKvHeads * headDim

	if config.QKVRMSNorm && bw.AttentionQNorm != nil && bw.AttentionKNorm != nil {
		for i := 0; i < seqLen; i++ {
			for h := 0; h < nHeads; h++ {
				qStart := i*qd + h*headDim
				qNormed := llmmath.RMSNorm(q[qStart:qStart+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
				copy(q[qStart:], qNormed)
			}
			for h := 0; h < nKvHeads; h++ {
				kStart := i*kd + h*headDim
				kNormed := llmmath.RMSNorm(k[kStart:kStart+headDim], bw.AttentionKNorm, config.LayerNormRmsEps)
				copy(k[kStart:], kNormed)
			}
		}
	}
	for i := 0; i < seqLen; i++ {
		pos := startPos + i
		for h := 0; h < nKvHeads; h++ {
			kStart := i*kd + h*headDim
			_, kRope := llmmath.RoPE(nil, k[kStart:kStart+headDim], pos, headDim, config.RopeFreqBase)
			copy(k[kStart:], kRope)
		}
		for h := 0; h < nHeads; h++ {
			qStart := i*qd + h*headDim
			qRope, _ := llmmath.RoPE(q[qStart:qStart+headDim], nil, pos, headDim, config.RopeFreqBase)
			copy(q[qStart:], qRope)
		}
	}
}

func rms32(v []float32) float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return float32(math.Sqrt(sum / float64(len(v))))
}

func maxAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return -1
	}
	var maxDiff float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}
