package math

import (
	"math"
	"math/rand"
)

func MatMul(A, B []float32, M, N, K int) []float32 {
	if len(A) != M*K || len(B) != K*N {
		return nil
	}

	C := make([]float32, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32 = 0.0
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}

	return C
}

func MatMulTransposed(A, B []float32, M, N, K int) []float32 {
	if len(A) != M*K || len(B) != N*K {
		return nil
	}

	C := make([]float32, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32 = 0.0
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[j*K+k]
			}
			C[i*N+j] = sum
		}
	}

	return C
}

func VectorAdd(a, b []float32) []float32 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func VectorMul(a, b []float32) []float32 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

func VectorScale(a []float32, scale float32) []float32 {
	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] * scale
	}
	return result
}

func VectorDot(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var sum float32 = 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func VectorNorm(a []float32) float32 {
	var sum float32 = 0.0
	for i := range a {
		sum += a[i] * a[i]
	}
	return float32(math.Sqrt(float64(sum)))
}

func VectorSoftmax(a []float32) []float32 {
	if len(a) == 0 {
		return nil
	}

	maxVal := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > maxVal {
			maxVal = a[i]
		}
	}

	var sum float32 = 0.0
	expValues := make([]float32, len(a))
	for i := range a {
		expValues[i] = float32(math.Exp(float64(a[i] - maxVal)))
		sum += expValues[i]
	}

	for i := range expValues {
		expValues[i] /= sum
	}

	return expValues
}

func RMSNorm(x, weight []float32, eps float32) []float32 {
	if len(x) != len(weight) {
		return nil
	}

	result := make([]float32, len(x))
	RMSNormInPlace(result, x, weight, eps)
	return result
}

// RMSNormInPlace computes the same RMSNorm but writes directly into dst.
// dst must be the same length as x and weight.
func RMSNormInPlace(dst, x, weight []float32, eps float32) {
	var sum float32 = 0.0
	for i := range x {
		sum += x[i] * x[i]
	}
	mean := sum / float32(len(x))
	rms := float32(math.Sqrt(float64(mean + eps)))

	for i := range x {
		dst[i] = (x[i] / rms) * weight[i]
	}
}

func Silu(x float32) float32 {
	return x * Sigmoid(x)
}

func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func SwiGLU(x1, x2 []float32) []float32 {
	if len(x1) != len(x2) {
		return nil
	}

	result := make([]float32, len(x1))
	for i := range x1 {
		result[i] = Silu(x1[i]) * x2[i]
	}
	return result
}

type RoPEMode int

const (
	RoPE_NORMAL RoPEMode = iota // consecutive pairs: (0,1), (2,3), ...
	RoPE_NEOX                    // first-half with second-half: (0, dim/2), (1, dim/2+1), ...
)

// RoPE applies Rotary Position Embedding.
// mode=RoPE_NORMAL: pairs (i, i+1) with freq_base^(-i/dim)
// mode=RoPE_NEOX:   pairs (i, i+dim/2) with freq_base^(-2i/dim))
func RoPE(q, k []float32, pos int, dim int, ropeFreqBase float32, mode RoPEMode) ([]float32, []float32) {
	if (q != nil && len(q) != dim) || (k != nil && len(k) != dim) {
		return nil, nil
	}

	var qOut []float32
	var kOut []float32

	if q != nil {
		qOut = make([]float32, dim)
	}
	if k != nil {
		kOut = make([]float32, dim)
	}

	half := dim / 2
	for j := 0; j < half; j++ {
		i := 2 * j
		freq := 1.0 / float32(math.Pow(float64(ropeFreqBase), float64(i)/float64(dim)))
		angle := float32(pos) * freq

		cos := float32(math.Cos(float64(angle)))
		sin := float32(math.Sin(float64(angle)))

		switch mode {
		case RoPE_NORMAL:
			if q != nil {
				qOut[i] = q[i]*cos - q[i+1]*sin
				qOut[i+1] = q[i]*sin + q[i+1]*cos
			}
			if k != nil {
				kOut[i] = k[i]*cos - k[i+1]*sin
				kOut[i+1] = k[i]*sin + k[i+1]*cos
			}
		case RoPE_NEOX:
			if q != nil {
				qOut[j] = q[j]*cos - q[j+half]*sin
				qOut[j+half] = q[j]*sin + q[j+half]*cos
			}
			if k != nil {
				kOut[j] = k[j]*cos - k[j+half]*sin
				kOut[j+half] = k[j]*sin + k[j+half]*cos
			}
		}
	}

	return qOut, kOut
}

func ComputeAttentionScores(q, k []float32, dim int) []float32 {
	scale := 1.0 / float32(math.Sqrt(float64(dim)))

	scores := make([]float32, len(q))
	for i := range q {
		scores[i] = q[i] * k[i] * scale
	}

	return scores
}

func ApplyCausalMask(attention []float32, seqLen int) {
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			attention[i*seqLen+j] = float32(math.Inf(-1))
		}
	}
}

func Argmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}

	maxIdx := 0
	maxVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func SampleCategorical(probs []float32) int {
	if len(probs) == 0 {
		return -1
	}

	r := rand.Float32()
	cumulative := float32(0.0)
	
	for i := range probs {
		cumulative += probs[i]
		if r < cumulative {
			return i
		}
	}
	
	return len(probs) - 1
}

func EmbeddingLookupTokenFirst(embeddings []float32, vocabSize, embeddingDim int, tokenId int) []float32 {
	if tokenId < 0 || tokenId >= vocabSize {
		return nil
	}

	result := make([]float32, embeddingDim)
	for i := 0; i < embeddingDim; i++ {
		result[i] = embeddings[tokenId*embeddingDim+i]
	}
	return result
}

func EmbeddingLookupDimFirst(embeddings []float32, vocabSize, embeddingDim int, tokenId int) []float32 {
	if tokenId < 0 || tokenId >= vocabSize {
		return nil
	}

	result := make([]float32, embeddingDim)
	for i := 0; i < embeddingDim; i++ {
		result[i] = embeddings[i*vocabSize+tokenId]
	}
	return result
}

func RepeatKV(kv []float32, nKvHeads, nHeads int, headDim, seqLen int) []float32 {
	repeat := nHeads / nKvHeads
	if repeat == 1 {
		return kv
	}

	result := make([]float32, nHeads*headDim*seqLen)

	for s := 0; s < seqLen; s++ {
		for h := 0; h < nKvHeads; h++ {
			for r := 0; r < repeat; r++ {
				srcIdx := s*nKvHeads*headDim + h*headDim
				dstIdx := s*nHeads*headDim + (h*repeat+r)*headDim

				for d := 0; d < headDim; d++ {
					result[dstIdx+d] = kv[srcIdx+d]
				}
			}
		}
	}

	return result
}

func SplitHeads(x []float32, nHeads, headDim, seqLen int) [][][]float32 {
	result := make([][][]float32, seqLen)
	for s := 0; s < seqLen; s++ {
		result[s] = make([][]float32, nHeads)
		for h := 0; h < nHeads; h++ {
			result[s][h] = make([]float32, headDim)
			for d := 0; d < headDim; d++ {
				result[s][h][d] = x[s*nHeads*headDim+h*headDim+d]
			}
		}
	}
	return result
}

func MergeHeads(x [][][]float32, nHeads, headDim, seqLen int) []float32 {
	result := make([]float32, seqLen*nHeads*headDim)
	for s := 0; s < seqLen; s++ {
		for h := 0; h < nHeads; h++ {
			for d := 0; d < headDim; d++ {
				result[s*nHeads*headDim+h*headDim+d] = x[s][h][d]
			}
		}
	}
	return result
}
