package math

import (
	"testing"
)

func TestEmbeddingLookupTokenFirst(t *testing.T) {
	vocabSize := 4
	embeddingDim := 3

	embeddings := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}

	token0 := EmbeddingLookupTokenFirst(embeddings, vocabSize, embeddingDim, 0)
	if len(token0) != embeddingDim {
		t.Errorf("Expected length %d, got %d", embeddingDim, len(token0))
	}
	for i := 0; i < embeddingDim; i++ {
		if token0[i] != float32(i+1) {
			t.Errorf("token0[%d] = %f, expected %f", i, token0[i], float32(i+1))
		}
	}

	token2 := EmbeddingLookupTokenFirst(embeddings, vocabSize, embeddingDim, 2)
	expected := []float32{7, 8, 9}
	for i := 0; i < embeddingDim; i++ {
		if token2[i] != expected[i] {
			t.Errorf("token2[%d] = %f, expected %f", i, token2[i], expected[i])
		}
	}
}

func TestEmbeddingLookupDimFirst(t *testing.T) {
	vocabSize := 4
	embeddingDim := 3

	embeddings := []float32{
		1, 4, 7, 10,
		2, 5, 8, 11,
		3, 6, 9, 12,
	}

	token0 := EmbeddingLookupDimFirst(embeddings, vocabSize, embeddingDim, 0)
	if len(token0) != embeddingDim {
		t.Errorf("Expected length %d, got %d", embeddingDim, len(token0))
	}
	expected := []float32{1, 2, 3}
	for i := 0; i < embeddingDim; i++ {
		if token0[i] != expected[i] {
			t.Errorf("token0[%d] = %f, expected %f", i, token0[i], expected[i])
		}
	}
}

func TestMatMulTransposedOutput(t *testing.T) {
	embeddingDim := 3
	vocabSize := 4

	hidden := []float32{1, 0, 0}

	weights := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}

	logits := MatMulTransposed(hidden, weights, 1, vocabSize, embeddingDim)

	expected := []float32{1, 4, 7, 10}
	for i := 0; i < vocabSize; i++ {
		if logits[i] != expected[i] {
			t.Errorf("logits[%d] = %f, expected %f", i, logits[i], expected[i])
		}
	}
}

func TestArgmax(t *testing.T) {
	v := []float32{1, 3, 2, 5, 4}
	idx := Argmax(v)
	if idx != 3 {
		t.Errorf("Argmax returned %d, expected 3", idx)
	}

	v2 := []float32{0, 0, 0}
	idx2 := Argmax(v2)
	if idx2 != 0 {
		t.Errorf("Argmax returned %d, expected 0", idx2)
	}
}
