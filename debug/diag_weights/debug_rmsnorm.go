package main

import (
    "fmt"
    "math"
    "os"
    "gguf/gguf"
    "gguf/model"
    llmmath "gguf/math"
)

func main() {
    modelPath := os.Args[1]
    reader := gguf.NewGGUFReader()
    mmapFile, err := reader.LoadFromFileMMap(modelPath)
    if err != nil { fmt.Printf("Failed to load: %v\n", err); os.Exit(1) }
    defer mmapFile.Close()
    config, err := model.LoadQwen3Config(reader)
    if err != nil { fmt.Printf("Failed to load config: %v\n", err); os.Exit(1) }
    weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
    if err != nil { fmt.Printf("Failed to load weights: %v\n", err); os.Exit(1) }

    nHeads := config.AttentionHeadCount
    nKvHeads := config.AttentionHeadCountKv
    headDim := config.AttentionKeyLength
    qDim := nHeads * headDim
    kvDim := nKvHeads * headDim
    embeddingDim := config.EmbeddingLength

    tokenID := 6023
    embedding := llmmath.EmbeddingLookupTokenFirst(weights.TokenEmbedding, config.VocabSize, embeddingDim, tokenID)
    hiddenStates := make([]float32, embeddingDim)
    copy(hiddenStates, embedding)

    bw := weights.BlockWeights[0]
    normed := llmmath.RMSNorm(hiddenStates, bw.AttentionNorm, config.LayerNormRmsEps)
    q := llmmath.MatMulTransposed(normed, bw.AttentionQ, 1, qDim, embeddingDim)
    k := llmmath.MatMulTransposed(normed, bw.AttentionK, 1, kvDim, embeddingDim)

    // Check Q norm weight per head
    fmt.Printf("Q norm weight: len=%d, RMS=%.6f\n", len(bw.AttentionQNorm), rms32(bw.AttentionQNorm))
    fmt.Printf("K norm weight: len=%d, RMS=%.6f\n", len(bw.AttentionKNorm), rms32(bw.AttentionKNorm))

    // Per-head Q/K stats before and after norm
    for h := 0; h < nKvHeads && h < 4; h++ {
        qStart := h * headDim
        kStart := h * headDim

        qHead := q[qStart : qStart+headDim]
        kHead := k[kStart : kStart+headDim]

        fmt.Printf("\nHead %d:\n", h)
        fmt.Printf("  Q pre-norm:  RMS=%.6f\n", rms32(qHead))
        fmt.Printf("  K pre-norm:  RMS=%.6f\n", rms32(kHead))

        // Compute manually what the post-norm RMS should be
        // After RMSNorm: out = (x / rms(x)) * weight
        qRMS := rms32(qHead)
        qWeighted := make([]float32, headDim)
        for i := range qHead {
            qWeighted[i] = (qHead[i] / qRMS) * bw.AttentionQNorm[i]
        }
        fmt.Printf("  Q manual norm: RMS=%.6f (expected ~%.6f)\n", rms32(qWeighted), rms32(bw.AttentionQNorm))

        kRMS := rms32(kHead)
        kWeighted := make([]float32, headDim)
        for i := range kHead {
            kWeighted[i] = (kHead[i] / kRMS) * bw.AttentionKNorm[i]
        }
        fmt.Printf("  K manual norm: RMS=%.6f (expected ~%.6f)\n", rms32(kWeighted), rms32(bw.AttentionKNorm))

        // Now use the actual RMSNorm function
        qActual := llmmath.RMSNorm(qHead, bw.AttentionQNorm, config.LayerNormRmsEps)
        kActual := llmmath.RMSNorm(kHead, bw.AttentionKNorm, config.LayerNormRmsEps)
        fmt.Printf("  Q actual norm: RMS=%.6f\n", rms32(qActual))
        fmt.Printf("  K actual norm: RMS=%.6f\n", rms32(kActual))

        // Compare manual vs actual
        for i := 0; i < 4; i++ {
            fmt.Printf("    q[%d]: manual=%.6f actual=%.6f\n", i, qWeighted[i], qActual[i])
        }
    }
}

func min32(v []float32) float32 {
    if len(v) == 0 { return 0 }
    m := v[0]
    for _, x := range v[1:] { if x < m { m = x } }
    return m
}

func max32(v []float32) float32 {
    if len(v) == 0 { return 0 }
    m := v[0]
    for _, x := range v[1:] { if x > m { m = x } }
    return m
}

func rms32(v []float32) float32 {
    if len(v) == 0 { return 0 }
    var sum float32
    for _, x := range v { sum += x * x }
    return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
