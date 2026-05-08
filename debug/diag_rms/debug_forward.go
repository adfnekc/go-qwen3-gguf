package main

import (
    "fmt"
    "math"
    "os"
    "github.com/adfnekc/go-qwen3-gguf/gguf"
    "github.com/adfnekc/go-qwen3-gguf/model"
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

    m := model.NewQwen3Model(config, weights)

    tokens := []int{6023} // "hi"
    logits, err := m.Forward(tokens, 0)
    if err != nil { fmt.Printf("Forward failed: %v\n", err); os.Exit(1) }

    topToken := 0
    topVal := logits[0]
    for i, v := range logits {
        if v > topVal { topVal = v; topToken = i }
    }
    fmt.Printf("Top token: #%d = %.4f\n", topToken, topVal)

    // Print top 10
    type scored struct { id int; score float32 }
    var top10 []scored
    for i := 0; i < 10; i++ {
        best := -1
        bestVal := float32(math.Inf(-1))
        for j, v := range logits {
            skip := false
            for _, t := range top10 { if t.id == j { skip = true; break } }
            if !skip && v > bestVal { bestVal = v; best = j }
        }
        if best >= 0 { top10 = append(top10, scored{best, bestVal}) }
    }
    for _, t := range top10 {
        fmt.Printf("  #%d: %.4f\n", t.id, t.score)
    }

    // Also check: what if we use Generate with temp=0?
    fmt.Println("\n=== Generate with temp=0, 5 tokens ===")
    m2 := model.NewQwen3Model(config, weights)
    result, err := m2.Generate(tokens, 5, 0)
    if err != nil { fmt.Printf("Generate failed: %v\n", err); os.Exit(1) }
    fmt.Printf("Generated IDs: %v\n", result)
    for _, id := range result[len(tokens):] {
        fmt.Printf("  Generated: #%d\n", id)
    }
}
