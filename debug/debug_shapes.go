package main

import (
    "fmt"
    "os"
    "gguf/gguf"
)

func main() {
    modelPath := os.Args[1]
    reader := gguf.NewGGUFReader()
    mmapFile, err := reader.LoadFromFileMMap(modelPath)
    if err != nil {
        fmt.Printf("Failed to load: %v\n", err)
        os.Exit(1)
    }
    defer mmapFile.Close()

    // Print key tensor shapes
    keyTensors := []string{
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "output_norm.weight",
        "output.weight",
    }

    for _, name := range keyTensors {
        tensor, data, ok := reader.GetTensor(name)
        if ok {
            fmt.Printf("%s: shape=%v, type=%s, data_len=%d\n", name, tensor.Shape, tensor.Type.String(), len(data))
        } else {
            fmt.Printf("%s: NOT FOUND\n", name)
        }
    }

    // Also print all tensors for the first block
    fmt.Println("\n=== All blk.0 tensors ===")
    for _, t := range reader.Tensors {
        if len(t.Name) >= 5 && t.Name[:5] == "blk.0" {
            fmt.Printf("%s: shape=%v, type=%s\n", t.Name, t.Shape, t.Type.String())
        }
    }
}
