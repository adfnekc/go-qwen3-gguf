package main

import (
    "fmt"
    "os"
    "github.com/adfnekc/go-qwen3-gguf/gguf"
    "github.com/adfnekc/go-qwen3-gguf/tokenizer"
)

func main() {
    modelPath := os.Args[1]
    reader := gguf.NewGGUFReader()
    mmapFile, err := reader.LoadFromFileMMap(modelPath)
    if err != nil { fmt.Printf("Failed to load: %v\n", err); os.Exit(1) }
    defer mmapFile.Close()

    tok, err := tokenizer.NewTokenizerFromGGUF(reader)
    if err != nil {
        fmt.Printf("Failed to create tokenizer: %v\n", err)
        os.Exit(1)
    }

    testWords := []string{"hi", "hello", "Hi", "Hello", "world", "the", "user", "system", "assistant"}
    for _, word := range testWords {
        tokens := tok.Encode(word)
        fmt.Printf("%-12s -> tokens: %v", word, tokens)
        for _, t := range tokens {
            dec, _ := tok.GetToken(t)
            fmt.Printf(" [%q]", dec)
        }
        fmt.Println()
    }

    // Check specific token IDs that might be special
    fmt.Println("\nSpecial tokens:")
    for _, id := range []int{6023, 4999, 1479, 198, 151644, 151645, 151643} {
        s, ok := tok.GetToken(id)
        if ok {
            fmt.Printf("  #%d -> %q\n", id, s)
        } else {
            fmt.Printf("  #%d -> NOT FOUND\n", id)
        }
    }

    // Check what "hi" encodes to with simple tokenizer
    fmt.Println("\nWith simple tokenizer:")
    simpleTok := tokenizer.NewSimpleTokenizer()
    for _, word := range testWords {
        tokens := simpleTok.Encode(word)
        fmt.Printf("  %-12s -> %v\n", word, tokens)
    }
}
