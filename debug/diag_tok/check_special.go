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
    mmapFile, _ := reader.LoadFromFileMMap(modelPath)
    defer mmapFile.Close()

    tok, _ := tokenizer.NewTokenizerFromGGUF(reader)

    // Check what <think> token is
    for _, id := range []int{151667, 151666, 151668, 151664, 151640, 151641, 151642} {
        s, ok := tok.GetToken(id)
        if ok {
            fmt.Printf("Token #%d: %q\n", id, s)
        }
    }
    
    // Try to find <think> token
    fmt.Println("\nSearching for think related tokens:")
    for i := 0; i < tok.GetVocabSize(); i++ {
        s, ok := tok.GetToken(i)
        if ok && (s == "<think>" || s == "Think" || s == "think" || s == "[Start thinking]") {
            fmt.Printf("  Token #%d: %q\n", i, s)
        }
    }
    
    // Decode what the model generates
    tokens := []int{151667, 198, 198, 151667, 198, 198}
    fmt.Printf("\nDecoded: %q\n", tok.Decode(tokens))
    
    // Also check <|im_start|>user\n = ?
    chatTokens := []int{151644}
    chatTokens = append(chatTokens, tok.Encode("user\n")...)
    chatTokens = append(chatTokens, tok.Encode("hi")...)
    chatTokens = append(chatTokens, 151645)
    chatTokens = append(chatTokens, 198)
    chatTokens = append(chatTokens, 151644)
    chatTokens = append(chatTokens, tok.Encode("assistant\n")...)
    fmt.Printf("\nFull chat prompt decoded: %q\n", tok.Decode(chatTokens))
}
