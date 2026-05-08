package main

import (
    "fmt"
    "os"
    "gguf/gguf"
    "gguf/tokenizer"
)

func main() {
    modelPath := os.Args[1]
    reader := gguf.NewGGUFReader()
    mmapFile, _ := reader.LoadFromFileMMap(modelPath)
    defer mmapFile.Close()

    tok, _ := tokenizer.NewTokenizerFromGGUF(reader)

    for _, s := range []string{"user\n", "assistant\n", "user", "assistant", "\n", " system", "user\nhi", "hi<|im_end|>"} {
        tokens := tok.Encode(s)
        fmt.Printf("%q -> %v", s, tokens)
        for _, t := range tokens {
            dec, _ := tok.GetToken(t)
            fmt.Printf(" [%q]", dec)
        }
        fmt.Println()
    }
    
    // Check: <|im_start|>user\n = ?
    fmt.Println("\nChat template test:")
    userTokens := tok.Encode("user\n")
    fmt.Printf("<|im_start|> + user\\n = [151644")
    for _, t := range userTokens {
        d, _ := tok.GetToken(t)
        fmt.Printf(", %d (%q)", t, d)
    }
    fmt.Println("]")
    
    // Also check what raw IDs these produce
    // <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n
    chatPrompt := []int{151644}
    chatPrompt = append(chatPrompt, tok.Encode("user")...)
    chatPrompt = append(chatPrompt, tok.Encode("\n")...)
    chatPrompt = append(chatPrompt, tok.Encode("hi")...)
    chatPrompt = append(chatPrompt, 151645) // im_end
    chatPrompt = append(chatPrompt, 198) // UTF-8 \n
    chatPrompt = append(chatPrompt, 151644) // im_start
    chatPrompt = append(chatPrompt, tok.Encode("assistant")...)
    chatPrompt = append(chatPrompt, tok.Encode("\n")...)
    fmt.Printf("\nChat prompt: %v\n", chatPrompt)
    
    // Compare with the infer.go approach which encodes "user\n" together
    fmt.Println("\n=== infer.go approach ===")
    inferPrompt := []int{151644}
    inferPrompt = append(inferPrompt, tok.Encode("user\n")...)
    inferPrompt = append(inferPrompt, tok.Encode("hi")...)
    inferPrompt = append(inferPrompt, 151645)
    inferPrompt = append(inferPrompt, 198)
    inferPrompt = append(inferPrompt, 151644)
    inferPrompt = append(inferPrompt, tok.Encode("assistant\n")...)
    fmt.Printf("infer.go prompt: %v\n", inferPrompt)
}
