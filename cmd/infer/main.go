package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/adfnekc/go-qwen3-gguf/internal"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run ./cmd/infer <model_path> [input_text] [--no-mmap] [--temp=0.7] [--tokens=10]")
		fmt.Println("Example: go run ./cmd/infer <model.gguf> \"Hello\"")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	inputText := "Hello, world!"
	var opts = internal.RunOptions{UseChat: true, Temperature: 0.7}

	for i := 2; i < len(os.Args); i++ {
		arg := os.Args[i]
		switch {
		case arg == "--no-mmap":
			opts.ForceNoMMap = true
		case arg == "--no-chat":
			opts.UseChat = false
		case arg == "--debug":
			opts.Debug = true
		case len(arg) > 7 && arg[:7] == "--temp=":
			if v, err := strconv.ParseFloat(arg[7:], 32); err == nil {
				opts.Temperature = float32(v)
			}
		case len(arg) > 9 && arg[:9] == "--tokens=":
			if v, err := strconv.Atoi(arg[9:]); err == nil {
				opts.MaxTokens = v
			}
		default:
			inputText = arg
		}
	}

	runner := internal.NewRunner()
	if err := runner.Run(modelPath, inputText, opts); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
