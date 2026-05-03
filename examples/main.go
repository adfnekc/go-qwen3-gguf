package main

import (
	"fmt"
	"log"
	"os"

	"gguf/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <command> [arguments]")
		fmt.Println("Commands:")
		fmt.Println("  create <filename.gguf>  - Create a test GGUF file")
		fmt.Println("  read <filename.gguf>    - Read and display GGUF file info")
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "create":
		if len(os.Args) < 3 {
			fmt.Println("Usage: go run main.go create <filename.gguf>")
			os.Exit(1)
		}
		createTestFile(os.Args[2])
	case "read":
		if len(os.Args) < 3 {
			fmt.Println("Usage: go run main.go read <filename.gguf>")
			os.Exit(1)
		}
		readAndDisplayFile(os.Args[2])
	default:
		fmt.Printf("Unknown command: %s\n", command)
		os.Exit(1)
	}
}

func createTestFile(filename string) {
	fmt.Printf("Creating test GGUF file: %s\n", filename)
	
	reader := gguf.NewGGUFReader()
	
	if err := reader.CreateTestGGUF(filename); err != nil {
		log.Fatalf("Failed to create test GGUF file: %v", err)
	}
	
	fmt.Println("Test GGUF file created successfully!")
	fmt.Println()
	
	readAndDisplayFile(filename)
}

func readAndDisplayFile(filename string) {
	fmt.Printf("Reading GGUF file: %s\n", filename)
	fmt.Println()
	
	reader := gguf.NewGGUFReader()
	
	if err := reader.LoadFromFile(filename); err != nil {
		log.Fatalf("Failed to load GGUF file: %v", err)
	}
	
	reader.PrintInfo()
	
	demonstrateAPIs(reader)
}

func demonstrateAPIs(reader *gguf.GGUFReader) {
	fmt.Println("=== API Usage Examples ===")
	
	if modelName, ok := reader.GetMetadataString("general.name"); ok {
		fmt.Printf("Model Name: %s\n", modelName)
	}
	
	if arch, ok := reader.GetMetadataString("general.architecture"); ok {
		fmt.Printf("Architecture: %s\n", arch)
	}
	
	if ctxLen, ok := reader.GetMetadataInt("llama.context_length"); ok {
		fmt.Printf("Context Length: %d\n", ctxLen)
	}
	
	fmt.Printf("\nTotal Tensors: %d\n", len(reader.Tensors))
	
	for i, tensor := range reader.Tensors {
		info, data, ok := reader.GetTensor(tensor.Name)
		if !ok {
			continue
		}
		
		fmt.Printf("\nTensor %d: %s\n", i, info.Name)
		fmt.Printf("  Type: %s\n", info.Type.String())
		fmt.Printf("  Shape: %v\n", info.Shape)
		fmt.Printf("  Data Size: %d bytes\n", len(data))
	}
}
