package main

import (
	"fmt"
	"os"
	"sort"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
)

func main() {
	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(os.Args[1])
	if err != nil {
		fmt.Printf("Failed to load: %v\n", err)
		os.Exit(1)
	}
	defer mmapFile.Close()

	// Print all metadata
	keys := make([]string, 0, len(reader.Metadata))
	for k := range reader.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	fmt.Println("=== All Metadata ===")
	for _, k := range keys {
		v := reader.Metadata[k]
		fmt.Printf("  %s (%s): %v\n", k, v.Type, v.Value)
	}

	// Check specific keys
	fmt.Println("\n=== Key values ===")
	if arch, ok := reader.GetMetadataString("general.architecture"); ok {
		fmt.Printf("architecture: %s\n", arch)
	}
	if ropeDim, ok := reader.GetMetadataInt("qwen3.rope.dimension_count"); ok {
		fmt.Printf("qwen3.rope.dimension_count: %d\n", ropeDim)
	}
	if ropeDim, ok := reader.GetMetadataInt("llama.rope.dimension_count"); ok {
		fmt.Printf("llama.rope.dimension_count: %d\n", ropeDim)
	}
}
