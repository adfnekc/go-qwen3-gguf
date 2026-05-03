package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"gguf/gguf"
)

func main() {
	modelPath := filepath.Join("src", "models", "qwen3-reranker-0.6b-q8_0.gguf")

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s", modelPath)
	}

	fmt.Println("========================================")
	fmt.Println("Testing GGUF Model Parser")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Println("========================================")
	fmt.Println()

	reader := gguf.NewGGUFReader()

	if err := reader.LoadFromFile(modelPath); err != nil {
		log.Fatalf("Failed to load GGUF file: %v", err)
	}

	reader.PrintInfo()

	fmt.Println("\n========================================")
	fmt.Println("Detailed Model Information")
	fmt.Println("========================================")

	demonstrateAPIs(reader)
}

func demonstrateAPIs(reader *gguf.GGUFReader) {
	fmt.Println("\n--- Key Metadata ---")

	if modelName, ok := reader.GetMetadataString("general.name"); ok {
		fmt.Printf("Model Name: %s\n", modelName)
	}

	if arch, ok := reader.GetMetadataString("general.architecture"); ok {
		fmt.Printf("Architecture: %s\n", arch)
	}

	if fileType, ok := reader.GetMetadataString("general.file_type"); ok {
		fmt.Printf("File Type: %s\n", fileType)
	}

	if author, ok := reader.GetMetadataString("general.author"); ok {
		fmt.Printf("Author: %s\n", author)
	}

	if quantVersion, ok := reader.GetMetadataInt("general.quantization_version"); ok {
		fmt.Printf("Quantization Version: %d\n", quantVersion)
	}

	if alignment, ok := reader.GetMetadataInt("general.alignment"); ok {
		fmt.Printf("Alignment: %d\n", alignment)
	}

	fmt.Println("\n--- Architecture Specific ---")

	if arch, ok := reader.GetMetadataString("general.architecture"); ok {
		archPrefix := arch + "."

		if contextLength, ok := reader.GetMetadataInt(archPrefix + "context_length"); ok {
			fmt.Printf("Context Length: %d\n", contextLength)
		}

		if embeddingLength, ok := reader.GetMetadataInt(archPrefix + "embedding_length"); ok {
			fmt.Printf("Embedding Length: %d\n", embeddingLength)
		}

		if blockCount, ok := reader.GetMetadataInt(archPrefix + "block_count"); ok {
			fmt.Printf("Block Count: %d\n", blockCount)
		}

		if feedForwardLength, ok := reader.GetMetadataInt(archPrefix + "feed_forward_length"); ok {
			fmt.Printf("Feed Forward Length: %d\n", feedForwardLength)
		}

		if headCount, ok := reader.GetMetadataInt(archPrefix + "attention.head_count"); ok {
			fmt.Printf("Attention Head Count: %d\n", headCount)
		}

		if headCountKv, ok := reader.GetMetadataInt(archPrefix + "attention.head_count_kv"); ok {
			fmt.Printf("Attention Head Count KV: %d\n", headCountKv)
		}

		if layerNormRmsEps, ok := reader.GetMetadataFloat(archPrefix + "attention.layer_norm_rms_epsilon"); ok {
			fmt.Printf("Layer Norm RMS Epsilon: %g\n", layerNormRmsEps)
		}

		if ropeFreqBase, ok := reader.GetMetadataFloat(archPrefix + "rope.freq_base"); ok {
			fmt.Printf("RoPE Frequency Base: %g\n", ropeFreqBase)
		}
	}

	fmt.Println("\n--- Tokenizer ---")

	if tokenizerModel, ok := reader.GetMetadataString("tokenizer.ggml.model"); ok {
		fmt.Printf("Tokenizer Model: %s\n", tokenizerModel)
	}

	if tokenizerPre, ok := reader.GetMetadataString("tokenizer.ggml.pre"); ok {
		fmt.Printf("Tokenizer Pre: %s\n", tokenizerPre)
	}

	if vocabSize, ok := reader.GetMetadataInt("tokenizer.ggml.vocab_size"); ok {
		fmt.Printf("Vocabulary Size: %d\n", vocabSize)
	}

	fmt.Println("\n--- Tensors Summary ---")
	fmt.Printf("Total Tensors: %d\n", len(reader.Tensors))

	typeCounts := make(map[string]int)
	for _, tensor := range reader.Tensors {
		typeCounts[tensor.Type.String()]++
	}

	fmt.Println("\nTensor Types Distribution:")
	for typ, count := range typeCounts {
		fmt.Printf("  %s: %d\n", typ, count)
	}

	if len(reader.Tensors) > 0 {
		fmt.Println("\nFirst 5 Tensors:")
		for i := 0; i < 5 && i < len(reader.Tensors); i++ {
			tensor := reader.Tensors[i]
			info, data, ok := reader.GetTensor(tensor.Name)
			if !ok {
				continue
			}

			fmt.Printf("\n  Tensor %d: %s\n", i+1, info.Name)
			fmt.Printf("    Type: %s\n", info.Type.String())
			fmt.Printf("    Shape: %v\n", info.Shape)
			fmt.Printf("    Data Size: %d bytes\n", len(data))
		}

		if len(reader.Tensors) > 5 {
			fmt.Printf("\n... and %d more tensors\n", len(reader.Tensors)-5)
		}
	}
}
