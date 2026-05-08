package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"gguf/gguf"
	"gguf/model"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	reader := gguf.NewGGUFReader()
	mmapFile, err := reader.LoadFromFileMMap(modelPath)
	if err != nil {
		panic(err)
	}
	defer mmapFile.Close()

	config, err := model.LoadQwen3Config(reader)
	if err != nil {
		panic(err)
	}

	// Get raw data for layer 2 ffn_gate
	tensorInfo, rawData, ok := reader.GetTensor("blk.2.ffn_gate.weight")
	if !ok {
		// Try alternative naming
		tensorInfo, rawData, ok = reader.GetTensor("model.layers.2.mlp.gate_proj.weight")
	}
	if !ok {
		panic("blk.2.ffn_gate.weight not found")
	}

	fmt.Printf("Tensor: %s\n", tensorInfo.Name)
	fmt.Printf("Shape: %v\n", tensorInfo.Shape)
	fmt.Printf("Type: %s (code=%d)\n", tensorInfo.Type, tensorInfo.Type)
	fmt.Printf("Num elements: %d\n", tensorInfo.ElementCount)
	fmt.Printf("Raw data length: %d bytes\n", len(rawData))

	numElements := uint64(1)
	for _, d := range tensorInfo.Shape {
		numElements *= d
	}

	// Q8_0: 34 bytes per 32-element block
	const blockSize = 32
	const blockBytes = 34
	numBlocks := int((numElements + blockSize - 1) / blockSize)

	fmt.Printf("\nNumber of Q8_0 blocks: %d (expected: %d)\n", numBlocks, len(rawData)/blockBytes)
	fmt.Printf("Expected raw size: %d bytes\n", numBlocks*blockBytes)

	// Dequantize
	dequantized := gguf.DequantizeQ8_0(rawData, int(numElements))
	if dequantized == nil {
		panic("dequantization failed")
	}
	fmt.Printf("Dequantized length: %d\n", len(dequantized))
	fmt.Printf("RMS: %.6f\n", rms32(dequantized))

	// Print first 5 blocks: scale and first few int8 values
	fmt.Println("\n=== First 5 Q8_0 blocks ===")
	for b := 0; b < 5 && b < numBlocks; b++ {
		blockData := rawData[b*blockBytes : (b+1)*blockBytes]
		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)
		fmt.Printf("Block %d: scale=%.6f (uint16=0x%04x)  q = [", b, scale, scaleBits)
		for i := 0; i < 8 && i < blockSize; i++ {
			fmt.Printf(" %d", int8(blockData[2+i]))
		}
		fmt.Printf(" ...] → dequant = [")
		for i := 0; i < 8 && i < blockSize; i++ {
			fmt.Printf(" %.4f", float32(int8(blockData[2+i]))*scale)
		}
		fmt.Printf(" ...]\n")
	}

	// Also check layer 2's ffn_norm weight
	normInfo, normRaw, ok := reader.GetTensor("blk.2.ffn_norm.weight")
	if ok {
		normF32 := gguf.DequantizeF32(normRaw)
		fmt.Printf("\n=== Layer 2 ffn_norm ===\n")
		fmt.Printf("Type: %s, length: %d\n", normInfo.Type, len(normF32))
		fmt.Printf("First 5 values:\n")
		for i := 0; i < 5 && i < len(normF32); i++ {
			fmt.Printf("  [%d] = %.6f\n", i, normF32[i])
		}
		fmt.Printf("RMS: %.6f\n", rms32(normF32))
	}

	// Load the weights and check vs manual dequant
	fmt.Println("\n=== Weight loading verification ===")
	if config.BlockCount > 2 {
		bw := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
		if bw != nil { // Hmm, this returns different type. Skip for now.
			_ = bw
		}
	}
}

// Read the actual loaded weights instead
func init() {
	// We'll do this differently
}

func float16ToFloat32(h uint16) float32 {
	sign := int32(h >> 15)
	exp := int32((h >> 10) & 0x1F)
	mant := int32(h & 0x3FF)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(uint32(sign << 31))
		}
		for (mant & 0x400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(uint32((sign << 31) | 0x7F800000))
		}
		return math.Float32frombits(uint32((sign << 31) | 0x7F800000 | (mant << 13)))
	}
	exp += 127 - 15
	mant <<= 13
	return math.Float32frombits(uint32((sign << 31) | (exp << 23) | mant))
}

func rms32(v []float32) float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return float32(math.Sqrt(sum / float64(len(v))))
}
