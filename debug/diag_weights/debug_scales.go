package main

import (
	"encoding/binary"
	"fmt"
	"math"
	
	"github.com/adfnekc/go-qwen3-gguf/model"
	"github.com/adfnekc/go-qwen3-gguf/gguf"
)

func main() {
	modelPath := "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
	reader := gguf.NewGGUFReader()
	mmapFile, _ := reader.LoadFromFileMMap(modelPath)
	defer mmapFile.Close()
	config, _ := model.LoadQwen3Config(reader)
	weights, _ := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	
	// Compare first few values of attn_q.weight for layer 0 and layer 1
	fmt.Println("=== Layer 0 attn_q first 10 values ===")
	q0 := weights.BlockWeights[0].AttentionQ
	qDim := config.AttentionHeadCount * config.AttentionKeyLength // 2048
	for i := 0; i < 10; i++ {
		fmt.Printf("  q0[%d] = %f\n", i, q0[i])
	}
	
	fmt.Println("=== Layer 1 attn_q first 10 values ===")
	q1 := weights.BlockWeights[1].AttentionQ
	for i := 0; i < 10; i++ {
		fmt.Printf("  q1[%d] = %f\n", i, q1[i])
	}
	
	// Check RMS of attn_q weights per layer
	fmt.Println("\n=== Per-layer attn_q RMS ===")
	for l := 0; l < 28; l++ {
		q := weights.BlockWeights[l].AttentionQ
		var sum float32 = 0
		for _, v := range q {
			sum += v * v
		}
		rms := math.Sqrt(float64(sum / float32(len(q))))
		fmt.Printf("  Layer %2d attn_q RMS: %f\n", l, rms)
	}
	
	// Check RMS of ffn_gate weights per layer
	fmt.Println("\n=== Per-layer ffn_gate RMS ===")
	for l := 0; l < 28; l++ {
		g := weights.BlockWeights[l].FeedForwardGate
		var sum float32 = 0
		for _, v := range g {
			sum += v * v
		}
		rms := math.Sqrt(float64(sum / float32(len(g))))
		fmt.Printf("  Layer %2d ffn_gate RMS: %f\n", l, rms)
	}
	
	// Check raw Q8_0 scale for layer 0 attn_q first few blocks
	fmt.Println("\n=== Raw Q8_0 scales for layer 0 attn_q (first 5 blocks) ===")
	tensorInfo, _, _ := reader.GetTensor("blk.0.attn_q.weight")
	data, _ := mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(model.CalculateTensorSize(tensorInfo)))
	for b := 0; b < 5; b++ {
		scaleBits := binary.LittleEndian.Uint16(data[b*34 : b*34+2])
		scale := float32(math.Float32frombits(uint32(float16ToFloat32(scaleBits))))
		fmt.Printf("  Block %d: scale_f16=0x%04x, scale_f32=%f\n", b, scaleBits, scale)
	}
	
	fmt.Println("\n=== Raw Q8_0 scales for layer 1 attn_q (first 5 blocks) ===")
	tensorInfo, _, _ = reader.GetTensor("blk.1.attn_q.weight")
	data, _ = mmapFile.GetSlice(tensorAbsOffset(reader, tensorInfo), int(model.CalculateTensorSize(tensorInfo)))
	for b := 0; b < 5; b++ {
		scaleBits := binary.LittleEndian.Uint16(data[b*34 : b*34+2])
		scale := float32(math.Float32frombits(uint32(float16ToFloat32(scaleBits))))
		fmt.Printf("  Block %d: scale_f16=0x%04x, scale_f32=%f\n", b, scaleBits, scale)
	}
}
