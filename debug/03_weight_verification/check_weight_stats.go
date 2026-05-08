package main

import (
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

	weights, err := model.LoadQwen3WeightsMMap(reader, config, mmapFile)
	if err != nil {
		panic(err)
	}

	fmt.Println("=== Per-Layer Weight Statistics ===")
	fmt.Printf("%3s | %12s %12s %12s | %12s %12s %12s | %12s\n",
		"L", "attn_q RMS", "attn_k RMS", "attn_v RMS",
		"ffn_gate RMS", "ffn_up RMS", "ffn_down RMS",
		"output RMS")
	fmt.Println("-----|--------------+--------------+--------------|--------------+--------------+--------------|--------------")

	for l := 0; l < config.BlockCount; l++ {
		bw := weights.BlockWeights[l]
		qRms := rms32(bw.AttentionQ)
		kRms := rms32(bw.AttentionK)
		vRms := rms32(bw.AttentionV)
		gRms := rms32(bw.FeedForwardGate)
		uRms := rms32(bw.FeedForwardUp)
		dRms := rms32(bw.FeedForwardDown)
		oRms := rms32(bw.AttentionOutput)

		fmt.Printf("%3d | %12.6f %12.6f %12.6f | %12.6f %12.6f %12.6f | %12.6f\n",
			l, qRms, kRms, vRms, gRms, uRms, dRms, oRms)
	}

	// Also check ffn_norm weights
	fmt.Println("\n=== FFN Norm Weights ===")
	for l := 0; l < config.BlockCount; l++ {
		r := rms32(weights.BlockWeights[l].FeedForwardNorm)
		if l < 5 || r > 2.0 {
			fmt.Printf("Layer %d ffn_norm RMS: %.6f\n", l, r)
		}
	}

	// Check: what's the expected FFN output scale?
	// For ffn_gate: out[j] = sum_k input[k] * weight[j*1024 + k]
	// If input has RMS ≈ 1 (after norm) and weight has RMS ≈ σ_gate for each row:
	// E[|out|] ≈ sqrt(1024) * σ_gate = 32 * σ_gate
	fmt.Println("\n=== Expected Output Scale (per layer FFN) ===")
	fmt.Println("Gate/Up expected output RMS (assuming normed input RMS=1):")
	for l := 0; l < 5; l++ {
		bw := weights.BlockWeights[l]
		// Per-row RMS of gate weight
		gateRowRMS := avgRowRMS(bw.FeedForwardGate, config.EmbeddingLength, config.FeedForwardLength)
		upRowRMS := avgRowRMS(bw.FeedForwardUp, config.EmbeddingLength, config.FeedForwardLength)
		downRowRMS := avgRowRMS(bw.FeedForwardDown, config.FeedForwardLength, config.EmbeddingLength)

		gateOutRMS := float32(math.Sqrt(float64(config.EmbeddingLength))) * gateRowRMS
		upOutRMS := float32(math.Sqrt(float64(config.EmbeddingLength))) * upRowRMS
		// SwiGLU output: silu(gate) * up, both with similar RMS
		swigluRMS := gateOutRMS * upOutRMS // rough estimate
		downOutRMS := float32(math.Sqrt(float64(config.FeedForwardLength))) * downRowRMS * swigluRMS

		fmt.Printf("Layer %2d: gate_row_rms=%.4f  up_row_rms=%.4f  down_row_rms=%.4f → gate_out≈%.1f  swiglu≈%.0f  down_out≈%.0f\n",
			l, gateRowRMS, upRowRMS, downRowRMS, gateOutRMS, swigluRMS, downOutRMS)
	}
}

func avgRowRMS(w []float32, innerDim, outerDim int) float32 {
	if len(w) != innerDim*outerDim {
		return 0
	}
	var totalRMS float32
	for j := 0; j < outerDim; j++ {
		start := j * innerDim
		var sum float32
		for k := 0; k < innerDim; k++ {
			sum += w[start+k] * w[start+k]
		}
		totalRMS += float32(math.Sqrt(float64(sum / float32(innerDim))))
	}
	return totalRMS / float32(outerDim)
}

func rms32(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum / float32(len(v)))))
}
