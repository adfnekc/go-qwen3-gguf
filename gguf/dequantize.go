package gguf

import (
	"encoding/binary"
	"math"
)

func DequantizeF32(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}
	result := make([]float32, len(data)/4)
	for i := 0; i < len(result); i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		result[i] = math.Float32frombits(bits)
	}
	return result
}

func DequantizeF16(data []byte) []float32 {
	if len(data)%2 != 0 {
		return nil
	}
	result := make([]float32, len(data)/2)
	for i := 0; i < len(result); i++ {
		bits := binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
		result[i] = float16ToFloat32(bits)
	}
	return result
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

func DequantizeQ8_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockBytes = 34

	numBlocks := (numElements + blockSize - 1) / blockSize
	if len(data) != numBlocks*blockBytes {
		return nil
	}

	result := make([]float32, numElements)

	for block := 0; block < numBlocks; block++ {
		blockData := data[block*blockBytes : (block+1)*blockBytes]

		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)

		for i := 0; i < blockSize; i++ {
			idx := block*blockSize + i
			if idx >= numElements {
				break
			}
			q := int8(blockData[2+i])
			result[idx] = float32(q) * scale
		}
	}

	return result
}

func DequantizeQ4_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockBytes = 18

	numBlocks := (numElements + blockSize - 1) / blockSize
	if len(data) != numBlocks*blockBytes {
		return nil
	}

	result := make([]float32, numElements)

	for block := 0; block < numBlocks; block++ {
		blockData := data[block*blockBytes : (block+1)*blockBytes]

		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)

		for i := 0; i < blockSize; i++ {
			idx := block*blockSize + i
			if idx >= numElements {
				break
			}

			byteIdx := 2 + (i / 2)
			byteVal := blockData[byteIdx]

			var q int8
			if i%2 == 0 {
				q = int8(byteVal & 0x0F)
			} else {
				q = int8((byteVal >> 4) & 0x0F)
			}

			q = q - 8
			result[idx] = float32(q) * scale
		}
	}

	return result
}

func DequantizeQ4_1(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockBytes = 20

	numBlocks := (numElements + blockSize - 1) / blockSize
	if len(data) != numBlocks*blockBytes {
		return nil
	}

	result := make([]float32, numElements)

	for block := 0; block < numBlocks; block++ {
		blockData := data[block*blockBytes : (block+1)*blockBytes]

		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)

		minBits := binary.LittleEndian.Uint16(blockData[2:4])
		minVal := float16ToFloat32(minBits)

		for i := 0; i < blockSize; i++ {
			idx := block*blockSize + i
			if idx >= numElements {
				break
			}

			byteIdx := 4 + (i / 2)
			byteVal := blockData[byteIdx]

			var q uint8
			if i%2 == 0 {
				q = byteVal & 0x0F
			} else {
				q = (byteVal >> 4) & 0x0F
			}

			result[idx] = float32(q)*scale + minVal
		}
	}

	return result
}

func DequantizeQ5_0(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockBytes = 24

	numBlocks := (numElements + blockSize - 1) / blockSize
	if len(data) != numBlocks*blockBytes {
		return nil
	}

	result := make([]float32, numElements)

	for block := 0; block < numBlocks; block++ {
		blockData := data[block*blockBytes : (block+1)*blockBytes]

		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)

		qh := binary.LittleEndian.Uint32(blockData[2:6])

		for i := 0; i < blockSize; i++ {
			idx := block*blockSize + i
			if idx >= numElements {
				break
			}

			ql := blockData[6+(i/2)]
			var qLow uint8
			if i%2 == 0 {
				qLow = ql & 0x0F
			} else {
				qLow = (ql >> 4) & 0x0F
			}

			qHigh := uint8((qh >> i) & 0x01)
			q := int8((qHigh << 4) | qLow)
			q = q - 16

			result[idx] = float32(q) * scale
		}
	}

	return result
}

func DequantizeQ5_1(data []byte, numElements int) []float32 {
	const blockSize = 32
	const blockBytes = 26

	numBlocks := (numElements + blockSize - 1) / blockSize
	if len(data) != numBlocks*blockBytes {
		return nil
	}

	result := make([]float32, numElements)

	for block := 0; block < numBlocks; block++ {
		blockData := data[block*blockBytes : (block+1)*blockBytes]

		scaleBits := binary.LittleEndian.Uint16(blockData[0:2])
		scale := float16ToFloat32(scaleBits)

		minBits := binary.LittleEndian.Uint16(blockData[2:4])
		minVal := float16ToFloat32(minBits)

		qh := binary.LittleEndian.Uint32(blockData[4:8])

		for i := 0; i < blockSize; i++ {
			idx := block*blockSize + i
			if idx >= numElements {
				break
			}

			ql := blockData[8+(i/2)]
			var qLow uint8
			if i%2 == 0 {
				qLow = ql & 0x0F
			} else {
				qLow = (ql >> 4) & 0x0F
			}

			qHigh := uint8((qh >> i) & 0x01)
			q := uint8((qHigh << 4) | qLow)

			result[idx] = float32(q)*scale + minVal
		}
	}

	return result
}

func DequantizeTensor(tensor *GGUFTensorInfo, data []byte) []float32 {
	if tensor == nil || len(data) == 0 {
		return nil
	}

	numElements := uint64(1)
	for _, dim := range tensor.Shape {
		numElements *= dim
	}

	switch tensor.Type {
	case GGML_TYPE_F32:
		return DequantizeF32(data)
	case GGML_TYPE_F16:
		return DequantizeF16(data)
	case GGML_TYPE_Q8_0:
		return DequantizeQ8_0(data, int(numElements))
	case GGML_TYPE_Q4_0:
		return DequantizeQ4_0(data, int(numElements))
	case GGML_TYPE_Q4_1:
		return DequantizeQ4_1(data, int(numElements))
	case GGML_TYPE_Q5_0:
		return DequantizeQ5_0(data, int(numElements))
	case GGML_TYPE_Q5_1:
		return DequantizeQ5_1(data, int(numElements))
	default:
		return nil
	}
}

func QuantizeF32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := int32((bits >> 31) & 0x1)
	exp := int32((bits >> 23) & 0xFF)
	mant := int32(bits & 0x7FFFFF)

	exp -= 127 - 15

	if exp <= 0 {
		if exp < -10 {
			return uint16(sign << 15)
		}
		mant = (mant | 0x800000) >> uint32(1-exp)
		if mant&0x1000 != 0 {
			mant += 0x2000
		}
		return uint16((sign << 15) | (mant >> 13))
	} else if exp == 0xff-127+15 {
		if mant == 0 {
			return uint16((sign << 15) | 0x7C00)
		}
		mant >>= 13
		if mant == 0 {
			mant = 1
		}
		return uint16((sign << 15) | 0x7C00 | mant)
	} else {
		if mant&0x1000 != 0 {
			mant += 0x2000
			if mant&0x800000 != 0 {
				mant = 0
				exp++
			}
		}
		if exp > 30 {
			return uint16((sign << 15) | 0x7C00)
		}
		return uint16((sign << 15) | (exp << 10) | (mant >> 13))
	}
}
