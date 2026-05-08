package gguf
import (
	"encoding/binary"
	"fmt"
)

type GGUFType uint32

const (
	GGUF_TYPE_UINT8   GGUFType = 0
	GGUF_TYPE_INT8    GGUFType = 1
	GGUF_TYPE_UINT16  GGUFType = 2
	GGUF_TYPE_INT16   GGUFType = 3
	GGUF_TYPE_UINT32  GGUFType = 4
	GGUF_TYPE_INT32   GGUFType = 5
	GGUF_TYPE_FLOAT32 GGUFType = 6
	GGUF_TYPE_BOOL    GGUFType = 7
	GGUF_TYPE_STRING  GGUFType = 8
	GGUF_TYPE_ARRAY   GGUFType = 9
	GGUF_TYPE_UINT64  GGUFType = 10
	GGUF_TYPE_INT64   GGUFType = 11
	GGUF_TYPE_FLOAT64 GGUFType = 12
)

type GGMLQuantizationType uint32

const (
	GGML_TYPE_F32   GGMLQuantizationType = 0
	GGML_TYPE_F16   GGMLQuantizationType = 1
	GGML_TYPE_Q4_0  GGMLQuantizationType = 2
	GGML_TYPE_Q4_1  GGMLQuantizationType = 3
	GGML_TYPE_Q5_0  GGMLQuantizationType = 6
	GGML_TYPE_Q5_1  GGMLQuantizationType = 7
	GGML_TYPE_Q8_0  GGMLQuantizationType = 8
	GGML_TYPE_Q8_1  GGMLQuantizationType = 9
	GGML_TYPE_Q2_K  GGMLQuantizationType = 10
	GGML_TYPE_Q3_K  GGMLQuantizationType = 11
	GGML_TYPE_Q4_K  GGMLQuantizationType = 12
	GGML_TYPE_Q5_K  GGMLQuantizationType = 13
	GGML_TYPE_Q6_K  GGMLQuantizationType = 14
	GGML_TYPE_Q8_K  GGMLQuantizationType = 15
	GGML_TYPE_IQ2_XXS GGMLQuantizationType = 16
	GGML_TYPE_IQ2_XS  GGMLQuantizationType = 17
	GGML_TYPE_IQ3_XXS GGMLQuantizationType = 18
	GGML_TYPE_IQ1_S   GGMLQuantizationType = 19
	GGML_TYPE_IQ4_NL  GGMLQuantizationType = 20
	GGML_TYPE_IQ3_S   GGMLQuantizationType = 21
	GGML_TYPE_IQ2_S   GGMLQuantizationType = 22
	GGML_TYPE_IQ4_XS  GGMLQuantizationType = 23
	GGML_TYPE_I8      GGMLQuantizationType = 24
	GGML_TYPE_I16     GGMLQuantizationType = 25
	GGML_TYPE_I32     GGMLQuantizationType = 26
	GGML_TYPE_I64     GGMLQuantizationType = 27
	GGML_TYPE_F64     GGMLQuantizationType = 28
	GGML_TYPE_IQ1_M   GGMLQuantizationType = 29
	GGML_TYPE_BF16    GGMLQuantizationType = 30
	GGML_TYPE_Q4_0_4_4 GGMLQuantizationType = 31
	GGML_TYPE_Q4_0_4_8 GGMLQuantizationType = 32
	GGML_TYPE_Q4_0_8_8 GGMLQuantizationType = 33
	GGML_TYPE_TQ1_0    GGMLQuantizationType = 34
	GGML_TYPE_TQ2_0    GGMLQuantizationType = 35
)

const (
	GGUF_MAGIC   = "GGUF"
	GGUF_VERSION = 3
)

type GGUFHeader struct {
	Magic           [4]byte
	Version         uint32
	TensorCount     uint64
	MetadataKVCount uint64
}

type GGUFMetadataItem struct {
	Key   string
	Type  GGUFType
	Value interface{}
}

type GGUFTensorInfo struct {
	Name       string
	NDimensions uint32
	Shape      []uint64
	Type       GGMLQuantizationType
	Offset     uint64
}

type GGUFReader struct {
	Header     GGUFHeader
	Metadata   map[string]GGUFMetadataItem
	Tensors    []GGUFTensorInfo
	TensorData map[string][]byte
	DataStart  int64
	byteOrder  binary.ByteOrder
}

func (g GGUFType) String() string {
	switch g {
	case GGUF_TYPE_UINT8:
		return "uint8"
	case GGUF_TYPE_INT8:
		return "int8"
	case GGUF_TYPE_UINT16:
		return "uint16"
	case GGUF_TYPE_INT16:
		return "int16"
	case GGUF_TYPE_UINT32:
		return "uint32"
	case GGUF_TYPE_INT32:
		return "int32"
	case GGUF_TYPE_FLOAT32:
		return "float32"
	case GGUF_TYPE_BOOL:
		return "bool"
	case GGUF_TYPE_STRING:
		return "string"
	case GGUF_TYPE_ARRAY:
		return "array"
	case GGUF_TYPE_UINT64:
		return "uint64"
	case GGUF_TYPE_INT64:
		return "int64"
	case GGUF_TYPE_FLOAT64:
		return "float64"
	default:
		return fmt.Sprintf("unknown(%d)", g)
	}
}

func (g GGMLQuantizationType) String() string {
	switch g {
	case GGML_TYPE_F32:
		return "f32"
	case GGML_TYPE_F16:
		return "f16"
	case GGML_TYPE_Q4_0:
		return "q4_0"
	case GGML_TYPE_Q4_1:
		return "q4_1"
	case GGML_TYPE_Q5_0:
		return "q5_0"
	case GGML_TYPE_Q5_1:
		return "q5_1"
	case GGML_TYPE_Q8_0:
		return "q8_0"
	case GGML_TYPE_Q8_1:
		return "q8_1"
	case GGML_TYPE_Q2_K:
		return "q2_k"
	case GGML_TYPE_Q3_K:
		return "q3_k"
	case GGML_TYPE_Q4_K:
		return "q4_k"
	case GGML_TYPE_Q5_K:
		return "q5_k"
	case GGML_TYPE_Q6_K:
		return "q6_k"
	case GGML_TYPE_Q8_K:
		return "q8_k"
	case GGML_TYPE_I8:
		return "i8"
	case GGML_TYPE_I16:
		return "i16"
	case GGML_TYPE_I32:
		return "i32"
	case GGML_TYPE_I64:
		return "i64"
	case GGML_TYPE_F64:
		return "f64"
	case GGML_TYPE_BF16:
		return "bf16"
	default:
		return fmt.Sprintf("unknown(%d)", g)
	}
}

func (g GGMLQuantizationType) ElementSize() uint64 {
	switch g {
	case GGML_TYPE_F32, GGML_TYPE_I32:
		return 4
	case GGML_TYPE_F16, GGML_TYPE_I16, GGML_TYPE_BF16:
		return 2
	case GGML_TYPE_I8:
		return 1
	case GGML_TYPE_I64, GGML_TYPE_F64:
		return 8
	case GGML_TYPE_Q4_0, GGML_TYPE_Q4_1:
		return 18
	case GGML_TYPE_Q5_0, GGML_TYPE_Q5_1:
		return 24
	case GGML_TYPE_Q8_0:
		return 34
	case GGML_TYPE_Q2_K:
		return 20
	case GGML_TYPE_Q3_K:
		return 22
	case GGML_TYPE_Q4_K:
		return 26
	case GGML_TYPE_Q5_K:
		return 32
	case GGML_TYPE_Q6_K:
		return 38
	default:
		return 0
	}
}
