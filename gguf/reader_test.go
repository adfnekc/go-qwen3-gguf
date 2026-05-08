package gguf

import (
	"os"
	"testing"
)

func TestGGUFReader(t *testing.T) {
	testFile := "test_model.gguf"
	defer os.Remove(testFile)

	reader := NewGGUFReader()
	if err := reader.CreateTestGGUF(testFile); err != nil {
		t.Fatalf("Failed to create test GGUF file: %v", err)
	}

	if _, err := os.Stat(testFile); os.IsNotExist(err) {
		t.Fatalf("Test GGUF file was not created")
	}

	readReader := NewGGUFReader()
	if err := readReader.LoadFromFile(testFile); err != nil {
		t.Fatalf("Failed to load GGUF file: %v", err)
	}

	if string(readReader.Header.Magic[:]) != GGUF_MAGIC {
		t.Errorf("Expected magic %s, got %s", GGUF_MAGIC, string(readReader.Header.Magic[:]))
	}

	if readReader.Header.Version != GGUF_VERSION {
		t.Errorf("Expected version %d, got %d", GGUF_VERSION, readReader.Header.Version)
	}

	if readReader.Header.TensorCount != 2 {
		t.Errorf("Expected tensor count 2, got %d", readReader.Header.TensorCount)
	}

	if readReader.Header.MetadataKVCount != 3 {
		t.Errorf("Expected metadata KV count 3, got %d", readReader.Header.MetadataKVCount)
	}

	if modelName, ok := readReader.GetMetadataString("general.name"); !ok || modelName != "TestModel" {
		t.Errorf("Expected model name 'TestModel', got '%s'", modelName)
	}

	if arch, ok := readReader.GetMetadataString("general.architecture"); !ok || arch != "llama" {
		t.Errorf("Expected architecture 'llama', got '%s'", arch)
	}

	if ctxLen, ok := readReader.GetMetadataInt("llama.context_length"); !ok || ctxLen != 2048 {
		t.Errorf("Expected context length 2048, got %d", ctxLen)
	}

	if len(readReader.Tensors) != 2 {
		t.Errorf("Expected 2 tensors, got %d", len(readReader.Tensors))
	}

	tensor1, data1, ok1 := readReader.GetTensor("test.tensor1")
	if !ok1 {
		t.Error("Failed to find tensor 'test.tensor1'")
	} else {
		if tensor1.Type != GGML_TYPE_F32 {
			t.Errorf("Expected tensor1 type f32, got %s", tensor1.Type.String())
		}
		if tensor1.NDimensions != 2 {
			t.Errorf("Expected tensor1 dimensions 2, got %d", tensor1.NDimensions)
		}
		if len(data1) != 24 {
			t.Errorf("Expected tensor1 data size 24, got %d", len(data1))
		}
	}

	tensor2, data2, ok2 := readReader.GetTensor("test.tensor2")
	if !ok2 {
		t.Error("Failed to find tensor 'test.tensor2'")
	} else {
		if tensor2.Type != GGML_TYPE_F16 {
			t.Errorf("Expected tensor2 type f16, got %s", tensor2.Type.String())
		}
		if tensor2.NDimensions != 1 {
			t.Errorf("Expected tensor2 dimensions 1, got %d", tensor2.NDimensions)
		}
		if len(data2) != 8 {
			t.Errorf("Expected tensor2 data size 8, got %d", len(data2))
		}
	}

	_, _, notFound := readReader.GetTensor("non_existent_tensor")
	if notFound {
		t.Error("Expected to not find non-existent tensor")
	}

	_, notFoundMeta := readReader.GetMetadata("non_existent_key")
	if notFoundMeta {
		t.Error("Expected to not find non-existent metadata key")
	}
}

func TestGGUFTypeString(t *testing.T) {
	tests := []struct {
		typ      GGUFType
		expected string
	}{
		{GGUF_TYPE_UINT8, "uint8"},
		{GGUF_TYPE_INT8, "int8"},
		{GGUF_TYPE_UINT16, "uint16"},
		{GGUF_TYPE_INT16, "int16"},
		{GGUF_TYPE_UINT32, "uint32"},
		{GGUF_TYPE_INT32, "int32"},
		{GGUF_TYPE_FLOAT32, "float32"},
		{GGUF_TYPE_BOOL, "bool"},
		{GGUF_TYPE_STRING, "string"},
		{GGUF_TYPE_ARRAY, "array"},
		{GGUF_TYPE_UINT64, "uint64"},
		{GGUF_TYPE_INT64, "int64"},
		{GGUF_TYPE_FLOAT64, "float64"},
		{GGUFType(99), "unknown(99)"},
	}

	for _, test := range tests {
		result := test.typ.String()
		if result != test.expected {
			t.Errorf("Expected '%s', got '%s'", test.expected, result)
		}
	}
}

func TestGGMLQuantizationTypeString(t *testing.T) {
	tests := []struct {
		typ      GGMLQuantizationType
		expected string
	}{
		{GGML_TYPE_F32, "f32"},
		{GGML_TYPE_F16, "f16"},
		{GGML_TYPE_Q4_0, "q4_0"},
		{GGML_TYPE_Q4_1, "q4_1"},
		{GGML_TYPE_Q8_0, "q8_0"},
		{GGML_TYPE_I8, "i8"},
		{GGML_TYPE_I16, "i16"},
		{GGML_TYPE_I32, "i32"},
		{GGML_TYPE_I64, "i64"},
		{GGML_TYPE_F64, "f64"},
		{GGML_TYPE_BF16, "bf16"},
		{GGMLQuantizationType(99), "unknown(99)"},
	}

	for _, test := range tests {
		result := test.typ.String()
		if result != test.expected {
			t.Errorf("Expected '%s', got '%s'", test.expected, result)
		}
	}
}

func TestGGMLQuantizationTypeElementSize(t *testing.T) {
	tests := []struct {
		typ      GGMLQuantizationType
		expected uint64
	}{
		{GGML_TYPE_F32, 4},
		{GGML_TYPE_F16, 2},
		{GGML_TYPE_I8, 1},
		{GGML_TYPE_I16, 2},
		{GGML_TYPE_I32, 4},
		{GGML_TYPE_I64, 8},
		{GGML_TYPE_F64, 8},
		{GGML_TYPE_BF16, 2},
		{GGML_TYPE_Q4_0, 18},
		{GGML_TYPE_Q8_0, 34},
		{GGMLQuantizationType(99), 0},
	}

	for _, test := range tests {
		result := test.typ.ElementSize()
		if result != test.expected {
			t.Errorf("Expected element size %d for type %s, got %d", 
				test.expected, test.typ.String(), result)
		}
	}
}

func TestNewGGUFReader(t *testing.T) {
	reader := NewGGUFReader()
	
	if reader.Metadata == nil {
		t.Error("Metadata map should be initialized")
	}
	
	if reader.Tensors == nil {
		t.Error("Tensors slice should be initialized")
	}
	
	if reader.TensorData == nil {
		t.Error("TensorData map should be initialized")
	}
	
	if len(reader.Metadata) != 0 {
		t.Error("Metadata should be empty initially")
	}
	
	if len(reader.Tensors) != 0 {
		t.Error("Tensors should be empty initially")
	}
	
	if len(reader.TensorData) != 0 {
		t.Error("TensorData should be empty initially")
	}
}

func TestDefaultGGUFReaderOptions(t *testing.T) {
	options := DefaultGGUFReaderOptions()
	
	if options == nil {
		t.Error("Default options should not be nil")
	}
	
	if !options.LoadTensors {
		t.Error("LoadTensors should be true by default")
	}
}

func TestDequantizeQ8_0Format(t *testing.T) {
	blockData := make([]byte, 34)
	
	scale := float32(0.5)
	scaleBits := QuantizeF32ToF16(scale)
	
	blockData[0] = byte(scaleBits & 0xFF)
	blockData[1] = byte((scaleBits >> 8) & 0xFF)
	
	for i := 0; i < 32; i++ {
		blockData[2+i] = byte(int8(i - 16))
	}
	
	result := DequantizeQ8_0(blockData, 32)
	
	if result == nil {
		t.Fatal("DequantizeQ8_0 returned nil")
	}
	
	if len(result) != 32 {
		t.Errorf("Expected 32 elements, got %d", len(result))
	}
	
	for i := 0; i < 32; i++ {
		expected := float32(int8(i-16)) * scale
		if result[i] != expected {
			t.Errorf("result[%d] = %f, expected %f", i, result[i], expected)
		}
	}
}
