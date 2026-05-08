package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type GGUFReaderOptions struct {
	LoadTensors bool
}

func DefaultGGUFReaderOptions() *GGUFReaderOptions {
	return &GGUFReaderOptions{
		LoadTensors: true,
	}
}

func NewGGUFReader() *GGUFReader {
	return &GGUFReader{
		Metadata:   make(map[string]GGUFMetadataItem),
		Tensors:    make([]GGUFTensorInfo, 0),
		TensorData: make(map[string][]byte),
		byteOrder:  binary.LittleEndian,
	}
}

func (r *GGUFReader) LoadFromFile(filePath string) error {
	return r.LoadFromFileWithOptions(filePath, DefaultGGUFReaderOptions())
}

func (r *GGUFReader) LoadFromFileWithOptions(filePath string, options *GGUFReaderOptions) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	return r.LoadFromReaderWithOptions(file, options)
}

func (r *GGUFReader) LoadFromReader(reader io.Reader) error {
	return r.LoadFromReaderWithOptions(reader, DefaultGGUFReaderOptions())
}

func (r *GGUFReader) LoadFromReaderWithOptions(reader io.Reader, options *GGUFReaderOptions) error {
	if err := r.readHeader(reader); err != nil {
		return err
	}

	if err := r.readMetadata(reader); err != nil {
		return err
	}

	if err := r.readTensorInfo(reader); err != nil {
		return err
	}

	if options.LoadTensors {
		if seeker, ok := reader.(io.Seeker); ok {
			if err := r.loadTensorData(seeker); err != nil {
				return err
			}
		}
	}

	return nil
}

func (r *GGUFReader) readHeader(reader io.Reader) error {
	header := GGUFHeader{}

	if err := binary.Read(reader, binary.LittleEndian, &header.Magic); err != nil {
		return fmt.Errorf("failed to read magic: %w", err)
	}

	magicStr := string(header.Magic[:])
	if magicStr != GGUF_MAGIC {
		return fmt.Errorf("invalid magic number: expected %s, got %s", GGUF_MAGIC, magicStr)
	}

	if err := binary.Read(reader, binary.LittleEndian, &header.Version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}

	if header.Version != GGUF_VERSION {
		return fmt.Errorf("unsupported version: expected %d, got %d", GGUF_VERSION, header.Version)
	}

	if err := binary.Read(reader, binary.LittleEndian, &header.TensorCount); err != nil {
		return fmt.Errorf("failed to read tensor count: %w", err)
	}

	if err := binary.Read(reader, binary.LittleEndian, &header.MetadataKVCount); err != nil {
		return fmt.Errorf("failed to read metadata KV count: %w", err)
	}

	r.Header = header
	return nil
}

func (r *GGUFReader) alignReader(reader io.Reader) error {
	if seeker, ok := reader.(io.Seeker); ok {
		currentPos, err := seeker.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		alignment := int64(32)
		alignedPos := (currentPos + alignment - 1) & ^(alignment - 1)

		if alignedPos > currentPos {
			_, err = seeker.Seek(alignedPos, io.SeekStart)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (r *GGUFReader) loadTensorData(seeker io.Seeker) error {
	for i := range r.Tensors {
		tensor := &r.Tensors[i]

		dataSize := r.calculateTensorSize(tensor)
		if dataSize == 0 {
			continue
		}

		absOffset := r.DataStart + int64(tensor.Offset)
		_, err := seeker.Seek(absOffset, io.SeekStart)
		if err != nil {
			return fmt.Errorf("failed to seek to tensor %s data: %w", tensor.Name, err)
		}

		data := make([]byte, dataSize)
		_, err = seeker.(io.Reader).Read(data)
		if err != nil && err != io.EOF {
			return fmt.Errorf("failed to read tensor %s data: %w", tensor.Name, err)
		}

		r.TensorData[tensor.Name] = data
	}
	return nil
}

func (r *GGUFReader) calculateTensorSize(tensor *GGUFTensorInfo) uint64 {
	elementSize := tensor.Type.ElementSize()
	if elementSize == 0 {
		return 0
	}

	numElements := uint64(1)
	for _, dim := range tensor.Shape {
		numElements *= dim
	}

	switch tensor.Type {
	case GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
		GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
		GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K:
		blockSize := uint64(32)
		return (numElements + blockSize - 1) / blockSize * elementSize
	default:
		return numElements * elementSize
	}
}

func (r *GGUFReader) PrintInfo() {
	fmt.Println("=== GGUF File Information ===")
	fmt.Printf("Magic: %s\n", string(r.Header.Magic[:]))
	fmt.Printf("Version: %d\n", r.Header.Version)
	fmt.Printf("Tensor Count: %d\n", r.Header.TensorCount)
	fmt.Printf("Metadata KV Count: %d\n", r.Header.MetadataKVCount)
	fmt.Println()

	fmt.Println("=== Metadata ===")
	for key, item := range r.Metadata {
		fmt.Printf("  %s: ", key)
		switch v := item.Value.(type) {
		case []uint8:
			fmt.Printf("[%d]uint8\n", len(v))
		case []int8:
			fmt.Printf("[%d]int8\n", len(v))
		case []uint16:
			fmt.Printf("[%d]uint16\n", len(v))
		case []int16:
			fmt.Printf("[%d]int16\n", len(v))
		case []uint32:
			fmt.Printf("[%d]uint32\n", len(v))
		case []int32:
			fmt.Printf("[%d]int32\n", len(v))
		case []float32:
			fmt.Printf("[%d]float32\n", len(v))
		case []bool:
			fmt.Printf("[%d]bool\n", len(v))
		case []string:
			fmt.Printf("[%d]string\n", len(v))
		case []uint64:
			fmt.Printf("[%d]uint64\n", len(v))
		case []int64:
			fmt.Printf("[%d]int64\n", len(v))
		case []float64:
			fmt.Printf("[%d]float64\n", len(v))
		default:
			fmt.Printf("%v\n", v)
		}
	}
	fmt.Println()

	fmt.Println("=== Tensors ===")
	for i, tensor := range r.Tensors {
		fmt.Printf("  Tensor %d: %s\n", i, tensor.Name)
		fmt.Printf("    Type: %s\n", tensor.Type.String())
		fmt.Printf("    Dimensions: %d\n", tensor.NDimensions)
		fmt.Printf("    Shape: %v\n", tensor.Shape)
		fmt.Printf("    Offset: 0x%x\n", tensor.Offset)
		if data, ok := r.TensorData[tensor.Name]; ok {
			fmt.Printf("    Data Size: %d bytes\n", len(data))
		}
		fmt.Println()
	}
}

func (r *GGUFReader) CreateTestGGUF(filePath string) error {
	buf := new(bytes.Buffer)

	header := GGUFHeader{
		Magic:           [4]byte{'G', 'G', 'U', 'F'},
		Version:         GGUF_VERSION,
		TensorCount:     2,
		MetadataKVCount: 3,
	}

	if err := binary.Write(buf, binary.LittleEndian, header.Magic); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, header.Version); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, header.TensorCount); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, header.MetadataKVCount); err != nil {
		return err
	}

	metadata := []struct {
		key   string
		typ   GGUFType
		value interface{}
	}{
		{"general.name", GGUF_TYPE_STRING, "TestModel"},
		{"general.architecture", GGUF_TYPE_STRING, "llama"},
		{"llama.context_length", GGUF_TYPE_UINT32, uint32(2048)},
	}

	for _, md := range metadata {
		if err := binary.Write(buf, binary.LittleEndian, uint64(len(md.key))); err != nil {
			return err
		}
		if _, err := buf.WriteString(md.key); err != nil {
			return err
		}

		if err := binary.Write(buf, binary.LittleEndian, uint32(md.typ)); err != nil {
			return err
		}

		switch md.typ {
		case GGUF_TYPE_STRING:
			str := md.value.(string)
			if err := binary.Write(buf, binary.LittleEndian, uint64(len(str))); err != nil {
				return err
			}
			if _, err := buf.WriteString(str); err != nil {
				return err
			}
		case GGUF_TYPE_UINT32:
			if err := binary.Write(buf, binary.LittleEndian, md.value.(uint32)); err != nil {
				return err
			}
		}
	}

	tensors := []struct {
		name  string
		typ   GGMLQuantizationType
		shape []uint64
		data  []byte
	}{
		{"test.tensor1", GGML_TYPE_F32, []uint64{2, 3}, make([]byte, 24)},
		{"test.tensor2", GGML_TYPE_F16, []uint64{4}, make([]byte, 8)},
	}

	var tensorDataOffset uint64 = uint64(buf.Len())

	for i := range tensors {
		tensorDataOffset += uint64(8 + len(tensors[i].name) + 4 + 8*len(tensors[i].shape) + 4 + 8)
	}

	alignment := uint64(32)
	tensorDataOffset = (tensorDataOffset + alignment - 1) & ^(alignment - 1)

	for _, tensor := range tensors {
		if err := binary.Write(buf, binary.LittleEndian, uint64(len(tensor.name))); err != nil {
			return err
		}
		if _, err := buf.WriteString(tensor.name); err != nil {
			return err
		}

		if err := binary.Write(buf, binary.LittleEndian, uint32(len(tensor.shape))); err != nil {
			return err
		}

		for _, dim := range tensor.shape {
			if err := binary.Write(buf, binary.LittleEndian, dim); err != nil {
				return err
			}
		}

		if err := binary.Write(buf, binary.LittleEndian, uint32(tensor.typ)); err != nil {
			return err
		}

		if err := binary.Write(buf, binary.LittleEndian, tensorDataOffset); err != nil {
			return err
		}

		tensorDataOffset += uint64(len(tensor.data))
	}

	currentLen := buf.Len()
	alignLen := int(alignment) - (currentLen % int(alignment))
	if alignLen != int(alignment) {
		buf.Write(make([]byte, alignLen))
	}

	for _, tensor := range tensors {
		buf.Write(tensor.data)
	}

	return os.WriteFile(filePath, buf.Bytes(), 0644)
}
