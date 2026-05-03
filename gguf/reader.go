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

func (r *GGUFReader) readMetadata(reader io.Reader) error {
	for i := uint64(0); i < r.Header.MetadataKVCount; i++ {
		item, err := r.readMetadataItem(reader)
		if err != nil {
			return fmt.Errorf("failed to read metadata item %d: %w", i, err)
		}
		r.Metadata[item.Key] = item
	}
	return nil
}

func (r *GGUFReader) readMetadataItem(reader io.Reader) (GGUFMetadataItem, error) {
	item := GGUFMetadataItem{}

	keyLen, err := r.readUint64(reader)
	if err != nil {
		return item, fmt.Errorf("failed to read key length: %w", err)
	}

	keyBytes := make([]byte, keyLen)
	if _, err := io.ReadFull(reader, keyBytes); err != nil {
		return item, fmt.Errorf("failed to read key: %w", err)
	}
	item.Key = string(keyBytes)

	valueType, err := r.readUint32(reader)
	if err != nil {
		return item, fmt.Errorf("failed to read value type: %w", err)
	}
	item.Type = GGUFType(valueType)

	value, err := r.readValue(reader, item.Type)
	if err != nil {
		return item, fmt.Errorf("failed to read value: %w", err)
	}
	item.Value = value

	return item, nil
}

func (r *GGUFReader) readValue(reader io.Reader, valueType GGUFType) (interface{}, error) {
	switch valueType {
	case GGUF_TYPE_UINT8:
		return r.readUint8(reader)
	case GGUF_TYPE_INT8:
		return r.readInt8(reader)
	case GGUF_TYPE_UINT16:
		return r.readUint16(reader)
	case GGUF_TYPE_INT16:
		return r.readInt16(reader)
	case GGUF_TYPE_UINT32:
		return r.readUint32(reader)
	case GGUF_TYPE_INT32:
		return r.readInt32(reader)
	case GGUF_TYPE_FLOAT32:
		return r.readFloat32(reader)
	case GGUF_TYPE_BOOL:
		return r.readBool(reader)
	case GGUF_TYPE_STRING:
		return r.readString(reader)
	case GGUF_TYPE_ARRAY:
		return r.readArray(reader)
	case GGUF_TYPE_UINT64:
		return r.readUint64(reader)
	case GGUF_TYPE_INT64:
		return r.readInt64(reader)
	case GGUF_TYPE_FLOAT64:
		return r.readFloat64(reader)
	default:
		return nil, fmt.Errorf("unsupported value type: %d", valueType)
	}
}

func (r *GGUFReader) readUint8(reader io.Reader) (uint8, error) {
	var val uint8
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readInt8(reader io.Reader) (int8, error) {
	var val int8
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readUint16(reader io.Reader) (uint16, error) {
	var val uint16
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readInt16(reader io.Reader) (int16, error) {
	var val int16
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readUint32(reader io.Reader) (uint32, error) {
	var val uint32
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readInt32(reader io.Reader) (int32, error) {
	var val int32
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readUint64(reader io.Reader) (uint64, error) {
	var val uint64
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readInt64(reader io.Reader) (int64, error) {
	var val int64
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readFloat32(reader io.Reader) (float32, error) {
	var val float32
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readFloat64(reader io.Reader) (float64, error) {
	var val float64
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return 0, err
	}
	return val, nil
}

func (r *GGUFReader) readBool(reader io.Reader) (bool, error) {
	var val uint8
	if err := binary.Read(reader, r.byteOrder, &val); err != nil {
		return false, err
	}
	return val != 0, nil
}

func (r *GGUFReader) readString(reader io.Reader) (string, error) {
	strLen, err := r.readUint64(reader)
	if err != nil {
		return "", err
	}

	strBytes := make([]byte, strLen)
	if _, err := io.ReadFull(reader, strBytes); err != nil {
		return "", err
	}

	return string(strBytes), nil
}

func (r *GGUFReader) readArray(reader io.Reader) (interface{}, error) {
	elemType, err := r.readUint32(reader)
	if err != nil {
		return nil, err
	}

	elemCount, err := r.readUint64(reader)
	if err != nil {
		return nil, err
	}

	switch GGUFType(elemType) {
	case GGUF_TYPE_UINT8:
		arr := make([]uint8, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readUint8(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_INT8:
		arr := make([]int8, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readInt8(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_UINT16:
		arr := make([]uint16, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readUint16(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_INT16:
		arr := make([]int16, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readInt16(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_UINT32:
		arr := make([]uint32, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readUint32(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_INT32:
		arr := make([]int32, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readInt32(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_FLOAT32:
		arr := make([]float32, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readFloat32(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_BOOL:
		arr := make([]bool, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readBool(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_STRING:
		arr := make([]string, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readString(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_UINT64:
		arr := make([]uint64, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readUint64(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_INT64:
		arr := make([]int64, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readInt64(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	case GGUF_TYPE_FLOAT64:
		arr := make([]float64, elemCount)
		for i := uint64(0); i < elemCount; i++ {
			arr[i], err = r.readFloat64(reader)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unsupported array element type: %d", elemType)
	}
}

func (r *GGUFReader) readTensorInfo(reader io.Reader) error {
	r.Tensors = make([]GGUFTensorInfo, r.Header.TensorCount)

	for i := uint64(0); i < r.Header.TensorCount; i++ {
		tensor, err := r.readSingleTensorInfo(reader)
		if err != nil {
			return fmt.Errorf("failed to read tensor %d: %w", i, err)
		}
		r.Tensors[i] = tensor
	}

	return r.alignReader(reader)
}

func (r *GGUFReader) readSingleTensorInfo(reader io.Reader) (GGUFTensorInfo, error) {
	tensor := GGUFTensorInfo{}

	nameLen, err := r.readUint64(reader)
	if err != nil {
		return tensor, fmt.Errorf("failed to read tensor name length: %w", err)
	}

	nameBytes := make([]byte, nameLen)
	if _, err := io.ReadFull(reader, nameBytes); err != nil {
		return tensor, fmt.Errorf("failed to read tensor name: %w", err)
	}
	tensor.Name = string(nameBytes)

	nDims, err := r.readUint32(reader)
	if err != nil {
		return tensor, fmt.Errorf("failed to read tensor dimensions: %w", err)
	}
	tensor.NDimensions = nDims

	tensor.Shape = make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		tensor.Shape[i], err = r.readUint64(reader)
		if err != nil {
			return tensor, fmt.Errorf("failed to read tensor shape %d: %w", i, err)
		}
	}

	tensorType, err := r.readUint32(reader)
	if err != nil {
		return tensor, fmt.Errorf("failed to read tensor type: %w", err)
	}
	tensor.Type = GGMLQuantizationType(tensorType)

	tensor.Offset, err = r.readUint64(reader)
	if err != nil {
		return tensor, fmt.Errorf("failed to read tensor offset: %w", err)
	}

	return tensor, nil
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

		_, err := seeker.Seek(int64(tensor.Offset), io.SeekStart)
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

func (r *GGUFReader) GetTensor(name string) (*GGUFTensorInfo, []byte, bool) {
	for i := range r.Tensors {
		if r.Tensors[i].Name == name {
			return &r.Tensors[i], r.TensorData[name], true
		}
	}
	return nil, nil, false
}

func (r *GGUFReader) GetMetadata(key string) (interface{}, bool) {
	item, ok := r.Metadata[key]
	if !ok {
		return nil, false
	}
	return item.Value, true
}

func (r *GGUFReader) GetMetadataString(key string) (string, bool) {
	value, ok := r.GetMetadata(key)
	if !ok {
		return "", false
	}
	str, ok := value.(string)
	return str, ok
}

func (r *GGUFReader) GetMetadataInt(key string) (int64, bool) {
	value, ok := r.GetMetadata(key)
	if !ok {
		return 0, false
	}

	switch v := value.(type) {
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint8:
		return int64(v), true
	case uint16:
		return int64(v), true
	case uint32:
		return int64(v), true
	case uint64:
		return int64(v), true
	default:
		return 0, false
	}
}

func (r *GGUFReader) GetMetadataFloat(key string) (float64, bool) {
	value, ok := r.GetMetadata(key)
	if !ok {
		return 0, false
	}

	switch v := value.(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
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
