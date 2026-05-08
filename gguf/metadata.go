package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
)

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
