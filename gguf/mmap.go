package gguf

import (
	"fmt"
	"os"
	"reflect"
	"unsafe"

	"golang.org/x/sys/unix"
)

type MMapFile struct {
	file *os.File
	data []byte
	size int64
}

func NewMMapFile(filePath string) (*MMapFile, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	size := info.Size()
	if size == 0 {
		file.Close()
		return nil, fmt.Errorf("file is empty")
	}

	data, err := unix.Mmap(int(file.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap file: %w", err)
	}

	return &MMapFile{
		file: file,
		data: data,
		size: size,
	}, nil
}

func (m *MMapFile) Close() error {
	if m.data != nil {
		if err := unix.Munmap(m.data); err != nil {
			return fmt.Errorf("failed to munmap: %w", err)
		}
		m.data = nil
	}
	if m.file != nil {
		if err := m.file.Close(); err != nil {
			return fmt.Errorf("failed to close file: %w", err)
		}
		m.file = nil
	}
	return nil
}

func (m *MMapFile) Data() []byte {
	return m.data
}

func (m *MMapFile) Size() int64 {
	return m.size
}

func (m *MMapFile) ReadAt(offset int64, length int) ([]byte, error) {
	if offset < 0 || offset >= m.size {
		return nil, fmt.Errorf("offset out of bounds")
	}
	if offset+int64(length) > m.size {
		length = int(m.size - offset)
	}
	return m.data[offset : offset+int64(length)], nil
}

func (m *MMapFile) GetSlice(offset int64, length int) ([]byte, error) {
	if offset < 0 || offset >= m.size {
		return nil, fmt.Errorf("offset out of bounds")
	}
	end := offset + int64(length)
	if end > m.size {
		end = m.size
	}
	return m.data[offset:end], nil
}

type MMapReader struct {
	mmapFile *MMapFile
	offset   int64
}

func NewMMapReader(mmapFile *MMapFile) *MMapReader {
	return &MMapReader{
		mmapFile: mmapFile,
		offset:   0,
	}
}

func (r *MMapReader) Read(p []byte) (int, error) {
	if r.offset >= r.mmapFile.size {
		return 0, fmt.Errorf("EOF")
	}
	remaining := r.mmapFile.size - r.offset
	length := len(p)
	if int64(length) > remaining {
		length = int(remaining)
	}
	copy(p, r.mmapFile.data[r.offset:r.offset+int64(length)])
	r.offset += int64(length)
	return length, nil
}

func (r *MMapReader) Seek(offset int64, whence int) (int64, error) {
	var newOffset int64
	switch whence {
	case 0:
		newOffset = offset
	case 1:
		newOffset = r.offset + offset
	case 2:
		newOffset = r.mmapFile.size + offset
	default:
		return 0, fmt.Errorf("invalid whence")
	}
	if newOffset < 0 || newOffset > r.mmapFile.size {
		return 0, fmt.Errorf("invalid offset")
	}
	r.offset = newOffset
	return newOffset, nil
}

func (r *GGUFReader) LoadFromFileMMap(filePath string) (*MMapFile, error) {
	mmapFile, err := NewMMapFile(filePath)
	if err != nil {
		return nil, err
	}

	reader := NewMMapReader(mmapFile)

	if err := r.readHeader(reader); err != nil {
		mmapFile.Close()
		return nil, err
	}

	if err := r.readMetadata(reader); err != nil {
		mmapFile.Close()
		return nil, err
	}

	if err := r.readTensorInfo(reader); err != nil {
		mmapFile.Close()
		return nil, err
	}

	return mmapFile, nil
}

func (r *GGUFReader) LoadTensorDataFromMMap(mmapFile *MMapFile) error {
	for i := range r.Tensors {
		tensor := &r.Tensors[i]

		dataSize := r.calculateTensorSize(tensor)
		if dataSize == 0 {
			continue
		}

		data, err := mmapFile.GetSlice(int64(tensor.Offset), int(dataSize))
		if err != nil {
			return fmt.Errorf("failed to get tensor %s data: %w", tensor.Name, err)
		}

		r.TensorData[tensor.Name] = data
	}
	return nil
}

func BytesToFloat32Unsafe(data []byte) []float32 {
	if len(data)%4 != 0 {
		return nil
	}

	header := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Len /= 4
	header.Cap /= 4

	return *(*[]float32)(unsafe.Pointer(&header))
}
