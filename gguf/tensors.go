package gguf

import (
	"fmt"
	"io"
)

func (r *GGUFReader) readTensorInfo(reader io.Reader) error {
	r.Tensors = make([]GGUFTensorInfo, r.Header.TensorCount)

	for i := uint64(0); i < r.Header.TensorCount; i++ {
		tensor, err := r.readSingleTensorInfo(reader)
		if err != nil {
			return fmt.Errorf("failed to read tensor %d: %w", i, err)
		}
		r.Tensors[i] = tensor
	}

	if seeker, ok := reader.(io.Seeker); ok {
		currentPos, err := seeker.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		alignment := int64(32)
		if v, ok := r.GetMetadataInt("general.alignment"); ok && v > 0 {
			alignment = v
		}

		alignedPos := (currentPos + alignment - 1) & ^(alignment - 1)
		r.DataStart = alignedPos

		if alignedPos > currentPos {
			_, err = seeker.Seek(alignedPos, io.SeekStart)
			if err != nil {
				return err
			}
		}
	}
	return nil
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

func (r *GGUFReader) GetTensor(name string) (*GGUFTensorInfo, []byte, bool) {
	for i := range r.Tensors {
		if r.Tensors[i].Name == name {
			return &r.Tensors[i], r.TensorData[name], true
		}
	}
	return nil, nil, false
}
