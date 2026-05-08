package model

import (
	"fmt"
	"log"

	"github.com/adfnekc/go-qwen3-gguf/gguf"
)

type ModelLoader struct {
	Reader    *gguf.GGUFReader
	Config    *Qwen3Config
	Weights   *Qwen3Weights
	MMapFile  *gguf.MMapFile
	UsingMMap bool
}

type LoadOptions struct {
	PreferMMap bool
}

func DefaultLoadOptions() *LoadOptions {
	return &LoadOptions{
		PreferMMap: true,
	}
}

func LoadModel(modelPath string) (*ModelLoader, error) {
	return LoadModelWithOptions(modelPath, DefaultLoadOptions())
}

func LoadModelWithOptions(modelPath string, options *LoadOptions) (*ModelLoader, error) {
	loader := &ModelLoader{}

	if options.PreferMMap {
		log.Printf("Attempting to load model with mmap...")
		reader := gguf.NewGGUFReader()
		mmapFile, err := reader.LoadFromFileMMap(modelPath)
		if err == nil {
			loader.Reader = reader
			loader.MMapFile = mmapFile
			loader.UsingMMap = true
			log.Printf("Successfully loaded model with mmap")
			return loader, nil
		}

		log.Printf("Warning: Failed to load with mmap: %v", err)
		log.Printf("Falling back to regular file loading...")
	}

	reader := gguf.NewGGUFReader()
	if err := reader.LoadFromFile(modelPath); err != nil {
		return nil, fmt.Errorf("failed to load GGUF file: %w", err)
	}

	loader.Reader = reader
	loader.UsingMMap = false
	log.Printf("Successfully loaded model with regular file loading")

	return loader, nil
}

func (loader *ModelLoader) LoadConfig() (*Qwen3Config, error) {
	if loader.Reader == nil {
		return nil, fmt.Errorf("reader not initialized")
	}

	config, err := LoadQwen3Config(loader.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to load model config: %w", err)
	}

	loader.Config = config
	return config, nil
}

func (loader *ModelLoader) LoadWeights() (*Qwen3Weights, error) {
	if loader.Reader == nil {
		return nil, fmt.Errorf("reader not initialized")
	}

	if loader.Config == nil {
		return nil, fmt.Errorf("config not loaded, call LoadConfig first")
	}

	var weights *Qwen3Weights
	var err error

	if loader.UsingMMap && loader.MMapFile != nil {
		weights, err = LoadQwen3WeightsMMap(loader.Reader, loader.Config, loader.MMapFile)
	} else {
		weights, err = LoadQwen3Weights(loader.Reader, loader.Config)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	loader.Weights = weights
	return weights, nil
}

func (loader *ModelLoader) LoadConfigAndWeights() (*Qwen3Config, *Qwen3Weights, error) {
	config, err := loader.LoadConfig()
	if err != nil {
		return nil, nil, err
	}

	weights, err := loader.LoadWeights()
	if err != nil {
		return nil, nil, err
	}

	return config, weights, nil
}

func (loader *ModelLoader) Close() {
	if loader.MMapFile != nil {
		loader.MMapFile.Close()
		loader.MMapFile = nil
	}
}
