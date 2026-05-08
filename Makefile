.PHONY: build test lint run clean

BINARY=bin/infer

build:
	go build -o $(BINARY) ./cmd/infer

test:
	go test ./gguf/... ./math/... ./model/... ./tokenizer/... -v

lint:
	golint ./cmd/infer

run:
	go run ./cmd/infer

clean:
	rm -rf bin/
