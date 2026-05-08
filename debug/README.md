# Qwen3 GGUF Inference — Debug Programs

This directory contains diagnostic programs used during development of a pure-Go
Qwen3 GGUF inference engine. The model is **Qwen3-0.6B** (Q8_0 quantized).

## Model Architecture (Qwen3-0.6B)

| Parameter          | Value     |
|--------------------|-----------|
| Layers             | 28        |
| Embedding dim      | 1024      |
| Heads              | 16        |
| KV heads           | 8         |
| Head dim           | 128       |
| FFN dim (gate/up)  | 3072      |
| RoPE type          | NEOX (half-split pairing) |
| RoPE freq base     | 1,000,000 |
| QK norm            | enabled (per-head RMSNorm) |
| Weight tying       | enabled (output = tok_embd) |

## Investigation Subdirectories

| Directory | Topic |
|-----------|-------|
| `03_weight_verification/` | Verify dequantization, weight stats, and forward pass correctness |
| `04_chat_inference/`      | KV cache correctness, layer-by-layer comparison |
| `05_q8_dequant_verify/`   | Python-based Q8_0 dequantization cross-verification |
| `06_rope_fix/`           | NEOX RoPE fix and logits comparison with llama.cpp |

## Top-Level Diagnostic Files

### Tokenizer

| File | Description |
|------|-------------|
| `check_tok.go` | Encode/decode various strings; verify chat template construction |
| `check_special.go` | Investigate special tokens (<think>, <|im_start|>, etc.) |
| `debug_tok.go` | Basic tokenizer encoding tests across common words |

### Model Metadata

| File | Description |
|------|-------------|
| `debug_meta.go` | Print all GGUF metadata key-value pairs |
| `debug_shapes.go` | Print tensor shapes for key weight tensors |

### Weight Diagnostics

| File | Description |
|------|-------------|
| `debug_model.go` | Model loading diagnostics and weight stats |
| `debug_scales.go` | Per-layer weight scaling analysis |
| `debug_norm.go` | Output norm weight stats and token embedding values |
| `debug_norms.go` | QK norm weight analysis |

### Per-Layer Diagnostics

| File | Description |
|------|-------------|
| `debug_layers.go` | Forward one layer and check output |
| `debug_layers2.go` | Extended layer diagnostics with top-10 token predictions |
| `debug_alllayers.go` | All-layer diagnostic trace |
| `debug_isolate.go` | Isolate RMS explosion per layer |
| `debug_embedding.go` | Embedding lookup and output projection test |

### RMS Diagnostics

| File | Description |
|------|-------------|
| `debug_rms.go` | Full 28-layer RMS trace after each op |
| `debug_layer2.go` | Detailed per-operation trace for layers 0-2 |
| `debug_no_qknorm.go` | Test with QK norm weights disabled |
| `debug_rmsnorm.go` | RMSNorm verification |
| `debug_forward.go` | Forward() method output — top token predictions |

### Comparison

| File | Description |
|------|-------------|
| `debug_compare.go` | Compare implementations |

## How to Run

```bash
go run debug/<name>.go /path/to/model.gguf
```

## History of Fixed Bugs

1. **MatMul direction**: Original used `MatMul` (weight layout
   output_dim×input_dim); ggml uses `MatMulTransposed`. Fixed by switching.

2. **Embedding layout**: Original used `EmbeddingLookupDimFirst`; ggml stores
   token embeddings as `[vocab_size, embedding_dim]` → `EmbeddingLookupTokenFirst`.

3. **RoPE type**: Used NORMAL (consecutive pairing (0,1), (2,3), ...) but Qwen3
   requires NEOX (half-split pairing (0,64), (1,65), ..., (63,127)). This was
   the root cause of garbled output.

4. **GGUF tensor offset**: MMAP tensor loading was missing `DataStart` offset,
   causing wrong weights to be loaded from mmap.
