# Go Qwen3 GGUF 推理引擎 — 重构设计

## 概述

对现有 Qwen3 GGUF 推理引擎进行系统性重构，从底层到上层分层改造。
目标是提升代码质量、可维护性、可测试性和工程化水平。

## 重构策略

自底向上分层，每层完成后确保 `go test ./...` 通过，再进行下一层。

## 第1层：Module Path + 包结构

### Module path 修正

```
gguf  →  github.com/adfnekc/go-qwen3-gguf
```

- 修改 `go.mod` 的 `module` 行
- 全局替换所有 `import "gguf/..."` → `import "github.com/adfnekc/go-qwen3-gguf/..."`
- 纯机械替换，无行为变化

### reader.go 拆分

将 `gguf/reader.go`（804 行）按职责拆分为：

| 文件 | 职责 | 预估行数 |
|------|------|----------|
| `reader.go` | GGUFReader 结构体、header 解析、文件加载入口 | ~300 |
| `metadata.go` | metadata KV 读取、类型转换 | ~200 |
| `tensors.go` | tensor info 读取、偏移排序 | ~300 |

## 第2层：Weights 去重

### 现状

`weights.go` 和 `weights_mmap.go` 有约 80% 重复代码：
- 相同的 tensor 名称映射（`token_embd.weight` / `model.embed_tokens.weight` 等 fallback）
- 相同的 prefix 检测逻辑（`blk.N.` / `model.layers.N.`）
- 相同的按 block 遍历和张量加载顺序

### 方案

不引入额外接口抽象，直接在 `model/` 包内合并为一个文件 `weights_load.go`：

```go
// tensorLoadFunc 定义为从 reader 加载并反量化一个 tensor 的策略
type tensorLoadFunc func(reader GGUFReader, name string) ([]float32, error)
```

- `loadTensorDequant` — 非 mmap：从 reader 的 `[]byte` 反量化
- `loadTensorMMap` — mmap：从 `mmap slice` 反量化

共享的函数：
- `resolveTensorName(reader, names...)` — 按顺序查找 tensor，返回第一个匹配
- `resolveBlockPrefix(reader, blockIdx)` — 检测 `blk.N.` 或 `model.layers.N.`
- `loadBlockWeights(reader, prefix, loadFn)` — 加载一个 block 的所有权重

## 第3层：Forward() 重构

### 现状

`Qwen3Model.Forward()` 200+ 行，多个职责混在一起：
- Embedding lookup
- Per-layer attention（QKV 投影、QK norm、RoPE、score 计算、因果掩码、加权求和）
- FFN（SwiGLU gate/up/down）
- 多处空 `if layer == 0 { }` 死代码

### 方案

```go
// Forward 入口——只做编排
func (m *Qwen3Model) Forward(tokens []int, startPos int) ([]float32, error)

// 提取为方法
func (m *Qwen3Model) embeddingLookup(tokens []int) []float32
func (m *Qwen3Model) forwardLayer(hs []float32, bw *Qwen3BlockWeights, layer, startPos, seqLen int) []float32
func applyQKNorm(q, k []float32, bw *Qwen3BlockWeights, config *Qwen3Config, seqLen, nHeads, nKvHeads, headDim int) ([]float32, []float32)
func applyRoPEToHeads(q, k []float32, pos, seqLen, nHeads, nKvHeads, headDim int, ropeFreqBase float32) ([]float32, []float32)
func applyAttention(q, k, v []float32, config *Qwen3Config, cache *KVCache, layer, seqLen, pastSeqLen int) []float32
func applyFFN(hs []float32, bw *Qwen3BlockWeights, config *Qwen3Config, seqLen int) []float32

// 删除的内容
- 删除所有空的 if layer == 0 { } 块
- 删除重复的 hidden 拷贝逻辑（residual 管理集中化）
```

### 好处

- 每个函数 20-40 行，职责单一
- 可单独测试 attention 和 FFN
- 注意力计算和 FFN 解耦，后续支持其他架构只需替换对应组件

## 第4层：CLI + Tokenizer 重构

### Tokenizer 接口化

```go
// tokenizer/tokenizer.go
type Tokenizer interface {
    Encode(text string) []int
    Decode(tokens []int) string
    GetToken(id int) (string, bool)
    GetVocabSize() int
}
```

- `BPETokenizer`（原 `Tokenizer`）实现该接口
- `SimpleTokenizer` 也实现该接口
- `runInference` 参数从 `interface{}` 改为 `Tokenizer`

### Chat Template 提取

```go
// model/chat.go
func ApplyQwen3ChatTemplate(tok Tokenizer, userInput string) []int
```

- 从 model metadata 读取特殊 token ID（`tokenizer.ggml.bos_token_id`、`chat_template` 等）
- 硬编码的 151644/151645 改为配置驱动

### infer.go 拆分

```
infer.go  →  cmd/infer/main.go  +  internal/runner.go
```

| 文件 | 职责 |
|------|------|
| `cmd/infer/main.go` | CLI 入口、flag 解析、调用 runner |
| `internal/runner.go` | 加载 → 推理 → 输出 的编排逻辑 |

## 第5层：工程化基础设施

### Makefile

```makefile
.PHONY: build test lint run clean

build:
    go build -o bin/infer ./cmd/infer

test:
    go test ./... -v

lint:
    golangci-lint run

run:
    go run ./cmd/infer

clean:
    rm -rf bin/
```

### CI (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
      - run: go build ./...
      - run: go test ./... -v
```

## 第6层：性能优化

### KV Cache 动态增长

现状：预分配 `maxSeqLen * nLayers * nKvHeads * headDim` float32。

改为按需 `append`，避免大块内存预分配。

```go
type KVCache struct {
    Keys   [][][]float32  // [layer][position][head*dim]
    Values [][][]float32
    Size   []int
}
// NewKVCache 改为接受初始容量，不预分配 position
// Update 时 append
// GetKV 返回 slice（减少拷贝）
```

### Buffer 复用

- `attentionScores`（per-head per-position）— 复用 per-layer buffer
- `outputHead` — 复用 per-head buffer
- `GetKV` 返回值改为 `(keys, values []float32)` 直接引用 cache 内 slice，而非 copy

## 执行顺序

```
第1层 ──→ 第2层 ──→ 第3层 ──→ 第4层 ──→ 第5层 ──→ 第6层
（module）  （weights） （forward）  （CLI/）    （Make/）   （perf）
                             ↓
                     每层后 go test ./...
```

## 不包含的范围

- 不支持 Qwen3 以外的模型架构
- 不改 `math/math.go` 中的数学运算实现（后续可按需优化）
- 不改 GGUF 解析核心逻辑（reader/metadata/tensor 格式）
- 不引入第三方依赖（除已有的 `golang.org/x/sys`）

## 验收标准

1. `go test ./...` 全部通过
2. `go run ./cmd/infer <model> "hi" --temp=0` 输出与重构前一致
3. `go build ./...` 无错误
4. Makefile 所有 target 正常工作
5. 无空 `if layer == 0 { }` 等死代码
6. 无 `fmt.Printf("DEBUG: ...")` 调试输出
