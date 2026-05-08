# Qwen3 GGUF 推理引擎

## 项目概述

纯 Go 实现的 Qwen3 模型推理引擎，支持 GGUF 格式的 Q8_0 量化模型。
模型文件：`/mnt/d/model/Qwen3-0.6B-Q8_0.gguf`

## 模型参数

| Parameter | Value |
|-----------|-------|
| Architecture | Qwen3ForCausalLM |
| Layers | 28 |
| Embedding Dim | 1024 |
| Attention Heads | 16 |
| KV Heads | 8 |
| Head Dim | 128 |
| FFN Dim | 3072 |
| RoPE Type | NEOX |
| RoPE freq_base | 1000000 |
| QK Norm | per-head RMSNorm |
| Weight Tying | output = token_embedding |

## 关键实现说明

### GGUF Tensor 布局
GGUF 使用 `[dim0, dim1]` 格式，dim0 变化最快。
例如 shape `{1024, 3072}` 表示 3072 行 × 1024 列。
`MatMulTransposed` 匹配 ggml_mul_mat 语义：
`C[i,j] = sum_k A[i*K+k] * B[j*K+k]`

### Q8_0 反量化
每个 block 34 字节：2 字节 fp16 scale + 32 个 int8 值。

### QK 归一化顺序
projection → QK RMSNorm（per-head） → RoPE（NEOX） → cache → attention

### SwiGLU FFN
`silu(gate) * up` → down projection

## 已修复 Bug

### NEOX RoPE（关键修复）

Qwen3 使用 `LLAMA_ROPE_TYPE_NEOX` (=2)，与常见的 NORMAL 模式不同。
这是乱码输出的根因。

**NORMAL 配对**（错误的）：
```
(0,1), (2,3), (4,5), ..., (126,127)
```

**NEOX 配对**（正确的，head_dim=128）：
```
(0,64), (1,65), (2,66), ..., (63,127)
```

修复文件：
- `math/math.go` — `RoPE()` 函数新增 mode 参数
- `model/qwen3.go` — 全部使用 `llmmath.RoPE_NEOX`

### 其他修复
- **MatMul 方向**：从 `MatMul` 改为 `MatMulTransposed` 匹配 ggml_mul_mat
- **Embedding 布局**：从 `EmbeddingLookupDimFirst` 改为 `EmbeddingLookupTokenFirst`
- **GGUF tensor offset**：mmap 加载缺失 `DataStart` 偏移

## 验证方法

1. **Python 交叉验证**：`uv run --with gguf python3 debug/*/check_*.py`
2. **llama.cpp 对比**：编译 C++ 程序 dump logits 直接逐元素对比
3. **推理测试**：`go run infer.go <model> "Hello" --temp=0 --tokens=50`

## 调试目录约定

- 所有调试程序放在 `debug/` 下
- 主题分类使用 `diag_*` 目录
- 结构化调查使用 `<编号>_<描述>/` 目录
- 每个目录有中文 README.md
- Python 脚本使用 `uv run` 执行

---

## 测试规范

### 每次代码修改后的必做检查

1. ✅ 运行 unit tests：`go test ./... -v`
2. ✅ 运行实际命令行程序验证输出
3. ✅ 检查 regressions

**MOCK TESTS 不够。必须用真实模型运行。**

### 命令行验证示例

```bash
go run infer.go /mnt/d/model/Qwen3-0.6B-Q8_0.gguf "hi" --temp=0
```

验证要点：
- 输出不是乱码
- Token IDs 合理（不全零、不随机噪音）
- Decode 后的文本有意义
- 不崩溃、不 panic

### 测试清单

1. **Unit Tests**：为新功能编写单元测试
2. **Integration Tests**：在全系统上下文中测试
3. **Command-Line Test**：用真实输入运行实际程序（必须）
4. **Edge Cases**：测试边界条件
5. **Regression Tests**：确保已有功能仍然正常

## 经验教训

### Embedding Layout 注意事项
不同模型格式使用不同的 tensor 布局：
- **Token-first**：`[vocab_size, embedding_dim]` — token 0 的 embedding 是前 embeddingDim 个元素
- **Dim-first**：`[embedding_dim, vocab_size]` — dim 0 的所有 token 值是前 vocabSize 个元素

### RoPE 类型很重要
不同模型使用不同的 RoPE type。Qwen3 使用 NEOX（half-split pairing），不是标准的 NORMAL（consecutive pairing）。**必须检查 llama.cpp 参考实现确认正确的 RoPE type。**

### 调试推理问题的方法
当推理输出异常时，系统性地逐个验证：
**dequantization → weights → KV cache → 逐层 hidden states → logits**

最可靠的方式是写一个 C++ 程序用 llama.cpp API dump logits，和你的实现逐元素对比。

### 最小修改原则
修复 bug 时只做最小必要的改动。不要引入可能破坏已有功能的额外修改。

### 不完整的测试的标志
- ❌ "All tests pass" — 但只跑了 mock tests
- ❌ "The logic looks correct" — 但没有实际验证
- ✅ "运行 infer.go 输出是 X" — 用真实模型验证过
- ✅ "修复前是 A，修复后是 B" — 有对比

---

> **NO CODE CHANGE IS COMPLETE UNTIL YOU HAVE:**
> 1. Written and run unit tests
> 2. Run the actual command-line program
> 3. Verified the output is correct and meaningful
