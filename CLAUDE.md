# Qwen3 GGUF 推理引擎 — 项目知识

## 项目概述

纯 Go 实现的 Qwen3 模型推理引擎，支持 GGUF 格式的 Q8_0 量化模型。
模型文件：`/mnt/d/model/Qwen3-0.6B-Q8_0.gguf`

## 关键修复记录

### NEOX RoPE（最重要的修复）

Qwen3 使用 `LLAMA_ROPE_TYPE_NEOX` (=2)，与常见的 NORMAL 模式不同。

**NORMAL 配对**（错误的）：
```
(0,1), (2,3), (4,5), ..., (126,127)
```

**NEOX 配对**（正确的，对 head_dim=128）：
```
(0,64), (1,65), (2,66), ..., (63,127)
```

修复位置：
- `math/math.go` — `RoPE()` 函数新增 `mode` 参数
- `model/qwen3.go` — 全部使用 `llmmath.RoPE_NEOX`

## 模型参数

| 参数 | 值 |
|------|------|
| 架构 | Qwen3ForCausalLM |
| 层数 | 28 |
| 嵌入维度 | 1024 |
| 注意力头数 | 16 |
| KV 头数 | 8 |
| 头维度 | 128 |
| FFN 维度 | 3072 |
| RoPE 类型 | NEOX |
| RoPE 频率基数 | 1000000 |
| QK 归一化 | 启用（逐头 RMSNorm） |
| 权重绑定 | output = token_embedding |

## 关键实现说明

### GGUF Tensor 布局
GGUF 使用 `[dim0, dim1]` 格式，dim0 变化最快。
例如 shape `{1024, 3072}` 表示 3072 行 × 1024 列。
`MatMulTransposed` 匹配 ggml_mul_mat 语义：
`C[i,j] = sum_k A[i*K+k] * B[j*K+k]`

### Q8_0 反量化
每个 block 34 字节：2 字节 fp16 scale + 32 个 int8 值。

### QK 归一化顺序
投影 → QK RMSNorm（逐头） → RoPE（NEOX） → 缓存 → 注意力

### SwiGLU FFN
`silu(gate) * up` → down 投影

## 验证方法

1. **Python 交叉验证**：`uv run --with gguf python3 debug/*/check_*.py`
2. **llama.cpp 对比**：编译 C++ 程序直接对比 logits
3. **推理测试**：`go run infer.go <model> "Hello" --temp=0 --tokens=50`

## 调试目录约定

- 所有调试程序放在 `debug/` 下
- 主题分类：`diag_*` 目录
- 结构化调查：`<编号>_<描述>/` 目录
- 每个目录有中文 README.md
- Python 脚本使用 `uv run` 执行
