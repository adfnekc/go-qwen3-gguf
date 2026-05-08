# Qwen3 是如何工作的：从 Token 到输出

> 面向资深开发者的大模型入门。假设你理解软件工程和基本线性代数，但对 Transformer 内部机制了解有限。我们用 Qwen3-0.6B 的具体数值逐一拆解。

## 概述

大语言模型本质上是一个 **极度巨大的条件概率分布**。给定一串 token，它预测下一个 token 是什么。

```
输入:  "hi"                    → token IDs [44, 402]
输出:  P(next token | "hi")    → 151936 个概率
采样:  argmax → token 2080 ("<think>")
```

整个推理过程可以看作一条流水线：

```
Tokens → Embedding → ×28 Transformer Layers → Norm → Linear → Probabilities
                          ↕                ↕
                    Self-Attention    FFN (SwiGLU)
```

下面我们从输入"hi"开始，跟随数据一路走到输出。

---

## 1. Tokenization：从文本到整数

模型不认识字符，只认识整数。Tokenizer 的作用是把文本映射到一个固定大小的词汇表中的 ID。

Qwen3 使用 BPE（Byte-Pair Encoding）tokenizer，词汇表大小 **151936**。

```
"hi" → BPE encode → [44, 402]   (具体 ID 依 tokenizer 实现而异)
```

注意：**BPE tokenizer 不是字符级编码**。`"hi"` 可能是一个 token、两个 token、甚至带前缀空格的变体。这里假设编码为 `[44, 402]` 作为示例，不表示实际对应关系。

每个 token 对应词汇表中的一个条目。词汇表里包含了从单个字节到完整词语的各种片段。

> **对资深开发者的类比**：Tokenizer 就像协议中的序列化/反序列化层。输入文本被"序列化"成整数序列，模型输出整数序列再"反序列化"回文本。

---

## 2. Embedding：从整数到向量

模型无法直接处理离散的整数 ID。每个 token ID 被映射到一个 **dense vector**（稠密向量）。

Qwen3-0.6B 的 embedding 维度是 **1024**。也就是说，每个 token 变成一个 1024 维的浮点数向量。

```
Token 44 → lookup → 向量 h_0, 长度 1024
Token 402 → lookup → 向量 h_1, 长度 1024
```

这个查找表 `TokenEmbedding` 的形状是 `[151936, 1024]`——词汇表中每个 token 对应一个 1024 维向量。这些向量是 **可学习的参数**，在训练过程中不断调整。

两行代码等价于：

```go
emb := llmmath.EmbeddingLookupTokenFirst(
    m.Weights.TokenEmbedding,  // shape [151936, 1024]
    m.Config.VocabSize,        // 151936
    m.Config.EmbeddingLength,  // 1024
    token,                     // 44 or 402
)
```

现在我们有输入矩阵 `hidden_states`，形状 `[2, 1024]`（seq_len=2, embedding_dim=1024）。

> 以下沿用 `[44, 402]` 作为示例 token ID，不表示它们的实际对应字符。

---

## 3. Transformer Layer（核心）

这是模型的核心，重复 **28 次**。每一层做的事情基本相同：对输入做 self-attention 然后经过 FFN。

每一层的结构：

```
hidden_states
    │
    ▼
┌─────────────────────────┐
│  RMSNorm (pre-attn)     │
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  Self-Attention         │
│  QKV Projection         │
│  QK RMSNorm             │
│  RoPE                   │
│  Score → Softmax → Agg  │
└─────────┬───────────────┘
          ▼
    ┌─────┴─────┐    ← Residual connection (add)
    ▼           ▼
hidden_states  RMSNorm (pre-ffn)
    │           │
    │           ▼
    │    ┌──────────┐
    │    │  SwiGLU  │
    │    │  FFN     │
    │    └────┬─────┘
    │         │
    └────┬────┘     ← Residual connection (add)
         ▼
   next hidden_states
```

### 3.1 RMSNorm：比 LayerNorm 更简单

原始 Transformer 使用 LayerNorm。Qwen3 使用 **RMSNorm**，它去掉了 LayerNorm 中的均值中心化步骤，只保留均方根归一化：

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
```

为什么可以去掉均值？经验上，Transformer 中向量的均值对训练影响很小，稳定向量尺度（scale）比中心化更重要。因此 RMSNorm 去除了 mean subtraction，只保留 RMS 缩放，同时节省了一次归约计算的成本。

对于 1024 维向量 x，RMSNorm 计算：

```
sum = x[0]² + x[1]² + ... + x[1023]²
rms = sqrt(sum / 1024 + 1e-6)
x_normalized[i] = x[i] / rms * weight[i]
```

> **对资深开发者的类比**：RMSNorm 相当于把向量缩放到单位长度（L2 归一化），但保留方向信息。这防止了深层网络中数值的指数级增长或衰减。

### 3.2 Self-Attention：让 Token 互相"看"对方

Attention 是 Transformer 中最核心的机制，决定了 token 之间如何交换信息。它的核心思想是：**每个 token 应该根据上下文来决定关注哪些其他 token**。

#### QKV Projection（投影）

输入 `hidden_states [2, 1024]` 通过三个不同的权重矩阵投影：

- **Q (Query)**：`[2, 1024] × [1024, 2048]` → `[2, 2048]`（16 heads × 128 dim）
- **K (Key)**：`[2, 1024] × [1024, 1024]` → `[2, 1024]`（8 heads × 128 dim，GQA）
- **V (Value)**：`[2, 1024] × [1024, 1024]` → `[2, 1024]`（8 heads × 128 dim）

为什么 Q 是 2048 维，K/V 是 1024 维？这是 **Grouped Query Attention (GQA)**，稍后会解释。

> **对资深开发者的类比**：Q/K/V 可以理解为三种不同角色的表示——Q（Query）是当前 token 用来"检索"上下文的表示；K（Key）是被检索时用于匹配的表示；V（Value）是真正被聚合的信息表示。Attention 计算 Q 与各 K 的匹配分数，然后用分数加权汇总对应的 V。

#### QK RMSNorm（Qwen3 的创新）

在计算 attention 分数之前，Qwen3 对每个 head 的 Q 和 K 独立做 RMSNorm。这不同于原始 Transformer：

```
// 每个 head 独立归一化
for h := 0; h < nHeads; h++ {
    start := i*qDim + h*headDim
    normed := llmmath.RMSNorm(q[start:start+headDim], bw.AttentionQNorm, config.LayerNormRmsEps)
    copy(q[start:], normed)
}
```

为什么要加这一步？原始的 attention 分数是 Q·K^T，Q 和 K 的幅度会影响分数。在深层网络中，Q/K 的幅度可能不稳定，导致 attention 分布过于尖锐（接近 one-hot）或过于平滑（接近 uniform）。QK RMSNorm 稳定了训练。

#### RoPE：位置编码（NEOX 变体）

原始 Transformer 使用正弦波位置编码（直接加到 embedding 上）。Qwen3 使用 **Rotary Position Embedding (RoPE)**，它在 attention 分数计算中编码位置信息。

RoPE 的核心思想：**对 Q 和 K 向量进行旋转，旋转角度由位置决定**。

向量在 2D 平面上旋转的公式：

```
rotate((x₁, x₂), θ) = (x₁·cos θ - x₂·sin θ, x₁·sin θ + x₂·cos θ)
```

对于 128 维的 head，RoPE 将其分成 64 对 (2D 平面)，每对旋转不同的角度：

```
对于 pair p (p = 0, 1, ..., 63)：
    θₚ = position / (1000000^(2p/128))
    x₁' = x₁·cos θₚ - x₂·sin θₚ
    x₂' = x₁·sin θₚ + x₂·cos θₚ
```

**Qwen3 使用的是 NEOX 变体**，它的配对方式与标准 RoPE 不同：

| 模式 | 配对方式 |
|------|---------|
| **NORMAL** | (0,1), (2,3), (4,5), ..., (126,127) |
| **NEOX** | **(0,64), (1,65), (2,66), ..., (63,127)** |

NEOX 配对将向量的前半部分和后半部分对应位置配对。这对于某些架构（如 GPT-NeoX/Qwen）更自然，因为它们的 FFN 层也是用类似方式 split 的。

为什么 RoPE 比绝对位置编码好？Q、K 乘以旋转矩阵后，**Q·K^T 自然依赖于 Q 和 K 之间的相对位置**（旋转角度的差）。这意味着模型可以泛化到更长的序列。

RoPE 相比绝对位置编码具有更好的长度泛化能力，因此模型有可能在超过训练长度的序列上继续工作。但长上下文能力并非天然无限——随着位置继续增大，高频维度会出现 aliasing，attention 分布逐渐退化，实际仍依赖 RoPE scaling（如 NTK-aware、YaRN）和长上下文微调等额外技术。

#### 计算 Attention 分数

```
score(q, k) = q·k / sqrt(128)  (scaled dot-product)

// 实际计算：
score[h][0] = (q_head · k_head_0) / sqrt(128)
score[h][1] = (q_head · k_head_1) / sqrt(128)
...
score[h][t] = (q_head · k_head_t) / sqrt(128)
```

除以 sqrt(128) 是关键——防止高维度下点积过大导致 softmax 进入饱和区。

#### Causal Mask

生成文本时，模型不能看到"未来的 token"。在预填充阶段（处理输入"hi"），我们对第一个位置的 mask：

```
     k_0    k_1
q_0  [0]   [-inf]    (token "h" 只能看自己)
q_1  [0]    [0]      (token "i" 可以看到 "h" 和 "i")
```

-inf 经过 softmax 后变成 0，相当于忽略。在自回归生成阶段，一次只生成一个 token，所以 mask 只是一个点积。

#### Softmax + Weighted Sum

```
probs = softmax(scores)  // [past_seq_len]，归一化到 [0, 1] 且 sum=1

output[h] = Σₚ probs[p] × V[p][h]
          = probs[0] × V[0][h] + probs[1] × V[1][h] + ...
```

输出是 value 向量的加权平均，权重由 Q-K 匹配程度决定。注意 softmax 的输出每项严格在 (0,1) 之间且总和为 1——它不会精确取到 0 或 1。

#### 合并多头

所有 16 个 heads 的输出拼接起来：

```
output = [head_0_output | head_1_output | ... | head_15_output]
       = [128维 | 128维 | ... | 128维]
       = 2048 维
```

然后通过 Output Projection 矩阵投影回 1024 维：

```
attention_out = [2, 2048] × [2048, 1024] → [2, 1024]
```

#### GQA（Grouped Query Attention）详解

Qwen3 有 16 个 Query Heads 但只有 8 个 Key/Value Heads。这意味着：

```
Q: [16 heads] → 16 个独立的"问题"
K: [8 heads]  → 8 个"键"（每个被 2 个 Q head 共享）
V: [8 heads]  → 8 个"值"（每个被 2 个 Q head 共享）
```

在 attention 计算中，KV heads 通过 **RepeatKV** 扩展到 16 个：

```go
repeat := nHeads / nKvHeads  // = 16/8 = 2
// 每个 KV head 被复制 2 次
pastK = llmmath.RepeatKV(pastK, nKvHeads, nHeads, headDim, pastSeqLen)
// K[0] → K_repeated[0], K_repeated[1]
// K[1] → K_repeated[2], K_repeated[3]
// ...
```

为什么这样做？**推理效率**。KV Cache 的大小与 KV heads 数成正比。8 个 KV heads 比 16 个节省一半的缓存。同时质量损失很小，因为多个 Query Head 可以共享相同的 Key/Value 空间（它们可能关注相同的特征，只是"提问角度"不同）。

从 MHA（Multi-Head Attention）到 GQA 的演进：

```
MHA:  Q=16, K=16, V=16  → 完整多头，计算量大，KV cache 大
MQA:  Q=16, K=1,  V=1   → 所有 Q 共享一个 KV，KV cache 最小，质量有损
GQA:  Q=16, K=8,  V=8   → 折中方案，质量接近 MHA，效率接近 MQA
```

### 3.3 Residual Connection（残差连接）

Attention 层的输出与输入相加：

```
hidden_states = hidden_states + attention_out
```

这是 Transformer 能堆到 28 层（甚至 70B 模型的 80 层）的关键设计之一。残差连接为梯度提供了"高速公路"，使梯度可以绕过深层直接流回浅层，显著改善了深层网络的优化稳定性。同时它也帮助保留了输入中的原始信息，让每一层只需学习"残差"而非完整映射。

```go
// 实际代码
hs = llmmath.VectorAdd(residual, attnOut)
```

### 3.4 SwiGLU FFN：更强大的特征变换

原始 Transformer 使用两层线性变换加 ReLU：

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

Qwen3（和 Llama/Qwen 等现代模型）使用 **SwiGLU**：

```
gate = x · W_gate  → [2, 3072]
up   = x · W_up    → [2, 3072]
swiglu = silu(gate) * up      → [2, 3072]  (逐元素相乘)
down = swiglu · W_down        → [2, 1024]
```

其中 `silu(x) = x * sigmoid(x)`。silu 是 sigmoid 线性单元，比 ReLU 更平滑。

SwiGLU 的关键：**引入了门控机制**。gate 决定哪些信息可以通过，up 提供"内容"，两者逐元素相乘。这相当于给 FFN 增加了一个可学习的"开关"。

为什么 SwiGLU 比 ReLU 好：
- ReLU 在负数区域完全截断（梯度为 0），可能导致"死神经元"
- silu 在负数区域有非零梯度，信息流动更好
- 门控机制提供了更灵活的特征选择

代价：SwiGLU 有三组权重（gate/up/down）而不是两组（W₁/W₂），参数量增加了 50%。Qwen3-0.6B 的 FFN dim 是 3072（而 embedding dim 是 1024），低于传统 4× 放大的 4096。这是因为 SwiGLU 引入了门控分支，其有效容量高于传统 ReLU FFN，因此 hidden dim 不需要扩到 4× embedding_dim，通常在 2.5×~3.5× 之间即可。

---

## 4. 逐层传播

28 层堆叠（每一层都包含 attention + FFN）后，hidden_states 在每层都被"精炼"一次：

```
Layer 0:  hidden_states → attention → + → FFN → + → hidden_states
Layer 1:  hidden_states → attention → + → FFN → + → hidden_states
...
Layer 27: hidden_states → attention → + → FFN → + → hidden_states
```

**层数的影响**：每层增加模型的"深度"。经验上，较低层通常更偏局部/语法模式（如"这个形容词修饰哪个名词"），较高层更偏抽象语义模式（如"这个代词指代什么"）。这个分层趋势已被大量 probing 实验观察到，但不是严格规律。

---

## 5. 输出：从向量到 Token

28 层之后，我们取 **最后一个 token** 的 hidden state（因为我们在做自回归生成，只需要预测下一个 token）：

```go
lastHidden := hiddenStates[(seqLen-1)*embDim : seqLen*embDim]  // [1024]
```

然后用 Output Norm（RMSNorm）归一化：

```go
normedOutput := llmmath.RMSNorm(lastHidden, m.Weights.OutputNorm, m.Config.LayerNormRmsEps)
```

最后通过 **Unembedding**（或叫 Language Model Head）映射到词汇表：

```go
logits := llmmath.MatMulTransposed(
    normedOutput,           // [1024]
    outputWeights,          // [151936, 1024] 与 token_embedding 共享权重
    1,                      // seq_len = 1
    m.Config.VocabSize,    // 151936
    embDim,                 // 1024
)
// 结果: [151936] float32 — 每个 token ID 的分数
```

**Weight Tying**：这里的 `outputWeights` 就是 `TokenEmbedding`（如果 Output 权重为 nil）。因为文本中"相似的词经常出现在相似的上下文中"，embedding 矩阵和 unembedding 矩阵可以共享参数。这显著减少了参数量，尤其在大词表模型中效果明显。

### 采样

logits（原始分数）通过 softmax 变成概率：

```go
if temperature <= 0 {
    nextToken = llmmath.Argmax(logits)    // 贪心解码
} else {
    scaledLogits[j] = logits[j] / temperature
    probs = llmmath.VectorSoftmax(scaledLogits)
    nextToken = llmmath.SampleCategorical(probs)  // 随机采样
}
```

- **greedy (temp=0)**：总是选概率最高的 token，输出确定但容易重复
- **sampling**：按概率分布随机采样，temperature 控制分布的"尖锐程度"（温度高 → 分布更均匀 → 输出更多样）

生成的 token ID 拼接到输入序列末尾，然后继续下一次前向传播：

```
Iteration 1:  [44, 402] → forward → logits → sample → 2080 ("<think>")
Iteration 2:  [44, 402, 2080] → forward → logits → sample → 486 ("\n")
Iteration 3:  [44, 402, 2080, 486] → forward → logits → sample → 32313 ("Okay")
...
```

---

## 6. KV Cache：避免重复计算

如果每次生成都重新计算整个序列的 attention，复杂度是 O(n²)——序列越长越慢。

**KV Cache** 缓存了之前所有 token 的 K 和 V 值。生成新 token 时：

```
预填充 (prefill):
  K = [k₀, k₁], V = [v₀, v₁]  ← 缓存

生成 token 2:
  输入: [token_{t-1}]  (仅最后一个 token)
  Q = [q₂]                     ← 只算新的 query
  K = [k₀, k₁, k₂]             ← k₂ 是新的，k₀, k₁ 从缓存取
  V = [v₀, v₁, v₂]
  cache.update(k₂, v₂)         ← 更新缓存

生成 token 3:
  输入: [token₂]  (仅最后一个 token)
  Q = [q₃]                     ← 只算新的
  K = [k₀, k₁, k₂, k₃]        ← 从缓存取
  ...
```

每次生成只计算一个 token 的 QKV，代价降至 O(n)。

Qwen3 的 KV Cache 实现：

```go
type KVCache struct {
    Keys     [][][]float32  // [n_layers][position][kv_dim]
    Values   [][][]float32
    Size     []int          // 每层已缓存的 token 数
}

func (c *KVCache) Update(layer int, key, value []float32, nKvHeads, headDim int) {
    k := make([]float32, len(key))
    copy(k, key)
    c.Keys[layer] = append(c.Keys[layer], k)
    c.Size[layer]++
}
```

缓存的大小与序列长度 × KV heads × head dim × 层数 成正比（Keys + Values 各一份）：

```
Qwen3-0.6B: 2 × 40960 × 8 × 128 × 28 = 2.35B float32 ≈ 9.4 GB (单精度)
             (fp16 下约 4.7 GB)
```

这就是为什么 KV cache 是推理引擎的主要内存瓶颈，也是 GQA（减少 KV heads）如此重要的原因。

---

## 7. 训练与推理的区别

全文默认在讲推理（inference），但了解训练有助于理解为什么模型是"这样"的。

| 方面 | 训练 | 推理 |
|------|------|------|
| **输入输出** | 对整个序列计算每个位置的 next-token 预测 | 自回归逐 token 生成 |
| **Teacher Forcing** | 无论模型预测什么，下一时间步的输入都是真实 token | 模型自己的输出作为输入 |
| **并行度** | 一次性处理整个序列（prefill 模式） | 逐 token，依赖前一个输出 |
| **Loss** | Cross-entropy loss，反向传播更新参数 | 无 loss，不更新参数 |
| **KV Cache** | 不必使用（训练时序列已知） | 必须使用（否则 O(n²) 太重） |
| **随机性** | Dropout 等正则化 | 一般无 dropout，temperature 控制采样 |

核心差异：训练时模型一次"看"完整序列，每一步都预测下一个 token，然后通过反向传播修正权重。推理时模型权重冻结，只看自己生成的 token。

---

## 8. Prefill 与 Decode：两个性能阶段

这是推理引擎最重要的性能概念，直接决定优化策略。

### Prefill（预填充）

处理输入 prompt（如 `"hi"` 的 `[44, 402]`）的阶段：

- 一次性计算所有 token 的 attention
- **compute-bound**：大矩阵乘（`[2, 1024] × [1024, 2048]` 等），GPU 利用率高
- 可以使用 Tensor Core / SIMD 高效执行
- 耗时与输入长度近似线性增长

### Decode（生成）

逐 token 生成的阶段：

- 每次只计算一个 token
- **memory-bound**：矩阵乘很小，瓶颈在 KV Cache 读写带宽
- GPU 利用率低（小矩阵乘喂不饱计算单元）
- 耗时与生成长度线性增长，但每步的延迟比 prefill 高（按 token 算）

### 对系统工程师的意义

```
Prefill:  优化计算吞吐 → 更大的 batch、Tensor Core、FP8/INT8
Decode:   优化内存带宽 → KV Cache 量化、PagedAttention、GQA

一个推理请求的延迟 = prefill_time + decode_time × num_tokens
```

Prefill 和 Decode 的硬件瓶颈完全不同，因此现代推理引擎（vLLM、TensorRT-LLM 等）对这两个阶段分别采用不同的优化策略，而非统一处理。

---

## 9. 现代 LLM 相比原始 Transformer 的演进

| 组件 | 原始 Transformer (Vaswani 2017) | Qwen3 |
|------|------|-------|
| **归一化** | LayerNorm（均值和方差） | RMSNorm（仅均方根） |
| **归一化位置** | Post-norm（residual 后归一化） | Pre-norm（在 attention/FFN 前归一化） |
| **激活函数** | ReLU | SwiGLU (silu × gate) |
| **位置编码** | 正弦波（加在 embedding 上） | RoPE（旋转 Q/K 向量） |
| **RoPE 变体** | — | NEOX（half-split 配对） |
| **Attention 类型** | MHA (Multi-Head) | GQA (Grouped Query Attention) |
| **QK 归一化** | 无 | Per-head RMSNorm on Q/K |
| **Weight Tying** | 无 | Output = Token Embedding |
| **FFN 结构** | 2 层线性 + ReLU | 3 层线性 (gate/up/down) + SiLU |

这些改进的累积效果：

- **RMSNorm + Pre-norm**：训练更稳定，允许更深网络
- **RoPE**：更好的位置泛化，支持长上下文
- **GQA**：推理效率翻倍（KV Cache 减半），质量几乎无损
- **SwiGLU**：更强的特征表达能力
- **QK RMSNorm**：稳定 attention 分数，防止 outlier

---

## 10. 实际推理中的完整数据流（Qwen3-0.6B）

```
输入: "hi" → tokens [44, 402]

预填充阶段 (seq_len=2):
  1. Token Embedding: [44,402] × [151936,1024] → [2,1024]
  2. × 28 layers:
     a. RMSNorm: [2,1024]
     b. QKV Proj: Q [2,2048], K [2,1024], V [2,1024]
     c. QK RMSNorm (per-head) + RoPE (NEOX)
     d. KV Cache Update: store K[2,1024], V[2,1024]
     e. Attention: [2,2048] → Output Proj → [2,1024]
     f. Residual Add: [2,1024] + [2,1024] = [2,1024]
     g. RMSNorm: [2,1024]
     h. SwiGLU FFN: [2,1024] → [2,3072] → [2,1024]
     i. Residual Add: [2,1024] + [2,1024] = [2,1024]
  3. Output Norm: [1024] (取最后一个位置)
  4. Unembedding: [1024] × [151936,1024] → [151936]
  5. Argmax → token 2080 ("<think>")

生成阶段 (seq_len=1, 循环):
  Token 3: input=[2080], cache_size=2 → forward → token 486 ("\n")
  Token 4: input=[486],  cache_size=3 → forward → token 32313 ("Okay")
  Token 5: input=[32313], cache_size=4 → forward → token 11 (",")
  Token 6: input=[11],   cache_size=5 → forward → token 279 (" the")

输出: "<think>\nOkay, the"
```

---

## 参数量计算

```
Token Embedding:       151936 × 1024 = 155,582,464
Output (weight tied):  0 (共享)
28 × Attention:
  Q: 1024 × 2048 = 2,097,152     × 28 = 58,720,256
  K: 1024 × 1024 = 1,048,576     × 28 = 29,360,128
  V: 1024 × 1024 = 1,048,576     × 28 = 29,360,128
  Output: 2048 × 1024 = 2,097,152 × 28 = 58,720,256
  Q Norm: 128 × 16 = 2048        × 28 = 57,344
  K Norm: 128 × 8 = 1024         × 28 = 28,672
  Attn Norm: 1024                 × 28 = 28,672
28 × FFN:
  Gate: 1024 × 3072 = 3,145,728  × 28 = 88,080,384
  Up:   1024 × 3072 = 3,145,728  × 28 = 88,080,384
  Down: 3072 × 1024 = 3,145,728  × 28 = 88,080,384
  FFN Norm: 1024                  × 28 = 28,672
Output Norm: 1024

总计 ≈ 597M 参数
```

与 0.6B（6 亿）一致。

---

## 进一步阅读

- 本项目代码：[`model/qwen3.go`](model/qwen3.go) — 完整的推理实现
- Vaswani et al. "Attention Is All You Need" (2017) — 原始 Transformer
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE 论文
- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- Shazeer "GLU Variants Improve Transformer" (2020)
- Zhang & Sennrich "RMS Normalization" (2019)
- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — KV Cache 内存管理
