# 🚀 PageAttention → Real LLM Inference Engine Roadmap

> Goal: 从 toy PageAttention demo 出发，逐步构建一个支持真实模型推理的最小推理引擎（类 vLLM）

---

# 🧩 Milestone 0 — Toy Demo 校正（数学 & 语义正确）

> 目标：确保当前实现“逻辑正确 + 数学正确”，为后续一切打基础

## ✅ Core Fixes

- [x] 修正 `decode_step` 顺序（先写 KV，再 attention）
- [ ] 确认 attention 计算包含当前 token（K<=t）
- [ ] 检查 `seq_len` 与 offset 逻辑是否严格一致
- [ ] 确认 block 切换边界（block_size）无 off-by-one bug

## ✅ Memory 管理

- [x] 实现 `release_request(req, block_manager)`
- [ ] 请求结束后归还所有 block
- [ ] 确认 free list 不泄漏

## ✅ 单元测试（非常关键）

- [ ] 测试 `append_token` 跨 block 写入
- [ ] 测试 `block_table` 扩展逻辑
- [ ] 测试 `seq_len` 增长
- [ ] 测试 `PageAttention.forward` 输出 shape
- [ ] 构造 baseline（连续 KV cache）对比 attention 输出
- [ ] 使用 `torch.allclose` 验证数值一致性

---

# 🧠 Milestone 1 — 接入真实 Attention 层（不再用随机 qkv）

> 目标：从“随机张量”升级为“真实 transformer attention”

## ✅ Attention 模块

- [x] 实现 `SelfAttention`（最小版本）
- [x] 添加 `q_proj / k_proj / v_proj / o_proj`
- [x] 支持 multi-head reshape（[H, D]）

## ✅ KV Cache 结构升级

- [x] 将 KVCache 支持 `num_kv_heads`（而非 num_heads）
- [x] 修改 KV tensor shape：
  - [x] K: `[num_blocks, block_size, num_kv_heads, head_dim]`
  - [x] V: 同上

## ✅ Attention 适配

- [x] 支持 `num_heads != num_kv_heads`（GQA / MQA）
- [x] 实现 head mapping / repeat KV heads

## ✅ 对齐验证

- [x] 使用同一输入 hidden state：
  - [x] 标准 attention
  - [x] PageAttention
- [x] 验证输出一致

---

# 📐 Milestone 2 — 加入 Position Encoding（RoPE）

> 目标：让 attention 具备真实 LLM 的位置信息能力

## ✅ Position 管理

- [x] 为每个 token 维护 `position_id`
- [x] prefill: 0 → T-1
- [x] decode: 使用当前 `seq_len`

## ✅ RoPE 实现

- [x] 实现 `apply_rope(q, pos)`
- [x] 实现 `apply_rope(k, pos)`
- [x] 确保 k 写入 cache 前已应用 RoPE

## ✅ 验证

- [x] 对比：
  - [x] 标准 RoPE attention
  - [x] paged attention + RoPE
- [x] 确认一致性

---

# 🏗️ Milestone 3 — 单请求真实模型推理（最关键）

> 目标：真正“生成文本”，哪怕模型很小

## ✅ 模型组件

- [ ] 实现 Tokenizer（或接 HuggingFace tokenizer）
- [ ] 实现 Embedding 层
- [ ] 实现 Decoder Block：
  - [ ] LayerNorm / RMSNorm
  - [ ] SelfAttention（接 PageAttention）
  - [ ] MLP
  - [ ] residual

## ✅ 多层 KV Cache

- [ ] 改造 KVCache → 支持多层：
  - [ ] `List[KVCache]`
  - [ ] 或 `[num_layers, ...]` tensor
- [ ] 每层独立 block_table（或共享 + layer index）

## ✅ Prefill 流程

- [ ] 输入 prompt token ids
- [ ] embedding → 多层 forward
- [ ] 每层写入 KV cache

## ✅ Decode 流程

- [ ] 每步输入一个 token
- [ ] 逐层 forward
- [ ] 使用 paged KV cache
- [ ] 输出 logits

## ✅ 采样

- [ ] 实现 greedy decoding
- [ ] 输出 token → text

## ✅ 验证

- [ ] 能从 prompt 生成连续文本
- [ ] 无 crash / 无 NaN

---

# 🔗 Milestone 4 — 对齐 HuggingFace 模型

> 目标：不再是 toy 模型，而是加载真实权重

## ✅ 模型加载

- [ ] 读取 config（hidden_size, heads, kv_heads 等）
- [ ] 加载权重（q_proj / k_proj / v_proj / mlp / norm / lm_head）

## ✅ 权重映射

- [ ] 映射 HF 权重 → 自己的模块
- [ ] 确认 tensor shape 对齐

## ✅ 对齐测试（非常关键）

- [ ] 同一输入：
  - [ ] HF forward logits
  - [ ] 自己实现 logits
- [ ] 对比：
  - [ ] prefill 最后一 token
  - [ ] decode 单步
- [ ] 允许小误差，但整体一致

---

# ⚙️ Milestone 5 — 推理引擎（多请求 + 调度）

> 目标：从“模型”进化为“系统”

## ✅ 多请求支持

- [ ] 支持 `List[Request]`
- [ ] 每个 request 独立 block_table
- [ ] 每个 request 独立 seq_len

## ✅ Scheduler（最简版）

- [ ] 实现循环 decode：
  - [ ] 遍历所有 active request
  - [ ] 每个 request decode 一步
- [ ] 区分：
  - [ ] prefill 阶段
  - [ ] decode 阶段

## ✅ 生命周期管理

- [ ] request 完成后释放 block
- [ ] 标记 finished request
- [ ] 从 active list 移除

## ✅ 基础并发

- [ ] 多 request 同时生成文本
- [ ] 输出 interleaved（流式）

---

# 🧠 Milestone 6 — 高级特性（可选进阶）

## ✅ Prefix Cache

- [ ] 支持共享 prompt 前缀
- [ ] 避免重复 prefill
- [ ] block_table 可复用

## ✅ Sampling

- [ ] top-k
- [ ] top-p
- [ ] temperature

## ✅ Request 控制

- [ ] max_new_tokens
- [ ] early stop（eos）

---

# ⚡ Milestone 7 — 性能优化（接近 vLLM）

> 目标：从“能跑”变成“跑得快”

## ❌ 当前问题（需要解决）

- [ ] Python for-loop gather block
- [ ] 每步 `torch.cat`
- [ ] 重复构造完整 KV

## ✅ 优化方向

- [ ] 减少 Python 循环（vectorize）
- [ ] 避免 `torch.cat`
- [ ] 使用 block index 直接 gather

## ✅ Kernel 加速

- [ ] 接入 FlashInfer（prefill + decode）
- [ ] 或 Triton kernel
- [ ] 或自写 CUDA kernel（可选）

## ✅ GPU 优化

- [ ] memory layout 优化（coalescing）
- [ ] block_size 调优
- [ ] head 排布优化

## ✅ Profiling

- [ ] 使用 torch profiler
- [ ] 找 bottleneck（attention / memory）

---

# 🎯 最终目标

- [ ] 支持真实 LLM（如 LLaMA / Qwen）
- [ ] 支持 paged KV cache
- [ ] 支持多请求并发
- [ ] 支持流式生成
- [ ] 性能接近推理引擎（vLLM / TGI baseline）

---

# 📌 当前进度标记

- [x] KVCache（block-based）
- [x] BlockManager
- [x] Request（block_table + seq_len）
- [x] append_token
- [x] naive PageAttention
- [x] prefill + decode demo

---
