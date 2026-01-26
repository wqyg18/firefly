# 🔥 Firefly

A **minimal vLLM-inspired LLM inference engine**, built on top of **Hugging Face Transformers**, focusing on understanding and experimenting with **KV cache management, batching strategies, and scheduling**.

This project is intentionally small and educational:
- to deeply understand how vLLM-style inference works
- to experiment with different batching and KV cache designs
- to serve as a clean, readable codebase for learning, interviews, and open-source discussion

---

## ✨ Features

- 🚀 Minimal end-to-end LLM inference engine
- 🧠 Explicit **KV cache lifecycle management**
- 🔄 Clear separation of **prefill** and **decode**
- 📦 Support for **static batching** and **continuous batching**
- 🧪 Built on **Hugging Face Transformers** using `DynamicCache`
- 🧩 Easy to modify and extend

---

## 🧠 Design Philosophy

Firefly is **not** a production-ready inference engine.

Instead, it aims to:
- make every design decision explicit
- keep abstractions thin
- prioritize readability over performance
- closely mirror the *core ideas* behind vLLM, without re-implementing everything

If you want something fast and battle-tested, use **vLLM**.
If you want to understand *why* vLLM works, this repo is for you.

---

## 🏗 Architecture Overview

At a high level, Firefly models inference as:

```

Request → Prefill → KV Cache → Decode (step-by-step)

````

Each request owns:
- its prompt tokens
- its attention mask
- its **independent KV cache**
- its generation state

Batching is achieved by:
- merging KV caches for a decode step
- running a single forward pass
- splitting the updated KV cache back to each request

This makes the control flow explicit and easy to reason about.

---

## 📌 Milestones

- [x] **Minimal inference engine**
- [x] **Prefill / Decode separation**
- [x] **Scheduler (request lifecycle management)**
- [x] **Static batching (no lifecycle)**
- [x] **Continuous batching**

---

## 🚧 Current Limitations / TODO

### Performance
- `clone()` during KV cache `split / merge` is **too slow**
- Current implementation is correctness-first, not performance-first

### Planned Improvements
- Replace split/merge with **reference-based KV cache**
- Or implement a **minimal PageAttention-style KV manager**
- Reduce unnecessary memory copies

These changes are the next major direction of the project.

---

## 🔍 Example

A minimal example demonstrating:
- batched prefill
- per-request KV cache
- step-by-step batched decoding

```python
prefill_batch(model, states, pad_token_id)

for _ in range(10):
    decode_step_batch(model, states)
````

See `main()` in the code for a complete runnable example.

---

## 🧰 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* `DynamicCache` for KV cache management

---

## 🎯 Motivation

This project started as:

* a way to **learn vLLM internals properly**
* a hands-on exploration of **LLM inference systems**
* a compact repo suitable for **interviews and discussion**

If it helps others learn or sparks ideas, that’s a big win.

---

## 🤝 Contributions & Discussion

Issues, ideas, and discussions are welcome.

This is a learning-oriented project — clarity and correctness matter more than benchmarks.

---

## 📄 License

MIT