import torch
import time
from dataclasses import dataclass, field
from typing import Optional, Any, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from icecream import ic

# ===============================
# ⭐ Stage 0 核心：请求状态封装
# ===============================
@dataclass
class RequestState:
    input_ids: torch.Tensor                  # prompt ids [1, S]
    past_key_values: Optional[Any] = None    # HF KV cache
    current_token: Optional[torch.Tensor] = None  # [1, 1]
    generated_ids: List[int] = field(default_factory=list)
    done: bool = False


device = "cuda"
model_id = "Qwen/Qwen2.5-0.5B"

# 1. 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype="auto", device_map=device
)
model.eval()  # ⭐ 推理模式

prompt = "hello, can you tell me what is the meaning of life?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# ⭐ RequestState：Stage 1 / 2 / 3 的锚点
state = RequestState(input_ids=input_ids)

ic(state.input_ids.shape)

print(f"--- 开始推理 ---")
print(f"输入文本: {prompt}")

# ---------------------------------------------------------
# ⭐ 阶段 1: Prefill（一次性 prompt 前向）
# ---------------------------------------------------------
@torch.inference_mode()
def prefill(model, state: RequestState):
    outputs = model(state.input_ids, use_cache=True)

    # ⭐ 明确 Prefill 的“产物”
    state.past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # [1, vocab]
    state.current_token = torch.argmax(logits, dim=-1, keepdim=True)
    state.generated_ids.append(int(state.current_token.item()))


t0 = time.time()
prefill(model, state)
prefill_time = time.time() - t0

first_token_text = tokenizer.decode(
    [state.generated_ids[0]], skip_special_tokens=False
)

print(f"\n[Prefill 阶段完成] 耗时: {prefill_time:.4f}s")
print(f"生成首词: {first_token_text}")

# ---------------------------------------------------------
# ⭐ 阶段 2: Decode（逐 token）
# ---------------------------------------------------------
print(f"[开始 Decode 阶段] ...")
t1 = time.time()

max_new_tokens = 50
eos_token_id = tokenizer.eos_token_id

# ⭐ 先输出 prefill 生成的第一个 token
print(first_token_text, end="", flush=True)

@torch.inference_mode()
def decode_step(model, state: RequestState):
    outputs = model(
        input_ids=state.current_token,
        past_key_values=state.past_key_values,
        use_cache=True,
    )

    state.past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    state.current_token = torch.argmax(logits, dim=-1, keepdim=True)

    new_id = int(state.current_token.item())
    state.generated_ids.append(new_id)

    if eos_token_id is not None and new_id == eos_token_id:
        state.done = True


# ⭐ Decode loop：一次一步，完全可调度
for _ in range(max_new_tokens - 1):
    if state.done:
        break

    decode_step(model, state)

    # ⭐ 关键修复：只 decode 新 token（不 O(n²)）
    new_text = tokenizer.decode(
        [state.generated_ids[-1]], skip_special_tokens=False
    )
    print(new_text, end="", flush=True)

decode_time = time.time() - t1

print(f"\n\n[Decode 阶段完成] 耗时: {decode_time:.4f}s")
print(f"生成 token 数: {len(state.generated_ids)}")
print(f"平均每个 Token 耗时: {decode_time/len(state.generated_ids):.4f}s")
