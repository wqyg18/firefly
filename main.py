import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RequestState:
    req_id: int
    prompt: str
    prompt_ids: torch.Tensor  # [1, S]
    past_key_values: Optional[Any] = None
    current_token: Optional[torch.Tensor] = (
        None  # [1, 1] token used as input for next decode step
    )
    generated_ids: List[int] = field(default_factory=list)
    done: bool = False
    printed_text: str = ""  # for debugging / optional


@torch.inference_mode()
def prefill(model, tokenizer, state: RequestState) -> str:
    """Run one prompt forward, create KV cache and the first generated token."""
    outputs = model(state.prompt_ids, use_cache=True)
    state.past_key_values = outputs.past_key_values

    logits = outputs.logits[:, -1, :]  # [1, vocab]
    state.current_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

    tok_id = int(state.current_token.item())
    state.generated_ids.append(tok_id)

    piece = tokenizer.decode([tok_id], skip_special_tokens=False)
    return piece


@torch.inference_mode()
def decode_step(model, state: RequestState, eos_token_id: Optional[int]) -> int:
    """Decode exactly one token for this request."""
    outputs = model(
        input_ids=state.current_token,
        past_key_values=state.past_key_values,
        use_cache=True,
    )
    state.past_key_values = outputs.past_key_values

    logits = outputs.logits[:, -1, :]
    state.current_token = torch.argmax(logits, dim=-1, keepdim=True)

    tok_id = int(state.current_token.item())
    state.generated_ids.append(tok_id)

    if eos_token_id is not None and tok_id == eos_token_id:
        state.done = True

    return tok_id


def make_state(req_id: int, prompt: str, tokenizer, device: str) -> RequestState:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return RequestState(req_id=req_id, prompt=prompt, prompt_ids=prompt_ids)


def main():
    device = "cuda"
    model_id = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map=device
    )
    model.eval()

    eos_token_id = tokenizer.eos_token_id

    # -----------------------------
    # Stage 1: queues + scheduler
    # -----------------------------
    waiting: Deque[RequestState] = deque()
    active: List[RequestState] = []
    finished: List[RequestState] = []

    # 你可以把这里换成真实服务的“到达请求”
    prompts = [
        "Tell me a short joke about GPUs.",
        "Write a 1-sentence summary of what a KV cache is.",
        "Give me three tips for debugging CUDA OOM in PyTorch.",
    ]
    for i, p in enumerate(prompts):
        waiting.append(make_state(i, p, tokenizer, device))

    # 关键：Stage 1 不做 batching，但要限制同时活跃请求数（模拟并发窗口）
    max_active = 2
    max_new_tokens = 60  # per request

    print("=== Stage 1: Multi-Request Decode (no batching) ===")

    # 记录每个请求已生成 token 数（不含 prompt）
    gen_budget = {}  # req_id -> remaining steps

    # 主调度循环：每一“轮”推进 active 里的每个请求一步
    round_idx = 0
    t0 = time.time()

    while waiting or active:
        # 1) 补充活跃请求（continuous admission，但仍是 no-batching）
        while waiting and len(active) < max_active:
            req = waiting.popleft()  # 左边为头部
            active.append(req)
            gen_budget[req.req_id] = max_new_tokens

            first_piece = prefill(model, tokenizer, req)
            gen_budget[req.req_id] -= 1

            print(f"\n[ADMIT req={req.req_id}] prompt: {req.prompt}")
            print(f"[req={req.req_id}] {first_piece}")

            # prefill 后就可能 eos（极少见，但处理一下）
            if req.done or gen_budget[req.req_id] <= 0:
                req.done = True
                print(f"\n[req={req.req_id}] <DONE after prefill>")
        if not active:
            continue

        # 2) 轮询推进：对 active 中每个请求 decode 一步
        round_idx += 1
        # 注意：遍历时可能移除，所以用索引方式更安全
        i = 0
        while i < len(active):
            req = active[i]

            # 如果已经没预算，直接结束
            if gen_budget[req.req_id] <= 0:
                req.done = True

            if req.done:
                print(
                    f"\n[FINISH req={req.req_id}] generated={len(req.generated_ids)} tokens"
                )
                finished.append(req)
                active.pop(i)
                continue  # 不 i+=1，因为 pop 后当前 i 已是下一个元素

            tok_id = decode_step(model, req, eos_token_id)
            gen_budget[req.req_id] -= 1

            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            print(f"[req={req.req_id}] {piece}")

            # 完成则移出 active
            if req.done or gen_budget[req.req_id] <= 0:
                req.done = True
                print(
                    f"\n[FINISH req={req.req_id}] generated={len(req.generated_ids)} tokens"
                )
                finished.append(req)
                active.pop(i)
                continue

            i += 1

    total = time.time() - t0
    print(f"\n\n=== All done. finished={len(finished)} total_time={total:.3f}s ===")


if __name__ == "__main__":
    main()