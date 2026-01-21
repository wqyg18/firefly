import time
from dataclasses import dataclass, field
from typing import Optional, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===============================
# Request State
# ===============================
@dataclass
class RequestState:
    req_id: int
    prompt: str
    prompt_ids: torch.Tensor        # [1, S_i]
    attention_mask: torch.Tensor    # [1, S_i]
    past_key_values: Optional[Any] = None
    current_token: Optional[torch.Tensor] = None  # [1, 1]
    generated_ids: List[int] = field(default_factory=list)


# ===============================
# Batched Prefill with Padding
# ===============================
@torch.inference_mode()
def prefill_batch_with_padding(model, states: List[RequestState], pad_token_id: int):
    """
    Pad prompt_ids to the same length, then run one batched forward.
    """
    # 1. 找最大 prompt 长度
    max_len = max(s.prompt_ids.shape[1] for s in states)

    padded_ids = []
    padded_masks = []

    for s in states:
        seq_len = s.prompt_ids.shape[1]
        pad_len = max_len - seq_len

        # pad input_ids
        if pad_len > 0:
            pad_ids = torch.full(
                (1, pad_len),
                pad_token_id,
                device=s.prompt_ids.device,
                dtype=s.prompt_ids.dtype,
            )
            ids = torch.cat([s.prompt_ids, pad_ids], dim=1)
        else:
            ids = s.prompt_ids

        # pad attention_mask (0 = pad, 1 = real token)
        if pad_len > 0:
            pad_mask = torch.zeros((1, pad_len), device=s.attention_mask.device)
            mask = torch.cat([s.attention_mask, pad_mask], dim=1)
        else:
            mask = s.attention_mask

        padded_ids.append(ids)
        padded_masks.append(mask)

    # [B, max_len]
    batched_ids = torch.cat(padded_ids, dim=0)
    batched_mask = torch.cat(padded_masks, dim=0)

    outputs = model(
        input_ids=batched_ids,
        attention_mask=batched_mask,
        use_cache=True,
    )

    batched_past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]        # [B, vocab]
    next_tokens = torch.argmax(logits, dim=-1)  # [B]

    for i, s in enumerate(states):
        s.past_key_values = batched_past_kv
        tok = next_tokens[i].view(1, 1)
        s.current_token = tok
        s.generated_ids.append(int(tok.item()))


# ===============================
# Batched Decode Step (no padding needed here)
# ===============================
@torch.inference_mode()
def decode_step_batch(model, states: List[RequestState]):
    """
    Decode one token for each request using a single batched forward.
    Assumes all requests are aligned in decode steps.
    """
    # [B, 1]
    batched_tokens = torch.cat([s.current_token for s in states], dim=0)

    batched_past_kv = states[0].past_key_values  # static batch assumption

    outputs = model(
        input_ids=batched_tokens,
        past_key_values=batched_past_kv,
        use_cache=True,
    )

    batched_past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(logits, dim=-1)

    for i, s in enumerate(states):
        s.past_key_values = batched_past_kv
        tok = next_tokens[i].view(1, 1)
        s.current_token = tok
        s.generated_ids.append(int(tok.item()))


# ===============================
# Main
# ===============================
def main():
    device = "cuda"
    model_id = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map=device
    )
    model.eval()

    prompts = [
        "Tell me a short joke about GPUs.",
        "Write a 1-sentence summary of what a KV cache is in transformers.",
    ]

    states: List[RequestState] = []
    for i, p in enumerate(prompts):
        enc = tokenizer(p, return_tensors="pt")
        states.append(
            RequestState(
                req_id=i,
                prompt=p,
                prompt_ids=enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
            )
        )

    print("=== Stage 2: Static Batching with Padding ===")

    # -------- Prefill --------
    t0 = time.time()
    prefill_batch_with_padding(model, states, tokenizer.pad_token_id)
    print(f"[Prefill done] time={time.time() - t0:.4f}s")

    for s in states:
        print(f"[req={s.req_id}] {tokenizer.decode([s.generated_ids[-1]])}")

    # -------- Decode --------
    max_new_tokens = 30
    t1 = time.time()

    for step in range(max_new_tokens - 1):
        decode_step_batch(model, states)
        for s in states:
            print(f"[req={s.req_id}] {tokenizer.decode([s.generated_ids[-1]])}")

    total = time.time() - t1
    print(f"\n[Decode done] tokens/request={max_new_tokens} time={total:.4f}s")
    print(f"Avg per-token time (batched): {total / max_new_tokens:.4f}s")


if __name__ == "__main__":
    main()
