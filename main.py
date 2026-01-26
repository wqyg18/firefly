from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class RequestState:
    req_id: int
    prompt: str
    prompt_ids: torch.Tensor
    attention_mask: torch.Tensor
    kv_cache: Optional[DynamicCache] = None
    current_token: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    max_tokens: int = 128


class SimpleScheduler:
    def __init__(self, max_concurrency: int, eos_token_id: int):
        self.max_concurrency = max_concurrency
        self.eos_token_id = eos_token_id
        self.waiting: Deque[RequestState] = deque()
        self.running: List[RequestState] = []
        self.finished: List[RequestState] = []

    def add_request(self, state: RequestState):
        self.waiting.append(state)

    def _admit(self):
        admitted = []
        while len(self.running) < self.max_concurrency and self.waiting:
            s = self.waiting.popleft()
            self.running.append(s)
            admitted.append(s)
        return admitted

    def schedule(self):
        return list(self.running)

    def update(self, states: List[RequestState]):
        still_running = []
        for s in self.running:
            if s.current_token is not None and (
                s.current_token.item() == self.eos_token_id
                or len(s.generated_ids) >= s.max_tokens
            ):
                s.is_finished = True
                self.finished.append(s)
            else:
                still_running.append(s)
        self.running = still_running


@torch.inference_mode()
def prefill_individually(model, states: List[RequestState]):
    """Process prompt prefill for new requests one by one to avoid padding."""
    for s in states:
        outputs = model(
            input_ids=s.prompt_ids,
            attention_mask=s.attention_mask,
            use_cache=True,
        )
        s.kv_cache = outputs.past_key_values
        token = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(1, 1)
        s.current_token = token
        s.generated_ids.append(int(token.item()))


@torch.inference_mode()
def decode_step_batch(model, states: List[RequestState]):
    device = states[0].prompt_ids.device
    batch_size = len(states)

    input_ids = torch.cat([s.current_token for s in states], dim=0)
    lengths = [s.kv_cache.get_seq_length() for s in states]
    max_len = max(lengths)
    num_layers = len(states[0].kv_cache)

    # Merge individual KV caches into a single batched DynamicCache
    merged_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k_list, v_list = [], []
        for s in states:
            k, v = s.kv_cache[layer_idx]
            # Right-pad KV to match max sequence length in current batch
            diff = max_len - k.shape[2]
            if diff > 0:
                k = torch.nn.functional.pad(k, (0, 0, 0, diff))
                v = torch.nn.functional.pad(v, (0, 0, 0, diff))
            k_list.append(k)
            v_list.append(v)

        merged_cache.update(
            torch.cat(k_list, dim=0), torch.cat(v_list, dim=0), layer_idx=layer_idx
        )

    pos_ids = torch.tensor([[l] for l in lengths], device=device, dtype=torch.long)
    attention_mask = torch.zeros(
        (batch_size, max_len + 1), device=device, dtype=torch.long
    )
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1
    attention_mask[:, -1] = 1

    outputs = model(
        input_ids=input_ids,
        past_key_values=merged_cache,
        attention_mask=attention_mask,
        position_ids=pos_ids,
        use_cache=True,
    )

    # Extract only the newly generated KV token and write back to private caches
    new_batch_cache = outputs.past_key_values
    for layer_idx in range(num_layers):
        full_k, full_v = new_batch_cache[layer_idx]
        for i, s in enumerate(states):
            k_new = full_k[i : i + 1, :, max_len : max_len + 1, :]
            v_new = full_v[i : i + 1, :, max_len : max_len + 1, :]
            s.kv_cache.update(k_new, v_new, layer_idx)

    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for i, s in enumerate(states):
        token = next_tokens[i].view(1, 1)
        s.current_token = token
        s.generated_ids.append(int(token.item()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B"
    model_id = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)

    scheduler = SimpleScheduler(max_concurrency=2, eos_token_id=tokenizer.eos_token_id)

    prompts = [
        "Tell me a short joke about GPUs.",
        "What is KV cache?",
        "Explain attention like I'm five.",
        "Why is batching important for LLMs?",
    ]

    for i, p in enumerate(prompts):
        enc = tokenizer(p, return_tensors="pt").to(device)
        scheduler.add_request(RequestState(i, p, enc.input_ids, enc.attention_mask))

    step = 0
    while True:
        newly_admitted = scheduler._admit()
        if newly_admitted:
            prefill_individually(model, newly_admitted)

        active = scheduler.schedule()
        if not active:
            break

        decode_step_batch(model, active)
        scheduler.update(active)

        status = [(s.req_id, tokenizer.decode([s.generated_ids[-1]])) for s in active]
        print(f"[step {step}] running ids: {status}")
        step += 1

    print("\n=== Final Results ===")
    for s in sorted(scheduler.finished, key=lambda x: x.req_id):
        print(
            f"[req {s.req_id}] {tokenizer.decode(s.generated_ids, skip_special_tokens=True)}"
        )


if __name__ == "__main__":
    main()
