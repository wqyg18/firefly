from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class RequestState:
    """
    Per-request state used to simulate independent decoding streams.
    Each request owns its own KV cache.
    """

    req_id: int
    prompt: str
    prompt_ids: torch.Tensor  # [1, seq_len]
    attention_mask: torch.Tensor  # [1, seq_len]
    kv_cache: Optional[DynamicCache] = None
    current_token: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False


def split_to_states(
    batch_cache: DynamicCache, attention_mask: torch.Tensor, states: List[RequestState]
):
    """
    Split a batched DynamicCache into per-request caches.

    The attention_mask is used to determine the valid sequence length
    for each request, since the batch may be padded.
    """
    batch_size = attention_mask.shape[0]

    for i in range(batch_size):
        new_cache = DynamicCache()
        seq_len = int(attention_mask[i].sum().item())

        for layer_idx in range(len(batch_cache)):
            k, v = batch_cache[layer_idx]
            k_slice = k[i : i + 1, :, :seq_len, :].clone()
            v_slice = v[i : i + 1, :, :seq_len, :].clone()
            new_cache.update(k_slice, v_slice, layer_idx=layer_idx)

        states[i].kv_cache = new_cache


def merge_from_states(states: List[RequestState]):
    """
    Merge per-request KV caches into a single batched cache.

    Since different requests may have different sequence lengths,
    shorter caches are right-padded to the maximum length.
    """
    caches = [s.kv_cache for s in states]
    batch_size = len(caches)
    num_layers = len(caches[0])
    lengths = [c.get_seq_length() for c in caches]
    max_len = max(lengths)

    merged_cache = DynamicCache()
    device = caches[0][0][0].device
    dtype = caches[0][0][0].dtype

    for layer_idx in range(num_layers):
        k_list, v_list = [], []

        for i in range(batch_size):
            k, v = caches[i][layer_idx]
            cur_len = k.shape[2]

            if cur_len < max_len:
                pad_shape = (1, k.shape[1], max_len - cur_len, k.shape[3])
                k = torch.cat(
                    [k, torch.zeros(pad_shape, device=device, dtype=dtype)], dim=2
                )
                v = torch.cat(
                    [v, torch.zeros(pad_shape, device=device, dtype=dtype)], dim=2
                )

            k_list.append(k)
            v_list.append(v)

        merged_cache.update(
            torch.cat(k_list, dim=0), torch.cat(v_list, dim=0), layer_idx=layer_idx
        )

    # Build the merged attention mask
    attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1

    return merged_cache, attention_mask


@torch.inference_mode()
def prefill_batch(model, states: List[RequestState], pad_token_id: int):
    """
    Run the prefill stage for multiple requests in a single batch.

    This computes the initial KV cache for each prompt and then
    splits the batched cache back into per-request caches.
    """
    max_len = max(s.prompt_ids.shape[1] for s in states)
    ids_list, masks_list = [], []

    for s in states:
        seq_len = s.prompt_ids.shape[1]
        pad_len = max_len - seq_len

        if pad_len > 0:
            ids = torch.cat(
                [
                    s.prompt_ids,
                    torch.full((1, pad_len), pad_token_id, device=s.prompt_ids.device),
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    s.attention_mask,
                    torch.zeros((1, pad_len), device=s.attention_mask.device),
                ],
                dim=1,
            )
        else:
            ids = s.prompt_ids
            mask = s.attention_mask

        ids_list.append(ids)
        masks_list.append(mask)

    input_ids = torch.cat(ids_list, dim=0)
    attention_mask = torch.cat(masks_list, dim=0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    split_to_states(outputs.past_key_values, attention_mask, states)

    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for i, s in enumerate(states):
        token = next_tokens[i].view(1, 1)
        s.current_token = token
        s.generated_ids.append(int(token.item()))


@torch.inference_mode()
def decode_step_batch(model, states: List[RequestState]):
    """
    Perform one decoding step for all active requests.

    Each request contributes exactly one token, but decoding
    is executed as a single batched forward pass.
    """
    merged_cache, prev_mask = merge_from_states(states)

    input_ids = torch.cat([s.current_token for s in states], dim=0)
    step_mask = torch.ones((len(states), 1), device=prev_mask.device)
    attention_mask = torch.cat([prev_mask, step_mask], dim=-1)

    outputs = model(
        input_ids=input_ids,
        past_key_values=merged_cache,
        attention_mask=attention_mask,
        use_cache=True,
    )

    split_to_states(outputs.past_key_values, attention_mask, states)

    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for i, s in enumerate(states):
        token = next_tokens[i].view(1, 1)
        s.current_token = token
        s.generated_ids.append(int(token.item()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)

    prompts = ["Tell me a short joke about GPUs.", "What is KV cache?"]

    states = [
        RequestState(
            req_id=i,
            prompt=p,
            prompt_ids=tokenizer(p, return_tensors="pt").input_ids.to(device),
            attention_mask=tokenizer(p, return_tensors="pt").attention_mask.to(device),
        )
        for i, p in enumerate(prompts)
    ]

    print("--- Prefill ---")
    prefill_batch(model, states, tokenizer.pad_token_id)

    print("--- Decoding ---")
    for _ in range(10):
        decode_step_batch(model, states)
        tokens = [tokenizer.decode([s.generated_ids[-1]]) for s in states]
        print(tokens)


if __name__ == "__main__":
    main()
