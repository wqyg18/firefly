import time
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ===============================
# 1. 结构定义
# ===============================
@dataclass
class RequestState:
    req_id: int
    prompt: str
    prompt_ids: torch.Tensor        # [1, S_i]
    attention_mask: torch.Tensor    # [1, S_i]
    # 核心：每个请求拥有自己独立的 DynamicCache 实例
    kv_cache: Optional[DynamicCache] = None 
    current_token: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False

# ===============================
# 2. KV Cache 工具函数 (集成你提供的逻辑)
# ===============================

def split_to_states(batch_cache: DynamicCache, attention_mask: torch.Tensor, states: List[RequestState]):
    """将 Batch 推理后的 Cache 拆解并分发给各个 RequestState"""
    batch_size = attention_mask.shape[0]
    
    for i in range(batch_size):
        new_cache = DynamicCache()
        # 强制转换为 int
        actual_length = int(attention_mask[i].sum().item()) 
        
        for layer_idx in range(len(batch_cache)):
            k, v = batch_cache[layer_idx]
            # 现在 actual_length 是 int，切片不会报错
            k_slice = k[i:i+1, :, :actual_length, :].clone()
            v_slice = v[i:i+1, :, :actual_length, :].clone()
            new_cache.update(k_slice, v_slice, layer_idx=layer_idx)
        
        states[i].kv_cache = new_cache

def merge_from_states(states: List[RequestState]):
    """从多个 RequestState 中提取 Cache 并合并为推理用的 Batch Cache"""
    cache_list = [s.kv_cache for s in states]
    batch_size = len(cache_list)
    num_layers = len(cache_list[0])
    lengths = [cache.get_seq_length() for cache in cache_list]
    max_len = max(lengths)
    
    merged_cache = DynamicCache()
    device = cache_list[0][0][0].device
    dtype = cache_list[0][0][0].dtype
    
    for layer_idx in range(num_layers):
        k_list, v_list = [], []
        for i in range(batch_size):
            k, v = cache_list[i][layer_idx]
            cur_len = k.shape[2]
            
            if cur_len < max_len:
                padding_shape = (1, k.shape[1], max_len - cur_len, k.shape[3])
                k_padded = torch.cat([k, torch.zeros(padding_shape, device=device, dtype=dtype)], dim=2)
                v_padded = torch.cat([v, torch.zeros(padding_shape, device=device, dtype=dtype)], dim=2)
                k_list.append(k_padded)
                v_list.append(v_padded)
            else:
                k_list.append(k)
                v_list.append(v)
        
        merged_cache.update(torch.cat(k_list, dim=0), torch.cat(v_list, dim=0), layer_idx=layer_idx)
        
    # 生成合并后的 mask
    new_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, l in enumerate(lengths):
        new_mask[i, :l] = 1
        
    return merged_cache, new_mask

# ===============================
# 3. 推理逻辑
# ===============================

@torch.inference_mode()
def prefill_batch(model, states: List[RequestState], pad_token_id: int):
    # 1. Padding 准备
    max_len = max(s.prompt_ids.shape[1] for s in states)
    padded_ids, padded_masks = [], []

    for s in states:
        seq_len = s.prompt_ids.shape[1]
        pad_len = max_len - seq_len
        ids = torch.cat([s.prompt_ids, torch.full((1, pad_len), pad_token_id, device=s.prompt_ids.device)], dim=1) if pad_len > 0 else s.prompt_ids
        mask = torch.cat([s.attention_mask, torch.zeros((1, pad_len), device=s.attention_mask.device)], dim=1) if pad_len > 0 else s.attention_mask
        padded_ids.append(ids)
        padded_masks.append(mask)

    batched_ids = torch.cat(padded_ids, dim=0)
    batched_mask = torch.cat(padded_masks, dim=0)

    # 2. 推理
    outputs = model(input_ids=batched_ids, attention_mask=batched_mask, use_cache=True)
    
    # 3. 拆分 KV Cache 到各自的 State
    split_to_states(outputs.past_key_values, batched_mask, states)

    # 4. 更新 Token
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for i, s in enumerate(states):
        tok = next_tokens[i].view(1, 1)
        s.current_token = tok
        s.generated_ids.append(int(tok.item()))

@torch.inference_mode()
def decode_step_batch(model, states: List[RequestState]):
    # 1. 合并各个请求的 Cache
    merged_cache, old_mask = merge_from_states(states)
    
    # 2. 准备当前 Input 和更新后的 Mask
    batched_tokens = torch.cat([s.current_token for s in states], dim=0)
    # 当前 step 的 mask 都是 1，拼在 old_mask 后面
    current_mask = torch.ones((len(states), 1), device=old_mask.device)
    full_mask = torch.cat([old_mask, current_mask], dim=-1)

    # 3. 推理
    outputs = model(
        input_ids=batched_tokens,
        past_key_values=merged_cache,
        attention_mask=full_mask,
        use_cache=True
    )

    # 4. 再次拆分回各自的 State (因为 DynamicCache 在推理后增加了新 token 的 KV)
    split_to_states(outputs.past_key_values, full_mask, states)

    # 5. 更新 Token
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    for i, s in enumerate(states):
        tok = next_tokens[i].view(1, 1)
        s.current_token = tok
        s.generated_ids.append(int(tok.item()))

# ===============================
# 4. 主函数
# ===============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)

    prompts = [
        "Tell me a short joke about GPUs.",
        "What is KV cache?"
    ]

    states = [
        RequestState(i, p, 
                     tokenizer(p, return_tensors="pt").input_ids.to(device),
                     tokenizer(p, return_tensors="pt").attention_mask.to(device)) 
        for i, p in enumerate(prompts)
    ]

    print("--- Prefill ---")
    prefill_batch(model, states, tokenizer.pad_token_id)
    
    print("--- Decoding ---")
    for _ in range(10):
        decode_step_batch(model, states)
        res = [tokenizer.decode([s.generated_ids[-1]]) for s in states]
        print(f"Step: {res}")

if __name__ == "__main__":
    main()