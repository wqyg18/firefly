import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from icecream import ic

device = "cuda"
model_id = "Qwen/Qwen2.5-0.5B"

# 1. 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=device)

prompt = "hello, can you tell me what is the meaning of life?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) # torch.Size([1, 13])
# 这里的 dim0 就是batch_size, 如果直接使用那么就是static batching技术了
ic(input_ids.shape)

print(f"--- 开始推理 ---")
print(f"输入文本: {prompt}")

# ---------------------------------------------------------
# 阶段 1: Prefill (预填充)
# ---------------------------------------------------------
t0 = time.time()
with torch.no_grad():
    # 我们一次性把所有 input_ids 丢进去
    outputs = model(input_ids, use_cache=True)
    
    # 获取第一个生成的 Token 和初始 KV Cache
    # ic| outputs.logits.shape: torch.Size([1, 13, 151936])
    logits = outputs.logits[:, -1, :]
    ic(outputs.logits.shape)
    ic(logits.shape)
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(logits, dim=-1, keepdim=True)

prefill_time = time.time() - t0
print(f"\n[Prefill 阶段完成] 耗时: {prefill_time:.4f}s")
print(f"生成首词: {tokenizer.decode(next_token[0])}")

# ---------------------------------------------------------
# 阶段 2: Decode (逐字解码)
# ---------------------------------------------------------
print(f"[开始 Decode 阶段] ...")
t1 = time.time()

generated_tokens = [next_token.item()]
max_new_tokens = 50
current_token = next_token # torch.Size([1, 1])
ic(current_token.shape)

decoded_text = tokenizer.decode(next_token[0])
print(decoded_text, end="", flush=True)

for i in range(max_new_tokens - 1):
    with torch.no_grad():
        # 注意！这里只传入最新的一个 token，以及之前的 KV Cache
        outputs = model(
            input_ids=current_token, 
            past_key_values=past_key_values, 
            use_cache=True
        )
        
        # 更新 KV Cache 和预测下一个词
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        current_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 实时打印（处理中文残缺字节）
        new_token_id = current_token.item()
        generated_tokens.append(new_token_id)
        
        # 增量解码防止乱码
        full_text = tokenizer.decode(generated_tokens)
        print(full_text[len(decoded_text):], end="", flush=True)
        decoded_text = full_text

        # 遇到结束符停止
        if new_token_id == tokenizer.eos_token_id:
            break

decode_time = time.time() - t1
print(f"\n\n[Decode 阶段完成] 耗时: {decode_time:.4f}s")
print(f"平均每个 Token 耗时: {decode_time/len(generated_tokens):.4f}s")