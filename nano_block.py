import torch
from typing import List, Tuple
import math
import torch.nn as nn


class KVCache:
    """
    Global KV cache pool organized in fixed-size blocks.

    This acts like the "physical memory" in PageAttention.

    Attributes:
        num_blocks: Total number of blocks available.
        block_size: Number of tokens per block.
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.

        K: Tensor storing all key vectors.
           Shape: [num_blocks, block_size, num_kv_heads, head_dim]

        V: Tensor storing all value vectors.
           Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
    ):
        self.k = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim, device=device
        )
        self.v = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim, device=device
        )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def write(
        self,
        block_id: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write a single token's KV into a specific block and offset.

        Args:
            block_id: Which block to write to.
            offset: Position inside the block (0 <= offset < block_size).
            k: Key tensor of shape [num_kv_heads, head_dim]
            v: Value tensor of shape [num_kv_heads, head_dim]
        """
        self.k[block_id][offset] = k
        self.v[block_id][offset] = v

    def read_block(
        self,
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the entire block (all tokens in it).

        Returns:
            K_block: [block_size, num_kv_heads, head_dim]
            V_block: [block_size, num_kv_heads, head_dim]
        """
        return self.k[block_id], self.v[block_id]


def repeat_kv_heads(kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Expand KV heads for GQA/MQA: num_heads can be larger than num_kv_heads.

    Args:
        kv: [T, num_kv_heads, head_dim]
        num_heads: query heads
    Returns:
        [T, num_heads, head_dim]
    """
    num_kv_heads = kv.shape[1]
    if num_heads == num_kv_heads:
        return kv
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads({num_heads}) must be divisible by num_kv_heads({num_kv_heads})."
        )
    repeat_factor = num_heads // num_kv_heads
    return kv.repeat_interleave(repeat_factor, dim=1)


class BlockManager:
    """
    Manages allocation and deallocation of blocks.

    This acts like a simple memory allocator.
    """

    def __init__(self, num_blocks: int):
        """
        Initialize with all blocks free.
        """
        self.free_block = list(range(num_blocks))

    def alloc(self) -> int:
        """
        Allocate a free block.

        Returns:
            block_id: int

        Raises:
            RuntimeError if no free blocks available.
        """
        if not self.free_block:
            raise RuntimeError("no free blocks available")
        block_id = self.free_block.pop()
        return block_id

    def free(self, block_id: int) -> None:
        """
        Return a block back to the free pool.
        """
        self.free_block.append(block_id)


class Request:
    """
    Represents a single sequence (request).

    Each request maintains its own block table (page table).
    """

    def __init__(self):
        """
        Attributes:
            block_table: List of block IDs (logical -> physical mapping)
            seq_len: Current sequence length (number of tokens)
        """
        self.block_table: List[int] = []
        self.seq_len: int = 0
        self.position_ids: List[int] = []

    def last_block(self) -> int:
        """
        Get the current last block.

        Returns:
            block_id
        """
        return self.block_table[-1]

    def append_block(self, block_id: int) -> None:
        """
        Append a newly allocated block to this request.
        """
        self.block_table.append(block_id)


def append_token(
    req: Request,
    kv_cache: KVCache,
    block_manager: BlockManager,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """
    Append a new token's KV to the request.

    This handles:
        - Allocating new blocks when needed
        - Writing KV into correct block and offset

    Args:
        req: The request
        kv_cache: Global KV cache
        block_manager: Block allocator
        k: [num_heads, head_dim]
        v: [num_heads, head_dim]
    """
    # 1. Check if new block is needed
    # 2. Allocate block if necessary
    # 3. Compute (block_id, offset)
    # 4. Write KV
    # 5. Update seq_len

    # 其实只会出现小于或者等于, 并不会出现大于的情况
    if req.seq_len >= (kv_cache.block_size * len(req.block_table)):
        last_block = block_manager.alloc()
        req.append_block(last_block)

    last_block = req.last_block()
    offset = req.seq_len % kv_cache.block_size

    kv_cache.write(last_block, offset, k, v)

    req.position_ids.append(req.seq_len)
    req.seq_len += 1


def release_request(req: Request, block_manager: BlockManager):
    release_len = len(req.block_table)
    # 释放所有的block
    for block_id in req.block_table:
        block_manager.free(block_id)

    req.seq_len = 0
    req.block_table.clear()
    req.position_ids.clear()
    print(f"release current seq, return {release_len} blocks to kvcache")


class PageAttention:
    """
    Naive PageAttention implementation (no CUDA optimization).

    This version gathers all KV blocks and performs standard attention.
    """

    def __init__(self):
        pass

    def forward(
        self,
        q: torch.Tensor,
        req: Request,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Compute attention for a single query token.

        Args:
            q: Query tensor, shape [num_heads, head_dim]
            req: Request containing block_table
            kv_cache: Global KV cache

        Returns:
            output: [num_heads, head_dim]
        """
        # 1. Iterate over block_table
        # 2. Collect K/V blocks
        # 3. Concatenate
        # 4. Trim to seq_len
        # 5. Compute attention

        if req.seq_len == 0:
            raise RuntimeError("Cannot attend with empty sequence.")

        # 取出所有的kvcache, 然后做一次attention
        k_blocks = []
        v_blocks = []

        for block_id in req.block_table:
            k_block, v_block = kv_cache.read_block(block_id)
            # k_block: [block_size, num_heads, head_dim]

            k_blocks.append(k_block)
            v_blocks.append(v_block)

        k_all = torch.cat(k_blocks, dim=0)
        # k_all: [num_blocks_used * block_size, num_heads, head_dim]
        v_all = torch.cat(v_blocks, dim=0)

        # 由于最后一个block可能是不满的, 所以需要trim到真实的seq_len
        k_all = k_all[: req.seq_len]
        v_all = v_all[: req.seq_len]
        # 当使用 GQA/MQA 时, q 的头数和 kv 头数不同.
        # 这里将 kv 头扩展到 q 头维度, 便于沿用同一个注意力公式.
        k_all = repeat_kv_heads(k_all, q.shape[0])
        v_all = repeat_kv_heads(v_all, q.shape[0])

        # 基于kv计算q的attention分数
        # attention(q,k,v) = softmax((q \cdot k^T) / sqrt(head_dim)) \cdot v
        scores = torch.einsum("hd,thd->ht", q, k_all) / math.sqrt(kv_cache.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.einsum("ht,thd->hd", attn, v_all)

        return output


class SelfAttention(nn.Module):
    """
    Minimal single-token self-attention block.

    目标:
    1) 提供真实的 q_proj/k_proj/v_proj/o_proj
    2) 支持 num_heads != num_kv_heads (GQA/MQA)
    3) 复用 PageAttention + KVCache 做 decode
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def project(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project one token hidden state to q/k/v.
        Args:
            hidden: [hidden_size]
        Returns:
            q: [num_heads, head_dim]
            k: [num_kv_heads, head_dim]
            v: [num_kv_heads, head_dim]
        """
        q = self.q_proj(hidden).view(self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(self.num_kv_heads, self.head_dim)
        return q, k, v

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_heads, head_dim]
        Returns:
            [hidden_size]
        """
        return x.reshape(self.hidden_size)

    @staticmethod
    def _build_rope_cache(
        head_dim: int, position_id: int, base: float, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even when using RoPE.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        pos = torch.tensor(position_id, device=device, dtype=torch.float32)
        freqs = pos * inv_freq
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def apply_rope(
        self, x: torch.Tensor, position_id: int, base: float = 10000.0
    ) -> torch.Tensor:
        """
        Apply RoPE to per-token head vectors.
        Args:
            x: [num_heads_or_kv_heads, head_dim]
            position_id: scalar position index for this token
        Returns:
            tensor with same shape as x
        """
        cos, sin = self._build_rope_cache(
            head_dim=x.shape[-1],
            position_id=position_id,
            base=base,
            device=x.device,
            dtype=x.dtype,
        )
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward_standard(
        self, q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Reference attention on contiguous KV.
        用于对齐验证 PageAttention 的输出.
        """
        k_all = repeat_kv_heads(k_all, self.num_heads)
        v_all = repeat_kv_heads(v_all, self.num_heads)
        scores = torch.einsum("hd,thd->ht", q, k_all) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("ht,thd->hd", attn, v_all)


class DecoderEngine:
    """
    Minimal decoding engine using PageAttention.

    This simulates:
        - Prefill
        - Decode loop
    """

    def __init__(
        self,
        kv_cache: KVCache,
        block_manager: BlockManager,
        attention: PageAttention,
    ):
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.attention = attention

    def prefill(
        self,
        req: Request,
        k_list: List[torch.Tensor],
        v_list: List[torch.Tensor],
    ) -> None:
        """
        Load an initial prompt into KV cache.

        Args:
            k_list: list of K tensors
            v_list: list of V tensors
        """
        # 这里是计算好的每一个的K,V
        for k, v in zip(k_list, v_list):
            append_token(req, self.kv_cache, self.block_manager, k, v)

    def decode_step(
        self,
        req: Request,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one decode step.

        Steps:
            1. Compute attention output
            2. Append new KV

        Returns:
            output tensor
        """
        append_token(req, self.kv_cache, self.block_manager, new_k, new_v)
        output = self.attention.forward(q, req, self.kv_cache)
        return output


def run_toy_example():
    torch.manual_seed(0)

    num_blocks = 8
    block_size = 4
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    hidden_size = num_heads * head_dim

    kv_cache = KVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    block_manager = BlockManager(num_blocks)
    attention = PageAttention()
    engine = DecoderEngine(kv_cache, block_manager, attention)
    self_attn = SelfAttention(
        hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
    )

    req = Request()

    # prefill 6 tokens
    prefill_len = 6
    hidden_list = [torch.randn(hidden_size) for _ in range(prefill_len)]
    k_list = []
    v_list = []
    for i, h in enumerate(hidden_list):
        _, k, v = self_attn.project(h)
        k = self_attn.apply_rope(k, position_id=i)
        k_list.append(k)
        v_list.append(v)

    engine.prefill(req, k_list, v_list)

    print("After prefill:")
    print("seq_len =", req.seq_len)
    print("block_table =", req.block_table)

    # decode 3 steps + RoPE/page attention alignment check
    all_hidden = hidden_list[:]
    for step in range(3):
        hidden = torch.randn(hidden_size)
        all_hidden.append(hidden)
        q, new_k, new_v = self_attn.project(hidden)
        pos_id = req.seq_len
        q = self_attn.apply_rope(q, position_id=pos_id)
        new_k = self_attn.apply_rope(new_k, position_id=pos_id)

        out = engine.decode_step(req, q, new_k, new_v)
        # baseline: contiguous KV with same RoPE logic
        k_all = []
        v_all = []
        for p, h in enumerate(all_hidden):
            _, k_ref, v_ref = self_attn.project(h)
            k_ref = self_attn.apply_rope(k_ref, position_id=p)
            k_all.append(k_ref)
            v_all.append(v_ref)
        k_all = torch.stack(k_all, dim=0)
        v_all = torch.stack(v_all, dim=0)
        out_ref = self_attn.forward_standard(q, k_all, v_all)
        if not torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"RoPE alignment check failed at decode step {step}.")

        out_hidden = self_attn.o_proj(self_attn.combine_heads(out))

        print(f"\nDecode step {step}:")
        print("output shape =", out.shape)
        print("output hidden shape =", out_hidden.shape)
        print("seq_len =", req.seq_len)
        print("block_table =", req.block_table)
        print("position_ids =", req.position_ids)

    # decode结束, 释放block
    release_request(req, block_manager)


if __name__ == "__main__":
    run_toy_example()
