import torch
from typing import List, Tuple
import math


class KVCache:
    """
    Global KV cache pool organized in fixed-size blocks.

    This acts like the "physical memory" in PageAttention.

    Attributes:
        num_blocks: Total number of blocks available.
        block_size: Number of tokens per block.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

        K: Tensor storing all key vectors.
           Shape: [num_blocks, block_size, num_heads, head_dim]

        V: Tensor storing all value vectors.
           Shape: [num_blocks, block_size, num_heads, head_dim]
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu",
    ):
        self.k = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device)
        self.v = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device)

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
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
            k: Key tensor of shape [num_heads, head_dim]
            v: Value tensor of shape [num_heads, head_dim]
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
            K_block: [block_size, num_heads, head_dim]
            V_block: [block_size, num_heads, head_dim]
        """
        return self.k[block_id], self.v[block_id]


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

    req.seq_len += 1


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

        # 基于kv计算q的attention分数
        # attention(q,k,v) = softmax((q \cdot k^T) / sqrt(head_dim)) \cdot v
        scores = torch.einsum("hd,thd->ht", q, k_all) / math.sqrt(kv_cache.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.einsum("ht,thd->hd", attn, v_all)

        return output


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
    head_dim = 8

    kv_cache = KVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    block_manager = BlockManager(num_blocks)
    attention = PageAttention()
    engine = DecoderEngine(kv_cache, block_manager, attention)

    req = Request()

    # prefill 6 tokens
    prefill_len = 6
    k_list = [torch.randn(num_heads, head_dim) for _ in range(prefill_len)]
    v_list = [torch.randn(num_heads, head_dim) for _ in range(prefill_len)]

    engine.prefill(req, k_list, v_list)

    print("After prefill:")
    print("seq_len =", req.seq_len)
    print("block_table =", req.block_table)

    # decode 3 steps
    for step in range(3):
        q = torch.randn(num_heads, head_dim)
        new_k = torch.randn(num_heads, head_dim)
        new_v = torch.randn(num_heads, head_dim)

        out = engine.decode_step(req, q, new_k, new_v)

        print(f"\nDecode step {step}:")
        print("output shape =", out.shape)
        print("seq_len =", req.seq_len)
        print("block_table =", req.block_table)


if __name__ == "__main__":
    run_toy_example()
