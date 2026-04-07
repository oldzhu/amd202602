"""
Mixed-MLA: Use AITER's installed Triton MLA kernel (mla_decode_rope).
No FP8 quantization. bf16 Q + bf16 KV. Minimal overhead.
Uses decode_attention_fwd_grouped_rope with use_rope=False.
"""

import torch
from task import input_t, output_t

from aiter.ops.triton.attention.mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

Lv = 512

_INDPTR_CACHE = {}
_KV_INDEX_CACHE = {}
_ATT_LOGITS_CACHE = {}
_OUTPUT_CACHE = {}
_DUMMY_CACHE = {}

_SPLIT_MAP = {
    (4, 1024): 8,
    (4, 8192): 16,
    (32, 1024): 8,
    (32, 8192): 16,
    (64, 1024): 8,
    (64, 8192): 16,
    (256, 1024): 8,
    (256, 8192): 16,
}


def _device_key(device):
    return device.type, -1 if device.index is None else device.index


def _get_indptrs(batch_size, kv_seq_len, device):
    key = (_device_key(device), batch_size, kv_seq_len)
    cached = _INDPTR_CACHE.get(key)
    if cached is None:
        steps = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        cached = steps * kv_seq_len
        _INDPTR_CACHE[key] = cached
    return cached


def _get_kv_indices(total_kv_len, device):
    key = (_device_key(device), total_kv_len)
    cached = _KV_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.arange(total_kv_len, dtype=torch.int32, device=device)
        _KV_INDEX_CACHE[key] = cached
    return cached


def _get_attn_logits(batch_size, num_heads, num_kv_splits, device):
    key = (_device_key(device), batch_size, num_heads, num_kv_splits)
    cached = _ATT_LOGITS_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (batch_size, num_heads, num_kv_splits, Lv + 1),
            dtype=torch.float32, device=device)
        _ATT_LOGITS_CACHE[key] = cached
    return cached


def _get_output(batch_size, num_heads, device):
    key = (_device_key(device), batch_size, num_heads)
    cached = _OUTPUT_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (batch_size, num_heads, V_HEAD_DIM),
            dtype=torch.bfloat16, device=device)
        _OUTPUT_CACHE[key] = cached
    return cached


def _get_dummies(device):
    key = _device_key(device)
    cached = _DUMMY_CACHE.get(key)
    if cached is None:
        # Dummy tensors for the rope API (use_rope=False, so these are ignored)
        cos_sin_cache = torch.empty((1, QK_ROPE_HEAD_DIM), dtype=torch.float32, device=device)
        positions = torch.zeros(1, dtype=torch.int64, device=device)
        k_pe_tokens = None
        cached = (cos_sin_cache, positions, k_pe_tokens)
        _DUMMY_CACHE[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, _, _, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_bf16 = kv_data["bf16"]
    total_kv_len = kv_buffer_bf16.shape[0]

    # kv_buffer_bf16: (total_kv_len, nkv, 576)
    if kv_buffer_bf16.dim() == 3 and kv_buffer_bf16.shape[1] == 1:
        k_buffer = kv_buffer_bf16.squeeze(1)  # (total_kv_len, 576)
    elif kv_buffer_bf16.dim() == 2:
        k_buffer = kv_buffer_bf16  # already (total_kv_len, 576)
    else:
        k_buffer = kv_buffer_bf16

    # For the rope API: k_buffer needs shape (total_kv, nkv, head_dim)
    if k_buffer.dim() == 2:
        k_buffer = k_buffer.unsqueeze(1)  # (total_kv, 1, 576)

    v_buffer = k_buffer  # Same buffer; kernel masks Lv=512

    q_reshaped = q.reshape(batch_size, nq, -1)  # (batch_size, nq, 576)

    kv_indptr = _get_indptrs(batch_size, kv_seq_len, q.device)
    kv_indices = _get_kv_indices(total_kv_len, q.device)

    num_kv_splits = _SPLIT_MAP.get((batch_size, kv_seq_len), 8)
    attn_logits = _get_attn_logits(batch_size, nq, num_kv_splits, q.device)
    o = _get_output(batch_size, nq, q.device)

    cos_sin_cache, positions, k_pe_tokens = _get_dummies(q.device)

    decode_attention_fwd_grouped_rope(
        q_reshaped,           # [batch, nq, 576]
        k_buffer,             # [total_kv, nkv, 576]
        v_buffer,             # [total_kv, nkv, 576] (kernel uses Lv=512 mask)
        o,                    # [batch, nq, 512]
        kv_indptr,            # [batch+1]
        kv_indices,           # [total_kv]
        k_pe_tokens,          # None (no rope)
        KV_LORA_RANK,         # 512
        QK_ROPE_HEAD_DIM,     # 64
        cos_sin_cache,        # dummy
        positions,            # dummy
        attn_logits,          # [batch, nq, num_kv_splits, 513]
        num_kv_splits,
        SM_SCALE,
        logit_cap=0.0,
        use_rope=False,
    )

    return o
