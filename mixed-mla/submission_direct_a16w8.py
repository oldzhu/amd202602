"""
Mixed-MLA: Direct stage1_asm + mla_reduce_v1, bf16 Q + fp8 KV (a16w8).
- kv_granularity=32
- bf16 Q (no quantization needed)
- Pre-allocate logits/attn_lse
- Skip wrapper overhead
"""

import torch
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.attention import mla_decode_stage1_asm_fwd, mla_reduce_v1


NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

PAGE_SIZE = 1
KV_GRANULARITY = 32

FP8_DTYPE = aiter_dtypes.fp8

_WORKSPACE_CACHE = {}
_INDPTR_CACHE = {}
_KV_INDEX_CACHE = {}
_METADATA_CACHE = {}
_OUTPUT_CACHE = {}


def _device_key(device):
    return device.type, -1 if device.index is None else device.index


def _get_workspace(batch_size, max_q_len, nhead, nhead_kv, q_dtype, kv_dtype, num_splits, device):
    key = (_device_key(device), batch_size, max_q_len, nhead, nhead_kv, q_dtype, kv_dtype, num_splits)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is None:
        info = get_mla_metadata_info_v1(
            batch_size, max_q_len, nhead, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=True,
            num_kv_splits=num_splits, intra_batch_mode=True,
        )
        cached = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in info)
        _WORKSPACE_CACHE[key] = cached
    return cached


def _get_kv_indices(total_kv_len, device):
    key = (_device_key(device), total_kv_len)
    cached = _KV_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.arange(total_kv_len, dtype=torch.int32, device=device)
        _KV_INDEX_CACHE[key] = cached
    return cached


def _get_indptrs(batch_size, q_seq_len, kv_seq_len, device):
    key = (_device_key(device), batch_size, q_seq_len, kv_seq_len)
    cached = _INDPTR_CACHE.get(key)
    if cached is None:
        steps = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        cached = (
            steps * q_seq_len,
            steps * kv_seq_len,
            torch.full((batch_size,), kv_seq_len, dtype=torch.int32, device=device),
        )
        _INDPTR_CACHE[key] = cached
    return cached


def _get_output(shape, device):
    key = (_device_key(device), shape)
    cached = _OUTPUT_CACHE.get(key)
    if cached is None:
        cached = torch.empty(shape, dtype=torch.bfloat16, device=device)
        _OUTPUT_CACHE[key] = cached
    return cached


_SPLIT_MAP = {
    (4, 1024): 12,
    (4, 8192): 16,
    (32, 1024): 16,
    (32, 8192): 24,
    (64, 1024): 16,
    (64, 8192): 24,
    (256, 1024): 20,
    (256, 8192): 24,
}


def _make_metadata(batch_size, max_q_len, kv_seq_len, nq, nkv, q_dtype, kv_dtype,
                   qo_indptr, kv_indptr, kv_last_page_len, num_splits):
    key = (_device_key(qo_indptr.device), batch_size, max_q_len, kv_seq_len, nq, nkv,
           q_dtype, kv_dtype, num_splits)
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    ws = _get_workspace(batch_size, max_q_len, nq, nkv, q_dtype, kv_dtype, num_splits, qo_indptr.device)
    work_metadata, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map = ws
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nq // nkv, nkv,
        True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=KV_GRANULARITY,
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=True,
        max_split_per_batch=num_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    num_partial = reduce_partial_map.size(0)
    logits = torch.empty(
        (num_partial * max_q_len, 1, nq, KV_LORA_RANK),
        dtype=torch.float32, device=qo_indptr.device)
    attn_lse = torch.empty(
        (num_partial * max_q_len, 1, nq, 1),
        dtype=torch.float32, device=qo_indptr.device)

    cached = {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "logits": logits,
        "attn_lse": attn_lse,
    }
    _METADATA_CACHE[key] = cached
    return cached


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, _, _, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    qo_indptr, kv_indptr, kv_last_page_len = _get_indptrs(
        batch_size, q_seq_len, kv_seq_len, q.device)

    # bf16 Q — no quantization needed (a16w8 kernel)
    q_bf16 = q if q.is_contiguous() else q.contiguous()

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    total_kv_len = kv_buffer_fp8.shape[0]
    kv_indices = _get_kv_indices(total_kv_len, q.device)
    kv_buffer_4d = kv_buffer_fp8.reshape(total_kv_len, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    num_kv_splits = _SPLIT_MAP.get((batch_size, kv_seq_len), 16)

    meta = _make_metadata(
        batch_size, q_seq_len, kv_seq_len, nq, nkv,
        q_bf16.dtype, kv_buffer_4d.dtype,
        qo_indptr, kv_indptr, kv_last_page_len, num_kv_splits)

    o = _get_output((q.shape[0], nq, dv), q.device)

    q_reshaped = q_bf16.reshape(-1, nq, dq)

    mla_decode_stage1_asm_fwd(
        q_reshaped,
        kv_buffer_4d,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        None,  # num_kv_splits_indptr=None → persistent mode
        meta["work_meta_data"],
        meta["work_indptr"],
        meta["work_info_set"],
        q_seq_len,
        PAGE_SIZE,
        nkv,
        SM_SCALE,
        meta["logits"],
        meta["attn_lse"],
        o,
        None,  # q_scale (bf16 Q doesn't need it)
        kv_scale,
    )

    mla_reduce_v1(
        meta["logits"],
        meta["attn_lse"],
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        q_seq_len,
        o,
        None,
    )

    return o
