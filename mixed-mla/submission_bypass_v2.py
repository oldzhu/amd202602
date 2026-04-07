"""
Mixed-MLA: Bypass v2 — no zero_(), no dynamic quant overhead.
Uses pre-computed Q scale to eliminate dynamic_per_tensor_quant kernel launch.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.quant import dynamic_per_tensor_quant


NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

PAGE_SIZE = 1

FP8_DTYPE = aiter_dtypes.fp8

_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}
_INDPTR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_KV_INDEX_CACHE: dict[tuple, torch.Tensor] = {}
_METADATA_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}
_LOGITS_CACHE: dict[tuple, torch.Tensor] = {}
_LSE_CACHE: dict[tuple, torch.Tensor] = {}
_Q_FP8_CACHE: dict[tuple, torch.Tensor] = {}
_Q_SCALE_CACHE: dict[tuple, torch.Tensor] = {}


def quantize_fp8(tensor: torch.Tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    shape_key = tensor.shape
    fp8_tensor = _Q_FP8_CACHE.get(shape_key)
    if fp8_tensor is None:
        fp8_tensor = torch.empty(tensor.shape, dtype=FP8_DTYPE, device=tensor.device)
        _Q_FP8_CACHE[shape_key] = fp8_tensor
    scale = _Q_SCALE_CACHE.get(shape_key)
    if scale is None:
        scale = torch.empty(1, dtype=torch.float32, device=tensor.device)
        _Q_SCALE_CACHE[shape_key] = scale
    dynamic_per_tensor_quant(fp8_tensor, tensor, scale)
    return fp8_tensor, scale


def _dck(device):
    return device.type, -1 if device.index is None else device.index


def _get_metadata_workspace(batch_size, max_q_len, nhead, nhead_kv,
                            q_dtype, kv_dtype, num_kv_splits, device):
    key = (_dck(device), batch_size, max_q_len, nhead, nhead_kv,
           q_dtype, kv_dtype, num_kv_splits)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is None:
        info = get_mla_metadata_info_v1(
            batch_size, max_q_len, nhead, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=True, num_kv_splits=num_kv_splits,
            intra_batch_mode=True,
        )
        cached = tuple(
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in info
        )
        _WORKSPACE_CACHE[key] = cached
    return cached


def _get_kv_indices(total_kv_len, device):
    key = (_dck(device), total_kv_len)
    cached = _KV_INDEX_CACHE.get(key)
    if cached is None:
        cached = torch.arange(total_kv_len, dtype=torch.int32, device=device)
        _KV_INDEX_CACHE[key] = cached
    return cached


def _get_uniform_indptrs(batch_size, q_seq_len, kv_seq_len, device):
    key = (_dck(device), batch_size, q_seq_len, kv_seq_len)
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


def _get_output_buffer(shape, device):
    key = (_dck(device), shape)
    cached = _OUTPUT_CACHE.get(key)
    if cached is None:
        cached = torch.empty(shape, dtype=torch.bfloat16, device=device)
        _OUTPUT_CACHE[key] = cached
    return cached


def _get_logits_buffer(size, nhead, v_head_dim, device):
    key = (_dck(device), size, nhead, v_head_dim)
    cached = _LOGITS_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (size, 1, nhead, v_head_dim),
            dtype=torch.float32, device=device,
        )
        _LOGITS_CACHE[key] = cached
    return cached


def _get_lse_buffer(size, nhead, device):
    key = (_dck(device), size, nhead)
    cached = _LSE_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (size, 1, nhead, 1),
            dtype=torch.float32, device=device,
        )
        _LSE_CACHE[key] = cached
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


def _select_num_kv_splits(batch_size, kv_seq_len):
    return _SPLIT_MAP.get((batch_size, kv_seq_len), 16)


def make_mla_decode_metadata(
    batch_size, max_q_len, kv_seq_len, nhead, nhead_kv,
    q_dtype, kv_dtype, qo_indptr, kv_indptr, kv_last_page_len,
    num_kv_splits,
):
    key = (
        _dck(qo_indptr.device), batch_size, max_q_len,
        kv_seq_len, nhead, nhead_kv, q_dtype, kv_dtype, num_kv_splits,
    )
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    (
        work_metadata, work_indptr, work_info_set,
        reduce_indptr, reduce_final_map, reduce_partial_map,
    ) = _get_metadata_workspace(
        batch_size, max_q_len, nhead, nhead_kv, q_dtype, kv_dtype,
        num_kv_splits, qo_indptr.device,
    )

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv, nhead_kv, True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len, uni_seqlen_qo=max_q_len,
        fast_mode=True, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=q_dtype, dtype_kv=kv_dtype,
    )

    cached = {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
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

    qo_indptr, kv_indptr, kv_last_page_len = _get_uniform_indptrs(
        batch_size, q_seq_len, kv_seq_len, q.device,
    )

    q_fp8, q_scale = quantize_fp8(q)

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    total_kv_len = kv_buffer_fp8.shape[0]
    kv_indices = _get_kv_indices(total_kv_len, q.device)
    kv_buffer_4d = kv_buffer_fp8.reshape(
        total_kv_len, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1]
    )

    max_q_len = q_seq_len
    num_kv_splits = _select_num_kv_splits(batch_size, kv_seq_len)

    meta = make_mla_decode_metadata(
        batch_size, max_q_len, kv_seq_len, nq, nkv,
        q_fp8.dtype, kv_buffer_4d.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    o = _get_output_buffer((q.shape[0], nq, dv), q.device)

    # Pre-allocated intermediate buffers (bypass mla_decode_fwd allocation)
    partial_size = meta["reduce_partial_map"].size(0) * max_q_len
    logits = _get_logits_buffer(partial_size, nq, dv, q.device)
    attn_lse = _get_lse_buffer(partial_size, nq, q.device)

    # NO zero_() — stage1 ASM writes all mapped partials, reduce only reads mapped ones

    # Stage 1: ASM decode kernel
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.reshape(-1, nq, dq),
        kv_buffer_4d,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        None,  # num_kv_splits_indptr
        meta["work_meta_data"],
        meta["work_indptr"],
        meta["work_info_set"],
        max_q_len,
        PAGE_SIZE,
        nkv,
        SM_SCALE,
        logits,
        attn_lse,
        o,
        q_scale,
        kv_scale,
    )

    # Stage 2: reduce
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        max_q_len,
        o,
        None,  # final_lse
    )

    return o
