"""
Mixed-MLA: Optimized split tuning for MI355X (304 CUs).
Reduces wave overhead for large batch sizes by using fewer KV splits.
Based on bypass v2 analysis: dominant shapes (bs=256) had 17-21 waves.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
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

_WORKSPACE_CACHE: dict = {}
_INDPTR_CACHE: dict = {}
_KV_INDEX_CACHE: dict = {}
_METADATA_CACHE: dict = {}
_OUTPUT_CACHE: dict = {}
_Q_FP8_CACHE: dict = {}
_Q_SCALE_CACHE: dict = {}


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


# Optimized split map: reduce waves for large batch sizes
# 304 CUs: B*S items → ceil(B*S/304) waves
# Key insight: for bs=256, fewer waves = less overhead per CU
_SPLIT_MAP = {
    (4, 1024):   16,   # single-wave, more CU util
    (4, 8192):   32,   # single-wave, halve kv_per_split
    (32, 1024):   8,   # 1 wave (256 items) vs 2 waves (512 items)
    (32, 8192):  16,   # 2 waves vs 3 waves
    (64, 1024):   4,   # 1 wave (256 items) vs 4 waves (1024)
    (64, 8192):  16,   # 4 waves vs 6 waves
    (256, 1024):  4,   # 4 waves vs 17 waves — biggest win
    (256, 8192):  8,   # 7 waves vs 21 waves
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

    mla_decode_fwd(
        q_fp8.reshape(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )

    return o
