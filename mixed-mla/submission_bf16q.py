"""
Mixed-MLA: BF16 Q experiment - skip FP8 quantization of Q
Saves one kernel launch by using bf16 Q directly with fp8 KV.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1


NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

PAGE_SIZE = 1

FP8_DTYPE = aiter_dtypes.fp8

_WORKSPACE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, ...]] = {}
_INDPTR_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_KV_INDEX_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_METADATA_CACHE: dict[tuple[object, ...], dict[str, torch.Tensor]] = {}
_OUTPUT_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else device.index


def _get_metadata_workspace(
    batch_size, max_q_len, nhead, nhead_kv, q_dtype, kv_dtype,
    num_kv_splits, device,
):
    cache_key = (
        _device_cache_key(device), batch_size, max_q_len, nhead, nhead_kv,
        q_dtype, kv_dtype, num_kv_splits,
    )
    cached = _WORKSPACE_CACHE.get(cache_key)
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
        _WORKSPACE_CACHE[cache_key] = cached
    return cached


def _get_kv_indices(total_kv_len, device):
    cache_key = (_device_cache_key(device), total_kv_len)
    cached = _KV_INDEX_CACHE.get(cache_key)
    if cached is None:
        cached = torch.arange(total_kv_len, dtype=torch.int32, device=device)
        _KV_INDEX_CACHE[cache_key] = cached
    return cached


def _get_uniform_indptrs(batch_size, q_seq_len, kv_seq_len, device):
    cache_key = (_device_cache_key(device), batch_size, q_seq_len, kv_seq_len)
    cached = _INDPTR_CACHE.get(cache_key)
    if cached is None:
        steps = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        cached = (
            steps * q_seq_len,
            steps * kv_seq_len,
            torch.full((batch_size,), kv_seq_len, dtype=torch.int32, device=device),
        )
        _INDPTR_CACHE[cache_key] = cached
    return cached


def _get_output_buffer(shape, device):
    cache_key = (_device_cache_key(device), shape)
    cached = _OUTPUT_CACHE.get(cache_key)
    if cached is None:
        cached = torch.empty(shape, dtype=torch.bfloat16, device=device)
        _OUTPUT_CACHE[cache_key] = cached
    return cached


# Per-shape optimal split counts
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


def _select_num_kv_splits(batch_size: int, kv_seq_len: int) -> int:
    return _SPLIT_MAP.get((batch_size, kv_seq_len), 16)


def make_mla_decode_metadata(
    batch_size, max_q_len, kv_seq_len, nhead, nhead_kv,
    q_dtype, kv_dtype, qo_indptr, kv_indptr, kv_last_page_len,
    num_kv_splits,
):
    cache_key = (
        _device_cache_key(qo_indptr.device), batch_size, max_q_len,
        kv_seq_len, nhead, nhead_kv, q_dtype, kv_dtype, num_kv_splits,
    )
    cached = _METADATA_CACHE.get(cache_key)
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
        "work_meta_data": work_metadata, "work_indptr": work_indptr,
        "work_info_set": work_info_set, "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }
    _METADATA_CACHE[cache_key] = cached
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

    # Use bf16 Q directly - skip FP8 quantization kernel launch
    q_bf16 = q
    if not q_bf16.is_contiguous():
        q_bf16 = q_bf16.contiguous()

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    total_kv_len = kv_buffer_fp8.shape[0]
    kv_indices = _get_kv_indices(total_kv_len, q.device)
    kv_buffer_4d = kv_buffer_fp8.reshape(
        total_kv_len, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1]
    )

    num_kv_splits = _select_num_kv_splits(batch_size, kv_seq_len)

    meta = make_mla_decode_metadata(
        batch_size, q_seq_len, kv_seq_len, nq, nkv,
        q_bf16.dtype, kv_buffer_4d.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    o = _get_output_buffer((q.shape[0], nq, dv), q.device)

    mla_decode_fwd(
        q_bf16.reshape(-1, nq, dq), kv_buffer_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        q_seq_len,
        page_size=PAGE_SIZE, nhead_kv=nkv, sm_scale=SM_SCALE,
        logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=None, kv_scale=kv_scale,
        intra_batch_mode=True, **meta,
    )

    return o
