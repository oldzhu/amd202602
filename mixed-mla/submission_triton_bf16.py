"""
Mixed-MLA: Triton flash-decoding kernel with bf16 Q + bf16 KV.
Eliminates FP8 quantization overhead and metadata computation.
Uses inline Triton kernels for stage1 (attention) and stage2 (reduce).
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t


NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

Lk = 576  # K dimension (512 nope + 64 rope)
Lv = 512  # V dimension (512 nope only)

_INDPTR_CACHE = {}
_KV_INDEX_CACHE = {}
_ATT_LOGITS_CACHE = {}
_OUTPUT_CACHE = {}

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


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q, K_Buffer, V_Buffer, sm_scale,
    kv_indptr, kv_indices, Att_Out,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end, other=0,
            )
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]), other=0.0)
            qk = tl.dot(q, k.to(q.dtype))

            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None]
                kpe = tl.load(K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]), other=0.0)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))

            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * tl.math.tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]), other=0.0)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                      + split_kv_id * stride_mid_os + offs_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]))

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh
                        + split_kv_id * stride_mid_os + Lv)
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2(
    Mid_O, O, kv_indptr,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_obs, stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum, mask=mask_d)


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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, _, _, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    kv_seq_len = config["kv_seq_len"]

    # Use bf16 KV directly — no quantization needed!
    kv_buffer_bf16 = kv_data["bf16"]
    total_kv_len = kv_buffer_bf16.shape[0]

    # K = full 576-dim, V = first 512-dim (shared buffer)
    # kv_buffer_bf16 shape: (total_kv_len, nkv, 576)
    if kv_buffer_bf16.dim() == 2:
        kv_buffer_bf16 = kv_buffer_bf16.unsqueeze(1)
    k_buffer = kv_buffer_bf16  # (total_kv_len, nkv, 576)
    v_buffer = kv_buffer_bf16  # same buffer, kernel uses Lv=512 mask

    # Q shape: (batch_size, nq, 576)
    q_reshaped = q.reshape(batch_size, nq, -1)

    kv_indptr = _get_indptrs(batch_size, kv_seq_len, q.device)
    kv_indices = _get_kv_indices(total_kv_len, q.device)

    num_kv_splits = _SPLIT_MAP.get((batch_size, kv_seq_len), 8)
    attn_logits = _get_attn_logits(batch_size, nq, num_kv_splits, q.device)
    o = _get_output(batch_size, nq, q.device)

    kv_group_num = nq // nkv  # 16
    BLOCK_H = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 512
    BLOCK_DPE = 64
    BLOCK_DV = 512

    grid_stage1 = (
        batch_size,
        triton.cdiv(nq, min(BLOCK_H, kv_group_num)),
        num_kv_splits,
    )

    extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 1}

    _fwd_grouped_kernel_stage1[grid_stage1](
        q_reshaped, k_buffer, v_buffer, SM_SCALE,
        kv_indptr, kv_indices, attn_logits,
        q_reshaped.stride(0), q_reshaped.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        kv_group_num=kv_group_num, q_head_num=nq,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, NUM_KV_SPLITS=num_kv_splits,
        logit_cap=0.0, Lk=Lk, Lv=Lv,
        num_warps=4, num_stages=1, **extra_kargs,
    )

    grid_stage2 = (batch_size, nq)
    extra_kargs2 = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 1}

    _fwd_kernel_stage2[grid_stage2](
        attn_logits, o, kv_indptr,
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        o.stride(0), o.stride(1),
        NUM_KV_SPLITS=num_kv_splits, BLOCK_DV=BLOCK_DV, Lv=Lv,
        num_warps=4, num_stages=2, **extra_kargs2,
    )

    return o
