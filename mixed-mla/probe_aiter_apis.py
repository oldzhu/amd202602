"""
Probe AITER MLA APIs on the remote runner.
This is a read-only info-gathering submission.
"""
import sys
import inspect
import torch

def custom_kernel(data):
    """Minimal kernel that returns expected output while probing APIs."""
    # Print probe info to stderr so it shows in logs
    probe_output = []
    
    def log(msg):
        probe_output.append(msg)
    
    # 1. Triton attention module
    log("=" * 80)
    log("1. aiter.ops.triton.attention contents")
    log("=" * 80)
    try:
        import aiter.ops.triton.attention as att
        for x in sorted(dir(att)):
            if not x.startswith('_'):
                log(f"  {x}")
    except Exception as e:
        log(f"  ERROR: {e}")
    
    # 2. aiter.ops.triton submodules via pkgutil
    log("\n" + "=" * 80)
    log("2. aiter.ops.triton.attention submodules")
    log("=" * 80)
    try:
        import pkgutil
        import aiter.ops.triton.attention as att_pkg
        for importer, modname, ispkg in pkgutil.walk_packages(att_pkg.__path__, att_pkg.__name__ + '.'):
            log(f"  {'PKG' if ispkg else 'MOD'}: {modname}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 3. aiter.mla module
    log("\n" + "=" * 80)
    log("3. aiter.mla module contents")
    log("=" * 80)
    try:
        import aiter.mla as mla
        for x in sorted(dir(mla)):
            if not x.startswith('_'):
                log(f"  {x}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 4. aiter top-level MLA/MXFP4/attention/decode/flash related
    log("\n" + "=" * 80)
    log("4. aiter top-level: mla/mxfp4/attention/decode/flash related")
    log("=" * 80)
    try:
        import aiter
        for x in sorted(dir(aiter)):
            xl = x.lower()
            if any(k in xl for k in ['mla', 'mxfp', 'attention', 'decode', 'flash']):
                log(f"  {x}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 5. decode_attention_fwd_grouped_rope signature
    log("\n" + "=" * 80)
    log("5. decode_attention_fwd_grouped_rope signature")
    log("=" * 80)
    try:
        from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope
        log(str(inspect.signature(decode_attention_fwd_grouped_rope)))
        doc = inspect.getdoc(decode_attention_fwd_grouped_rope)
        if doc:
            log(f"DOC: {doc[:500]}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 6. decode_attention_fwd_grouped (reference, no rope)
    log("\n" + "=" * 80)
    log("6. decode_attention_fwd_grouped (reference)")
    log("=" * 80)
    try:
        from aiter.ops.triton.attention import decode_attention_fwd_grouped
        log(str(inspect.signature(decode_attention_fwd_grouped)))
    except ImportError:
        try:
            from aiter.ops.triton.attention.mla_decode_ref import decode_attention_fwd_grouped
            log("(from mla_decode_ref)")
            log(str(inspect.signature(decode_attention_fwd_grouped)))
        except Exception as e2:
            log(f"  NOT FOUND: {e2}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 7. mla_decode_fwd full signature
    log("\n" + "=" * 80)
    log("7. mla_decode_fwd signature")
    log("=" * 80)
    try:
        from aiter.mla import mla_decode_fwd
        log(str(inspect.signature(mla_decode_fwd)))
    except Exception as e:
        log(f"  ERROR: {e}")

    # 8. mla_decode_stage1_asm_fwd signature
    log("\n" + "=" * 80)
    log("8. mla_decode_stage1_asm_fwd signature")
    log("=" * 80)
    try:
        from aiter.ops.attention import mla_decode_stage1_asm_fwd
        log(str(inspect.signature(mla_decode_stage1_asm_fwd)))
    except Exception as e:
        log(f"  ERROR: {e}")

    # 9. mla_reduce_v1 signature
    log("\n" + "=" * 80)
    log("9. mla_reduce_v1 signature")
    log("=" * 80)
    try:
        from aiter.ops.attention import mla_reduce_v1
        log(str(inspect.signature(mla_reduce_v1)))
    except Exception as e:
        try:
            from aiter.mla import mla_reduce_v1
            log("(from aiter.mla)")
            log(str(inspect.signature(mla_reduce_v1)))
        except Exception as e2:
            log(f"  NOT FOUND: {e2}")

    # 10. Metadata functions
    log("\n" + "=" * 80)
    log("10. Metadata functions")
    log("=" * 80)
    try:
        from aiter.ops.attention import get_mla_metadata_info_v1
        log(f"get_mla_metadata_info_v1: {inspect.signature(get_mla_metadata_info_v1)}")
    except Exception as e:
        log(f"  get_mla_metadata_info_v1 ERROR: {e}")
    try:
        from aiter.ops.attention import get_mla_metadata_v1
        log(f"get_mla_metadata_v1: {inspect.signature(get_mla_metadata_v1)}")
    except Exception as e:
        log(f"  get_mla_metadata_v1 ERROR: {e}")

    # 11. HK MLA decode (HipKittens)
    log("\n" + "=" * 80)
    log("11. HK MLA decode (HipKittens)")
    log("=" * 80)
    try:
        from aiter.ops.attention import hk_mla_decode_fwd
        log(str(inspect.signature(hk_mla_decode_fwd)))
    except Exception as e:
        log(f"  ERROR: {e}")

    # 12. Full aiter.ops.attention contents
    log("\n" + "=" * 80)
    log("12. aiter.ops.attention contents")
    log("=" * 80)
    try:
        import aiter.ops.attention as ops_att
        for x in sorted(dir(ops_att)):
            if not x.startswith('_'):
                log(f"  {x}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 13. MXFP4 in aiter
    log("\n" + "=" * 80)
    log("13. MXFP4 symbols in aiter")
    log("=" * 80)
    try:
        import aiter
        for x in sorted(dir(aiter)):
            if 'mxfp4' in x.lower() or 'fp4' in x.lower():
                log(f"  aiter.{x}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 14. mla_decode_rope.py source (first 150 lines)
    log("\n" + "=" * 80)
    log("14. mla_decode_rope.py source (first 150 lines)")
    log("=" * 80)
    try:
        import aiter.ops.triton.attention.mla_decode_rope as mdr
        src_file = inspect.getfile(mdr)
        log(f"FILE: {src_file}")
        with open(src_file) as f:
            lines = f.readlines()[:150]
            for i, line in enumerate(lines, 1):
                log(f"{i:4d}: {line.rstrip()}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 15. aiter.mla source (first 200 lines)
    log("\n" + "=" * 80)
    log("15. aiter.mla source (first 200 lines)")
    log("=" * 80)
    try:
        import aiter.mla as mla_mod
        src = inspect.getfile(mla_mod)
        log(f"FILE: {src}")
        with open(src) as f:
            lines = f.readlines()[:200]
            for i, line in enumerate(lines, 1):
                log(f"{i:4d}: {line.rstrip()}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 16. stage2 / reduce from mla_decode_rope module
    log("\n" + "=" * 80)
    log("16. Stage2/reduce in mla_decode_rope module")
    log("=" * 80)
    try:
        import aiter.ops.triton.attention.mla_decode_rope as mdr
        for name in sorted(dir(mdr)):
            nl = name.lower()
            if 'stage' in nl or 'reduce' in nl or 'fwd' in nl:
                obj = getattr(mdr, name)
                if callable(obj):
                    try:
                        log(f"  {name}: {inspect.signature(obj)}")
                    except (ValueError, TypeError):
                        log(f"  {name}: (callable, no signature)")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 17. Check for non-persistent path functions
    log("\n" + "=" * 80)
    log("17. Non-persistent decode functions")
    log("=" * 80)
    try:
        import aiter.mla as mla_mod
        for name in sorted(dir(mla_mod)):
            if 'decode' in name.lower() or 'stage' in name.lower() or 'reduce' in name.lower() or 'meta' in name.lower():
                obj = getattr(mla_mod, name)
                if callable(obj):
                    try:
                        log(f"  {name}: {inspect.signature(obj)}")
                    except (ValueError, TypeError):
                        log(f"  {name}: (callable, no signature)")
    except Exception as e:
        log(f"  ERROR: {e}")

    # 18. Check _triton_kernels subdirectory
    log("\n" + "=" * 80)
    log("18. aiter.ops.triton._triton_kernels submodules")
    log("=" * 80)
    try:
        import pkgutil
        import aiter.ops.triton._triton_kernels as tk
        for importer, modname, ispkg in pkgutil.walk_packages(tk.__path__, tk.__name__ + '.'):
            if 'attention' in modname.lower() or 'mla' in modname.lower():
                log(f"  {'PKG' if ispkg else 'MOD'}: {modname}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # Print all output
    full_output = "\n".join(probe_output)
    print(full_output, file=sys.stderr)
    
    # Return correct output for the task
    # We need to match the reference kernel output
    # Inspect what data we got
    q = data["q"]
    kv_data = data["kv_data"]
    
    log_extra = []
    log_extra.append(f"\nDATA KEYS: {sorted(data.keys())}")
    log_extra.append(f"Q shape: {q.shape}, dtype: {q.dtype}")
    if isinstance(kv_data, dict):
        for k, v in kv_data.items():
            log_extra.append(f"KV_DATA[{k}]: shape={v.shape}, dtype={v.dtype}")
    log_extra.append(f"metadata keys: {sorted(data.get('metadata', {}).keys()) if isinstance(data.get('metadata'), dict) else 'N/A'}")
    print("\n".join(log_extra), file=sys.stderr)
    
    # Actually compute the result using the standard path
    from aiter.mla import mla_decode_fwd
    from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter import dynamic_per_tensor_quant
    
    metadata = data["metadata"]
    batch_size = metadata["batch_size"]
    kv_seq_len = metadata["kv_seq_len"]
    num_heads = metadata["num_heads"]
    num_kv_heads = metadata.get("num_kv_heads", 1)
    head_dim = metadata["head_dim"]
    v_head_dim = metadata.get("v_head_dim", 512)
    qk_head_dim = metadata.get("qk_head_dim", 576)
    sm_scale = metadata.get("sm_scale", 1.0 / (qk_head_dim ** 0.5))
    
    kv_buffer_fp8 = kv_data["fp8"]
    kv_scale = kv_data["kv_scale"]
    
    total_q = batch_size
    total_kv_len = batch_size * kv_seq_len
    device = q.device
    
    q_fp8, q_scale = dynamic_per_tensor_quant(q.reshape(total_q, num_heads, qk_head_dim), torch.float8_e4m3fnuz)
    
    PAGE_SIZE = 1
    num_kv_splits = 16
    max_q_len = 1
    
    kv_buf_4d = kv_buffer_fp8.reshape(total_kv_len, PAGE_SIZE, num_kv_heads, head_dim)
    
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * kv_seq_len
    kv_indices = torch.arange(0, total_kv_len, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
    
    out = torch.empty(total_q, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)
    
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, num_heads, q_fp8.dtype, kv_buf_4d.dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    meta_bufs = [torch.empty(s, dtype=d, device=device) for s, d in info]
    
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        num_heads // num_kv_heads, num_kv_heads, True,
        *meta_bufs,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_fp8.dtype,
        dtype_kv=kv_buf_4d.dtype,
    )
    
    mla_decode_fwd(
        q_fp8, kv_buf_4d, out,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        max_q_len, PAGE_SIZE, num_kv_heads, sm_scale, 0.0,
        num_kv_splits, q_scale, kv_scale,
        True,
        *meta_bufs,
    )
    
    return out.reshape(batch_size, num_heads * v_head_dim)
