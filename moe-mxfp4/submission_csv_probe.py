"""
Probe: Dump AITER fused_moe tuning CSV, dispatch config, and FlyDSL availability on MI355X.
"""
import os
import sys
import torch


def custom_kernel(data):
    """Probe only - just return reference output."""
    (
        hidden_states, w1, w2, topk_weights, topk_ids,
        num_experts, renormalize, topk,
        w1_scale, w2_scale, a1_scale, a2_scale,
    ) = data

    # === 1. AITER installation info ===
    try:
        import aiter
        print(f"[PROBE] aiter.__file__: {aiter.__file__}")
        print(f"[PROBE] aiter version: {getattr(aiter, '__version__', 'N/A')}")
        aiter_root = os.path.dirname(os.path.dirname(aiter.__file__))
        print(f"[PROBE] AITER_ROOT_DIR: {aiter_root}")
    except Exception as e:
        print(f"[PROBE] aiter import error: {e}")
        aiter_root = None

    # === 2. Environment variables ===
    for var in [
        "AITER_BYPASS_TUNE_CONFIG", "AITER_ONLINE_TUNE",
        "AITER_CONFIG_FMOE", "AITER_REBUILD", "AITER_KSPLIT",
        "AITER_LOG_MORE",
    ]:
        print(f"[PROBE] {var}={os.environ.get(var, '<unset>')}")

    # === 3. Find and read tuned_fmoe.csv ===
    try:
        from aiter.jit.core import AITER_CONFIGS
        fmoe_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
        print(f"[PROBE] AITER_CONFIG_FMOE_FILE: {fmoe_file}")
        print(f"[PROBE] File exists: {os.path.exists(fmoe_file)}")
    except Exception as e:
        print(f"[PROBE] AITER_CONFIGS error: {e}")
        fmoe_file = None

    # Try multiple paths
    candidate_paths = []
    if fmoe_file:
        candidate_paths.append(fmoe_file)
    if aiter_root:
        candidate_paths.append(f"{aiter_root}/aiter/configs/tuned_fmoe.csv")
    candidate_paths.append("/tmp/aiter_configs/tuned_fmoe.csv")

    csv_found = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_found = p
            print(f"[PROBE] CSV found at: {p}")
            break
        else:
            print(f"[PROBE] CSV NOT at: {p}")

    if csv_found:
        import pandas as pd
        df = pd.read_csv(csv_found)
        print(f"[PROBE] CSV shape: {df.shape}")
        print(f"[PROBE] CSV columns: {list(df.columns)}")

        # Show unique values for key columns
        for col in ["cu_num", "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type"]:
            if col in df.columns:
                print(f"[PROBE] Unique {col}: {sorted(df[col].unique().tolist())}")

        # Filter for per_1x32 / MXFP4
        mxfp4_mask = df["q_type"].astype(str).str.contains("per_1x32")
        mxfp4_rows = df[mxfp4_mask]
        print(f"\n[PROBE] === MXFP4 (per_1x32) rows: {len(mxfp4_rows)} ===")
        if len(mxfp4_rows) > 0:
            print(mxfp4_rows.to_string())
        else:
            print("[PROBE] NO per_1x32 rows found in CSV!")

        # Also show Silu rows count
        silu_mask = df["act_type"].astype(str).str.contains("Silu")
        print(f"\n[PROBE] Total Silu rows: {len(df[silu_mask])}")

        # Show first 5 rows of CSV header
        print(f"\n[PROBE] CSV first 3 rows:\n{df.head(3).to_string()}")
    else:
        print("[PROBE] NO CSV file found at any candidate path!")

    # === 4. List all config files ===
    if aiter_root:
        config_dir = f"{aiter_root}/aiter/configs"
        if os.path.isdir(config_dir):
            print(f"\n[PROBE] Files in {config_dir}:")
            for f in sorted(os.listdir(config_dir)):
                fpath = os.path.join(config_dir, f)
                sz = os.path.getsize(fpath) if os.path.isfile(fpath) else "dir"
                print(f"  {f} ({sz})")

        # Also check model_configs subdirectory
        model_cfg_dir = f"{config_dir}/model_configs"
        if os.path.isdir(model_cfg_dir):
            print(f"\n[PROBE] Files in {model_cfg_dir}:")
            for f in sorted(os.listdir(model_cfg_dir)):
                fpath = os.path.join(model_cfg_dir, f)
                sz = os.path.getsize(fpath) if os.path.isfile(fpath) else "dir"
                print(f"  {f} ({sz})")

    # === 5. FlyDSL availability ===
    try:
        from aiter.ops.flydsl.utils import is_flydsl_available
        flydsl = is_flydsl_available()
        print(f"\n[PROBE] FlyDSL available: {flydsl}")
    except Exception as e:
        print(f"\n[PROBE] FlyDSL check error: {e}")

    # === 6. GPU info ===
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx
        print(f"[PROBE] get_gfx(): {get_gfx()}")
        print(f"[PROBE] get_cu_num(): {get_cu_num()}")
    except Exception as e:
        print(f"[PROBE] chip_info error: {e}")

    # === 7. Check what get_2stage_cfgs returns for our config ===
    try:
        from aiter.fused_moe import get_2stage_cfgs
        from aiter import ActivationType, QuantType, dtypes
        cu_num = get_cu_num()
        # Our MXFP4 config
        metadata = get_2stage_cfgs(
            cu_num=cu_num,
            token=4,
            model_dim=hidden_states.shape[-1],
            inter_dim=w1.shape[1],
            expert=num_experts,
            topk=topk,
            act_type=ActivationType.Silu,
            dtype=torch.bfloat16,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            q_type=QuantType.per_1x32,
            use_g1u1=True,
            doweight_stage1=False,
        )
        print(f"\n[PROBE] get_2stage_cfgs result: {metadata}")
        if metadata:
            print(f"[PROBE]   stage1_func: {metadata.stage1_func}")
            print(f"[PROBE]   stage2_func: {metadata.stage2_func}")
            print(f"[PROBE]   block_m: {metadata.block_m}")
            print(f"[PROBE]   ksplit: {metadata.ksplit}")
            print(f"[PROBE]   run_1stage: {metadata.run_1stage}")
    except Exception as e:
        print(f"\n[PROBE] get_2stage_cfgs error: {e}")

    # === 8. Check /tmp/aiter_configs ===
    tmp_dir = "/tmp/aiter_configs"
    if os.path.isdir(tmp_dir):
        print(f"\n[PROBE] Files in {tmp_dir}:")
        for f in sorted(os.listdir(tmp_dir)):
            fpath = os.path.join(tmp_dir, f)
            sz = os.path.getsize(fpath) if os.path.isfile(fpath) else "dir"
            print(f"  {f} ({sz})")

    # Return reference result
    from aiter.fused_moe import fused_moe
    return fused_moe(
        hidden_states, w1, w2,
        topk_weights, topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale, w2_scale=w2_scale,
        a1_scale=a1_scale, a2_scale=a2_scale,
    )
