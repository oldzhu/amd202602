import torch
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

from _mlir import ir
import flydsl
from flydsl.dialects.ext import flir, arith, gpu, buffer_ops, vector, rocdl, scf
from flydsl.lang.ir.types import T, memref
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils import SmemAllocator, SmemPtr

from flydsl.kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
)
from flydsl.kernels.mfma_epilogues import mfma_epilog


def _quant_mxfp4(x, shuffle=True):
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)


def compile_mxfp4_preshuffle_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = True,
):
    a_elem_vec_pack = 2
    b_elem_vec_pack = 2
    elem_bytes = 1
    pack_M = 2
    pack_N = 2
    pack_K_eff = 2

    cbsz = 4
    blgp = 4

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    num_waves = 4
    k_unroll = tile_k_bytes // 128
    k_unroll_packed = k_unroll // pack_K_eff
    n_per_wave = int(tile_n) // num_waves
    num_acc_n = n_per_wave // 16
    num_acc_n_packed = num_acc_n // pack_N

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    DYN = ir.ShapedType.get_dynamic_size()
    total_threads = 256
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_elem_vec_pack
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16
    lds_stride_bytes = tile_k_bytes

    def _a_elem_type():
        return T.ui8

    def _b_elem_type():
        return T.ui8

    def _scale_elem_type():
        return T.i32

    def _a_vec16_type():
        return T.vec(16, T.ui8)

    def _out_elem_type():
        return T.bf16

    module_name = f"mfma_preshuffle_{lds_stage}stages_fp4_fp4_bf16_cshuffle_tm{tile_m}_tn{tile_n}_tk{tile_k}"

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            lds_a_bytes = int(lds_stage) * int(tile_m) * int(lds_stride_bytes) // int(a_elem_vec_pack)
            lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
            lds_total_bytes = max(lds_a_bytes, lds_out_bytes)
            _state["lds_a_decl"] = allocator.allocate_array(_a_elem_type(), lds_total_bytes)
            allocator.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            acc_init = arith.unwrap(arith.constant_vector(0.0, T.f32x4))
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            c_k_div4bytes = c_k / 4 / a_elem_vec_pack
            layout_a_div4 = flir.make_layout((c_m, c_k_div4bytes), stride=(c_k_div4bytes, 1))

            c_k_b = c_k // b_elem_vec_pack
            kpack_bytes = 16
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k_b, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            ).layout_b

            layout_a_scale = make_preshuffle_scale_layout(flir, arith, c_mn=c_m, c_k=c_k)
            layout_b_scale = make_preshuffle_scale_layout(flir, arith, c_mn=c_n, c_k=c_k)

            shape_lds = flir.make_shape(tile_m, tile_k // a_elem_vec_pack)
            stride_lds = flir.make_stride(tile_k // a_elem_vec_pack, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            lds_k_bytes = tile_k_bytes // a_elem_vec_pack
            k_blocks16 = arith.index(lds_k_bytes // 16)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()
            lds_out = (
                SmemPtr(base_ptr, lds_a_ptr.byte_offset, T.bf16, shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )

            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

            bx_m = bx * tile_m
            by_n = by * tile_n

            layout_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            kpack_elems = 16
            col_offset_base_bytes = lane_div_16 * arith.constant(int(kpack_elems), index=True)

            m_repeat = tile_m // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

            c_n0 = c_n / 16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                coord_n = flir.idx2crd(global_n, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_n, 0))
                n_intra_list.append(flir.get(coord_n, 1))

            c64_b = 64
            c0_idx = 0

            def load_b_packs_k64(base_k, ku: int, ni: int):
                base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                k0_base = base_k_bytes / c64_b
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
                idx_pack = flir.crd2idx(coord_pack, layout_b)
                vec_elems = 16
                b_view = flir.TensorView(
                    arg_b, (vec_elems,), strides=(1,), base_indices=(idx_pack,), element_type=_b_elem_type()
                )
                b16 = flir.copy(
                    flir.make_copy_atom(_b_elem_type(), vector_size=vec_elems),
                    b_view,
                    None,
                    alignment=16,
                    return_vector=True,
                    src_buffer_resource=b_rsrc,
                    src_buffer_offset_in_bytes=True,
                )
                b_i64x2 = vector.bitcast(T.i64x2, b16)
                b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                return b0_i64, b1_i64

            def load_b_tile(base_k):
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            def load_scale(arg_scale, rsrc, layout, ku, mni):
                coord_pack = flir.make_coord(mni, ku, lane_div_16, lane_mod_16)
                idx_pack = flir.crd2idx(coord_pack, layout)
                scale_view = flir.TensorView(
                    arg_scale, (1,), strides=(1,), base_indices=(idx_pack,), element_type=_scale_elem_type()
                )
                return flir.copy(
                    flir.make_copy_atom(_scale_elem_type(), vector_size=1),
                    scale_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=rsrc,
                    src_buffer_offset_in_bytes=False,
                )

            def load_b_scale_tile(base_k):
                b_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        scale = load_scale(
                            arg_scale_b,
                            scale_b_rsrc,
                            layout_b_scale,
                            ku + base_k,
                            ni + (by_n + n_tile_base) // pack_N // 16,
                        )
                        b_scale_tile.append(scale)
                return b_scale_tile

            def load_a_scale_tile(base_k):
                a_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for mi in range_constexpr(m_repeat // pack_M):
                        scale = load_scale(
                            arg_scale_a,
                            scale_a_rsrc,
                            layout_a_scale,
                            ku + base_k,
                            mi + bx_m // pack_M // 16,
                        )
                        a_scale_tile.append(scale)
                return a_scale_tile

            def prefetch_ab_scale_tile(base_k):
                return [load_a_scale_tile(base_k), load_b_scale_tile(base_k)]

            def lds_load_16b(curr_row_a_lds, col_base, lds_base):
                col_base_swz = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                return vector.load_op(_a_vec16_type(), lds_a, [idx_a16 + lds_base])

            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_base)
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                return (
                    vector.extract(a_i64x2, static_position=[0], dynamic_position=[]),
                    vector.extract(a_i64x2, static_position=[1], dynamic_position=[]),
                )

            num_a_loads = bytes_per_thread_a // a_load_bytes
            tile_k_dwords = tile_k // 4 // a_elem_vec_pack
            layout_a_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            atom_a_g2r16 = flir.make_copy_atom(_a_elem_type(), vector_size=16)

            def load_a_16(idx_elem):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_a,
                    elem_type=_a_elem_type(),
                    idx_i32=idx_elem,
                    atom_g2r16=atom_a_g2r16,
                    rsrc=a_rsrc,
                    vec_elems=16,
                )

            def a_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4,
                )

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    a_16B = load_a_16(idx_i32)
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_base):
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_a,
                        vec16_ty=_a_vec16_type(),
                        elem_type=_a_elem_type(),
                        atom_s16=atom_a_g2r16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=vec_a_parts[i],
                        elem_bytes=elem_bytes,
                    )

            def prefetch_ab_tile(base_k):
                base_k_div4 = base_k / 4
                return load_a_tile(base_k_div4 // a_elem_vec_pack), load_b_tile(base_k // 2)

            def compute_tile(accs_in, b_tile_in, lds_base, *, a0_prefetch=None, a_scale=None, b_scale=None):
                current_accs_list = list(accs_in)
                mfma_res_ty = T.f32x4
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)
                c0_i64 = arith.constant(0, type=T.i64)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)

                for ku128 in range_constexpr(k_unroll_packed):
                    for mi in range_constexpr(m_repeat // pack_M):
                        a_scale_i32 = a_scale[ku128 * (m_repeat // pack_M) + mi]
                        a_scale_val = vector.extract(a_scale_i32, static_position=[0], dynamic_position=[])

                        for ni in range_constexpr(num_acc_n_packed):
                            b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                            b_scale_val = vector.extract(b_scale_i32, static_position=[0], dynamic_position=[])

                            for ikxdl in range_constexpr(pack_K_eff):
                                k_idx = ku128 * pack_K_eff + ikxdl
                                b_packs0, b_packs1 = b_tile_in[k_idx]
                                col_base = col_offset_base_bytes + (k_idx * 128) // a_elem_vec_pack

                                for imxdl in range_constexpr(pack_M):
                                    mi_idx = mi * pack_M + imxdl
                                    mi_val = arith.constant(mi_idx * 16, index=True)
                                    curr_row_a_lds = row_a_lds + mi_val

                                    if (a0_prefetch is not None) and (k_idx == 0) and (mi_idx == 0):
                                        a0, a1 = a0_prefetch
                                    else:
                                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                                    a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)

                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl
                                        b0 = b_packs0[ni_idx]
                                        b1 = b_packs1[ni_idx]
                                        b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)

                                        acc_idx = mi_idx * num_acc_n + ni_idx
                                        current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                            mfma_res_ty,
                                            [
                                                a128,
                                                b128,
                                                current_accs_list[acc_idx],
                                                cbsz,
                                                blgp,
                                                (ikxdl * pack_M + imxdl),
                                                a_scale_val,
                                                (ikxdl * pack_N + inxdl),
                                                b_scale_val,
                                            ],
                                        )
                return current_accs_list, None

            def store_output(final_accs):
                gpu.barrier()

                def write_row_to_lds(*, mi: int, ii: int, row_in_tile, row, row_base_lds, col_base_local, num_acc_n: int, lds_out):
                    c0_i32 = arith.constant(0, type=T.i32)
                    c1_i32 = arith.constant(1, type=T.i32)
                    cFE_i32 = arith.constant(0xFFFFFFFE, type=T.i32)
                    c2_i32 = arith.constant(2, type=T.i32)

                    lane_id_i32 = arith.index_cast(T.i32, lane_id)
                    lane_lsb = arith.andi(lane_id_i32, c1_i32)
                    is_odd = lane_lsb != c0_i32
                    nbr_lane = arith.xori(lane_id_i32, c1_i32)
                    nbr_lane_bytes = arith.shli(nbr_lane, c2_i32)

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])

                        v16 = arith.trunc_f(T.bf16, val)

                        v1_f16 = vector.from_elements(T.vec(1, T.bf16), [v16])
                        v1_i16 = vector.bitcast(T.vec(1, T.i16), v1_f16)
                        v16_i16 = vector.extract(v1_i16, static_position=[0], dynamic_position=[])
                        z16 = arith.constant(0, type=T.i16)
                        v2_i16 = vector.from_elements(T.vec(2, T.i16), [v16_i16, z16])
                        v16_i32 = vector.extract(vector.bitcast(T.vec(1, T.i32), v2_i16), static_position=[0], dynamic_position=[])

                        nbr_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(nbr_lane_bytes), arith.unwrap(v16_i32))
                        nbr_v1_i32 = vector.from_elements(T.vec(1, T.i32), [nbr_i32])
                        nbr_v2_i16 = vector.bitcast(T.vec(2, T.i16), nbr_v1_i32)
                        nbr_i16 = vector.extract(nbr_v2_i16, static_position=[0], dynamic_position=[])
                        nbr_v1_i16 = vector.from_elements(T.vec(1, T.i16), [nbr_i16])

                        nbr_v1_f16 = vector.bitcast(T.vec(1, T.bf16), nbr_v1_i16)
                        nbr_f16 = vector.extract(nbr_v1_f16, static_position=[0], dynamic_position=[])

                        even_f16 = arith.select(is_odd, nbr_f16, v16)
                        odd_f16 = arith.select(is_odd, v16, nbr_f16)

                        col_local_i32 = arith.index_cast(T.i32, col_local)
                        col_even_i32 = arith.andi(col_local_i32, cFE_i32)
                        col_even = arith.index_cast(T.index, col_even_i32)

                        lds_idx = row_base_lds + col_even

                        v2 = vector.from_elements(T.vec(2, T.bf16), [even_f16, odd_f16])
                        vector.store(v2, lds_out, [lds_idx])

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    is_valid = col_g0 < c_n
                    _if = scf.IfOp(is_valid)
                    with _if.then():
                        idx_out = flir.crd2idx(flir.make_coord(row, col_g0), layout_c)
                        byte_off = idx_out * arith.constant(2, index=True)
                        frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                        buffer_ops.buffer_store(frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True)

                mfma_epilog(
                    use_cshuffle=True,
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=4,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=T.bf16,
                    write_row_to_lds=write_row_to_lds,
                    store_pair=store_pair,
                )

            lds_tile_elems = arith.constant(tile_m * tile_k // a_elem_vec_pack, index=True)
            lds_base0 = arith.constant(0, index=True)
            lds_base1 = lds_tile_elems

            k0 = arith.constant(0, index=True)
            a_regs0, b_tile0 = prefetch_ab_tile(k0)
            a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // 256)

            store_a_tile_to_lds(a_regs0, lds_base0)
            gpu.barrier()
            accs = [acc_init] * (num_acc_n * m_repeat)

            lds_base_pong = lds_base0
            lds_base_ping = lds_base1
            b_tile_pong = b_tile0

            a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

            c_k_stop = c_k - (tile_k * 3)
            for k_iv in range(0, c_k_stop, tile_k * 2):
                next_k1 = k_iv + tile_k
                a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(next_k1 // 256)

                accs, _ = compute_tile(
                    accs,
                    b_tile_pong,
                    lds_base_pong,
                    a0_prefetch=a0_prefetch_pong,
                    a_scale=a_scale_pong,
                    b_scale=b_scale_pong,
                )
                a0_prefetch_pong = None

                store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                gpu.barrier()

                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

                next_k2 = k_iv + tile_k * 2
                a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(next_k2 // 256)

                accs, _ = compute_tile(
                    accs,
                    b_tile_ping,
                    lds_base_ping,
                    a0_prefetch=a0_prefetch_ping,
                    a_scale=a_scale_ping,
                    b_scale=b_scale_ping,
                )
                a0_prefetch_ping = None

                store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                gpu.barrier()

                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)

            last_k = c_k - tile_k
            a_regs_ping, b_tile_ping = prefetch_ab_tile(last_k)
            a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(last_k // 256)

            accs, _ = compute_tile(
                accs,
                b_tile_pong,
                lds_base_pong,
                a0_prefetch=a0_prefetch_pong,
                a_scale=a_scale_pong,
                b_scale=b_scale_pong,
            )

            a0_prefetch_pong = None

            store_a_tile_to_lds(a_regs_ping, lds_base_ping)
            gpu.barrier()

            a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)

            final_accs, _ = compute_tile(
                accs,
                b_tile_ping,
                lds_base_ping,
                a0_prefetch=a0_prefetch_ping,
                a_scale=a_scale_ping,
                b_scale=b_scale_ping,
            )

            store_output(final_accs)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            gx = (c_m + tm - one) / tm
            gy = (c_n + tn - one) / tn

            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, c_m, c_n, c_k],
            )

    m = _GEMM()
    return flydsl.compile(m)


_KERNEL_CACHE = {}
_STATIC_VIEW_CACHE = {}


def _tile_m_for_m(m):
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    if m <= 128:
        return 128
    return 256


def _flat_reinterpret(tensor, dtype):
    view = tensor.view(dtype)
    if not view.is_contiguous():
        view = view.contiguous()
    return view.view(-1)


def _flat_reinterpret_cached(tensor, dtype):
    cache_key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(dtype),
        str(tensor.device),
    )
    cached = _STATIC_VIEW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    flat_view = _flat_reinterpret(tensor, dtype)
    _STATIC_VIEW_CACHE[cache_key] = flat_view
    return flat_view


def get_flydsl_kernel(m, n, k):
    global _KERNEL_CACHE

    tm = _tile_m_for_m(m)

    tn = 256 if n >= 256 else 128
    tk = 256

    cache_key = (tm, tn, tk)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    try:
        exe = compile_mxfp4_preshuffle_gemm(
            M=m,
            N=n,
            K=k,
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
        )
        _KERNEL_CACHE[cache_key] = exe
        return exe
    except Exception as e:
        print(f"MLIR compilation failed: {e}")
        print("Full traceback:")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"FlyDSL Compilation failed: {e}")


def custom_kernel(data):
    A, B, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n, _ = B.shape

    tm = _tile_m_for_m(m)

    pad_m = (m + tm - 1) // tm * tm
    A_padded = torch.nn.functional.pad(A, (0, 0, 0, pad_m - m)) if m != pad_m else A

    A_q, A_scale_sh = _quant_mxfp4(A_padded, shuffle=True)

    A_q_u8 = _flat_reinterpret(A_q, torch.uint8)
    A_scale_i32 = _flat_reinterpret(A_scale_sh, torch.int32)
    B_shuffle_u8 = _flat_reinterpret_cached(B_shuffle, torch.uint8)
    B_scale_i32 = _flat_reinterpret_cached(B_scale_sh, torch.int32)

    C_padded = torch.empty((pad_m, n), device=A.device, dtype=torch.bfloat16)

    exe = get_flydsl_kernel(pad_m, n, k)

    exe(
        C_padded.view(-1),
        A_q_u8,
        B_shuffle_u8,
        A_scale_i32,
        B_scale_i32,
        pad_m,
        n,
        k,
    )
    return C_padded[:m, :]