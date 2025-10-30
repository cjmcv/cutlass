/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Utilities for mixed input data type kernels.
*/

#pragma once

#include <cuda.h>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cute/util/type_traits.hpp"

namespace cutlass {

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// <NT> dequantize_kernel
// 要点：
// 1) dq_buffer是反量化的输出，operand_layout是量化tensor的layout，layout中包含的shape和stride，这里为何能基于量化tensor的layout来构建反量化结果的tensor？
//  -- 量化tensor和反量化tensor的shape是一致的，只是stride因为数据格式不同会有差别。在 dequantize 里我们根本没用到这些 stride，
//   真正决定写回地址的是后面 local_tile + local_partition 产生的 新 Tensor 的指针和偏移；所以只要 shape 相同，layout 可以安全复用。
//   local_tile 会：1) 根据 blk_coord 算出 新的 base pointer（dq_buffer + offset）.
//                  2) 根据 blk_shape 算出 新的 shape（tpb,1,1）.
//                  3) 重新构造一个不含原 stride 的 ComposedLayout，旧 stride 被丢弃.
// 2) init_quantized_iterator
//  -- 当每个元素 ≥ 8 bit（int8、fp8 …）, 直接返回普通的 gmem_ptr<T>，GPU 全局内存天然支持 1 字节对齐访问。
//   当每个元素 < 8 bit（4-bit、2-bit、1-bit 等）返回 subbyte_iterator<const QuantizedElement>，
//   这个迭代器内部会：1) 把指针当成 uint8_t* 逐字节加载；
//                   2) 用位运算提取或写入对应偏移处的若干 bit；
//                   3) 对外仍然表现成“一个元素”的引用，语法上像普通指针。
//                  于是后面 make_tensor(init_quantized_iterator(), operand_layout) 得到的 gmem_op_q 无论元素多小，都能正确读出量化值，而 kernel 里剩下的代码无需再关心“位打包”细节。
// 3) broadcasted_scale_layout 是广播的layout，其shape是 (n, make_shape(group_size, group_num), l). 原scale的shape是 (n, group_num, l)，访问时按broadcasted的坐标访问，自动广播数据。
// 4）local_tile 是 block 级别划分，维度是(tpb, 1, 1, num_iters)
//    local_partition是线程级划分，维度是(1, 1, 1, num_iters)
//    make_fragment_like是寄存器划分，只需要一个实例就能推断出 元素类型 和 寄存器数组大小，此外除了num_iters，其他都是1，所以按(_, _, _, 0)来取即可。
//    其中的num_iters的意思：由auto blk_coord = cute::make_coord(_, blockIdx.x, blockIdx.y);// [MN,K,L]，一个block负责固定KL下的所有MN，blk_shape是(128,1,1)，所以num_iters=MN//128（向上取整）
// 5）for循环中，先根据坐标处理边界，逐个iter将量化的q/scale/zero都拷贝到寄存器，基于寄存器完成反量化：将q和zero都转到scale类型，计算 q*scale + zero，将类型转为反量化输出格式后保存会gmem。
//    注意：反量化公式是(q - zero) * scale，而这里计算的是 q*scale + zero。
//
// 补充：
// <NT> 分组量化逻辑，权重维度是src[N,K], 以group_size=128为例：
// 1）将[1,128]的数据作为一组，一组有一个scale值，即对于每个[1,128]的原权重矩阵，找到其最大绝对值作为该组的scale，所以共得到scale[N,K/128].  
// 2）对每个[1,128]的原权重矩阵，取出其scale，逐个元素除以scale，再乘以7放大到[-7,7]，即可完成整个量化计算。quant = round(w128 / scale * 7)
//  如有zero：
// 1）将[1,128]的数据作为一组，一组有一个scale值和zero值，先计算 zero=(该组的max+该组min)/2, 即为中心点center,而scale=max(abs(w-zero)),即减去zero后的绝对最大值，
//    scale和zero的维度都是[N,K/128]。
// 2）对每个[1,128]的原权重矩阵，取出其scale和zero，逐个元素先减去zero后除以scale，再乘以7放大到[-7,7]，即可完成整个量化计算。quant = round((w128 - zero) / scale * 7)
//
// <NT> blockwise / groupwise / channelwise / tokenwise / tensorwise 的区别
// blockwise 常取[128,128]为一组，scale维度是[N//128, K//128]，同时针对N和K维度。
// groupwise 常取[1,128]为一组，scale维度是[N, K//128]，针对K维度。
// channelwise 以整个输出channel为一组，即一个N的所有K为一组，所以scale维度是[N, 1]
// tokenwise 以一个token为一组，针对输入A矩阵[M,K], 即一个M的所有K为一组，所以scale维度是[M, 1]
// tensorwise 是整个tensor为一组，即scale只有一个值。layerwise同tensorwise。

template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleBroadCastLayout,
  class ThrLayout>
__global__ void dequantize_kernel(DequantizedElement* dq_buffer,
                                  QuantizedElement const* q_buffer,
                                  OperandLayout const operand_layout,
                                  ElementScale const* scale_buffer,
                                  ElementZero const* zero_buffer,
                                  ScaleBroadCastLayout const broadcasted_scale_layout,
                                  ThrLayout thr_layout) {
  using namespace cute;

  // Represent the full tensors to gmem elements.
  // These are expected to have shape [MN, K, L]
  cute::Tensor gmem_op_dq = cute::make_tensor(cute::make_gmem_ptr(dq_buffer), operand_layout);
  auto init_quantized_iterator = [&]() {
    if constexpr (cute::sizeof_bits_v<QuantizedElement> >= 8) {
      return cute::make_gmem_ptr(q_buffer);
    }
    else {
      return cute::subbyte_iterator<const QuantizedElement>(q_buffer);
    }
  };
  cute::Tensor gmem_op_q  = cute::make_tensor(init_quantized_iterator(), operand_layout);
  // While the scales are expected to have shape [MN, G, L] but with a stride to allow broadcasting
  // It is expected that K % G == 0
  cute::Tensor gmem_scale_broadcasted = cute::make_tensor(make_gmem_ptr(scale_buffer), broadcasted_scale_layout);
  cute::Tensor gmem_zero_broadcasted = cute::make_tensor(make_gmem_ptr(zero_buffer), broadcasted_scale_layout);

  // Assign 1 thread per element in the thread block
  auto blk_shape = cute::make_shape(size<0>(thr_layout), _1{}, _1{}); //
  auto blk_coord = cute::make_coord(_, blockIdx.x, blockIdx.y);  // (MN, K, L)

  // Tile across the block
  auto gOp_dq = cute::local_tile(gmem_op_dq, blk_shape, blk_coord);
  auto gScale = cute::local_tile(gmem_scale_broadcasted, blk_shape, blk_coord);
  auto gZero  = cute::local_tile(gmem_zero_broadcasted,  blk_shape, blk_coord);
  auto gOp_q  = cute::local_tile(gmem_op_q, blk_shape, blk_coord);

  auto tOpDq_gOpDq = cute::local_partition(gOp_dq, thr_layout, threadIdx.x);
  auto tScale_gScale = cute::local_partition(gScale, thr_layout, threadIdx.x);
  auto tZero_gZero = cute::local_partition(gZero, thr_layout, threadIdx.x);
  auto tOpQ_gOpQ = cute::local_partition(gOp_q, thr_layout, threadIdx.x);

  // Make a fragment of registers to hold gmem loads
  cute::Tensor rmem_op_q = cute::make_fragment_like(tOpQ_gOpQ(_, _, _, 0));
  cute::Tensor rmem_scale = cute::make_fragment_like(tScale_gScale(_, _, _, 0));
  cute::Tensor rmem_zero = cute::make_fragment_like(tZero_gZero(_, _, _, 0));
  cute::Tensor rmem_op_dq = cute::make_fragment_like(tOpDq_gOpDq(_, _, _, 0));
  cute::Tensor rmem_op_scaled = cute::make_fragment_like<ElementScale>(rmem_op_dq);
  cute::Tensor rmem_zero_buf = cute::make_fragment_like<ElementScale>(rmem_zero);

  cute::Tensor pred_id = cute::make_identity_tensor(shape(operand_layout));
  auto pred_blk_tile = cute::local_tile(pred_id, blk_shape, blk_coord);
  auto pred_thr_partition = cute::local_partition(pred_blk_tile, thr_layout, threadIdx.x);

  const auto num_iters = cute::size<3>(tOpDq_gOpDq);

  for (int ii = 0; ii < num_iters; ++ii) {
    const auto thread_offset = cute::get<0>(pred_thr_partition(0, 0, 0, ii));
    if (thread_offset < cute::size<0>(operand_layout)) {
      cute::copy(tOpQ_gOpQ(_, _, _, ii), rmem_op_q);
      cute::copy(tScale_gScale(_, _, _, ii), rmem_scale);
      cute::copy(tZero_gZero(_, _, _, ii), rmem_zero);
      cute::transform(rmem_op_q, rmem_op_scaled, [] (const QuantizedElement& elt) { return ElementScale(elt); } );
      cute::transform(rmem_zero, rmem_zero_buf, [] (const ElementZero& elt) { return ElementScale(elt); } );
      cute::transform(rmem_op_scaled, rmem_scale, rmem_op_scaled, cute::multiplies{});
      cute::transform(rmem_op_scaled, rmem_zero_buf, rmem_op_scaled, cute::plus{});
      cute::transform(rmem_op_scaled, rmem_op_dq, [] (const ElementScale& elt) { return DequantizedElement(elt); } );
      cute::copy(rmem_op_dq, tOpDq_gOpDq(_, _, _, ii));
    }
  }
}

// <NT> mix_gemm中的 dequantize 函数介绍 
// dq_buffer是输出的反量化结果， q_buffer/scale_buffer/zero_buffer是输入的量化数据。
// operand_layout 对应 q_buffer，如 make_layout(shape_B, stride_B) 
//                                 -> shape_B = cute::make_shape(n,k,l);
//                                 -> stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
// scale_layout 对应 scale_buffer, 与 operand_layout 类似，shape为(n, scale_k=k/g, l), 注意mnklg中的g是group_size.
// scale_k是group_num, group_size*scale_k = k, 每个 group 的 scale 值被广播复制 group_size 次，正好对应权重矩阵里同一组的 group_size 个元素。
// stride和shape会一一对应，shape中的group_size对应stride为0，即同一组里的 group_size 个元素都指向 同一份 scale 内存，实现“一次存放，多次复用”。
template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout>
static void dequantize(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       int const group_size,
                       cudaStream_t &stream) {
  using namespace cute;

  constexpr int tpb = 128;
  auto thr_layout = make_layout(make_shape(Int<tpb>{}));

  const auto num_rows = get<0>(shape(operand_layout));
  const auto gemm_k = get<1>(shape(operand_layout));   // [MN, K, L]
  const auto batches = get<2>(shape(operand_layout));  // [MN, K, L]
  const auto scale_k = get<1>(shape(scale_layout));    // [MN, Scale_K, L]

  if (num_rows != size<0>(scale_layout)) {
    std::cerr << "Invalid first dimension for scales. Must match first dim for weights."
              << " But got shapes " << shape(operand_layout) << " " << shape(scale_layout)
              << std::endl;
    exit(-1);
  }

  const auto scale_stride0 = get<0>(stride(scale_layout));
  const auto scale_stride1 = get<1>(stride(scale_layout));
  const auto scale_stride2 = get<2>(stride(scale_layout));

  auto scale_shape_bcast = make_shape(num_rows, make_shape(group_size, scale_k), batches);
  auto scale_stride_bcast = make_stride(scale_stride0, make_stride(0, scale_stride1), scale_stride2);
  auto scale_layout_bcast = make_layout(scale_shape_bcast, scale_stride_bcast);

  const auto blocks_x = gemm_k;
  const auto blocks_y = batches;

  dim3 blocks(blocks_x, blocks_y, 1);
  dequantize_kernel<<<blocks, tpb, 0, stream>>>(dq_buffer, q_buffer, operand_layout, scale_buffer, zero_buffer, scale_layout_bcast, thr_layout);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
class packed_scale_t {
public:
  static_assert(cute::is_same_v<T, cutlass::int8_t> ||
                cute::is_same_v<T, cutlass::uint8_t> ||
                cute::is_same_v<T, cutlass::float_e4m3_t> ||
                cute::is_same_v<T, cutlass::float_e5m2_t>,
                "only 8 bit arithmetic types are supported.");
  CUTLASS_HOST_DEVICE
  explicit packed_scale_t(T val) {
    if constexpr (!cute::is_unsigned_v<T>) {
      // Only pack negative values. The positive values are generated in flight in the mainloop.
      storage[0] = pack4(T(float(val) * -8.f), T(float(val) * -7.f), T(float(val) * -6.f), T(float(val) * -5.f));
      storage[1] = pack4(T(float(val) * -4.f), T(float(val) * -3.f), T(float(val) * -2.f), -val);
    }
    else {
      storage[0] = pack4(T(float(val) * 8.f), T(float(val) * 7.f), T(float(val) * 6.f), T(float(val) * 5.f));
      storage[1] = pack4(T(float(val) * 4.f), T(float(val) * 3.f), T(float(val) * 2.f), val);
    }
  }
  CUTLASS_HOST_DEVICE
  packed_scale_t() = default;
  CUTLASS_HOST_DEVICE
  explicit operator float() const {
    return float(get());
  }
  CUTLASS_HOST_DEVICE
  bool operator==(packed_scale_t const& rhs) const {
    return storage[0] == rhs.storage[0] && storage[1] == rhs.storage[1];
  }
  CUTLASS_HOST_DEVICE
  bool operator!=(packed_scale_t const& rhs) const {
    return !(*this == rhs);
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator+(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() + rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator-(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() - rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator*(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() * rhs.get());
  }
  CUTLASS_HOST_DEVICE
  friend packed_scale_t operator/(packed_scale_t const& lhs, packed_scale_t const& rhs) {
    return packed_scale_t(lhs.get() / rhs.get());
  }

private:
  using Storage = uint32_t;
  using Stage = uint8_t;

  Storage storage[2] {};

  CUTLASS_HOST_DEVICE
  static Storage pack4(T c1, T c2, T c3, T c4) {
    Storage result = 0;
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c4)) << 24);
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c3)) << 16);
    result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c2)) << 8);
    result |= static_cast<Storage>(reinterpret_cast<Stage const&>(c1));
    return result;
  }
  CUTLASS_HOST_DEVICE
  T get() const {
    auto stage = static_cast<Stage>(storage[0] >> 8);
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<T const&>(stage);
    #else
    T tmp;
    std::memcpy(&tmp, &stage, sizeof(Stage));
    return tmp;
    #endif
  }
  CUTLASS_HOST_DEVICE
  T get(int idx) const {
    Stage stage;
    if (idx < 4) stage = static_cast<Stage>(storage[0] >> (8 * idx));
    else         stage = static_cast<Stage>(storage[1] >> (8 * idx - 32));
    #if defined(__CUDA_ARCH__)
    return reinterpret_cast<T const&>(stage);
    #else
    T tmp;
    std::memcpy(&tmp, &stage, sizeof(Stage));
    return tmp;
    #endif
  }
};

// In the mainloop, PRMT selects 1 byte from only 8 bytes so the sign bit is handled in an extra PRMT.
// Here the encodings of positive values and negative values are unified (except for the sign bit).
// For instance, 1 becomes 0b0111, which is the same encoding as -1 (0b1111).
static bool unified_encode_int4b(cutlass::int4b_t const *block_in, cutlass::int4b_t *block_out, const size_t block_size) {

  using StorageType = cutlass::int4b_t::Storage;
  constexpr int pack = cute::sizeof_bits_v<StorageType> / 4;
  const size_t host_buf_size = block_size / pack;
  std::vector<StorageType> host_buf(host_buf_size);
  cutlass::device_memory::copy_to_host(host_buf.data(), (StorageType *) block_in, host_buf_size);

  for (auto&& d : host_buf) {
    StorageType out = 0;
    StorageType mask = 0x0f;
    for (int i = 0; i < pack; i++) {
      cutlass::int4b_t curr;
      curr.storage = (d >> (i * 4)) & 0x0f;
      switch (curr) {
        case 1: curr.storage = StorageType(0b0111); break; // 2's complement
        case 2: curr.storage = StorageType(0b0110); break; // 2's complement
        case 3: curr.storage = StorageType(0b0101); break; // 2's complement
        case 4: curr.storage = StorageType(0b0100); break; // 2's complement
        case 5: curr.storage = StorageType(0b0011); break; // 2's complement
        case 6: curr.storage = StorageType(0b0010); break; // 2's complement
        case 7: curr.storage = StorageType(0b0001); break; // 2's complement
        default: break;
      }
      out |= (curr.storage << (4 * i)) & mask;
      mask <<= 4;
    }
    d = out;
  }

  cutlass::device_memory::copy_to_device((StorageType*) block_out, host_buf.data(), host_buf_size);
  return true;
}

template <class ElementScale>
static bool pack_scale_fp8(ElementScale const *block_in, cutlass::Array<ElementScale, 8> *block_out, const size_t block_size) {
  std::vector<ElementScale> data_in(block_size);
  std::vector<cutlass::Array<ElementScale, 8>> data_out(block_size);

  try {
    cutlass::device_memory::copy_to_host(data_in.data(), block_in, block_size);
  }
  catch (cutlass::cuda_exception const& e) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
    return false;
  }

  for (size_t i = 0; i < block_size; i++) {
    cutlass::packed_scale_t<ElementScale> tmp(data_in[i]);
    data_out[i] = reinterpret_cast<cutlass::Array<ElementScale, 8> const&>(tmp);
  }

  try {
    cutlass::device_memory::copy_to_device(block_out, data_out.data(), block_size);
  }
  catch (cutlass::cuda_exception const& e) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
    return false;
  }
  return true;
}

template <class T, class = void>
struct UnderlyingElement {
  using type = T;
};

template <class T>
struct UnderlyingElement<T, cute::void_t<typename T::Element>> {
  using type = typename T::Element;
};

// <NT> compute_memory_reordering_atom: w4a16中权重shuffle的关键函数
// 把 tensor core 指令的寄存器-TV(thread-value) 布局 逆向映射为一个 共享内存重排原子，使得 每个线程在共享内存
// 中需要的数据连续且对齐，从而用 最宽向量加载 完成 mixed-dtype GEMM 中的窄类型加载，消除 sub-bank 冲突，最大化带宽。
// 具体做法:
//   1) 选一条 代表GMMA 指令 作为代表，把它的 TV 布局借出来。
//      rs_op_selector 返回最合适给定参数的指令，其中 ElementC 和 TileShape 不影响TV布局，可以随便填。
//      以MMA_Traits取得A矩阵TV布局 ALayout, 维度是(thr,val,mk)， 前两个维度就是 “哪个线程拥有哪几个值” 的映射表。
//      size<1>(tv_layout_mma)取的是 val 的size，即一条 GMMA 里 每个线程拥有的元素个数，需要能整除 size(val_layout) 才能换序。
//   2) tv_tiler_warp 是把线程数从128压到32，val不变，充当warp级别的切分器。
//      原来 128 线程分 64×16 元素，现在 32 线程分 16×16 元素（M 维除以 4，K 维不变）。
//      tv_layout_mma_warp: 把“全局 TV 布局”与“warp 拆分器”复合，得到 单 warp 的 TV 布局（仍是 thr,val → mk）
//      mk_layout_mma_warp：求逆布局，得到(m,k) 坐标 → (thr,val) 编号，与tv_layout_mma_warp相反对应。
//   3) 沿 K 方向把多个 warp 布局拼起来，换得更宽向量段。
//      atom_layout 典型为 _4、_8，表示 把 4 或 8 个相邻 GMMA 的 K 段拼成一块。
//      同一线程在共享内存中连续拥有的元素数 ×4/×8，后续可用 128-bit 加载 一次搬完。
//   4) val_to_offset: 值偏移。值 → 线性偏移，例如 val_layout = _2 且原每线程 8 元素，则变成 0,1,8,9,16,17… 这种交织地址。
//      thr_to_offset: 线程偏移。线程编号 → 线性偏移，简单 1-D 排列。
//      tv_to_offset: logical_product 先把 值偏移 与 线程偏移 拼成二维。再 select<1,0> 把 值维放前面，
//                    线程维放后面 → 保证 同一线程的所有值在地址上是相邻的。
//      layout_atom: 最后composition(tv_to_offset, mk_layout_mma_trgt)，把 “(m,k) → (thr,val)” 与 “(thr,val) → offset” 复合，
//                   得到 “(m,k) → offset” 的最终重排函数。
//  layout_atom里包含了(m,k) 逻辑坐标 -> 线性偏移的映射关系，基于该关系在reorder_tensor中进行拷贝即可。
//
// <NT> sub bank冲突 与 bank 冲突
//   回顾bank 冲突定义：一个 32-bank 共享内存，bank宽度 4 字节 (128 bit), 如果 warp 的 32 个线程同时访问 
//                   同一个 bank 的不同地址，就会触发 bank conflict，事务被串行化。
//   随着低精度（8-bit、4-bit）在 mixed-dtype GEMM 中大量出现，一个线程往往只读 1 字节甚至 0.5 字节。
// 因此从 Ampere 开始把 每个 4-byte bank 拆成 4 个 1-byte sub-bank（Hopper 也有 2-byte sub-bank 模式）。
// 如：
//   | 访问宽度 | 每 bank 拆成 | 冲突条件                |
//   | ------- | ------------ | ---------------------- |
//   | 32-bit  |    1 × 4 B   | 不同线程访问 同 bank 不同 4 B 地址   |  即是传统意义上的bank冲突
//   | 16-bit  |    2 × 2 B   | 不同线程访问 同 bank 同 2 B sub-bank |  
//   | 8-bit   |    4 × 1 B   | 不同线程访问 同 bank 同 1 B sub-bank | 
//
//   避免冲突：1）让同一 warp 的线程访问不同 bank：一个warp不同线程访问同一个 bank 的不同地址，会造成bank冲突。
//            2）让相邻线程访问不同 sub-bank：如每线程读取0.5B，两个0.5B处于同一个subbank，两个线程同时访问同一个subbank，即会存在subbank冲突。
//                                   对数据做shuffle，使其交错，即可避免subbank冲突

// Given a type of MMA instruction, compute a memory reordering atom that places all values
// owned by each thread in contiguous memory locations. This improves smem load vectorization,
// particularly for mixed dtype GEMMs where a narrow type is loaded in the thread/value order
// of the wider type and may result in inefficient sub-bank (8-bit or 16-bit) accesses.
// In addition, we can reorder the values across several MMA instructions to get even wider
// vectorization (AtomLayout parameter) and permute the values within each instruction to get
// more optimal conversion instruction sequences (ValLayout parameter).
template <class ElementMma,
         class AtomLayout = cute::Layout<cute::_1>,
         class ValLayout  = cute::Layout<cute::_1>>
constexpr auto compute_memory_reordering_atom(AtomLayout atom_layout = {}, ValLayout val_layout = {})
{
  using namespace cute;

  static_assert(is_static_v<ValLayout>, "ValLayout must be static");
  static_assert(is_static_v<AtomLayout>, "AtomLayout must be static");

  // 1. Choose an MMA atom to access TV layout and MN shape
  // Note: parameters like GMMA Major, TileShape, ElementC don't affect TV layout of A, use arbitrary
  using MmaAtom = decltype(SM90::GMMA::rs_op_selector<ElementMma, ElementMma, float, Shape<_64,_16,_32>>());
  using MmaTraits = MMA_Traits<MmaAtom>;
  auto mk_shape_mma = select<0,2>(typename MmaTraits::Shape_MNK{});
  auto tv_layout_mma = typename MmaTraits::ALayout{};
  static_assert(size<1>(tv_layout_mma) % size(val_layout) == 0, "Value layout must evenly divide the MMA value layout");

  // 2. Create a single warp's TV layout from that of the whole MMA and invert to get (m,k -> thr,val)
  // Note: this assumes A is partitioned between warps along M mode
  auto tv_tiler_warp = make_shape(Int<32>{}, size<1>(tv_layout_mma));
  auto mk_shape_warp = shape_div(mk_shape_mma, size(typename MmaTraits::ThrID{}) / Int<32>{});
  auto tv_layout_mma_warp = make_layout_like(composition(tv_layout_mma, tv_tiler_warp));
  auto mk_layout_mma_warp = right_inverse(tv_layout_mma_warp).with_shape(mk_shape_warp);

  // 3. Repeat the warp layout NumAtoms times along K mode to get wider vectorization
  auto mk_layout_mma_trgt = blocked_product(mk_layout_mma_warp, atom_layout);

  // 4. Compose with a contiguous layout of values in each thread (required for smem vectorization)
  auto val_to_offset = logical_product(val_layout, size<1>(tv_layout_mma) / size(val_layout) * size(atom_layout));
  auto thr_to_offset = make_layout(size<0>(tv_layout_mma_warp));
  auto tv_to_offset = select<1,0>(logical_product(val_to_offset, thr_to_offset));
  auto layout_atom = composition(tv_to_offset, mk_layout_mma_trgt);

  return layout_atom;
}

// <NT> reorder_tensor / reorder_tensor_kernel 介绍，如在hopper中的w4a16实现里，对w4做offline shuffle时用到。
// 1) 做shuffle的原因：
// -- 对于float16或更高位宽的数据, ldmatrix 可以在硬件中高效完成数据shuffle,。但对于4bit并没有，无法做硬件shuffle，
//    则需要 四次 8-bit 共享内存加载才能拼出一个符合 tensor core 布局的 32-bit 寄存器块。
// 2) w4a16中的使用方式：
//      using MmaType = cutlass::bfloat16_t;
//      using ElementB = cutlass::int4b_t;
//      using LayoutB = cutlass::layout::ColumnMajor;
//      using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
//      auto shape_B = cute::make_shape(n, k, l);
//      StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
//      auto layout_B = make_layout(shape_B, stride_B);
//
//      using ValueShuffle = cutlass::Layout<cutlass::Shape<cute::_2,cute::_4>, cutlass::Stride<cute::_4,cute::_1>>; // order [0,2,4,6,1,3,5,7]
//      static constexpr int NumShuffleAtoms = 1;
//      using MmaAtomShape = cutlass::Layout<cutlass::Shape<cute::_1,cute::Int<NumShuffleAtoms>>>;
//      using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
//      using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, cutlass::Layout<cutlass::Shape<int,int,int>, StrideB>{}));
// 
//      LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
//      cutlass::reorder_tensor((ElementB *)weight.data_ptr(), layout_B, layout_B_reordered);
//    关键函数是 compute_memory_reordering_atom 得到 (m,n)->offset关系 和 reorder_tensor 基于映射关系进行数据拷贝。

template <class TileShape, class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst, class TiledCopy>
__global__ void reorder_tensor_kernel(
  cute::Tensor<EngineSrc, LayoutSrc> S,
  cute::Tensor<EngineDst, LayoutDst> D,
  TiledCopy tiled_copy)
{
  using namespace cute;

  using T = typename EngineDst::value_type;

  Tensor gS = local_tile(S, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));
  Tensor gD = local_tile(D, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));

  auto thread_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor tS = thread_copy.partition_S(gS);
  Tensor tD = thread_copy.partition_D(gD);

  copy(tiled_copy, tS, tD);
}

template <class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst>
void reorder_tensor(
  cute::Tensor<EngineSrc, LayoutSrc> S,
  cute::Tensor<EngineDst, LayoutDst> D)
{
  using namespace cute;

  using T = typename EngineDst::value_type;
  static_assert(is_same_v<remove_const_t<typename EngineSrc::value_type>, T>, "Type mismatch");

  // Construct a value layout that assigns at least 8 bits of contiguous elements in destination tensor to a thread
  // This avoids a race condition when writing out subbyte types (e.g. int4b_t).
  auto has_major_mode = [](auto s) {
    return any_of(flatten(s), [](auto a){ return is_constant<1, decltype(a)>{}; });
  };
  static_assert(has_major_mode(stride<0>(LayoutDst{})) ^ has_major_mode(stride<1>(LayoutDst{})),
                "Could not find stride-1 mode in destination layout");
  constexpr int N = shape_div(Int<8>{}, Int<sizeof_bits_v<T>>{});
  auto val_layout = conditional_return<has_major_mode(stride<0>(LayoutDst{}))>(
    make_layout(make_shape(Int<N>{}, Int<1>{}), GenColMajor{}),
    make_layout(make_shape(Int<1>{}, Int<N>{}), GenRowMajor{}));

  // Make a tiled copy with a simple row-major thread order and above layout
  int constexpr NumThreads = 128;
  auto const thr_layout = make_layout(make_shape(Int<1>{}, Int<NumThreads>{}));
  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, T>{}, thr_layout, val_layout);

  // Assign a group of 16 rows to a threadblock; this matches the shuffle atom size for Hopper
  using TileShape = Shape<_16>;
  auto tiled_D = group_modes<3,rank_v<LayoutDst>>(tiled_divide(D, TileShape{}));
  dim3 blocks{unsigned(size<1>(tiled_D)), 1u, unsigned(size<3>(tiled_D))};

  reorder_tensor_kernel<TileShape><<<blocks, NumThreads>>>(S, D, tiled_copy);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// In-place version
template <class T, class LayoutSrc, class LayoutDst>
void reorder_tensor(
  T const* src,
  LayoutSrc const& layout_src,
  T * dst,
  LayoutDst const& layout_dst)
{
  using namespace cute;
  reorder_tensor(make_tensor(make_gmem_ptr<T>(src), layout_src),
                 make_tensor(make_gmem_ptr<T>(dst), layout_dst));
}

// In-place version
template <class T, class LayoutSrc, class LayoutDst>
void reorder_tensor(
  T * data,
  LayoutSrc const& layout_src,
  LayoutDst const& layout_dst)
{
  using namespace cute;
  cutlass::DeviceAllocation<T> temp(size(layout_src));
  reorder_tensor(data, layout_src, temp.get(), layout_dst);
  cutlass::device_memory::copy_device_to_device(data, temp.get(), static_cast<size_t>(size(layout_src)));
}

#undef CUDA_CHECK

}  // namespace cutlass
