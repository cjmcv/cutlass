/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      GEMM problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/gemm/threadblock/index_remat.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

// <NT> 对应最“轻量级”的一种 Threadblock → CTA 映射策略 —— GemmIdentityThreadblockSwizzle<N> (identity swizzle)，
// 1. 把 逻辑 tile 网格（M×N×K_split） 翻译成 CUDA grid 尺寸(blockIdx.x, blockIdx.y, blockIdx.z)；
// 2. 把 blockIdx.{x,y,z} 再反向映射回 逻辑 tile 坐标 (m,n,k)，但中间可以按模板参数 N 做一点 2^N 阶的 x-y 转置/swizzle，
// 目的是让 同一行（m 方向）的 tile 在物理上不要连续落地到相邻 SM，从而 缓解 L2 冲突、提高 L2 命中率，同时 保持零开销（纯位运算）。
//
// 背景：1）L2处于在 SM 外部，所有 SM 共享的一整块片上 SRAM。
//         内部会进一步切分，如共40MB的L2会被切成 1024+ 个 bank(芯片定制，用户不可见)，每个 bank 再分 64–256 个 set。
//         L2的bank和shared memory的bank仅是名字相同，用意完全不同。smem的bank固定为32个，在 SM 内部，与 L1 同一块物理 RAM。
//       2）L2中的set容量小（16-way × 128 B/line ≈ 2 kiB），如果一个wave的多个block同时命中一个set，会导致数据频繁被顶飞；
//         所以需要达到的目的是：把同一 wave 里并行活跃的 tile 地址打散到不同 set，让每条 line 都有机会在 L2 里“存活”到真正被写完，从而减少写回 DRAM 的次数，保住带宽和吞吐量。
//       3）如果不做swizzle，一个 wave 的 block 在 n 方向连续铺开 时，相邻 tile 的地址只差 tile_n * sizeof(float)（例如 128×4 B = 512 B）。
//         以ampere为例，地址位： [6:0] 128 B line 内字节偏移，
//                              [11:7] bank 索引（32-64 个)，（虽然bank索引只有32-64个，而实际bank数量有1024个，只是用户不可见）。
//                             [17:12] set 索引（64 个), 
//                              [X:18] tag. 
//         所以当相邻 tile的地址跨度在512B，二进制只去到了第 9 位，第 12 位以下都没进位，所以 [17:12] 这组 set 位纹丝不动 → set_id 完全相同。
//         以连续 16 个 n-tile 举例：16 条 line 全部哈希到 set-0，16-way 关联度刚好被一次性占满；第 17 个 tile 一来就 必踢掉一条，thrashing（抖动/换进换出） 开始。
//
// swizzle的做法：
//   把 blockIdx.x 的高 2 位和 blockIdx.y 的低 2 位交换，相当于 把 n 的“高位”挪到 m 的“低位”，于是 连续 n 号不再是连续地址，set 位被 XOR 打乱，
// 同一 wave 的 tile 落在不同 set，冲突消失。

/// Threadblock swizzling function for GEMMs
template <int N = 1>
struct GemmIdentityThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv3d problem size: conv_operator(NZPQK, NDHWC, KTRSC)
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv3dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  // <NT> 把 逻辑 tile 网格 变成 CUDA gridDim
  // tile对应8/4/2/1，k取原值不修改，则以4为例，原来的shape[m,n] => [m*4, (n+3)/4]，行数增加tile倍，列数减少tile倍。
  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  // <NT> 根据 模板上限 N 和 n 方向 tile 数 动态选 实际转置阶数. 分别对应8x8, 4x4, 2x2 和 不做。
  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  // <NT> get_tile_offset将(blockIdx.x, blockIdx.y, blockIdx.z)通过位运算反向解算成逻辑tile坐标(m,n,k)
  // 核心动作是 把 x-y 平面做 2^log_tile 阶转置（swizzle），让 连续 blockIdx.x 对应的 n 坐标不再连续，
  // 从而 打散 L2 set 冲突。
  // 
  // 例子：原始blockIdx(x,y)，如下
  // y\x |   0      1      2      3      4      5      6      7
  // ----|----------------------------------------------------------
  //  0  | (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  //  1  | (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  //  2  | (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  //  3  | (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
  //  4  | (4,0)  (4,1)  (4,2)  (4,3)  (4,4)  (4,5)  (4,6)  (4,7)
  //  5  | (5,0)  (5,1)  (5,2)  (5,3)  (5,4)  (5,5)  (5,6)  (5,7)
  //  6  | (6,0)  (6,1)  (6,2)  (6,3)  (6,4)  (6,5)  (6,6)  (6,7)
  //  7  | (7,0)  (7,1)  (7,2)  (7,3)  (7,4)  (7,5)  (7,6)  (7,7)
  //
  // 当log_tile为3，对应8x8的tile，逻辑坐标映射如下，得到一行64列的矩阵：
  // 从原先8x8的矩阵，按8x8的tile做swizzle看不出来用途，而如果原来矩阵的16x16，就可以看到，
  // 一行连续16个block会被拆分到两行，首8个block后接8个其他block后才会接后8个block。
  // y\x |   0       1      2      3      4      5      6      7
  // ----|----------------------------------------------------------
  //  0  | (0,0)   (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  //  1  | (0,8)   (0,9) (0,10) (0,11) (0,12) (0,13) (0,14) (0,15)
  //  2  | (0,16) (0,17) (0,18) (0,19) (0,20) (0,21) (0,22) (0,23)
  //  3  | (0,24) (0,25) (0,26) (0,27) (0,28) (0,29) (0,30) (0,31)
  //  4	 | (0,32)	(0,33) (0,34)	(0,35) (0,36)	(0,37) (0,38)	(0,39)
  //  5	 | (0,40)	(0,41) (0,42)	(0,43) (0,44)	(0,45) (0,46)	(0,47)
  //  6	 | (0,48)	(0,49) (0,50)	(0,51) (0,52)	(0,53) (0,54)	(0,55)
  //  7  | (0,56)	(0,57) (0,58)	(0,59) (0,60)	(0,61) (0,62)	(0,63)
  //
  // 当log_tile为2，对应4x4的tile，逻辑坐标映射如下，原本两行连续的8+8个block, 被转为了4x4的tile。
  // 开头连续4个block后，需要跳过4个才继续接上未做swizzle前的后4个block。8x8变成了4个4x4矩阵。
  // y\x |    0	     1	    2	     3	    4	     5	    6	     7
  // ----|----------------------------------------------------------
  //  0	 |  (0,0)	 (0,1)  (0,2)  (0,3)  (1,0)  (1,1)  (1,2)  (1,3)
  //  1	 |  (0,4)	 (0,5)  (0,6)  (0,7)  (1,4)  (1,5)  (1,6)  (1,7)
  //  2	 |  (0,8)	 (0,9) (0,10) (0,11)  (1,8)  (1,9) (1,10) (1,11)
  //  3	 | (0,12)	(0,13) (0,14) (0,15) (1,12) (1,13) (1,14) (1,15)
  //  4	 |  (2,0)	 (2,1)  (2,2)  (2,3)  (3,0)  (3,1)  (3,2)  (3,3)
  //  5	 |  (2,4)	 (2,5)  (2,6)  (2,7)  (3,4)  (3,5)  (3,6)  (3,7)
  //  6	 |  (2,8)  (2,9) (2,10) (2,11)  (3,8)  (3,9) (3,10) (3,11)
  //  7	 | (2,12)	(2,13) (2,14) (2,15) (3,12) (3,13) (3,14) (3,15)
  // 

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
struct GemmHorizontalThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMMs
struct GemmBatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int batch_count) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      batch_count % (1 << 16));
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxX(),
      RematerializeBlockIdxY(),
      RematerializeBlockIdxZ()
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Gets the batch index
  CUTLASS_DEVICE
  static int get_batch_idx() {
    return RematerializeBlockIdxZ();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
template <int N = 1>
struct GemmSplitKIdentityThreadblockSwizzle {

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
struct GemmSplitKHorizontalThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(int log_tile) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
    return GemmCoord{
      RematerializeBlockIdxY(),
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMVs
struct GemvBatchedStridedThreadblockDefaultSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static BatchedGemmCoord get_tiled_shape(
    BatchedGemmCoord problem_size,
    BatchedGemmCoord tile_size) {

    return BatchedGemmCoord(
      1, // M is always 1
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      (problem_size.k() + tile_size.k() - 1) / tile_size.k(),
      (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(BatchedGemmCoord tiled_shape) {
    return dim3(tiled_shape.n(), tiled_shape.batch(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord tiled_shape) {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static BatchedGemmCoord get_tile_offset(int log_tile) {
    return BatchedGemmCoord{
      0, // M is always 1
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ(),
      RematerializeBlockIdxY(),
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static BatchedGemmCoord get_tile_offset() {
    return BatchedGemmCoord{
      0, // M is always 1
      RematerializeBlockIdxX(),
      RematerializeBlockIdxZ(),
      RematerializeBlockIdxY(),
    };
  }

  /// Gets the batch tile index
  CUTLASS_DEVICE
  static int get_batch_tile_idx() {
    return RematerializeBlockIdxY();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  static int get_batch_idx() {
    return RematerializeBlockDimY()*RematerializeBlockIdxY() + RematerializeThreadIdxY();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

