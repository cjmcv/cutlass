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

// <NT> ��Ӧ�����������һ�� Threadblock �� CTA ӳ����� ���� GemmIdentityThreadblockSwizzle<N> (identity swizzle)��
// 1. �� �߼� tile ����M��N��K_split�� ����� CUDA grid �ߴ�(blockIdx.x, blockIdx.y, blockIdx.z)��
// 2. �� blockIdx.{x,y,z} �ٷ���ӳ��� �߼� tile ���� (m,n,k)�����м���԰�ģ����� N ��һ�� 2^N �׵� x-y ת��/swizzle��
// Ŀ������ ͬһ�У�m ���򣩵� tile �������ϲ�Ҫ������ص����� SM���Ӷ� ���� L2 ��ͻ����� L2 �����ʣ�ͬʱ �����㿪������λ���㣩��
//
// ������1��L2������ SM �ⲿ������ SM �����һ����Ƭ�� SRAM��
//         �ڲ����һ���з֣��繲40MB��L2�ᱻ�г� 1024+ �� bank(оƬ���ƣ��û����ɼ�)��ÿ�� bank �ٷ� 64�C256 �� set��
//         L2��bank��shared memory��bank����������ͬ��������ȫ��ͬ��smem��bank�̶�Ϊ32������ SM �ڲ����� L1 ͬһ������ RAM��
//       2��L2�е�set����С��16-way �� 128 B/line �� 2 kiB�������һ��wave�Ķ��blockͬʱ����һ��set���ᵼ������Ƶ�������ɣ�
//         ������Ҫ�ﵽ��Ŀ���ǣ���ͬһ wave �ﲢ�л�Ծ�� tile ��ַ��ɢ����ͬ set����ÿ�� line ���л����� L2 �����������д�꣬�Ӷ�����д�� DRAM �Ĵ�������ס�������������
//       3���������swizzle��һ�� wave �� block �� n ���������̿� ʱ������ tile �ĵ�ַֻ�� tile_n * sizeof(float)������ 128��4 B = 512 B����
//         ��ampereΪ������ַλ�� [6:0] 128 B line ���ֽ�ƫ�ƣ�
//                              [11:7] bank ������32-64 ��)������Ȼbank����ֻ��32-64������ʵ��bank������1024����ֻ���û����ɼ�����
//                             [17:12] set ������64 ��), 
//                              [X:18] tag. 
//         ���Ե����� tile�ĵ�ַ�����512B��������ֻȥ���˵� 9 λ���� 12 λ���¶�û��λ������ [17:12] ���� set λ��˿���� �� set_id ��ȫ��ͬ��
//         ������ 16 �� n-tile ������16 �� line ȫ����ϣ�� set-0��16-way �����ȸպñ�һ����ռ������ 17 �� tile һ���� ���ߵ�һ����thrashing������/���������� ��ʼ��
//
// swizzle��������
//   �� blockIdx.x �ĸ� 2 λ�� blockIdx.y �ĵ� 2 λ�������൱�� �� n �ġ���λ��Ų�� m �ġ���λ�������� ���� n �Ų�����������ַ��set λ�� XOR ���ң�
// ͬһ wave �� tile ���ڲ�ͬ set����ͻ��ʧ��

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

  // <NT> �� �߼� tile ���� ��� CUDA gridDim
  // tile��Ӧ8/4/2/1��kȡԭֵ���޸ģ�����4Ϊ����ԭ����shape[m,n] => [m*4, (n+3)/4]����������tile������������tile����
  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  // <NT> ���� ģ������ N �� n ���� tile �� ��̬ѡ ʵ��ת�ý���. �ֱ��Ӧ8x8, 4x4, 2x2 �� ������
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

  // <NT> get_tile_offset��(blockIdx.x, blockIdx.y, blockIdx.z)ͨ��λ���㷴�������߼�tile����(m,n,k)
  // ���Ķ����� �� x-y ƽ���� 2^log_tile ��ת�ã�swizzle������ ���� blockIdx.x ��Ӧ�� n ���겻��������
  // �Ӷ� ��ɢ L2 set ��ͻ��
  // 
  // ���ӣ�ԭʼblockIdx(x,y)������
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
  // ��log_tileΪ3����Ӧ8x8��tile���߼�����ӳ�����£��õ�һ��64�еľ���
  // ��ԭ��8x8�ľ��󣬰�8x8��tile��swizzle����������;�������ԭ�������16x16���Ϳ��Կ�����
  // һ������16��block�ᱻ��ֵ����У���8��block���8������block��Ż�Ӻ�8��block��
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
  // ��log_tileΪ2����Ӧ4x4��tile���߼�����ӳ�����£�ԭ������������8+8��block, ��תΪ��4x4��tile��
  // ��ͷ����4��block����Ҫ����4���ż�������δ��swizzleǰ�ĺ�4��block��8x8�����4��4x4����
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

