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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm80CpAsyncUnpredicated<Stages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_
  >
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm80CpAsyncUnpredicated<Stages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  // Follow the change in TestSmall: TileShape switch to CtaShape
  // For sm80 arch, CtaShape should equal to TileShape
  using CtaShape_MNK = TileShape;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value,    "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value,    "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");
    static_assert(cute::rank(SmemLayoutB{}) == 3,
      "MainloopSm80CpAsync must have a pipeline mode in the smem layout.");

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    (void) residue_mnk;
    //assert(residue_mnk == make_tuple(0,0,0));

    //
    // PREFETCH
    //

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
      copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K
    CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_A) == size(tiled_mma));
    CUTE_STATIC_ASSERT_V(size(gmem_tiled_copy_B) == size(tiled_mma));

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(DispatchPolicy::Stages-1))
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
          copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
          cp_async_fence();

          // Advance the tile
          --k_tile_count;
          if (k_tile_count > 0) { ++k_tile_iter; }

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});
        // Thread-level register gemm for k_block
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });

    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_
>
struct CollectiveMma<
    MainloopSm80CpAsync<
      Stages,
      ClusterShape_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_
   >
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm80CpAsync<
                          Stages,
                          ClusterShape_>;
  using TileShape = TileShape_;
  // Follow the change in TestSmall: TileShape switch to CtaShape
  // In legacy arch, it should be same
  using CtaShape_MNK = TileShape;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  // <NT> ���ô����ɲο������kernel�㣬��: include/cutlass/gemm/kernel/sm70_gemm.hpp
  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,                   // (BLK_M, BLK_K, K_TILES)
      TensorB gB,                   // (BLK_N, BLK_K, K_TILES)
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value,    "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value,    "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

    // <NT> smem_bufӦ�����洴����SharedStorage���͵�buffer�������������cute::array_aligned���ڴ�飬����Ӧsmem��
    // ʹ��make_smem_ptr��һ��ָ�������ת��������layoutΪA��B����ֱ𹹽������ڴ��Tensor��
    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // <NT> residue_mnk��δ�ܱ�tile������mnk������get<2>(residue_mnk)��ȡ��residue_k����ȡֵ��Χ��[0, BLK_K).
    // ��make_coord(0, residue_k, 0)������ƫ������
    // ��gAά����[BLK_M, BLK_K, K_TILES], BLK_K��ʾһ�� tile �ڲ���С K ά��K_TILES��һ�����ٸ� K tile��
    // ��make_coord(0, get<2>(residue_mnk), 0)����ȡ�����м�� BLK_K��
    // ����ִ��cute::domain_offset�󣬷���gA(m, k_in_tile, k_tile)���൱��ԭ����gA(m, k_in_tile + residue_k, k_tile)
    // k_in_tile ��ȡֵ��Χ�� [0, BLK_K) ����� [-residue_k, BLK_K - residue_k).
    // ����Ч������[0,residue_k) ����� [-residue_k, 0), ��[0,BLK_K - residue_k)������Ч���ݡ�
    // ����ʱ�Ϳ��Ը���������blockһ��������ͳһ��(>= -residue_k)��Ϊ�������й��˼��ɣ�
    // �߽�block��residue_kΪ�������Ǳ߽�block��residue_kΪ0���ж��߼�һ�¡�
    // ��: ��һ�θ�����ƽ�ƻ�һ����ȫ�޷�֧���������֧��ͳһѭ��.
    // 
    // ����: ��Χ����[-residue_k, BLK_K-residue_k), �Ǳ߽��residue_k��0���߽�block��residue_k������������Ч��Χ��[-residue_k, 0)
    //       ��ʱ��ʼ������һ���ˣ����ǽ���������һ�£��߽�block��[0, BLK_K-residue_k)��������Ч���ݣ���ô����
    //  ��: <NT-TODO>
    //      1�������Ǳ߽�block��ƫ�ƺ����겻���ٳ��ִ���0������������� if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) ������ʼ�㼴�ɣ���������Ȼ��Ȼ�ı��0.
    //      2����������predicate��tApA(_,k)�д�k���Ǹ�ϸ���ȵ�Ԫ�ؼ����ˣ�����kά�ȵ�����û���������������ˡ�
    // 
    // ע�⣺gA[BLK_M, BLK_K, K_TILES]��ֻ��һ��block�����ݣ���������Ǳ߽��ϵ�block��residue_mnk������0, �������κ�ƫ�ơ�
    //       �����ע�ͼ��趼�ǻ��ڸ�block�Ǵ��ڱ߽��ϵ�block��
    // 
    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gA);
    gB = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gB);

    // <NT> GmemTiledCopyAͨ����make_tiled_copy������get_slice������include/cute/atom/copy_atom.hpp
    // ���ڻ�ȡ�߳���������Ƭ�������ThrCopy������TiledCopy�õ�ThrCopy��ThrCopy��һ���̰߳󶨡�
    // ע: mma atom��Ҳ��get_slice��������;����һ�£������̼߳���
    //     һ������copyһ������mma (include/cute/atom/mma_atom.hpp)
    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    // <NT> ThrCopy���ͣ���ȡ���̸߳���Ķ�ȡ��srcƬ�κ�д���dstƬ��
    // tAgA[ACPY,ACPY_M,ACPY_K,k], 
    //     ����ACPY��ʾ�̸߳���ID��ÿ���߳��õ��Լ��Ƿ����ݣ�ÿ���̷߳���tAgAʱ����ACPYֵ����ͬ����ֱ��Ӧ0~ACPY����
    //     ACPY_M, ACPY_K	��ʾ��ǰ�̸߳��𿽱��� M��K��tile��С��
    //     ��Ϊ gA[BLK_M, BLK_K, K_TILES], ��K�з֣�����tAgA����kά�ȡ�
    // tAsA[ACPY,ACPY_M,ACPY_K,PIPE]
    //     ���е�PIPEͬ��, ��sA[BLK_M,BLK_K,PIPE]
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    // <NT> ����ν��tensor�����ڴ���m��nδ�ܱ������ı߽硣
    // size<1>(tAsA), size<2>(tAsA) => ACPY_M, ACPY_K��
    // ��Stride<_1,_0>��0�Ǵ��Ƶ��Ķ�̬����ռλ����������ʱ�ᱻ�滻�� ʵ�ʼ���õ��Ĳ���ֵ��
    // ���� auto layout = make_layout(make_shape(4,8), make_stride(8,1));
    //      auto layout2 = Layout<Shape<_4,_8>, Stride<_8,_0>>{};
    //      �ڱ�������� _0 �滻�� 1������ layout �� layout2 ������һ���ġ�
    // tApA������һ��ά��Ϊ[ACPY_M, ACPY_K]��bool��tensor����tAgA���ڲ���tile��Сһ�¡�
    // tApA����tAgA��tileһһ��Ӧ����ʾ�����Ԫ���Ƿ���Ҫ������
    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // <NT> Ϊtensor sA ��shape[M, K]����һ����shape�ڵ�����ӳ�䵽�����������cA, ��Ԫ��ֵ��������Ķ�ά�߼�������
    // �� Tensor cA = make_identity_tensor(make_shape(3, 4));
    // cA�������ǣ�
    //   (0,0) (0,1) (0,2) (0,3)
    //   (1,0) (1,1) (1,2) (1,3)
    //   (2,0) (2,1) (2,2) (2,3)
    // ����֮��make_identity_tensor(make_shape(M, K)) ������һ�� �����꾵�ӡ���ÿ��λ�õ�ֵ�������Լ��� (m, k) ���꣬����������ӳ��͵��ԡ�
    // ������;��
    //  1. ��������ӳ������ں�������任��scatter/gather������������ȡ�
    //  2. ���Դ�ӡ������ print_tensor(cA) ֱ�ӿ���ÿ��λ�õ��������ꡣ
    //  3. ��� transform/apply ������任������� (m, k) ӳ�䵽�����ڴ�ƫ�ơ�
    //
    // �������sA��sB��shape������������cA/cB��
    // Ȼ�������������cA/cB��ȥ��ȡ���̸߳���Ķ�ȡ��src fragment����������tAcA/tBcB��
    // ��������tAcA/tBcB������������tAgA/tBgB���Ӧ����������ʱ��Ҫ�õ�����������
    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_A.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_B.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // <NT> ����fragment���������������ν�����롣
    // tApA[ACPY_M, ACPY_K], ���m�ı߽磬ֻ��Ҫ���mά�ȼ��ɡ�
    // residue_mnk[m,n,k]����get<0>ȡ��residue_m; 
    // tAcA[ACPY,ACPY_M,ACPY_K]��Ԫ�������꣬��tAcA(0,m,0)ȡ��ÿһ��mά���ϵ����꣬
    //     ������tuple<int,int> �� Coord<2>������get<0>ȡ�������꣬��Ӧ��������sA��global_m.
    // ���ʱ��tApA(m,0)��kά�ȹ̶�Ϊ0��ʹ��ʱ����й㲥������(m,0)ΪTrue�ģ�(m,0~k)����ΪTrue��
    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    //
    // PREFETCH
    //

    // <NT> �������Ǵ�gmem��smem��Ŀ��tensor�����漰���߽磬��Ҫ������ٿ���������߽��������쳣����
    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);

    // <NT> k_tile_iter�Լ�1����ȡ����һ��k_tile, k_tile_count��¼ʣ����ٸ�k_tile��ȡ��һ��k_tile��Ӧһ��block��
    // tAgA[ACPY,ACPY_M,ACPY_K,k] => tAgAk[ACPY,ACPY_M,ACPY_K]
    // tAsA[ACPY,ACPY_M,ACPY_K,PIPE], size<2>(tAsA)����ACPY_K��
    // ���Կ����� k ��0��ACPY_K�� �� tAgAk[ACPY,ACPY_M,0~ACPY_K] => tAsA[ACPY,ACPY_M,0~ACPY_K,PIPE]
    // for ѭ��kʱ����ҪtAcA[ACPY,ACPY_M,ACPY_K]�е�k�������ĺ������꣬��kά��������Ҫ����-residue_k��
    // ��ΪgAʹ�ù�cute::domain_offset������ƫ����residue_k���߽�block��k��Χ��[0, residue_k) ��Ϊ [-residue_k, 0)��
    // ����get<1>(tAcA(0,0,k)��Ҫ����-residue_k��
    //
    // ��һ��tile��Ϊ�б߽�residue_k���ڣ���Ҫ�����ó����������ˡ�
    // �����ڴ�����tAsA(_,_,k,k_pipe)���һ��ά��k_pipe��ÿ��һ��k_pipe���ڴ�ͻ��һ�ݣ��ö���ڴ�ռ�ȥ���������������µ�ʱ�ӡ�
    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      constexpr int k_pipe = 0;

      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      // <NT> ����k_in_tile��ά��
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tAsA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_A, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          copy_if(gmem_tiled_copy_B, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
        }
      }
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    // <NT> �������˵�һ��tile���������tile, һ��block��kά�ȿ��ܻỮ�ֶ��k_tile,
    // ���k_tile����һ�����������࿽��Stages-2���ޱ߽��������k_tile.
    // ���û��������k_tile, k_tile_count�ήΪ0��ν�������ᱻ��գ�copy_if�ᱻֱ��������
    // �����Stages�����ܸ���k_tile_count���򿽱�һ�ֺ�����k_tile�����洢��smem���ˣ��������в���k_tileδ����ȡ����ʱk_tile_count>0��
    // ���⣺Ϊʲô���ϵ�һ��tile�����k_pipe=0��ܹ�ֻ��ȡStages-1��tile��������Stages����
    //   ��Ԥ������һ��������ѭ���ĵ�һ�ε�������smem_pipe_write�ĳ�ʼֵ������Stages-1��
    //      ��Ԥ���أ���� 1 �� Stages-2; ��ѭ����һ�Σ���� Stages-1
    //      ��ˣ�smem_pipe_read���Դ�0��ʼ����smem_pipe_write�����readһ��stage������ȡ�����һ�ݵ�Stages-1��
    //      ��read��write�ľ�����󣬵ȴ�ʱ�ӿ��Ը��õ����ء�
    // Start async loads for 1st k-tile onwards, no k-residue handling needed
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 1; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count <= 0) {
        clear(tApA);
        clear(tBpB);
      }
      copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));  // CpAsync
      copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));  // CpAsync
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // <NT> sA[BLK_M,BLK_K,PIPE], ��PIPEά�ȹ̶�Ϊ0�󣬰��߳̽�����Ƭ��
    // �õ���tCrA��ά����[MMA,MMA_M,MMA_K], ����ע����"<NT> partition_shape_A".
    // ����: MMA: ��ǰ�߳��� MMA tile �е��߳� ID ά, ͬcopy atom�е�ACPY����Ӧ��mmaָ���shape��
    //       MMA_M��Mά���ϣ�һ��block��������ݵ�mmaָ����ظ����ô���
    //       MMA_K��Kά���ϣ�mmaָ����ظ����ô���
    // ���ӣ�mma.sync.aligned.m16n8k32��һ�����m16n8k16�����ݼ��㣬���16x8��һ����Ҫһ��warp32���̲߳��롣
    //      ����cute::partition_shape_A(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{}))  => BLK_M=32, BLK_K=128
    //      ��mma using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
    //            using PermutationMNK = Tile<Int<1>, Int<1>, Int<1>>
    //      �� MMA_M = 32/16 = 2�� MMA_K = 128/32 = 4
    // ��Ϊ�зֳ�����fragment����û�����ݣ�����״ֻ�� tile ��[BLK_M, BLK_K]�йأ�
    // �������õ� PIPE ��λ�޹أ������ȹ̶� PIPE = 0��������һ�ۣ�����.
    // ��ʹ��ʱ���Ӳ�ͬ��pipe�п������ݵ����fragment����м��㡣
    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_,0));                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_,0));                    // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //
    // <NT> ǰ���GmemTiledCopyA���ڴ�gmem->smem, ����Ŀ�����smem->rmem.
    // GmemTiledCopyAͨ��Ҳ��ͨ��make_tiled_copy�����������õ���cp.asyncָ��磺
    //   using GmemTiledCopyA = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementInput>{},
    //                          Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{}));
    // SmemCopyAtomA���smem->rmem��ͨ���õ���ldmatrix����
    //   using SmemCopyAtomLoad = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
    //
    // smem_tiled_copy_A��tile����copy���󣬽��������̻߳�ȡ�̼߳����copy����
    // ��smem��sA��ȡ��srcƬ��tCsA���ӼĴ�����tCrAȡ��dstƬ�Ρ�
    // 
    // �ʣ�Ϊʲôȡ�Ĵ���Ƭ����retile_D������partition_D��
    // ��partition_D�������shared memory tensor���� shared memory tile �гɡ�ÿ���̸߳����һ�ݡ���
    //     ��	sA[BLK_M,BLK_K,PIPE] => tCsA[CPY,CPY_M,CPY_K,PIPE].
    //    ��retile_D�������register fragment���ѼĴ��� fragment �����п飬ʹ����״�� copy tile һһ��Ӧ��
    //     �� tCrA[MMA,MMA_M,MMA_K]	=> tCrA_copy_view[CPY,CPY_M,CPY_K]
    //    ��Ϊ tCrA �Ѿ��� MMA �̲߳����г� fragment���ǿ���ֱ�ӷ��뵽mmaָ���м���Ĳ��֣�������tCrA�Ĳ��֣��ǱȽ�����Ĳ��ַ�ʽ����Ҫת�����ܿ�������
    //    �� smem<->reg ��copyʹ�ò�ͬ���̲߳��֣�CPY �� MMA������Ҫ�ٰ� copy �̲߳��� ���»���һ�Ρ��������� retile_D �����¡�
    auto smem_tiled_copy_A   = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA           = smem_thr_copy_A.partition_S(sA);                   // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                    // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB              = smem_thr_copy_B.partition_S(sB);                // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);                 // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // <NT> smem_pipe_write ����� smem_pipe_read һ��stage��ʹ�������
    // smem_pipe_writeָ�����һ��stage����prefetch�У����һ��stage��û��д�룬������ѭ��д��������ʼǡ�
    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // <NT> size<2>(tCrA) �� MMA_K����Ӧ��block��K������Ҫ�ظ�����mma�Ĵ�����
    // �����k�����mma���ð�pipeline���д��ӣ��Դﵽ�Ĵ���Ԥȡ��Ч����
    // ���Ը�multi-stageʵ���������smem pipeline �� reg pipeline��
    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // <NT> �����block��k�������һ��mma�ͽ����ˣ��͹�����reg��pipeline��
    // 1����������ʹ��cp_async_wait�ȴ�smem pipeline�е�smem���ݵ�λ��
    // cp_async_wait<N> ��N=0����Ҫ����cp_asyncָ��ִ����ϣ���N>0, �ȴ���δ��ɵ������� �� N�������������<NT> cp_async_wait<N>��
    // ǰ��gmem->smem��Ԥȡ�����У�������A��B����������2*(stage-1)�ε�copy_if��
    // �ʣ�Ϊʲôʹ��cp_async_wait<stage-2>��ȷ����һ��Ԥȡtile��ɡ�����cp_async����һ�����ȷ�������Ƚ������ܺܶ�����Ӱ�졣
    // ��<NT-TODO> �²⣺����ʹ��ʱ���� cp.async ��Դ��ַ��Ŀ�ĵ�ַ����С��ȫһ�����롢�� bank-conflict��Ҳ�����룬��ÿһ�� cp.async ���ӳټ�����ͬ��
    // ���Ի�������ȷ���Ƚ�������ɡ�����ǰ���ܹ���2*(stage-1)�Σ�����ֻ��סstage-2������ʾǰ��Ŀ��������һ�뼴�ɣ���������£���һ��tile��������ȷ�����Ѿ�����ˡ�
    // ���ÿ������������������ͬ�����Ⱥ�˳����ܳ����ܴ�����Ҫʹ��cp.async_wait<0>��ǿ������cp_async�����ȷ����һ��tile����ɡ�
    // 2������Ԥȡ0��stage��0��k_tile���Ĵ���, ʹ��Int<0>{}���ڱ���ʱ�ػ������ڱ������Ż���
    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    // <NT> ��ѭ����k_tile_count ͨ������0����Ϊstageһ����3~4�����Խ�����ȫ���ᵽsmem�С�
    // ��ǰ��Ԥȡ��Stages-1�ݵ����ݣ�������һ�����������ѭ������ˡ�����ԭ k_tile_count ��6��stage��3��
    // ��prefetch�л����2��k_tile, ������ʱ k_tile_count Ϊ4. ѭ���𲽼���-2������ѭ��6�Σ�����Ԥȡ�˵ĺ�ûԤȡ�ġ�
    // 
    // ��� k_tile_count Ϊ0��������Ĵ�gmem->smem��Ԥȡ���ܹ���ά�ȵ��������ݰ���smem ()
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<N>.
      // <NT> make_int_sequence<K_BLOCK_MAX>������һ������ʱ����������������������һ���� 0 �� K_BLOCK_MAX-1 ���������С�
      //      for_each������һ��Ԫ��̹��ߣ���������������е�ÿ��ֵ������ÿ��ִֵ�и����� lambda ������
      //      [&] (auto k_block)������һ�����������ⲿ������ lambda ������������һ������ʱ��������k_block����ʾ��ǰ����������ֵ��
      //      �����������ڱ���ʱչ��һ��ѭ���������ɶ��ʵ�����汾�Ĵ��룬��ÿ�����������е�ִֵ����ͬ�Ĵ����߼���
      //  ������������ʱִ��ѭ������������������ʱ���������á�����k_block��ֱ�ȡ[0, K_BLOCK_MAX - 1]�е�һ��ֵ��
      //  �����ɵ�ѭ���ڴ����߼�����˳��չ���ģ���k_block=0,1,...,MMA_K-1����ִ�У���
      //
      // k_block ��Ϊ�ֿ���������ȡֵ��Χ��[0, K_BLOCK_MAX - 1]��������MMA_Kһ��(һ��block��kά�ȿɱ��зֵ�mmaָ��Ĵ���)��
      // �Զ��mmaָ���ѭ���������block��kά�ȵļ��㡣
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        // <NT> �����һ��k_block��׼����ʼ��һstage�ļ���.
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // <NT> cp_async_waitȷ������������ (<= DispatchPolicy::Stages-2) ��ָ��δ��ɡ�
          // �Ϳ��������ߣ����ȴ�ȷ��Ϊ��һ�� copy_if���� k_block == 0 ʱ�����ڳ��������ռ䡣
          // ͬʱҲ��ȷ�������һ�� copy_if ����ɡ�
          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // <NT> ����ǰ��Ԥȡ��һ��k_tile�ļ������ݵ�reg
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // <NT> ��һ��block����ʾ��k_tile�ļ���տ�ʼ�����Կ�ʼ������һ��k_tile��gmem->smemԤȡ�ˡ�
        // ����stage=4���ս������read 0�����м��㣬��ʱ��k_blockΪ0ʱ����ʼwrite 3. 
        // write��read��Զ���һ����ʹ�����󣬸�������ʱ�ӡ�
        //      ��k_tile_count <= 0 ʱ����ʾk_tile_iter�Ѿ��ܹ�һ�֣�
        // ʣ�µ�k_tile_count������Ӧ��k_tile��prefecth�׶��Ѿ���ɴ�gmem->smem, ����Ҫ��Ԥȡ�����ˡ�
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          // Set all predicates to false if we are going to overshoot bounds
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
          }
          copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
          copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
          cp_async_fence();
          ++k_tile_iter;

          // <NT> smem_pipe_read ʼ������ smem_pipe_write һ��������������read���ã�write�ǲ������ݡ�
          // ��k_block == 0����󣬾Ϳ��Ը��µ���һ��k_tile�ˣ���k_blockִ����0~K_BLOCK_MAX-1�󣬿�ʼ������һ��k_tile�ļ��㡣
          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // <NT> ���Ĵ�����tCrA/tCrB���е����ݴӴ洢��ʽת��Ϊ�����ʽ��
        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});
        // <NT> Tensor Core��MMAָ����ͬ����ԭ�ӵģ���Ҫһ�������������
        // Thread-level register gemm for k_block
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });

    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
