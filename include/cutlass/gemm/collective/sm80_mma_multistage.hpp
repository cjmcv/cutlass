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

  // <NT> 调用处，可参考外面的kernel层，如: include/cutlass/gemm/kernel/sm70_gemm.hpp
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

    // <NT> smem_buf应是外面创建的SharedStorage类型的buffer，里面包含两个cute::array_aligned的内存块，均对应smem。
    // 使用make_smem_ptr做一下指针的类型转换，并按layout为A和B矩阵分别构建共享内存的Tensor。
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

    // <NT> residue_mnk是未能被tile整除的mnk个数，get<2>(residue_mnk)即取出residue_k，其取值范围是[0, BLK_K).
    // 用make_coord(0, residue_k, 0)来构建偏移量。
    // 而gA维度是[BLK_M, BLK_K, K_TILES], BLK_K表示一个 tile 内部的小 K 维，K_TILES是一共多少个 K tile。
    // 从make_coord(0, get<2>(residue_mnk), 0)看，取的是中间的 BLK_K。
    // 所以执行cute::domain_offset后，访问gA(m, k_in_tile, k_tile)就相当于原来的gA(m, k_in_tile + residue_k, k_tile)
    // k_in_tile 的取值范围从 [0, BLK_K) 变成了 [-residue_k, BLK_K - residue_k).
    // 而有效数据是[0,residue_k) 变成了 [-residue_k, 0), 而[0,BLK_K - residue_k)会是无效数据。
    // 计算时就可以跟其他所有block一样，采用统一的(>= -residue_k)作为条件进行过滤即可，
    // 边界block上residue_k为正数，非边界block上residue_k为0，判断逻辑一致。
    // 即: 用一次负索引平移换一份完全无分支、无特殊分支的统一循环.
    // 
    // 问题: 范围都是[-residue_k, BLK_K-residue_k), 非边界的residue_k是0，边界block的residue_k是正数，且有效范围是[-residue_k, 0)
    //       此时起始条件是一致了，但是结束条件不一致，边界block上[0, BLK_K-residue_k)部分是无效数据，怎么处理？
    //  答: <NT-TODO>
    //      1）可能是边界block在偏移后，坐标不会再出现大于0的情况，所以用 if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) 管理起始点即可，结束点自然而然的变成0.
    //      2）可能是用predicate，tApA(_,k)中带k，是更细粒度的元素级过滤，但是k维度的掩码没发现在哪里设置了。
    // 
    // 注意：gA[BLK_M, BLK_K, K_TILES]就只是一个block的数据，如果它不是边界上的block，residue_mnk都会是0, 不会做任何偏移。
    //       上面的注释假设都是基于该block是处于边界上的block。
    // 
    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gA);
    gB = cute::domain_offset(make_coord(0, get<2>(residue_mnk), 0), gB);

    // <NT> GmemTiledCopyA通常由make_tiled_copy构建，get_slice函数在include/cute/atom/copy_atom.hpp
    // 用于获取线程所属的切片管理对象ThrCopy，即从TiledCopy得到ThrCopy，ThrCopy与一个线程绑定。
    // 注: mma atom中也有get_slice函数，用途大体一致，都是线程级别，
    //     一个用于copy一个用于mma (include/cute/atom/mma_atom.hpp)
    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    // <NT> ThrCopy类型，获取该线程负责的读取的src片段和写入的dst片段
    // tAgA[ACPY,ACPY_M,ACPY_K,k], 
    //     其中ACPY表示线程副本ID（每个线程拿到自己那份数据，每个线程访问tAgA时，其ACPY值都不同，会分别对应0~ACPY）。
    //     ACPY_M, ACPY_K	表示当前线程负责拷贝的 M×K子tile大小，
    //     因为 gA[BLK_M, BLK_K, K_TILES], 按K切分，所以tAgA还有k维度。
    // tAsA[ACPY,ACPY_M,ACPY_K,PIPE]
    //     其中的PIPE同理, 因sA[BLK_M,BLK_K,PIPE]
    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    // <NT> 创建谓语tensor，用于处理m和n未能被整除的边界。
    // size<1>(tAsA), size<2>(tAsA) => ACPY_M, ACPY_K。
    // 而Stride<_1,_0>中0是待推导的动态步长占位符。在运行时会被替换成 实际计算得到的步长值。
    // 如有 auto layout = make_layout(make_shape(4,8), make_stride(8,1));
    //      auto layout2 = Layout<Shape<_4,_8>, Stride<_8,_0>>{};
    //      在编译器会把 _0 替换成 1，所以 layout 和 layout2 最终是一样的。
    // tApA将会是一个维度为[ACPY_M, ACPY_K]的bool型tensor，跟tAgA的内部子tile大小一致。
    // tApA会与tAgA子tile一一对应，表示里面的元素是否需要拷贝。
    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // <NT> 为tensor sA 的shape[M, K]创建一个将shape内的坐标映射到其自身的张量cA, 其元素值就是自身的多维逻辑索引。
    // 如 Tensor cA = make_identity_tensor(make_shape(3, 4));
    // cA的内容是：
    //   (0,0) (0,1) (0,2) (0,3)
    //   (1,0) (1,1) (1,2) (1,3)
    //   (2,0) (2,1) (2,2) (2,3)
    // 简言之：make_identity_tensor(make_shape(M, K)) 创建了一个 “坐标镜子”：每个位置的值就是它自己的 (m, k) 坐标，常用于索引映射和调试。
    // 常见用途：
    //  1. 生成索引映射表，用于后续坐标变换、scatter/gather、迭代器构造等。
    //  2. 调试打印，可以 print_tensor(cA) 直接看到每个位置的行列坐标。
    //  3. 配合 transform/apply 做坐标变换，例如把 (m, k) 映射到物理内存偏移。
    //
    // 这里基于sA和sB的shape构建坐标张量cA/cB，
    // 然后基于坐标张量cA/cB，去获取该线程负责的读取的src fragment的坐标张量tAcA/tBcB。
    // 坐标张量tAcA/tBcB与数据张量的tAgA/tBgB相对应，构建掩码时需要用到坐标张量。
    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_A.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_B.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // <NT> 基于fragment的坐标张量，填充谓语掩码。
    // tApA[ACPY_M, ACPY_K], 针对m的边界，只需要填充m维度即可。
    // residue_mnk[m,n,k]，以get<0>取出residue_m; 
    // tAcA[ACPY,ACPY_M,ACPY_K]，元素是坐标，以tAcA(0,m,0)取出每一个m维度上的坐标，
    //     类型是tuple<int,int> 或 Coord<2>，再用get<0>取出行坐标，对应着是整个sA的global_m.
    // 填充时用tApA(m,0)，k维度固定为0，使用时会进行广播处理，即(m,0)为True的，(m,0~k)都会为True。
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

    // <NT> 这两个是从gmem到smem的目标tensor，因涉及到边界，需要先清空再拷贝，否则边界外会存在异常数据
    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);

    // <NT> k_tile_iter自加1即可取到下一个k_tile, k_tile_count记录剩余多少个k_tile可取。一个k_tile对应一个block。
    // tAgA[ACPY,ACPY_M,ACPY_K,k] => tAgAk[ACPY,ACPY_M,ACPY_K]
    // tAsA[ACPY,ACPY_M,ACPY_K,PIPE], size<2>(tAsA)则是ACPY_K。
    // 所以拷贝是 k 从0到ACPY_K， 从 tAgAk[ACPY,ACPY_M,0~ACPY_K] => tAsA[ACPY,ACPY_M,0~ACPY_K,PIPE]
    // for 循环k时，需要tAcA[ACPY,ACPY_M,ACPY_K]中第k点的坐标的横轴坐标，即k维度坐标需要大于-residue_k。
    // 因为gA使用过cute::domain_offset，整体偏移了residue_k，边界block的k范围从[0, residue_k) 变为 [-residue_k, 0)，
    // 所以get<1>(tAcA(0,0,k)需要大于-residue_k。
    //
    // 第一个tile因为有边界residue_k存在，需要单独拿出来处理并过滤。
    // 共享内存区的tAsA(_,_,k,k_pipe)会多一个维度k_pipe，每多一份k_pipe，内存就会多一份，用多份内存空间去减少数据依赖导致的时延。
    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      constexpr int k_pipe = 0;

      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      // <NT> 遍历k_in_tile的维度
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

    // <NT> 拷贝除了第一个tile以外的所有tile, 一个block在k维度可能会划分多个k_tile,
    // 如果k_tile多于一个，在这里会多拷贝Stages-2个无边界的完整的k_tile.
    // 如果没有这个多个k_tile, k_tile_count会降为0，谓语张量会被清空，copy_if会被直接跳过。
    // 即如果Stages数量能覆盖k_tile_count，则拷贝一轮后，所有k_tile都被存储到smem中了，否则仍有部分k_tile未被读取（此时k_tile_count>0）
    // 问题：为什么算上第一个tile存放在k_pipe=0里，总共只读取Stages-1个tile，而不是Stages个？
    //   答：预填充最后一级留给主循环的第一次迭代，则smem_pipe_write的初始值将会是Stages-1。
    //      即预加载：填充 1 … Stages-2; 主循环第一次：填充 Stages-1
    //      因此，smem_pipe_read可以从0开始，而smem_pipe_write落后于read一个stage，正好取到最后一份的Stages-1。
    //      则read和write的距离最大，等待时延可以更好地隐藏。
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

    // <NT> sA[BLK_M,BLK_K,PIPE], 把PIPE维度固定为0后，按线程进行切片。
    // 得到的tCrA的维度是[MMA,MMA_M,MMA_K], 更多注解搜"<NT> partition_shape_A".
    // 其中: MMA: 当前线程在 MMA tile 中的线程 ID 维, 同copy atom中的ACPY，对应着mma指令的shape。
    //       MMA_M：M维度上，一个block负责的数据的mma指令的重复调用次数
    //       MMA_K：K维度上，mma指令的重复调用次数
    // 例子：mma.sync.aligned.m16n8k32，一次完成m16n8k16的数据计算，输出16x8，一次需要一个warp32个线程参与。
    //      对于cute::partition_shape_A(mma, cute::make_shape(cute::Int<32>{}, cute::Int<128>{}))  => BLK_M=32, BLK_K=128
    //      且mma using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
    //            using PermutationMNK = Tile<Int<1>, Int<1>, Int<1>>
    //      则 MMA_M = 32/16 = 2； MMA_K = 128/32 = 4
    // 因为切分出来的fragment本身还没有数据，其形状只跟 tile 内[BLK_M, BLK_K]有关，
    // 与正在用的 PIPE 槽位无关，所以先固定 PIPE = 0（或任意一槽）即可.
    // 在使用时，从不同的pipe中拷贝数据到这个fragment里进行计算。
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
    // <NT> 前面的GmemTiledCopyA用于从gmem->smem, 这里的拷贝是smem->rmem.
    // GmemTiledCopyA通常也是通过make_tiled_copy函数构建，用的是cp.async指令。如：
    //   using GmemTiledCopyA = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementInput>{},
    //                          Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{}));
    // SmemCopyAtomA里的smem->rmem，通常用的是ldmatrix，如
    //   using SmemCopyAtomLoad = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
    //
    // smem_tiled_copy_A是tile级别copy对象，进而根据线程获取线程级别的copy对象，
    // 并smem的sA中取出src片段tCsA，从寄存器的tCrA取出dst片段。
    // 
    // 问：为什么取寄存器片段用retile_D而不是partition_D？
    // 答：partition_D输入的是shared memory tensor，把 shared memory tile 切成“每个线程负责的一份”。
    //     如	sA[BLK_M,BLK_K,PIPE] => tCsA[CPY,CPY_M,CPY_K,PIPE].
    //    而retile_D输入的是register fragment，把寄存器 fragment 重新切块，使其形状与 copy tile 一一对应。
    //     如 tCrA[MMA,MMA_M,MMA_K]	=> tCrA_copy_view[CPY,CPY_M,CPY_K]
    //    因为 tCrA 已经按 MMA 线程布局切成 fragment，是可以直接放入到mma指令中计算的布局（看上面tCrA的布局，是比较特殊的布局方式，需要转后后才能拷贝）。
    //    但 smem<->reg 的copy使用不同的线程布局（CPY ≠ MMA），需要再按 copy 线程布局 重新划分一次——这正是 retile_D 做的事。
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

    // <NT> smem_pipe_write 落后于 smem_pipe_read 一个stage，使其间距最大。
    // smem_pipe_write指向最后一个stage，在prefetch中，最后一个stage并没有写入，留给主循环写。见上面笔记。
    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // <NT> size<2>(tCrA) 是 MMA_K，对应该block在K方向需要重复调用mma的次数。
    // 这里对k方向的mma调用按pipeline进行串接，以达到寄存器预取的效果。
    // 所以该multi-stage实现里包含了smem pipeline 和 reg pipeline。
    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // <NT> 如果该block在k方向调用一次mma就结束了，就构不成reg的pipeline了
    // 1）这里首先使用cp_async_wait等待smem pipeline中的smem数据到位，
    // cp_async_wait<N> 当N=0，需要所有cp_async指令执行完毕；当N>0, 等待到未完成的组数量 ≤ N。更多解析看“<NT> cp_async_wait<N>”
    // 前面gmem->smem的预取过程中，包含了A和B，共调用了2*(stage-1)次的copy_if。
    // 问：为什么使用cp_async_wait<stage-2>来确保第一个预取tile完成。并且cp_async并不一定最先发起就最先结束，受很多条件影响。
    // 答：<NT-TODO> 猜测：这里使用时所有 cp.async 的源地址、目的地址、大小完全一样对齐、无 bank-conflict，也都对齐，则每一条 cp.async 的延迟几乎相同。
    // 所以基本上能确保先进的先完成。而且前面总共有2*(stage-1)次，这里只卡住stage-2，即表示前面的拷贝完成了一半即可，这种情况下，第一个tile基本可能确定是已经完成了。
    // 如果每条拷贝条件都不尽相同，则先后顺序可能出入会很大，则需要使用cp.async_wait<0>来强制所有cp_async完成来确定第一个tile的完成。
    // 2）单独预取0号stage的0号k_tile到寄存器, 使用Int<0>{}会在编译时特化，便于编译器优化。
    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    // <NT> 主循环，k_tile_count 通常大于0，因为stage一般是3~4，难以将数据全部搬到smem中。
    // 则前面预取了Stages-1份的数据，还留下一份在这里的主循环里搬运。假设原 k_tile_count 是6，stage是3。
    // 在prefetch中会搬运2个k_tile, 到这里时 k_tile_count 为4. 循环逐步减到-2，即会循环6次，包含预取了的和没预取的。
    // 
    // 如果 k_tile_count 为0，即上面的从gmem->smem的预取中能够把维度的所有数据搬至smem ()
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<N>.
      // <NT> make_int_sequence<K_BLOCK_MAX>：这是一个编译时整数序列生成器，会生成一个从 0 到 K_BLOCK_MAX-1 的整数序列。
      //      for_each：这是一个元编程工具，会遍历整数序列中的每个值，并对每个值执行给定的 lambda 函数。
      //      [&] (auto k_block)：这是一个捕获所有外部变量的 lambda 函数，它接受一个编译时常量参数k_block，表示当前迭代的整数值。
      //      整体作用是在编译时展开一个循环，在生成多个实例化版本的代码，对每个整数序列中的值执行相同的代码逻辑。
      //  而不是在运行时执行循环，可以起到消除运行时开销的作用。所以k_block会分别取[0, K_BLOCK_MAX - 1]中的一个值。
      //  且生成的循环在代码逻辑上是顺序展开的（如k_block=0,1,...,MMA_K-1依次执行）。
      //
      // k_block 作为分块索引，其取值范围是[0, K_BLOCK_MAX - 1]，正好与MMA_K一致(一个block在k维度可被切分的mma指令的次数)。
      // 以多次mma指令，来循环完成整个block的k维度的计算。
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        // <NT> 在最后一个k_block，准备开始下一stage的计算.
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // <NT> cp_async_wait确保缓冲区里有 (<= DispatchPolicy::Stages-2) 个指令未完成。
          // 就可以往下走，即等待确保为下一个 copy_if（在 k_block == 0 时发起）腾出缓冲区空间。
          // 同时也是确保最早的一个 copy_if 已完成。
          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // <NT> 计算前先预取下一个k_tile的计算数据到reg
        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // <NT> 第一个block，表示该k_tile的计算刚开始，可以开始进行下一份k_tile的gmem->smem预取了。
        // 如有stage=4，刚进来针对read 0，进行计算，此时的k_block为0时，开始write 3. 
        // write比read永远落后一批，使间距最大，更能隐藏时延。
        //      当k_tile_count <= 0 时，表示k_tile_iter已经跑够一轮，
        // 剩下的k_tile_count次数对应的k_tile在prefecth阶段已经完成从gmem->smem, 不需要再预取搬运了。
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

          // <NT> smem_pipe_read 始终先于 smem_pipe_write 一步，这样间隔最大。read是用，write是补新数据。
          // 在k_block == 0发起后，就可以更新到下一个k_tile了，等k_block执行完0~K_BLOCK_MAX-1后，开始进行下一个k_tile的计算。
          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // <NT> 将寄存器（tCrA/tCrB）中的数据从存储格式转换为计算格式。
        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});
        // <NT> Tensor Core的MMA指令是同步且原子的，需要一次完成数据输入
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
