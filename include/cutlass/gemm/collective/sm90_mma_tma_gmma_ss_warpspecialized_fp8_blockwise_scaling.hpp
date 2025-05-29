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
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/detail/blockwise_scale_layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// <NT> 参数分析
//  * ClusterShape: hopper架构提出了cluster smem（分布式共享内存，DSM）的概念，位于L1和L2 Cache之间新增了一层SM-to-SM网络，
// 使得同一个Cluster内的SM可以访问其他SM的Shared Memory，从而打破了传统上不同Thread Block之间无法直接通信的限制。
// cluster的组成单元是sm（cutlass编程定义会使用block），一个cluster通常会由2/4/8/16个sm组成，hopper多为2/4, 如Shape<_1,_2,_1>
//      
// 1) smem和L1与L2之间的关系：
//  smem和L1都在SM内，访存速度和数量都相近(L1通常比smem更接近计算核心)，二者共用部分资源，边界相对模糊，smem由开发者显式调用，L1由硬件自动管理(用于减少gmem访问次数)
//  L2位于sm和gmem之间，对所有sm可见，延迟较高，由硬件自动管理，用于减少gmem的访问次数。
// 2）DSM的数据流向：
//  新增的SM-to-SM网络虽然用于smem共享，但是放在了L1和L2之间，由硬件自动管理，不对开发者暴露过多细节.
//  基本路径: 源线程块的smem → 源SM的L1 Cache → 源SM的L2 Cache → 通过SM-to-SM网络 → 目标SM的L2 Cache → 目标SM的L1 Cache → 目标线程块的smem。
// 
// <NT> Kernel实现的底层函数，collective定义是 mma atoms和copy atoms被切分到的最大线程集合，提供了基础线程集合操作函数。
// 核心函数: 
// 1) can_implement: 每次计算前需要调用，检查当前参数能否正常运行。
// 2) prefetch_tma_descriptors:发射 tma描述符预取，通过 预取TMA描述符，可以将数据提前加载到L2缓存中，一个warp仅需要一个线程来调用，warp内其他线程会自动协同工作，不需要重复调用。
//                     调用例子：if ((warp_idx == 0) && lane_predicate) {
//                                CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
//                                CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
//                              }
// 3) load_init: 为该collective设置用于 load和 mma 操作所需的数据，返回一个张量元组。该元组至少会有两个tensor，
//               一个是gA_mkl，经过tile处理后的TMA张量，shape是[BLK_M,BLK_K,m,k,l]，另一个是gB_nkl[BLK_N,BLK_K,n,k,l]
// 4) load / load_auxiliary: (生产者视角) 执行一个collective范围内的矩阵乘累加操作, load 对应普通的 ProducerWarpRole::Mainloop，而load_auxiliary对应ProducerWarpRole::MainloopAux
// 5) load_tail: 执行一个生产者后处理操作，以防止集群中的线程块过早退出。
// 6) mma: (消费者视角) 执行一个collective范围内的矩阵乘累加操作.
// 7) mma_tail: 执行一个消费者后处理操作，以释放所有缓冲区。
// 
// kernel实现的外层函数位置，存放在kernel文件夹下，通过组合collective mainloop和collective epilogue构建而成的，当前文件实现的是collective mainloop。
// include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
//                             sm90_gemm_tma_warpspecialized_pingpong.hpp
//                             sm90_gemm_tma_warpspecialized.hpp
//
// device充当最外层封装，存放在device文件夹下，通过调用kernel的相关函数，完成计算。
// include/cutlass/gemm/device/gemm_universal.h
// 关系是 device => kernel => collective mainloop + collective epilogue

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
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
    MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = cute::tuple_element_t<0,StridePairA_>;
  using LayoutSFA = cute::tuple_element_t<1,StridePairA_>;
  using ElementB = ElementB_;
  using StrideB = cute::tuple_element_t<0,StridePairB_>;
  using LayoutSFB = cute::tuple_element_t<1,StridePairB_>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementBlockScale = ElementAccumulator;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScaleTMA = cute::SM90_TMA_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});

  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0);
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0);

  static constexpr int ScalePromotionInterval = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});
  static_assert(ScalePromotionInterval % 4 == 0, "ScalePromotionInterval must be a multiple of 4.");
  static_assert(ScalePromotionInterval >= size<2>(TileShape{}) / tile_size<2>(TiledMma{}),
    "ScalePromotionInterval must be greater than or equal to the number of stages of the MMA atom.");
  static_assert(ScalePromotionInterval % (size<2>(TileShape{}) / tile_size<2>(TiledMma{})) == 0,
    "ScalePromotionInterval must be a multiple of the number of stages of the MMA atom.");
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static constexpr int ScaleTmaThreshold = 32;
  static constexpr bool IsTmaLoadSFA = ScaleMsPerTile >= ScaleTmaThreshold && ScaleNsPerTile < ScaleTmaThreshold;
  static constexpr bool IsTmaLoadSFB = ScaleNsPerTile >= ScaleTmaThreshold && ScaleMsPerTile < ScaleTmaThreshold;
  // Two threads per CTA are producers (1 for operand tile `tma`, and 32 for scales `cp.async`)
  static constexpr int NumProducerThreadEvents = ((IsTmaLoadSFA && IsTmaLoadSFB)? 1 : 33);

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert((size<0>(TileShape{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  using ScaleConfig = ::cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // Block scaling gmem-to-smem copy atom 
  //  we can have partial tiles in M or N, so don't vectorize those loads
  using CopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
  using CopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;

  static constexpr int AlignmentSFA = 1;
  static constexpr int AlignmentSFB = 1;

  // Block scaling smem layout
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));


  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;  // TILE_M x PIPE_K
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;  // TILE_N x PIPE_K
      CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFA>> smem_SFA; // ScaleMsPerTile x PIPE_K
      CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFB>> smem_SFB; // ScaleNsPerTile x PIPE_K
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementBlockScale const* ptr_SFA; 
    LayoutSFA layout_SFA;
    ElementBlockScale const* ptr_SFB;
    LayoutSFB layout_SFB;
  };

  // Device side kernel params
  struct Params {
    static auto getTmaSFA() {
      if constexpr (IsTmaLoadSFA) {
        return make_tma_copy(
          GmemTiledCopyScaleTMA{},
          make_tensor(static_cast<ElementBlockScale const*>(nullptr), filter_zeros(LayoutSFA{})),
          filter_zeros(SmemLayoutSFA{}(_,_,_0{})),
          Shape<Int<ScaleMsPerTile>, Int<1>>{},
          _1{});
      }
      else {
        return nullptr;
      }
    }
    static auto getTmaSFB() {
      if constexpr (IsTmaLoadSFB) {
        return make_tma_copy(
          GmemTiledCopyScaleTMA{},
          make_tensor(static_cast<ElementBlockScale const*>(nullptr), filter_zeros(LayoutSFB{})),
          filter_zeros(SmemLayoutSFB{}(_,_,_0{})),
          Shape<Int<ScaleNsPerTile>, Int<1>>{},
          _1{});
      }
      else {
        return nullptr;
      }
    }
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_0{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_0{}),
        TileShape{},
        ClusterShape{}));
    // NOTE: Does make_tma_copy supports 0 stride?
    using TMA_SFA = decltype(getTmaSFA());
    using TMA_SFB = decltype(getTmaSFB());
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    // Block scaling factors for A and B
    ElementBlockScale const* ptr_SFA;
    ElementBlockScale const* ptr_SFB;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = reinterpret_cast<ElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<ElementB const*>(args.ptr_B);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale const*>(args.ptr_SFA);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale const*>(args.ptr_SFB);

    Tensor tensor_sfa = make_tensor(ptr_SFA, filter_zeros(args.layout_SFA));
    Tensor tensor_sfb = make_tensor(ptr_SFB, filter_zeros(args.layout_SFB));
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_SFA tma_load_sfa{};
    if constexpr (IsTmaLoadSFA) {
      tma_load_sfa = make_tma_copy(
          GmemTiledCopyScaleTMA{},
          tensor_sfa,
          filter_zeros(SmemLayoutSFA{})(_,_,cute::Int<0>{}),
          Shape<Int<ScaleMsPerTile>, Int<1>>{},
          _1{});
    }
    typename Params::TMA_SFB tma_load_sfb{};
    if constexpr (IsTmaLoadSFB) {
      tma_load_sfb = make_tma_copy(
          GmemTiledCopyScaleTMA{},
          tensor_sfb,
          filter_zeros(SmemLayoutSFB{})(_,_,cute::Int<0>{}),
          Shape<Int<ScaleNsPerTile>, Int<1>>{},
          _1{});
    }
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes_sfa = TmaTransactionBytesSFA;
    uint32_t transaction_bytes_sfb = TmaTransactionBytesSFB;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk + transaction_bytes_sfa + transaction_bytes_sfb;

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk,
      args.ptr_SFA,
      args.ptr_SFB,
      args.layout_SFA,
      args.layout_SFB,
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    if (!cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{})) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load tensor A.\n");
    }
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    if (!cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{})) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load tensor B.\n");
    }
    constexpr int min_tma_aligned_elements_S = tma_alignment_bits / cutlass::sizeof_bits<ElementBlockScale>::value;
    if (IsTmaLoadSFA && !cutlass::detail::check_alignment<min_tma_aligned_elements_S>(args.layout_SFA)) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load scale A.\n");
    }
    if (IsTmaLoadSFB && !cutlass::detail::check_alignment<min_tma_aligned_elements_S>(args.layout_SFB)) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load scale B.\n");
    }

    // We expect full tiles in K
    if (K % size<2>(TileShape{}) != 0) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size K is incompatible with tile size.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  static constexpr uint32_t TmaTransactionBytesSFA =
        (IsTmaLoadSFA? cutlass::bits_to_bytes(ScaleMsPerTile * static_cast<uint32_t>(sizeof_bits<ElementBlockScale>::value)): 0);
  static constexpr uint32_t TmaTransactionBytesSFB =
        (IsTmaLoadSFB? cutlass::bits_to_bytes(ScaleNsPerTile * static_cast<uint32_t>(sizeof_bits<ElementBlockScale>::value)): 0);
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesSFA + TmaTransactionBytesSFB;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params)
  {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    if constexpr (IsTmaLoadSFA) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfa.get_tma_descriptor());
    }
    if constexpr (IsTmaLoadSFB) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfb.get_tma_descriptor());
    }
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                             // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                             // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});         // (BLK_N,BLK_K,n,k,l)

    // Note that mSFA_mkl and mSFB_nkl are already blocked tiled in the `m` host and
    // gScaleA_mkl and gScaleB_nkl in `g` global memory are same as mSFA_mkl and mSFB_nkl.
    auto mSFA_mkl = [&]() {
      if constexpr (IsTmaLoadSFA) {
        return mainloop_params.tma_load_sfa.get_tma_tensor(shape(filter_zeros(mainloop_params.layout_SFA)));
      }
      else {
        return make_tensor(make_gmem_ptr(mainloop_params.ptr_SFA), mainloop_params.layout_SFA); // (scale_m,k,l)
      }
    }();
    auto mSFB_nkl = [&]() {
      if constexpr (IsTmaLoadSFB) {
        return mainloop_params.tma_load_sfb.get_tma_tensor(shape(filter_zeros(mainloop_params.layout_SFB)));
      }
      else {
        return make_tensor(make_gmem_ptr(mainloop_params.ptr_SFB), mainloop_params.layout_SFB); // (scale_n,k,l)
      }
    }();

    return cute::make_tuple(gA_mkl, gB_nkl, mSFA_mkl, mSFB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorScaleA, class TensorScaleB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    // <NT> lane_predicate表示当前线程是否被选取出，true为是false为否。
    // 一个warp里只需要一个线程调用具体拷贝指令，warp的其他线程会自动协同工作，不需要重复调用指令。
    // 
    // sA/sB中的s是smem的意思，对应存放到smem的A和B矩阵的tile，会使用tma来完成gmem到smem的数据加载；
    // sSFA/sSFB则对应A和B的ScaleFactor，会根据数据量由 IsTmaLoadSFA / IsTmaLoadSFB判断选择是否也使用tma；
    // 如果scale也使用tma，会在该load函数里一起加载，如果不使用tma，则会单独放到load_auxiliary中使用cp.async来加载scale。
    // 很多时候因为scale数据量少，使用cp.async可以更灵活，且与tma可以并行执行，也避免浪费tma，加快AB矩阵的加载。
    //
    int lane_predicate = cute::elect_one_sync();
    // Blockscaling: Tma loads for load_input and CpAsync for load_scale
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), filter_zeros(SmemLayoutSFA{})); // (ScaleMsPerTile,PIPE)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), filter_zeros(SmemLayoutSFB{})); // (ScaleNsPerTile,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);
    Tensor mSFA_mkl = get<2>(load_inputs);
    Tensor mSFB_nkl = get<3>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
    Tensor gSFA = local_tile(
      mSFA_mkl, make_tile(Int<ScaleMsPerTile>{}, Int<1>{}),
      make_coord(m_coord,_,l_coord));
    Tensor gSFB = local_tile(
      mSFB_nkl, make_tile(Int<ScaleNsPerTile>{}, Int<1>{}),
      make_coord(n_coord,_,l_coord));

    // <NT> S是src，D是dst，基于block_tma_a将数据分块，表示后续会将数据从tAgA搬运到tAsA。
    // A和B直接使用tma，scaleA和scaleB会按数据量判断是否使用tma，如使用，其才做与AB的加载类似。
    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                                 // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                                 // (TMA,TMA_N,TMA_K,PIPE)

    auto [tAgA_SFA, tAsA_SFA] = [&]() {
      if constexpr (IsTmaLoadSFA) {
        auto block_tma_sfa = mainloop_params.tma_load_sfa.get_slice(cluster_local_block_id.y);
        Tensor tAgA_SFA_ = block_tma_sfa.partition_S(gSFA);
        Tensor tAsA_SFA_ = block_tma_sfa.partition_D(sSFA);
        return cute::make_tuple(tAgA_SFA_, tAsA_SFA_);
      }
      else {
        return cute::make_tuple(0, 0);
      }
    }();
    auto [tBgB_SFB, tBsB_SFB] = [&]() {
      if constexpr (IsTmaLoadSFB) {
        auto block_tma_sfb = mainloop_params.tma_load_sfb.get_slice(cluster_local_block_id.y);
        Tensor tBgB_SFB_ = block_tma_sfb.partition_S(gSFB);
        Tensor tBsB_SFB_ = block_tma_sfb.partition_D(sSFB);
        return cute::make_tuple(tBgB_SFB_, tBsB_SFB_);
      }
      else {
        return cute::make_tuple(0, 0);
      }
    }();

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    uint16_t mcast_mask_sf = 0;

    // <NT> 如果采用了multicast的策略，则需要准备tma的multicast用的掩码，指示哪个线程块需要接收数据；
    // size<0>(block_layout)是cluster里一列有多少个线程块， size<1>(block_layout)是cluster里一行有多少个线程块，
    // 先遍历线程块布局的每一列，取出行为cluster_local_block_id.x列为n映射得到的block_id, 生成一个掩码位。
    // Issue TmaLoads for GEMM operands A/B and CpAsync for scale tensors
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    // <NT> 按k方向分块, 使用tma从gmem加载数据到smem。
    // pipeline是MainloopPipeline类型，内嵌mbarrier指令，实现生产者消费者的栅栏管理。
    // 在load           调用 pipeline.producer_acquire / producer_get_barrier
    //   load_auxiliary 调用 producer_acquire / producer_commit
    //   load_tail      调用 producer_tail
    // 在mma调用 pipeline.consumer_try_wait / consumer_wait / consumer_release.
    //   
    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //
      int write_stage = smem_pipe_write.index();
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      // Copy operands A and B from global memory to shared memory
      if (lane_predicate) copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      if (lane_predicate) copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

      // Copy scale tensors from global memory to shared memory
      if constexpr (IsTmaLoadSFA) {
        if (lane_predicate) {
          copy(mainloop_params.tma_load_sfa.with(*tma_barrier, mcast_mask_sf), tAgA_SFA(_,_,_,*k_tile_iter), tAsA_SFA(_,_,_,write_stage));
        }
      }
      if constexpr (IsTmaLoadSFB) {
        if (lane_predicate) {
          copy(mainloop_params.tma_load_sfb.with(*tma_barrier, mcast_mask_sf), tBgB_SFB(_,_,_,*k_tile_iter), tBsB_SFB(_,_,_,write_stage));
        }
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  template <
    class TensorA, class TensorB,
    class TensorScaleA, class TensorScaleB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_auxiliary(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    // Block scaling: load_scale has scaling tensors in global memory which are not tiled
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), SmemLayoutSFA{}); // (ScaleMsPerTile,k)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), SmemLayoutSFB{}); // (ScaleNsPerTile,k)

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor mSFA_mkl = get<2>(load_inputs);
    Tensor mSFB_nkl = get<3>(load_inputs);

    Tensor iSFA_mkl = make_identity_tensor(shape(mainloop_params.layout_SFA));                                // (m,k,l)
    Tensor iSFB_nkl = make_identity_tensor(shape(mainloop_params.layout_SFB));                                // (n,k,l)

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
    Tensor cSFA_mkl = local_tile(iSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
    Tensor cSFB_nkl = local_tile(iSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)

    Tensor gSFA_k = gSFA_mkl(_,_,m_coord,_,l_coord);
    Tensor cSFA_k = cSFA_mkl(_,_,m_coord,_,l_coord);
    Tensor gSFB_k = gSFB_nkl(_,_,n_coord,_,l_coord);
    Tensor cSFB_k = cSFB_nkl(_,_,n_coord,_,l_coord);

    TiledCopy scale_copy_a = make_tiled_copy(CopyAtomSFA{},
      Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
    TiledCopy scale_copy_b = make_tiled_copy(CopyAtomSFB{}, 
      Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(thread_idx);
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(thread_idx);

    Tensor tSFAgSFA_k = thr_scale_copy_a.partition_S(gSFA_k);
    Tensor tSFAcSFA_k = thr_scale_copy_a.partition_S(cSFA_k);
    Tensor tSFAsSFA   = thr_scale_copy_a.partition_D(sSFA);

    Tensor tSFBgSFB_k = thr_scale_copy_b.partition_S(gSFB_k);
    Tensor tSFBcSFB_k = thr_scale_copy_b.partition_S(cSFB_k);
    Tensor tSFBsSFB   = thr_scale_copy_b.partition_D(sSFB);

    Tensor tSFApSFA = make_tensor<bool>(shape(filter_zeros(tSFAsSFA(_,_,_,_0{}))));                 // (CPY,CPY_M,CPY_K)
    Tensor tSFBpSFB = make_tensor<bool>(shape(filter_zeros(tSFBsSFB(_,_,_,_0{}))));                 // (CPY,CPY_N,CPY_K)

    auto SFA_shape = shape(mainloop_params.layout_SFA);
    auto SFB_shape = shape(mainloop_params.layout_SFB);

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      // Since scale granularity K is multiple of BLK_K we do not have to consider if that is OOB
      bool load_sfa = thread_idx < ScaleMsPerTile;
      Tensor tSFAcSFA = tSFAcSFA_k(_,_,_,*k_tile_iter);
      Tensor tSFAcSFA_compact = filter_zeros(tSFAcSFA);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSFApSFA); ++i) {
        tSFApSFA(i) = load_sfa && elem_less(get<0>(tSFAcSFA_compact(i)), get<0>(SFA_shape));
      }

      bool load_sfb = thread_idx < ScaleNsPerTile;
      Tensor tSFBcSFB = tSFBcSFB_k(_,_,_,*k_tile_iter);
      Tensor tSFBcSFB_compact = filter_zeros(tSFBcSFB);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSFBpSFB); ++i) {
        tSFBpSFB(i) = load_sfb && elem_less(get<0>(tSFBcSFB_compact(i)), get<0>(SFB_shape));
      }
      int write_stage = smem_pipe_write.index();
      // Copy scale tensors from global memory to shared memory
      if constexpr (!IsTmaLoadSFA) {
        copy_if(scale_copy_a, tSFApSFA, filter_zeros(tSFAgSFA_k(_,_,_,*k_tile_iter)), filter_zeros(tSFAsSFA(_,_,_,write_stage)));
      }
      if constexpr (!IsTmaLoadSFB) {
        copy_if(scale_copy_b, tSFBpSFB, filter_zeros(tSFBgSFB_k(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,write_stage)));
      }
      if constexpr (!IsTmaLoadSFA || !IsTmaLoadSFB) {
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
      }

      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor scaleFactor) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor);
    }
  }
  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor1,
    class ScaleFactor2
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor1 scaleFactor1, ScaleFactor2 scaleFactor2) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor1, scaleFactor2);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor1, scaleFactor2);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {


    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // <NT> 同样拿到对应区域的共享内存块，生产者将数据从gmem搬运到smem后，mma这里直接从smem里消费数据做计算。
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Block scaling
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), make_layout(
        make_shape(get<0>(shape(SmemLayoutSFA{})), 
                   get<1>(TileShape{}), 
                   make_shape(get<1>(shape(SmemLayoutSFA{})), 
                   get<2>(shape(SmemLayoutSFA{})))),
        make_stride(get<0>(stride(SmemLayoutSFA{})), _0{}, 
                    make_stride(get<1>(stride(SmemLayoutSFA{})), get<2>(stride(SmemLayoutSFA{}))))
      ));                                                                                       // (BLK_M,BLK_N,(BLK_K,P))
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), make_layout(
        make_shape(get<0>(TileShape{}), 
                   get<0>(shape(SmemLayoutSFB{})), 
                   make_shape(get<1>(shape(SmemLayoutSFB{})), 
                   get<2>(shape(SmemLayoutSFB{})))),
        make_stride(_0{}, 
                    get<0>(stride(SmemLayoutSFB{})), 
                    make_stride(get<1>(stride(SmemLayoutSFB{})), 
                    get<2>(stride(SmemLayoutSFB{}))))
      ));                                                                                       // (BLK_M,BLK_N,(BLK_K,P))

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and 
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    // <NT> TiledMma通常与ClusterShape强关联，TiledMMA在ClusterShape定义的cluster上执行计算，TiledMMA里的元素是线程。
    // 一个warp组有128个线程，MmaWarpGroups看一个TileMMA里共有多少个warp组，以组的数量充当行，一组的128个线程充当列，得到warp_group_thread_layout。
    // 按warp group取出thread_mma，表示为当前线程要计算的部分。
    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, 
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))

    Tensor tCsA = thread_mma.partition_A(sA);                                                  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                            // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                            // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                          // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                          // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                           // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                          // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                          // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Per block scale values for operand A and B
    // Since scale factors always broadcast across MMA_K we slice that away
    Tensor tCrSFA = make_tensor_like<ElementBlockScale>(tCsSFA(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)
    Tensor tCrSFB = make_tensor_like<ElementBlockScale>(tCsSFB(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)

    // Prologue GMMAs

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    GmmaFP8Accumulation accumulation(accum, ScalePromotionInterval, size<2>(tCrA));
    warpgroup_fence_operand(accumulation());
    {
      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count)
    {
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_release); // Unlock previous tile
      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_release;
    }
    if (k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_release); // Unlock previous tile
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }
    }
    if constexpr (ScalePromotionInterval != 4) {
      // residues only exists when granularity is not the finnest
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        accumulation.scale_residue_if_needed(scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        accumulation.scale_residue_if_needed(tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrSFA, tCrSFB);
      }
    }

    warpgroup_fence_operand(accumulation());
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // The pipeline is not released in the first iteration
    smem_pipe_release.advance(k_tile_count - 1);
    pipeline.consumer_release(smem_pipe_release);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
