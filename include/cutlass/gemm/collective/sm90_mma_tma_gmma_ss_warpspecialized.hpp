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
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
// <NT> sm90 collective_mma 参数分析
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
//
// <NT> rs和ss的区别: 以 sm90_mma_tma_gmma_rs_warpspecialized 和 sm90_mma_tma_gmma_ss_warpspecialized 为例
// rs和ss表示着GMMA 的 A-operand 从哪儿来：r对应rf，s对应smem。
// | 阶段         | RS 行为                                     | SS 行为                        |
// | ----------- | ------------------------------------------- | ------------------------------ |
// | Producer    | TMA 把 A/B 搬到 Shared Memory（同 SS）       | 同左                            |
// | Consumer    | **额外做 smem→rmem 拷贝** → 寄存器片段 → GMMA | 直接 descriptor 绑定 GMMA，无拷贝|
// | 寄存器压力   | 高（A-tile 占 RF）                           | 低（A-tile 只占 SMEM）          |
// | 共享内存压力 | 低（A-tile 可立即覆盖）                       | 高（A/B 都要驻留）               |
//
// rs 适用于权重端位宽极低（W4A16、W8A16）且需要 反量化/查表 后再参与计算，反量化逻辑在 smem→rmem 拷贝时一并完成，寄存器里已是 FP16/BF16；
//    1）4-bit 权重先由 TMA 搬进 shared memory（仍是 packed 格式）
//    2）Consumer warps 用 ld.shared / ld.matrix 把 packed 数据 直接读到寄存器；
//    3）同一条寄存器流水线里立刻做 反量化 / 查表 / unpack，得到 FP16/BF16 值；
//    4）FP16/BF16 寄存器片段直接作为 wgmma 的 A-operand 使用，走 GMMA::RegisterIterator 路径，不需要写回到smem。
// ss 是直接放 Shared Memory 就算，因为 Hopper 的 wgmma 指令可以直接把 smem 当作 源数据，只要满足：
//    1）SMEM descriptor 路径； 2）数据布局符合 Tensor Core 的 swizzle 要求； 3）位宽合法，
//    不需要任何 smem→rmem 的搬运阶段，wgmma 指令内部会直接通过 shared memory descriptor 读取 A-tile，与 B-tile（同样来自 SMEM）一起完成矩阵乘。
// 
// 补充问题：为什么 SM90 还在 kernel 名字里区分 *_rs_* / *_ss_*，而 SM100 的示例代码已经看不到这对后缀。
//        -- wgmma 指令对两个 operand 的物理来源有硬连线限制，A 可以是寄存器（R）或共享内存描述符（S），B 只能是共享内存描述符（S）。
//           在hopper的代码里，单独区分开来写，而blackwell的umma代码里，则把 “A operand 要从哪来” 的决定完全下沉到 一条模板主循环 里，
//           4-bit → 内部走 RS（自动解包到寄存器）; 8-bit+ → 内部走 SS（直接用 SMEM-desc）.RS/SS 两条路径仍然物理存在，只是被封装在 TiledMma::make_fragment_A/B.
namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
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
    MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>,
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
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>;
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

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
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
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
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

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

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
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    return {
      tma_load_a,
      tma_load_b,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk
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
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  // <NT> MainloopSm90TmaGmmaWarpSpecialized的 load 函数介绍
  // 流程： for ( ; k_tile_count > 0; --k_tile_count)
  //          pipeline.producer_acquire
  //          pipeline.producer_get_barrier
  //          copy(mcast-a/b)
  //
  // multicast：
  //    如果采用了multicast的策略，则需要准备tma的multicast用的掩码，指示哪个线程块需要接收数据；
  // size<0>(block_layout)是cluster里一列有多少个线程块， size<1>(block_layout)是cluster里一行有多少个线程块，
  // 先遍历线程块布局的每一列，取出行为cluster_local_block_id.x列为n映射得到的block_id, 生成一个掩码位。
  // 
  //  1) 把 1-D 的 block_rank_in_cluster 映射到 2-D (m, n) 网格
  //     uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
  //  2) 生成 A 矩阵 mask（行广播）：固定当前行号 cluster_local_block_id.x，遍历 所有列号 n，把同一行的所有 block_id 置 1。
  //     形成 行广播掩码 一次 TMA load A 把同一份 A tile 同时写进该行所有 block 的 smem A buffer
  //     for (int n = 0; n < size<1>(block_layout); ++n)
  //       mcast_mask_a |= uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{});
  //  3）生成 B 矩阵 mask (列广播)：固定当前列号 cluster_local_block_id.y，遍历 所有行号 m，把同一列的所有 block_id 置 1。
  //     形成 列广播掩码 一次 TMA load B 把同一份 B tile 写进 该列所有 block 的 smem B buffer
  //     for (int m = 0; m < size<0>(block_layout); ++m)
  //       mcast_mask_b |= uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{});
  //  4) 基于掩码进行拷贝
  //     copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
  //
  //  * 掩码宽度 16 bit → 最多支持 16 个 block 的 Cluster，与 Hopper 硬件上限一致。
  //  * multicast可以使一次TMA 让一个cluster里多个block(掩码置1的) 都同时拿到相同的数据。
  //  对于A矩阵的读取, m相同的block可共享一份数据，对于B矩阵的读取，n相同的block可共享一份数据。
  //  如: for (int m=0; m<M; m++) 
  //        for (int n=0; n<N; n++) 
  //          for (int k=0; k<K; k++) 
  //            C[m,n] += A[m,k] * B[n,k]
  //  对于一个cluster(2,2)内的4个block而言：
  //    block(m  ,n  )负责C[m  ,n  ]的输出，循环读k，当k为0时，需要读取A[m,0]   / B[n,0]
  //    block(m  ,n+1)负责C[m  ,n+1]的输出，循环读k，当k为0时，需要读取A[m,0]   / B[n+1,0]
  //    block(m+1,n  )负责C[m+1,n  ]的输出，循环读k，当k为0时，需要读取A[m+1,0] / B[n,0]
  //    block(m+1,n+1)负责C[m+1,n+1]的输出，循环读k，当k为0时，需要读取A[m+1,0] / B[n+1,0]
  //  即A矩阵m相同则共享，B矩阵n相同则共享：
  //    block(m  ,n  ), block(m  ,n+1)   =>    A[m,0]/B[n,0]  ,  A[m,0]/B[n+1,0]
  //    block(m+1,n  ), block(m+1,n+1)   =>    A[m+1,0]/B[n,0],  A[m+1,0]/B[n+1,0]
  //
  // * lane_predicate 表示当前线程是否被选取出，true为是false为否。
  // 一个warp里只需要一个线程调用具体拷贝指令，warp的其他线程会自动协同工作，不需要重复调用指令。
  // 

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
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
      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
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

  // <NT> sm90 MainloopSm90TmaGmmaWarpSpecialized的mma函数介绍
  // 1）类型检查：
  // * FrgTensorC 计算结果需存放在寄存器，mma只管计算，不管写回
  // * SmemLayoutA / SmemLayoutB 为三维(BLK_M, BLK_K, PIPE)/(BLK_N, BLK_K, PIPE)，PIPE=Stages实现双缓冲。
  // * SmemCopyAtomA / SmemCopyAtomB 必须是 void。
  //   因为sm90的 GMMA 指令本身就从 shared memory 直接读 A/B，不需要再经过寄存器中转。
  //   而CopyAtom的含义中，当其不为void时，表示走 smem->reg->TensorCore 的路径；为void表示把smem提交给GMMA, 不经过reg。
  // 2）构建smem的视图 sA / sB
  // 3) WarpGroup 划分，以 线程号 / warpgroup内总线程数，换算得到当前线程对应的 warp_group_idx。
  //    * __shfl_sync(0xFFFFFFFF, ..., 0)	0xFFFFFFFF是掩码，表示所有32线程都参与广播。
  //    * 0 表示从 lane-0 广播这个值到整个 warp 的所有线程.
  //    auto thread_mma = tiled_mma.get_slice(warp_group_layout(warp_group_idx));
  //    每个 warp-group 只取自己那一 slice 的 A/B/C 子张量，互不重叠。
  // 4）分配寄存器描述符 tCrA/tCrB (MMA, MMA_M, MMA_K, PIPE)。
  // 5）检查pipeline变量：K_PIPE_MMAS 同时 inflight 的 GMMA 批次数量（通常 2-4），用来隐藏 GMMA latency。
  //                     warpgroup_wait<K_PIPE_MMAS>() 保证 不会同时提交超过硬件上限。
  // 6）Prologue与mainloop阶段：见下面的注释笔记 <NT>S
  //
  // <NT> TiledMma通常与ClusterShape强关联，TiledMMA在ClusterShape定义的cluster上执行计算，TiledMMA里的元素是线程。
  // 一个warp组有128个线程, MmaWarpGroups 看一个TileMMA里共有多少个warp组，以组的数量充当行，一组的128个线程充当列，得到 warp_group_thread_layout
  // 按warp group取出thread_mma，表示为当前线程要计算的部分。

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

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    assert(k_tile_count >= 1);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    // <NT> sm90 MainloopSm90TmaGmmaWarpSpecialized的mma函数里的 mainloop 阶段介绍
    //   mainloop阶段，直接进入steady-state 双缓冲：搬→算→搬→算…，每算完一 tile 就释放 smem 缓冲区；
    //   前面的prologue 把前 K_PIPE_MMAS 个 tile 算完，让 DMA 提前填下一批。即把 pipeline“灌满”到预定水位，
    // 让 TensorCore 和 TMA 搬运从第一拍就重叠；不释放缓冲区、不 wait GMMA，只为“预热”；mainloop 才带 
    // wait + release，保持 inflight 数量恒定。
    //
    // Mainloop
    // 1) pipeline.consumer_try_wait, 负责轻量级探针，只读一次 barrier 计数器/相位，立即返回，不调用arch::barrier_wait；
    //       大多数情况下producer 早已 arrive，token 直接命中，省一次 heavy barrier 调用，如多个warp同时冲击barrier寄存器会撑爆 LSU (Load-Store Unit)
    // 1) pipeline.consumer_wait(...)	根据 consumer_try_wait 返回结果里记录的“是否已满足” 决定路径.
    //       如已满足，则立即返回，否则才真正阻塞等待。等到 DMA warp 把 A/B 搬进 smem，phase 翻转即继续。
    // 3) warpgroup_fence_operand(accum)：表示在我真正要对 accum 寄存器做下一步操作（读、写、scale、store）之前，
    //       先把所有已经发射的 WGMMA 指令完成，并确保累加结果对当前线程可见。
    // 2) warpgroup_arrive()：向 TensorCore 发射 GMMA 描述符（非阻塞），表示当前 warp-group 的所有线程已经到达同一点，
    //       可以开始发射下一批 WGMMA（TensorCore）指令了。
    // 3) cute::gemm(...)	循环 K 维，一次提交 MMA_K 深度，其发射的wgmma指令，指令已进队列开始计算
    // 4) warpgroup_commit_batch()：“关门”，把前面的wgmma归为一组，保证后续 wait 能等到这一整组。
    // 5) warpgroup_wait<K_PIPE_MMAS>()	阻塞直到warpgroup_commit_batch提交的整组全部完成，保证 smem 缓冲区已空
    // 6) pipeline.consumer_release(...)	通知 DMA warp “这个 smem 阶段我已用完”，可再填新数据

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
