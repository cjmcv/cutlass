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
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"

///////////////////////////////////////////////////////////////////////////////
// <NT> GemmUniversal 的 KernelTmaWarpSpecializedPingpong 类型的偏特化实现。
//   sm90中使用tma的kernel有4类：KernelTmaWarpSpecializedPingpong / KernelTmaWarpSpecializedCooperative / KernelTmaWarpSpecialized / KernelTma.
// 以Pingpong和Cooperative两类warpspecialize的kernel最为常用。
// 
// Pingpong    = 两个 Consumer-WG 轮流啃 同一组 pipeline stage（一个吃奇数槽，一个吃偶数槽）。靠 barrier.phase 翻转 实现严格WG 之间交替，WG 级互斥。
//                 WG-0  Producer   : 一直发 TMA  
//                 WG-1  Consumer-A : 专吃 stage 0,2,4…  
//                 WG-2  Consumer-B : 专吃 stage 1,3,5…
//               需要2× pipeline stage 的 shared memory + 2× 寄存器堆 给两个 WG 轮流用。
// cooperative = 把“大块”再劈成“子块”，多组线程束一起上。同一块（甚至同一 WG）里的 4-8 组 warp 同时抢不同 k_tile 做 MMA，累加同一组输出寄存器。
//               只要 1× 寄存器堆，把省下的寄存器拿来再塞 1-2 组 warp，原理上cooperative性能远优于pingpong。(不再需要MathWarpGroupOrderBarrier)
//
// 问题：pingpong很多时候难以发挥cooperative的真实性能，大多以pingpong为主，cooperative为辅。而sm100以后则以cooperative为主，pingpong为辅。
// 答：1）sm90一个 warp 每周期只能发 1 条 MMA，若把 K 切片给 4 组 warp 同时算，前端端口立刻挤爆。
//        sm100 的 double-issue 前端 + mma.sp 新指令，单周期可发 2 条 MMA，带宽够了，多组 warp 一起算才不会堵队。
//     2) sm90 缺少 sub-warpgroup 轻量同步，cooperative 跨组同步开销大。
//        sm100 引入 cluster-level 细粒度 barrier，同步成本大幅下降。
//     3）sm90 每 SM 256 kB RegFile，看似比 Ampere 大，但 wgmma 专用 bank 与普通 RF 重叠；cooperative kernel 常需要 4-8 个 warpgroup 同时驻留，
//             Reg 用量 >192 kB/SM 时就会溢出到 local-memory，带宽瞬间被拉低.
//        sm100 增大 RF 并拆分 wgmma bank，溢出概率显著降低。
//
// <NT>M warpspecialize 介绍
// 定义：把一整张 GEMM 的流水线拆成几类专职 warp（group），让它们各自只盯一个子阶段、长期不换岗。
// 好处：1) 寄存器利用率：
//        传统的写法中，每个线程都需要负责读写和计算，都需要寄存器，甚至需要做double buffer等多缓冲区，寄存器压力大。
//        Warp-specialized 把 warp 分成 Producer、Consumer、Epilogue 三类，Producer 只用 smem + 少量地址寄存器，Consumer 只用 累加寄存器，Epilogue 只需要写回寄存器。
//      三类warp不会同时出现（barrier时序互斥），寄存器互不重叠。因为在同一个 warp-specialized CTA 内部，Producer warp 产生的数据（shared memory 中的 tile）只会被
//      该 CTA 内部的 Consumer warp 消费。不会、也绝不能被其他 CTA 的 Consumer warp 访问。一个stage的时间线如下：
//        | 周期区间   | 谁发指令                                   | 其余 warp 在干嘛                         | SM 资源视图                             |
//        | --------- | ------------------------------------------ | --------------------------------------- | --------------------------------------- |
//        | 0 … T0    | **Producer warp** 发 `cp.async`/`tma.load` | Consumer warp **停在 `barrier0.wait()`** | **所有 warp 的寄存器槽、SMEM 块都已分配** |
//        | T0+1 … T1 | **Consumer warp** 发 `mma.sync`            | Producer warp **停在 `barrier1.wait()`** | 同上，资源未释放                         |
//        | T1+2 … T2 | **Epilogue warp** 发 `st.global`           | 前两组都在等下一屏障                      | 同上                                    |
//      以4个stage为例（有几个stage，其资源就要分配几份，互不干扰，注意寄存器是复用的前一个stage的，不需要分多份）：
//        | 周期 | Producer warp 在干嘛      | Consumer warp 在干嘛    | 说明                                 |
//        | -- | -------------------------- | ----------------------- | ------------------------------------ |
//        | 0  | **stage-0 TMA load**       | `barrier0.wait()`       | 首次，Consumer 堵门                    |
//        | 1  | **stage-1 TMA load**       | **stage-0 mma.compute** | **Producer-Consumer 第一次重叠**       |
//        | 2  | **stage-2 TMA load**       | **stage-1 mma.compute** | 流水线已建立                           |
//        | 3  | **stage-3 TMA load**       | **stage-2 mma.compute** | 同上                                   |
//        | 4  | **stage-0 下一轮 TMA load** | **stage-3 mma.compute** | Producer **提前为下一条 k-loop 搬数据** |
//      指令发出的顺序是：stage-0 load -> stage-1 load -> stage-0 mma -> stage-2 load -> stage-1 mma，指令的发射完全串行的！！！所以三类warp不会同时存在。
//    而指令发射到操作完成有一定流水线延迟，如TMA/load 从发射到数据落进 shared memory 要 数百周期；mma 从发射到累加结果写回寄存器要 数十周期；
//    基于这段时间差完成load和mma的overlap。
//      另外Hopper 的 register file 是按 warp 槽位静态划分的（每个warp固定256寄存器）。只要一条 warp 的线程没有发射指令，它的整槽 256 个寄存器就不会被读/写，功耗和带宽都被门控掉——等价于“不存在”。 
//    所以三类warp不同时存在（在任何一个时钟周期，只有一类角色（Producer / Consumer / Epilogue）的 warp 处于 Ready/Active 状态并发射指令；
//    其余两类被 barrier 堵住，不发指令、不占寄存器读写带宽）的条件下，峰值寄存器用量 = max(P, C, E) × 256 寄存器/warp。
//      另外寄存器压力小，则 tile 可以更大，如把 tile 尺寸从 128×128 拉到 256×256 后，同样大小的矩阵乘法只需要 1/4 个 CTA 就能盖住整个输出矩阵；
//    CTA 数少了，launch 次数、grid-level 边界处理、L2 流量碎片都同步缩小。
//
//     2）TensorCore利用率：Consumer-warp 独占式发射 mma.sync，无 RAW 气泡、无地址计算穿插。传统 warp-uniform 因 地址/加载指令插入 mma 序列，通常只能到 70–80%。
//     3）指令缓存 & 分支 零压力：每个warp只编译 一段直线代码（搬/算/写），无 runtime 分支、无函数指针、无动态调度器，I-Cache miss 率 < 0.1%；
//            对比传统 kernel 里 “每个线程既做加载又做 mma”，导致 ICache 抖动明显下降。
//     4）完全隐藏 Epilogue 写回延迟：sm80中通过multistage可以重叠load和mma，warpspecialize也有这一层含义，关于这点二者收益相似。
//            而sm80的epi在mainloop之后，所有 warp 一起进入 store 阶段。此时 mainloop 已结束，没有后续计算可掩盖epi，使epi时延完全暴露。
//            但是在warpspecialize下，仅仅是错后一个stage，Consumer-A 算下一块时，Consumer-B 写上一块，mainloop 永不停，可以掩盖epi。
//
// <NT> 为什么sm80不使用warp specialize
// 1）没有TMA: SM80 只有 cp.async 线程级异步拷贝，每个 warp 都要自己算地址、发指令 ,Producer-warps 的指令流仍然挤占 ICache 与发射带宽；
//            SM90 的 TMA 是 SM 级 DMA，一个线程一条指令就把多维 tile 搬完，Producer-warps 代码量骤减 80%，才能真正“轻载”.
// 2）没有WGMMA：SM80 的 mma.sync 是 warp 内同步，每个warp自己算地址、自己发 ldmatrix/ld.shared 加载，再自己发 mma 计算，所以Consumer-warps 也无法做到“纯计算”；
//              SM90 的 wgmma.mma_async，由 Producer-warps（或 TMA）提前把数据搬进 shared memory，Consumer-warps 只发“纯计算”指令 wgmma.mma_async，
//            地址段已由硬件/TMA 隐式完成，计算段可异步继续；因此 Consumer-warps 的指令流里几乎只剩“纯计算”。
//              总之：一个是sync，一个是async。sync的需要自己来处理，没有很好的同步途径；而async的可以由他人帮忙搬数据，自己负责计算即可。SM90 的 async 让“加载-计算”首次在 指令级 解耦。
// 3）无 mbarrier + ordered sequence barrier：M80 只有 CTA 级 __syncthreads() 或 warp-vote 小把戏，想做“warp 级角色同步”得自己拼标志位，开销大且易翻车；
//              SM90 提供 SM 级 mbarrier 与 ordered sequence barrier，一条指令就完成“搬完→算→写完”跨角色同步，零分支零投票。
//
// <NT> H20下，bf16的KN=4096X4096，tuning选择大多是Cooperative，少数是Pingpong (会存在一个shape的tuning前几名都是pingpong的情况)，且未找到明显规律。
namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(!cute::is_same_v<TileScheduler_, StreamKScheduler>, "Ping-pong kernel does not currently support stream-K scheduler.");
  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
                                          TileSchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape,
                                          TileSchedulerPipelineStageCount
                                          >::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;

  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr uint32_t NumMMAThreads = size(TiledMma{});                 // 4 warp 
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads * NumMmaWarpGroups + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
  
  static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
  static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;
  using MathWarpGroupOrderBarrierSharedStorage =
    cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
      MathWarpGroupOrderBarrier::SequenceDepth,
      MathWarpGroupOrderBarrier::SequenceLength>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
    
    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles
      )
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // <NT> * WarpGroupRole 在这里是Producer/Consumer0/Consumer1，所以生产者和消费者的warp group比例是1:2.(注意是warp group而非warp)
    //      * ProducerWarpRole 一个warp group里的warp会进一步分工，里面NumWarpsPerWarpGroup 固定为4 (见include/cutlass/cutlass.h)
    //    所以 warp_idx_in_warp_group 会有0-3，则每一个warp group里的4个warp会分别负责 Mainloop/Warp1/Epilogue/MainloopAux 
    //    每个算法实现里warp组里warp的分工都不一定相同.
    //        其中编号为3的MainloopAux不一定需要(由IsMainloopAuxiliaryLoadNeeded判断)，当不需要Aux时，第4个warp就会没有工作量，
    //    将不会占用sm，但空的warp也会占用Warp Slots，可能会有轻微性能影响。
    //        另外编号为1的Warp1会IsSchedDynamicPersistent所覆盖，而目前sm90的kernel，其值均为false，因此Warp1也不生效。
    //    ProducerWarp里会分，而ConsumerWarp则不分。
    //      * shared_storage指向共享内存，由kernel外的device层级申请，此处kernel内使用。
    //      * lane_predicate 为true时表示在一个warp里被选取出来充当代表的一个线程，其他31个线程将会是false。
    //    如启动tma描述器只需要一个warp中的一个线程进行即可。

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // <NT> 
    // 1) TileScheduler 双流水线：任务领取 + 节流阀，见include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp。
    //      TileSchedulerPipeline 负责 “任务领取”——让 persistent 线程块安全地拿下一个 tile；（sm90的为PipelineEmpty）
    //      TileSchedulerThrottlePipeline 负责 “流量整形”——限制 producer 查询频率，防止 少数 CTA 狂拿任务 导致 负载倾斜。（sm90的为PipelineEmpty）
    //    IsSchedDynamicPersistent：sm90的kernel搭配的TileScheduler均设置为false。
    // 2) 实例化 epilogue 对象，询问是否需要 epilogue-load（C 矩阵预取）；
    // 3) 三条 TMA 流水线，Pipeline类型见 include/cutlass/pipeline/sm90_pipeline.hpp
    //       MainloopPipeline (PipelineTmaAsync类型，如gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)
    //       -> is_leader 为整个warp group中第0号线程
    //       -> num_consumers 为整个warp group的线程数量
    //       -> num_producers 由mainloop给出，通常为1，见NumProducerThreadEvents: include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
    //       EpiLoadPipeline (PipelineTransactionAsync类型，如epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp)
    //       -> 负责 把 global 里的 C/D 矩阵或 bias/scale 载入 smem，供 epilogue 计算单元使用。常常需要 1->N 广播（一个 tile 被 cluster 里多个 CTA 共享）。
    //          必须知道 最后一个消费者 也 arrive 了，才能复用该 pipeline slot -> 需要 arrival-count -> 选 PipelineTransactionAsync<Stages>。
    //       EpiStorePipeline (PipelineTmaStore类型，如epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp)
    //       -> 负责 把算好的结果从 smem 写回 global（C/D 或 O 矩阵）。每个 tile 只写一次，写完即作废；无消费者。
    //       -> 只要 TMA store 完成（scoreboard 位翻起）就能回收slot -> 不需要arrival计数 -> 选 PipelineTmaStore<Stages>（内部仅一条 bulk.commit + bulk.wait）。
    // 4) 两个 OrderedSequenceBarrier (pipeline/sm90_pipeline.hpp):
    //      load_order_barrier:  在producer mainloop中arrive，在producer epi中wait。为单缓冲 + 双生产者交替的最简顺序锁，即深度为1 长度为2。
    //          1) 因为由硬件条件约束，TMA bulk-load 一次只能发一个expect-tx (Hopper)，如果 Depth > 1，
    //          ProducerWarp 必须在同一 warp 内给两行 barrier 同时 expect-tx，硬件会丢包或返回错误，所以深度只能设置为1.
    //          2) producer有 mainloop读取 和 epi读取 两个，mainloop在前，epi在后，有两级顺序关系，所以长度为2.
    //      math_wg_order_barrier: 深度为2 (即stage总数)，长度为2. 在consumer中有两对wait/arrive。
    //          1）消费者需要真正的“双缓冲段间重叠” 来 掩盖 compute 延迟，ProducerWarp 才需要 expect-tx， ConsumerWarp 只做 compute + arrive/wait，
    //       完全不碰 expect-tx，因此ConsumerWarp的barrier行数(深度)可大于1。
    //          2）compute 比 load 慢，若消费者也 深度为1，compute 阶段会堵住 load 阶段，导致气泡 2-3 ?s；因此 消费者必须 Depth=2（或 3） 才能 让 load 与 compute完全重叠。
    //       这里深度是设置为2，也可以看到下面 消费者操作 里，有两对的wait/arrive，对应两个深度。
    //          3）consumer也有 mma计算 和 epi写回 两个，mma计算 在前，epi写回在后，有两级顺序关系，所以长度为2.
    //
    // *expect-tx: 就是“写给 TMA 的收货单”：等多少字节全部到齐，barrier 才翻完成位——写少了提前唤醒，写多了永远睡死。

    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) { 
      if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      }
      else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads;

      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
      bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      } 
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);

      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      }
      // set role when it is for DMA warp in Mainloop
      else if (warp_group_role == WarpGroupRole::Producer &&
               producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop 
        || producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    // <NT> 
    // 1) cluster_wait_fn: Cluster 级同步, >1 CTA 时用 cluster-arrive/cluster-wait，否则普通 __syncthreads
    // 2）problem_shape_MNKL: 把 3-D MNK 补成 4-D MNKL，方便统一处理 batch 维
    // 3) load_inputs：初始化 scheduler、主循环、输入张量切块
    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);
    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }

    if (warp_group_role == WarpGroupRole::Consumer1) {

      if constexpr (not IsSchedDynamicPersistent) {
        // Advance 2nd Math WG to the next work tile for the startup
        scheduler.advance_to_next_work();
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);
    }
    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    // <NT> Producer 和 Consumer 的工作概览
    // Producer Warp Group:
    //    * 使用 warpgroup_reg_dealloc<LoadRegisterRequirement> 按加载所需，当前 warp-group 的可用寄存器上限往下调，
    //      让 SM 可以把腾出来的寄存器重新分配给其它 warpgroup/CTA。
    //    1) ProducerWarpRole::Warp1：
    //        * Scheduler 子 Warp，负责领取新任务。（由IsSchedDynamicPersistent包装，sm90不启用）。
    //    2）ProducerWarpRole::Mainloop：TMA 搬 A/B
    //        * while (work_tile_info.is_valid()) {  // 循环加载所负责的所有块
    //            collective_mainloop.load(mainloop_pipeline, mainloop_pipe_producer_state)       ...
    //            第一个tile加载完成时，调用 load_order_barrier.arrive(); 告知 epilogue load(Producer) warp 可以开始了
    //            scheduler.advance_to_next_work()
    //            work_tile_info = scheduler.get_current_work();
    //          }
    //          collective_mainloop.load_tail()
    //    3）ProducerWarpRole::MainloopAux：（可选）
    //    4）ProducerWarpRole::Epilogue：TMA 搬 C
    //        * while (work_tile_info.is_valid()) {
    //            首次进入，等待Mainloop的第一个tile完成加载 load_order_barrier.wait();
    //            collective_epilogue.load(epi_load_pipeline, epi_load_pipe_producer_state)
    //            scheduler.advance_to_next_work();
    //            work_tile_info = scheduler.get_current_work();
    //          }
    //          collective_epilogue.load_tail();
    //        * 因为scheduler是每个warp都有一份，所以循环获取tile可以用相同的scheduler，每个warp各自从头到尾遍历。
    // Consumer Warp Groups：包含 Consumer0 与 Consumer1
    //    * 使用 warpgroup_reg_alloc<MmaRegisterRequirement> 按mma所需，将可用寄存器上限调高
    //    * while (work_tile_info.is_valid()) {
    //        math_wg_order_barrier.wait();
    //        collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state);
    //        math_wg_order_barrier.arrive();
    //        collective_mainloop.mma_tail();
    //
    //        math_wg_order_barrier.wait();
    //        collective_epilogue.store(epi_load_pipeline, epi_load_pipe_consumer_state，
    //                                  epi_store_pipeline, epi_store_pipe_producer_state)
    //        collective_epilogue.store_tail()
    //        math_wg_order_barrier.arrive();
    //
    //        scheduler.advance_to_next_work(NumMmaWarpGroups);
    //        work_tile_info = scheduler.get_current_work();
    //      }
    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    
      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) { 
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();

          while (work_tile_info.is_valid()) {
            
            if (requires_clc_query) {

              // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              // Query next work tile
              scheduler_pipe_producer_state = scheduler.advance_to_next_work(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            // Fetch next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
            
            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }

          // Terminal condition - if work_tile_info is end-of-grid, produce an extra invalid tile
          scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
          scheduler.store_invalid_response(scheduler_pipe_producer_state); // Push invalid tile to smem
          scheduler_pipeline.producer_commit(scheduler_pipe_producer_state); // Manual completion of transaction
          ++scheduler_pipe_producer_state;

          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        } 
      } // Scheduler Producer Warp End  
      else
      
      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            blk_coord,
            k_tile_iter, k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
        
      } // Mainloop Producer Warp End

      else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
        if constexpr (IsMainloopAuxiliaryLoadNeeded) {
          // Ensure that the prefetched kernel does not touch
          // unflushed global memory prior to this instruction
          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
            collective_mainloop.load_auxiliary(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter, k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop
            );
            // Update starting pipeline state for the next tile
            mainloop_pipe_producer_state.advance(k_tile_count);

            scheduler.advance_to_next_work();
            work_tile_info = scheduler.get_current_work();
          } // Scheduler work fetch loop

          // Make sure all Consumer Warp Groups have been waited upon
          collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

          if constexpr (IsSchedDynamicPersistent) {  
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                work_tile_info,
                scheduler_pipeline,
                scheduler_pipe_consumer_state
              );
          }
          
        }
      }

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {

        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();

        bool do_load_order_wait = true;
        while (work_tile_info.is_valid()) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          epi_load_pipe_producer_state =
          collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);

        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
      // It is possible to have work tiles start off invalid,
      // so we have to check that first.
      if (not work_tile_info.is_valid()) {
        // Hint on an early release of global memory resources.
        // The timing of calling this function only influences performance,
        // not functional correctness.
        cutlass::arch::launch_dependent_grids();

        return;
      }
      #endif
      
      if constexpr (IsSchedDynamicPersistent) {
        // Consumer0's initial tile is static. It starts consuming the 2nd tile.
        if (warp_group_role == WarpGroupRole::Consumer0) {
            ++scheduler_pipe_consumer_state;
        } 

        if (warp_group_role == WarpGroupRole::Consumer1) {
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        } 
      }

      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Allocate the accumulators for the (M,N) blk_shape
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();

        collective_mainloop.mma(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          accumulators,
          k_tile_count,
          warp_group_thread_idx,
          shared_storage.tensors.mainloop,
          params.mainloop
        );

        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();

        // Make sure the math instructions are done and free buffers before entering the epilogue
        collective_mainloop.mma_tail(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          k_tile_count
        );
        // Update starting mainloop pipeline state for the next tile
        mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups);

        #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
        if (scheduler.is_last_tile(work_tile_info, NumMmaWarpGroups)) {
          // Hint on an early release of global memory resources.
          // The timing of calling this function only influences performance,
          // not functional correctness.
          cutlass::arch::launch_dependent_grids();

        }
        #endif

        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();

        // Epilogue and write to gD
        auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
        collective_epilogue.store(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state,
          problem_shape_MNKL,
          blk_shape,
          blk_coord,
          accumulators,
          tiled_mma,
          warp_group_thread_idx,
          shared_storage.tensors.epilogue
        );

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer kernels
        // we need to wait for all TMA stores to complete before issuing consumer order barrier arrives
        // to ensure next math consumer doesn't overwrite smem of in-flight TMA stores of current consumer.
        auto [epi_load_pipe_consumer_state_next_, epi_store_pipe_producer_state_next_] =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state_next,
          epi_store_pipeline,
          epi_store_pipe_producer_state_next
        );

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for ping pong
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();

        if constexpr (IsSchedDynamicPersistent) {  
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        }
        else {
        // Get next work tile
        scheduler.advance_to_next_work(NumMmaWarpGroups);
        work_tile_info = scheduler.get_current_work();
        }
      } // Scheduler work fetch loop
    } // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
