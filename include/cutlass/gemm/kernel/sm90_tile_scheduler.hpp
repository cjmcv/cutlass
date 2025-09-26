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

#include "cutlass/gemm/kernel/static_tile_scheduler.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////
// <NT> PersistentTileSchedulerSm90: 派生于StaticPersistentTileScheduler基类，负责提供 persistent-loop 骨架（while 还有 tile 就继续领）。
// 只需要实现get_work_idx_m_and_n()，不需要写循环逻辑。因此 PersistentTileSchedulerSm90 具备 persistent 能力，但是否“真的持久”由 launch 端决定。
// * get_work_idx_m_and_n函数：给定一个线性全局 block ID（blk_per_grid_dim），算出它应该负责计算的是哪个输出 tile（m, n）。
//   -> log_swizzle_size： cluster swizzle尺寸，通常是3/4，目的是打散相邻 cluster 对 L2 cache 的冲突，如取3，则每8 cluster会被重新排列，
//                     避免连续的 cluster 访问连续的内存区域。其本质是block swizzle，只是有了cluster后，需要以cluster为单位，
//                     cluster内的block需要打包在一起，内部的 block->tile 的映射是线性连续的, 不需要再做swizzle。swizzle的对象只有cluster_id。
//                     * 从存储角度，一个 cluster 所有 CTA 共享同一块 L2 / TMA 资源，相邻 thread-block 访问相邻 tile 反而有利于 cache 预取.
//                         -> 更多block swizzle信息，见 include/cutlass/gemm/threadblock/threadblock_swizzle.h
//                         -> thread swizzle 见 include/cute/swizzle.hpp
//   -> raster_order: 沿 M 还是 N 方向优先遍历.
//   -> divmod_cluster_shape_major / divmod_cluster_shape_minor: major和minor是两个维度方向，如raster_order是AlongN，则major就是 N 维度, minor则是剩下的 M 维度.
//                    索引计算均以 major/minor 来计算。divmod_cluster_shape_major 表示整个grid在 major 方向 的 cluster 个数。
//   -> divmod_cluster_blk_major: 一个 major 行里有多少个 cluster, 与divmod_cluster_shape_major不相等。区别在于swizzle。
//
// * 其他函数见 include/cutlass/gemm/kernel/static_tile_scheduler.hpp
//              
// * IsDynamicPersistent = false 表示使用 静态persistent 模式，即每个 CTA/SM 负责固定范围的 tile，不依赖运行时动态分配。
//   sm90仅支持 静态persistent (如此处的PersistentTileSchedulerSm90和隔壁的PersistentTileSchedulerSm90StreamK/PersistentTileSchedulerSm90Group 均设为false)
//   但sm90的kernel层级(pingpong/collective)却留了 if constexpr (IsSchedDynamicPersistent) 的分支，主要是为了后续扩展动态调度器，
//   如sm100的tile scheduler就默认都设置 IsDynamicPersistent = true。
// 
// * ThrottlePipeline：动态限流机制，会在 SM 级维护一个剩余槽位计数器，当已驻留 CTA 数达到上限时，新 CTA 主动 wait() 在 grid 入口，
//   直到有前辈 CTA 退休才 arrive() 放行——避免一次性涌入过多 CTA 导致寄存器/SMEM 溢出、频率下降。sm90的TileScheduler的ThrottlePipeline
//   均为PipelineEmpty，即都不做限流处理。而sm100的TileScheduler开始有做限流。
//              
// <NT> presistent thread 模式的官方定义: A kernel is persistent if the number of CTAs launched is independent of the total amount of work, and each CTA iterates until all work is consumed.
// 关键词是iterates，只要存在 “for / while 循环” 让同一 CTA 连续处理 ≥1 个 work unit，就算 persistent；启动的CTA和工作量可以解耦。
// 如：最原始的CUDA写法DP模式，按grid给矩阵划分tile，每个block处理一个tile，每个block完成一次任务后退出，即可结束。
//    当矩阵增大时block的数量也更多，则不属于 persistent。属于最传统的 grid-level non-persistent 启动模型。
// 如：streamk的是 常驻不退出，自己领任务，block数量与任务量解耦，属于典型的 persistent。
//    SM90使用的PersistentTileSchedulerSm90StreamK，实现 K 段级 persistent。
// 如：这里的PersistentTileSchedulerSm90，针对的是与streamk同级别相对应的DP模式，
//    通过PersistentTileSchedulerSm90使DP也具备persistent的能力，CTA数量不随任务量增大而增大
//    从而实现tile级别的persistent。
//
// 补充：1）CTA==block，经常混用，二者描述的物理实体完全相同，只是语境不同，block出现在runtime API，而CTA出现在PTX手册与架构白皮书。
//      2）sm90的warp specialize与streamk无冲突，既可以选用此处的PersistentTileSchedulerSm90，也可以选用隔壁的PersistentTileSchedulerSm90StreamK。
//         只是目前Pingpong还不支持PersistentTileSchedulerSm90StreamK - cutlass 4.2

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90:
public StaticPersistentTileScheduler<PersistentTileSchedulerSm90> {

  using BaseScheduler = StaticPersistentTileScheduler<PersistentTileSchedulerSm90>;
public:
  using StaticPersistentTileScheduler::StaticPersistentTileScheduler;
  using Params = PersistentTileSchedulerSm90Params;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using Arguments = BaseScheduler::Arguments;

  static constexpr bool IsDynamicPersistent = false;

  using Pipeline = PipelineEmpty;
  using PipelineStorage = typename Pipeline::SharedStorage;
  using ThrottlePipeline = PipelineEmpty;
  using ThrottlePipelineStorage = typename ThrottlePipeline::SharedStorage;

  struct CLCResponse {};

  class SharedStorage {
  public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return PipelineStorage{}; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE CLCResponse* data() { return nullptr; }
  };

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    return get_work_idx_m_and_n(
      blk_per_grid_dim,
      divmod_cluster_shape_major,
      divmod_cluster_shape_minor,
      divmod_cluster_blk_major,
      log_swizzle_size,
      raster_order,
      cta_m_in_cluster,
      cta_n_in_cluster
    );
  }

  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order,
      uint64_t cta_m_in_cluster,
      uint64_t cta_n_in_cluster) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor +
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }

  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1, uint32_t = 1, CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

};

}
