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
// <NT> PersistentTileSchedulerSm90: ������StaticPersistentTileScheduler���࣬�����ṩ persistent-loop �Ǽܣ�while ���� tile �ͼ����죩��
// ��ʵ��ֻ��Ҫʵ��get_work_idx_m_and_n()������Ҫдѭ���߼������ PersistentTileSchedulerSm90 �߱� persistent ���������Ƿ���ĳ־á��� launch �˾�����
// IsDynamicPersistent = false ��ʾ �Ƕ�̬ persistent�������� ������ʱ��ʣ�� tile �������Ž� global memory��Ҳ��֧�ֿ� CTA ������
// TODO...
//
// <NT> presistent thread ģʽ�Ĺٷ�����: A kernel is persistent if the number of CTAs launched is independent of the total amount of work, and each CTA iterates until all work is consumed.
// �ؼ�����iterates��ֻҪ���� ��for / while ѭ���� ��ͬһ CTA �������� ��1 �� work unit������ persistent��������CTA�͹��������Խ��
// �磺��ԭʼ��CUDAд��DPģʽ����grid�����󻮷�tile��ÿ��block����һ��tile��ÿ��block���һ��������˳������ɽ�����
//    ����������ʱblock������Ҳ���࣬������ persistent�������ͳ�� grid-level non-persistent ����ģ�͡�
// �磺streamk���� ��פ���˳����Լ�������block������������������ڵ��͵� persistent��
//    SM90ʹ�õ�PersistentTileSchedulerSm90StreamK��ʵ�� K �μ� persistent��
// �磺�����PersistentTileSchedulerSm90����Ե�����streamkͬ�������Ӧ��DPģʽ��
//    ͨ��PersistentTileSchedulerSm90ʹDPҲ�߱�persistent��������CTA�����������������������
//    �Ӷ�ʵ��tile�����persistent��
//
// ���䣺1��CTA==block���������ã���������������ʵ����ȫ��ͬ��ֻ���ﾳ��ͬ��block������runtime API����CTA������PTX�ֲ���ܹ���Ƥ�顣
//      2��sm90��warp specialize�����PersistentTileSchedulerSm90�䵱tile scheduler����

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
