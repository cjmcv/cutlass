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
    \brief Implements streamk threadblock mapping blockIdx to GEMM problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/gemm/threadblock/index_remat.h"

#if !defined(__CUDACC_RTC__)
#include <iostream>
#include "cutlass/core_io.h"
#include "cutlass/trace.h"
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock mapping control for GEMMs
struct ThreadblockSwizzleStreamK {

  /// Advertise StreamkFeature
  using StreamkFeature = void;


  /// Kernel traits
  template <typename GemmKernel>
  struct KernelTraits {};


  /// Reduction strategy
  enum ReductionStrategy
  {
    kNone,      // Data-parallel strategy (no seams, fixup, etc.)

    kAtomic,    // Non-deterministic reduction of SK-block partials using atomic aggregation in L2

    kMixed,     // Deterministic reduction of SK-block partials employing either:
                //   (a) A separate wave of reduction thread blocks" (for scenarios with lots of
                //       SK-blocks per SK-tile)
                //   (b) Turnstile-ordered atomic aggregation in L2 (for scenarios with few
                //       SK-blocks per SK-tile)
  };

  static ReductionStrategy const kReductionStrategy = kMixed;


  //
  // Heuristics
  //
  // <NT> stream-k swizzle的启发式参数设置
  // kDpEfficiencyThreshold：Data-Parallel 效率阈值，实现里会先估算“如果完全走 DP (即每个 CTA 算一整条 K 维）”的负载均衡效率：
  //      eff_dp = (真正要做的 MAC 总数) / (ceil(M*K/(cta_m*k)) * ceil(N*K/(cta_n*k)) * cta_k)
  //      当 eff_dp ≥ 0.92 时，认为负载足够均匀，直接用传统 data-parallel 更快；否则才启用 Stream-K。
  //      0.92 是经验值，防止为了 8% 的负载不均去承担 Stream-K 的额外同步/修首尾开销。
  // kMinItersPerSkBlock：每个 Stream-K 块最少要认领的 MAC-iterations 数
  //      Stream-K 把 K 维切成若干“iter 段”，一个 CTA 一次领一段。如果某段太短，领取、写回 partial accumulator 的固定开销就会盖过计算收益。
  //      因此硬性规定：不管负载怎么切，一段不得少于 2 个迭代（即 2*k 个 K 维元素）。这个值与 kernel 里 accumulator 寄存器生命期、
  //      shared-memory 双缓冲粒度有关，2 是 CUTLASS 测出来的“最小不亏”长度。
  // kCtasPerCohort / kCohortCtasM / kCohortCtasN：将8x4个CTA作为一组，称为cohort。
  //      代码里用它一次性 malloc 出 32 个 CTA 的 partial accumulator buffer，也用来算“需要多少波 cohort 才能覆盖整个输出矩阵”.
  // kFixupStartupIterEquiv / kFixupPeerIterEquiv: 这两个把“修首尾”阶段的各种 memory 操作折算成“等效 MAC-iterations”，方便在启发式公式里与真正计算量直接比较。
  //      startup：一个 CTA 做“first wave”时要初始化为 0、写 global accumulator、读完后再读回做 reduce，CUTLASS 实测这串操作大约相当于 10 次 mac-iter 的计算时间。
  //      peer：同一个 cohort 里前后 CTA 之间把 partial acc 累加一次（shared→register→shared→global）的开销，大约等效 3 次 mac-iter。
  //        调度器在估算“用 Stream-K 到底划不划算”时，会把这些常量直接加到总代价里：
  //      cost_sk = mac_iters + n_cohort * kFixupStartupIterEquiv + n_peer * kFixupPeerIterEquiv
  //      只有当 cost_sk < cost_dp 时才真正走 Stream-K 路径。

  /// Data-parallel wave-quantization efficiency threshold (above which we go data-parallel)
  static float constexpr kDpEfficiencyThreshold = 0.92f;

  /// Minimum number of MAC-iterations per streamk block
  static int const kMinItersPerSkBlock = 2;

  /// Height in CTAs of a grid rasterization cohort
  static int const kCohortCtasM = 8;

  /// Width in CTAs of a grid rasterization cohort
  static int const kCohortCtasN = 4;

  /// Number of CTAs per cohort
  static int const kCtasPerCohort = kCohortCtasN * kCohortCtasM;

  /// Cost-equivalent number of SM-iterations for fixup I/O
  static int const kFixupStartupIterEquiv = 10;
  static int const kFixupPeerIterEquiv = 3;


  //
  // Member state
  //


  /// The 3D value-extents of the GEMM computation volume (m,n,k)
  GemmCoord problem_size;

  /// Div/mod accelerators
  FastDivmod div_mod_tiled_shape_m;
  FastDivmod div_mod_tiled_shape_n;
  FastDivmod div_mod_tiled_cohort_shape_n;
  FastDivmod div_mod_iters_per_tile;

  /// Whether to perform cohort CTA rasterization
  bool cohort_raster;

  // Whether to pad and remap block indices
  bool remap_block_indices;

  /// CTA occupancy per SM
  int sm_occupancy;

  /// Number of SMs for dispatch heuristics to load-balance using Stream-K CTAs (wave size)
  int avail_sms;

  int dp_blocks;                            /// Number of data-parallel thread blocks in the grid
  int dp_first_wave_tiles;                  /// Number of output tiles each CTA in the first DP wave will produce

  /// Number of reduction blocks in the grid
  int reduction_blocks;

  int sk_waves;
  int sk_tiles;
  int sk_big_blocks_per_region;
  int sk_iters_per_region;

  /// Div/mod accelerators
  FastDivmod div_mod_sk_iters_per_normal_block;
  FastDivmod div_mod_sk_iters_per_big_block;
  FastDivmod div_mod_sk_iters_per_region;
  FastDivmod div_mod_sk_regions;                      //!! used in block map
  FastDivmod div_mod_sk_blocks_per_region;            //!! used in block map

  /// The batch count
  int batch_count;


  //
  // Host+device interface
  //

  /// Constructor
  ThreadblockSwizzleStreamK() = default;

  /// Returns the GEMM volume in thread block tiles
  CUTLASS_HOST_DEVICE
  GemmCoord tiled_shape() const
  {
    return GemmCoord(
        static_cast<int>(div_mod_tiled_shape_m),
        static_cast<int>(div_mod_tiled_shape_n),
        batch_count);
  }

  /// Number of iterations per output tile
  CUTLASS_HOST_DEVICE
  int iters_per_tile() const
  {
    return static_cast<int>(div_mod_iters_per_tile);
  }

  /// Number of iterations for normal SK-blocks
  CUTLASS_HOST_DEVICE
  int sk_iters_per_normal_block() const
  {
    return static_cast<int>(div_mod_sk_iters_per_normal_block);
  }

  /// Number of SK regions
  CUTLASS_HOST_DEVICE
  int sk_regions() const
  {
    return static_cast<int>(div_mod_sk_regions);
  }

  /// Number of SK blocks per region (splitting factor)
  CUTLASS_HOST_DEVICE
  int sk_blocks_per_region() const
  {
    return static_cast<int>(div_mod_sk_blocks_per_region);
  }


  //
  // Host-side interface
  //

  /// Debug print
  void Print()
  {
#ifndef __CUDA_ARCH__
    auto tiles = tiled_shape().mn().product();
    std::cout <<
        "problem_size: (" << problem_size.m() << "," << problem_size.n() << ")" <<
        ", tiled_shape: (" << tiled_shape().m() << "," << tiled_shape().n() << ")" <<
        ", tiles: " << tiles <<
        ", dp_tiles: " << tiles - sk_tiles <<
        ", sk_tiles: " << sk_tiles <<
        ", iters_per_tile: " << iters_per_tile() <<
        ", reduction_blocks: " << reduction_blocks <<
        ", dp_blocks: " << dp_blocks <<
        ", dp_waves: " << dp_blocks / avail_sms <<
        ", dp_first_wave_tiles: " << dp_first_wave_tiles <<
        ", sk_blocks_per_region: " << sk_blocks_per_region() <<
        ", sk_regions: " << sk_regions() <<
        ", sk_waves: " << sk_waves <<
        ", sk_iters_per_normal_block: " << sk_iters_per_normal_block() <<
        ", sk_big_blocks_per_region: " << sk_big_blocks_per_region <<
        ", remap_block_indices: " << remap_block_indices <<
        ", cohort_raster: " << cohort_raster <<
        ", sm_occupancy: " << sm_occupancy <<
        ", avail_sms: " << avail_sms <<
        ", num_blocks: " << get_num_blocks() <<
        "\n\n";
#endif
  }


  // <NT> 给定一批要由 Stream-K 完成的 tile（sk_tiles），算出到底发射多少个 SK-CTA（sk_blocks）才最划算——划算的标准是“相比纯 DP 残尾，能省多少迭代开销”。
  // 参数：
  //   sk_tiles	准备交给 SK 的输出 tile 个数
  //   iters_per_tile	每个 tile 在 K 维必须串行完成的迭代次数
  //   avail_sms	物理 SM 数量
  //   max_sk_occupancy	每个 SM 最多能同时 resident 的 SK-CTA 数
  //   allow_partial_wave	是否允许 SK-CTA 数不足一整层波浪
  //   sk_blocks	出参 推荐发射的 SK-CTA 数量
  //   savings_iters	出参 相比纯 DP 残尾，预计能 节省的迭代次数（负值表示不划算）
  // 步骤：
  //   1）计算SK总迭代量并建立建立“纯 DP 残尾”对照组：如果不用 SK，这些残尾 tile 至少需要 dp_equiv_waves 层波浪，
  //      每层波浪里每个 CTA 都要跑满 iters_per_tile 次迭代，总工作量就是 dp_equiv_iters。
  //   2）确定搜索范围 min_sk_blocks：至少发一层 SM 或者干脆把每个 tile 分给一个 CTA。
  //                  max_sk_blocks：不能超过 SM 满载槽位，也不能让单个 SK-CTA 的迭代量低于 kMinItersPerSkBlock（否则 launch 开销不划算）。
  //   3）遍历所有 trial_sk_blocks 对每一个候选值，计算：
  //      sk_waves（SK 需要几层波浪）/ max_sk_iters_per_block（最忙的 CTA 要做几次迭代）/ sk_iter_equiv（SK 方案的总“等效迭代”（用来与 DP 对比）
  //   4）成本模型，所有开销都 折算成“等效迭代次数”，方便与 dp_equiv_iters 直接相减，选出最优
  //          base_cost  = 2.0f * sk_waves              // 启动更多 waves 的固定开销
  //          peer_cost  = 2.0f * num_peers             // 跨 tile reduce 的开销
  //          iter_cost  = 0.02f * num_peers * sk_iter_equiv  // 非对齐导致的不规则迭代惩罚
  //          fixup_iter_equiv = base_cost + peer_cost + iter_cost

  // Compute sk_blocks to dispatch for a given number of sk_tiles
  static void get_sk_blocks(
    int &sk_blocks,     /// [out]
    int &savings_iters, /// [out]
    int sk_tiles,
    int iters_per_tile,
    int avail_sms,
    int max_sk_occupancy,
    bool allow_partial_wave)
  {
    savings_iters = INT_MIN;
    sk_blocks = 0;

    if (sk_tiles == 0) {
      return;
    }

    int sk_iters = sk_tiles * iters_per_tile;

    int dp_equiv_waves = (sk_tiles + avail_sms - 1) / avail_sms;
    int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

    int min_sk_blocks = (allow_partial_wave) ? fast_min(avail_sms, sk_tiles + 1) : avail_sms;
    int max_sk_blocks = fast_min(avail_sms * max_sk_occupancy, sk_iters / kMinItersPerSkBlock);

    for (int trial_sk_blocks = min_sk_blocks; trial_sk_blocks <= max_sk_blocks; ++trial_sk_blocks)
    {
      int sk_waves = (trial_sk_blocks + avail_sms - 1) / avail_sms;
      int max_sk_iters_per_block = (sk_iters + trial_sk_blocks - 1) / trial_sk_blocks;
      int sk_iter_equiv = max_sk_iters_per_block * sk_waves;

      int num_peers = ((trial_sk_blocks + sk_tiles - 1) / sk_tiles) + 1;        // add one for alignment skew

      float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

      if (trial_sk_blocks % sk_tiles == 0)
      {
        // aligned
        num_peers = (trial_sk_blocks / sk_tiles);

        iter_cost = 0.0f;
      }

      float peer_cost = 2.0f * float(num_peers);

      float base_cost = 2.0f * float(sk_waves);

      int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

      int trial_savings_iters = dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

      if (trial_savings_iters >= savings_iters) {
          savings_iters = trial_savings_iters;
          sk_blocks = trial_sk_blocks;
      }
    }
  }

  // <NT> 给定输出tile数、每个tile的迭代次数-iters_per_tile、可用SM数、SM满载CTA数-sm_occupancy，决定多少 tile 走传统 DP，多少 tile 用 Stream-K.
  // 名词：iters_per_tile：针对的是输出tile，一个tile需要把整个K维度规约完。
  //                     如K=4096，k_tile=128，则iters_per_tile=4096/128=32。若把 40 个残尾 tile 交给 SK，则总迭代量 = 40 × 32 = 1280 次 scan-reduce 循环。
  //       sm_occupancy：在给定 寄存器用量 + 共享内存用量 + 线程块大小 后，一个 SM 同时能 resident 的 CTA 个数。
  //                     由硬件极限与内核资源同时决定：
  //                     occupancy = min(maxThreadsPerSM / blockThreads,
  //                                     maxRegistersPerSM / blockRegisters,
  //                                     maxSharedMemPerSM / blockSharedMem,
  //                                     maxCTAsPerSM)
  // 思路：
  // 1) 先算“完整波浪”和“残尾波浪”。如残尾为0，即Perfect quantization，则全部 DP，直接返回。
  // 2) 如果当前总 waves 不足一次满载（full_waves < sm_occupancy）：
  //      * 把残尾 tile 全部交给 SK，让 SK CTA 数刚好补齐到满载（max_sk_occupancy = sm_occupancy - full_waves）。
  //      * 用 get_sk_blocks() 评估“划算度”（score），若正收益则采纳；否则回退全部 DP。
  // 3) 如果已超过或等于一次满载，但满足“再补一层即可再次整除”的特殊情况（full_waves % sm_occupancy == sm_occupancy-1）：
  //      * 尝试用 1 层 SM 的 SK 去吃掉残尾，使总 waves 再次对齐。
  // 4) 其余情况（常见“最后一层缺一点”）：
  //      * 把倒数第二层完整波浪也拉出来（dp_tiles -= avail_sms），与残尾合并成一个大池子，全部交给 SK 去“填平”。
  //      * 此时要求 SK 必须发满一整层（max_sk_occupancy = ... 且最后一个参数 false），否则认为不划算，回退全部 DP。
  //
  // 举例：output_tiles = 1080, avail_sms = 80, sm_occupancy = 2 -> full_waves = 13, partial_wave_tiles = 40
  //      此时 full_waves (13) > sm_occupancy (2) 且 13 % 2 == 1，命中第 4 个分支：
  //    尝试用 1 层 SM（80 个）SK 去吃掉 40 个残尾 tile；
  //    若 get_sk_blocks 返回 score ≥ 0，则 dp_tiles = 13*80 = 1040, sk_blocks = 80；
  //    否则回退 dp_tiles = 1080, sk_blocks = 0。

  /// Determine the populations of DP and SK blocks to invoke for the given number of output tiles
  static void get_blocks(
    int &dp_tiles,      /// [out]
    int &sk_blocks,     /// [out]
    int output_tiles,
    int iters_per_tile,
    int avail_sms,
    int sm_occupancy)
  {
    int full_waves = output_tiles / avail_sms;
    int full_wave_tiles = full_waves * avail_sms;
    int partial_wave_tiles = output_tiles - full_wave_tiles;

    int score = -1;
    dp_tiles = output_tiles;
    sk_blocks = 0;

    if (partial_wave_tiles == 0)
    {
      // Perfect quantization
      return;
    }

    if (full_waves < sm_occupancy)
    {
        // We're less than full GPU occupancy

        // Form the SK wave from the partial wave to get us up to full GPU occupancy
        int max_sk_occupancy = sm_occupancy - full_waves;

        dp_tiles = full_wave_tiles;

        get_sk_blocks(
          sk_blocks,
          score,
          partial_wave_tiles,
          iters_per_tile,
          avail_sms,
          max_sk_occupancy,
          true);                 // we can run with less than a full wave of SK-blocks

        if (score < 0) {
          // not profitable
          sk_blocks = 0;
          dp_tiles = output_tiles;
        }

        return;
    }

    // We're at (or greater) than GPU occupancy

    if ((sm_occupancy > 1 ) && (full_waves % sm_occupancy == sm_occupancy - 1))
    {
        // If occupancy is more than one CTA per SM, form the SK wave from the partial
        // wave to get us to full GPU occupancy
        int max_sk_occupancy = 1;

        dp_tiles = full_wave_tiles;

        get_sk_blocks(
          sk_blocks,
          score,
          partial_wave_tiles,
          iters_per_tile,
          avail_sms,
          max_sk_occupancy,
          true);                 // we can run with less than a full wave of SK-blocks

        if (score >= 0) {
            return;
        }
    }

    // Form the SK wave by combining the last full wave and the partial wave
    // We're less than full GPU occupancy
    dp_tiles = full_wave_tiles - avail_sms;

    int max_sk_occupancy = sm_occupancy - ((full_waves - 1) % sm_occupancy);

    get_sk_blocks(
      sk_blocks,
      score,
      partial_wave_tiles + avail_sms,
      iters_per_tile,
      avail_sms,
      max_sk_occupancy,
      false);                 // we cannot run with less than a full wave of SK-blocks

    if (score < 0) {
      // not profitable
      sk_blocks = 0;
      dp_tiles = output_tiles;
    }

  }

  /// Constructor: *Gemm* problem size (m, n, k)
  ThreadblockSwizzleStreamK(
    GemmUniversalMode const mode_,
    GemmCoord const problem_size_,
    GemmCoord const tile_size_,
    int const batch_split_,                        /// Either (mode == GemmUniversalMode::kBatched) the batch count, or (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1 defaults to StreamK, >1 emulates Split-K)
    int const sm_occupancy_,
    int const device_sms_,
    int const avail_sms_,                          /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)
    size_t const element_A_bytes_,
    size_t const element_B_bytes_,
    size_t const element_C_bytes_,
    int const epilogue_acc_fragments_)
  :
    problem_size(problem_size_),
    batch_count((mode_ == GemmUniversalMode::kBatched || mode_ == GemmUniversalMode::kArray) ? batch_split_ : 1),
    reduction_blocks(0),
    dp_blocks(0),
    dp_first_wave_tiles(1),     // Default: one tile per DP-block in the first wave of DP blocks
    sk_tiles(0),
    sk_big_blocks_per_region(0),
    sk_iters_per_region(0),
    sk_waves(0),
    sm_occupancy(sm_occupancy_),
    remap_block_indices(false),
    avail_sms(fast_max(1, avail_sms_)),
    cohort_raster(false)
  {
    int gpu_occupancy = device_sms_ * sm_occupancy;
    int iters_per_tile = (problem_size.k() + tile_size_.k() - 1) / tile_size_.k();
    int sk_iters_per_normal_block = 0;

    int sk_regions = 1;              // Default: a single region of iteration space (across all SK tiles)
    int sk_blocks_per_region = 0;

    GemmCoord tiled_shape(
      (problem_size.m() + tile_size_.m() - 1) / tile_size_.m(),
      (problem_size.n() + tile_size_.n() - 1) / tile_size_.n(),
      batch_count);

    size_t problem_bytes =
              (element_C_bytes_ * problem_size.m() * problem_size.n()) +
              (element_A_bytes_ * problem_size.m() * problem_size.k()) +
              (element_B_bytes_ * problem_size.k() * problem_size.n());

    size_t problem_flops = size_t(problem_size.m()) * size_t(problem_size.n()) * size_t(problem_size.k()) * 2;

    [[maybe_unused]] float flops_per_byte = float(problem_flops) / float(problem_bytes);

    int output_tiles = tiled_shape.m() * tiled_shape.n();
    int waves = (output_tiles + avail_sms - 1) / avail_sms;
    [[maybe_unused]] float dp_efficiency = float(output_tiles) / float(waves * avail_sms);

    //
    // Determine dispatch composition of DP-tiles and SK-blocks
    //

    // Start with a DP-only configuration
    int dp_tiles = output_tiles;    // Number of data-parallel tiles
    int sk_blocks = 0;              // Number of thread blocks to produce the remaining SK tiles

    // Only kGemm mode allows for SK load balancing
    if (mode_ == GemmUniversalMode::kGemm)
    {
      int split_factor = batch_split_;
      if (split_factor > 1)
      {
        // Split-K override
        dp_tiles = 0;
        sk_blocks = output_tiles * split_factor;
      }
      else if ((kReductionStrategy != kNone) &&   // Load-balancing strategy statically enabled
        (avail_sms > 1))                         // Plurality of SMs to load balance across
      {
        // Use heuristics
        get_blocks(
          dp_tiles,      /// [out]
          sk_blocks,     /// [out]
          output_tiles,
          iters_per_tile,
          avail_sms,
          sm_occupancy);
      }
    }

    sk_tiles = output_tiles - dp_tiles;


    // Compute SK block iteration details
    if (sk_blocks > 0)
    {
      sk_waves = (sk_blocks + avail_sms - 1) / avail_sms;

      int sk_iters = sk_tiles * iters_per_tile;
      sk_blocks = fast_min(sk_blocks, sk_iters);

      sk_iters_per_normal_block = sk_iters / sk_blocks;
      int extra_sk_iters = sk_iters - (sk_iters_per_normal_block * sk_blocks);
      int sk_big_blocks = extra_sk_iters;

      if ((sk_blocks > sk_tiles) && (sk_blocks % sk_tiles == 0))
      {
        // Split-K decomposition
        sk_regions = sk_tiles;
      }

      sk_blocks_per_region = sk_blocks / sk_regions;
      sk_big_blocks_per_region = sk_big_blocks / sk_regions;
      sk_iters_per_region = sk_iters / sk_regions;

      // Use a separate reduction wave when all of:
      // - Non-atomic reduction stratgy
      // - The number of SK waves won't fully occupy the GPU (Otherwise we don't have
      //   a strong-scaling case for more parallel reduction)
      // - More than three peers working on an SK tile.  (This occurs when the ratio of
      //   SK-blocks to SK-tiles > 2, as a single tile may be covered by four SK-blocks,
      //   e.g.:[partial-block | block | block | partial-block] ).  With three or
      //   less peers, the two non-finishing SK-blocks are not expected to contend.
      if ((kReductionStrategy == kMixed) &&
          (sk_waves < sm_occupancy) &&
          (sk_blocks > 2 * sk_tiles))
      {
        // Launch a reduction block for every accumulator fragment in each SK-tile
        reduction_blocks = sk_tiles * epilogue_acc_fragments_;

      }

      // When we have a multi-occupancy kernel and at least two waves of active blocks (where
      // at least one wave is SK blocks), we need to (1) dispatch at least four waves, and (2)
      // remap the block indices so that we can reliably spread the SK blocks evenly across the
      // device's first SM occupancy valence. Also see get_num_blocks() and get_block_idx().
      remap_block_indices = (
          (sm_occupancy > 1) &&
          (device_sms_ == avail_sms) &&
          (get_num_active_blocks() > avail_sms * 2));

      // Initialize fast div/mod members related to SK
      div_mod_sk_iters_per_normal_block = FastDivmod(sk_iters_per_normal_block);
      div_mod_sk_iters_per_big_block = FastDivmod(sk_iters_per_normal_block + 1);
      div_mod_sk_iters_per_region = FastDivmod(sk_iters_per_region);
      div_mod_sk_regions = FastDivmod(sk_regions);
      div_mod_sk_blocks_per_region = FastDivmod(sk_blocks_per_region);
    }

    //
    // Compute DP blocks
    //

    dp_blocks = dp_tiles;

    cutlass::gemm::GemmCoord tiled_cohort_shape(
        (tiled_shape.m() + kCohortCtasM - 1) / kCohortCtasM,
        (tiled_shape.n() + kCohortCtasN - 1) / kCohortCtasN,
        tiled_shape.k());
    int cohort_blocks = (tiled_cohort_shape.m() * tiled_cohort_shape.n()) * kCtasPerCohort;
    float cohort_efficiency = float(dp_blocks) / float(cohort_blocks);

    // Check if the SK tiles would be in cohorts that are in-bounds
    bool sk_in_range = true;
    if (sk_tiles > 0)
    {
      int last_sk_tile = sk_tiles - 1;
      int cohort_tile_idx = last_sk_tile / kCtasPerCohort;
      int cohort_grid_m = cohort_tile_idx / tiled_cohort_shape.n();
      int cohort_grid_n = (cohort_grid_m > 0) ?
        tiled_cohort_shape.n() - 1 :
        cohort_tile_idx % tiled_cohort_shape.n();

      if ((((cohort_grid_m + 1) * kCohortCtasM) >= tiled_shape.m()) ||
          (((cohort_grid_n + 1) * kCohortCtasN) >= tiled_shape.n()))
      {
        sk_in_range = false;
      }

    }

    // Decide if we're going to be doing cohort raster
    if (sk_in_range &&
        (dp_blocks >= gpu_occupancy * 2) &&
        (cohort_efficiency > 0.85f))
    {
      cohort_raster = true;
      dp_blocks = cohort_blocks;
    }
    else if (sk_waves > 0)
    {
      // Update semi-persistence of first DP wave to ensure full grid wavesets
      // (Only applies when there's an SK component and we're not doing blocked cohort rasterization)
      int dp_tile_waves = (dp_tiles + avail_sms - 1) / avail_sms;
      int full_dp_tile_waves = dp_tiles / avail_sms;
      int waveset_excess = (sk_waves + dp_tile_waves) % sm_occupancy;

      if (dp_first_wave_tiles + waveset_excess <= full_dp_tile_waves)
      {
        dp_first_wave_tiles += waveset_excess;
        dp_blocks -= (waveset_excess * avail_sms);
      }
    }

    // Setup fast-div/mod for device-side usage
    div_mod_tiled_shape_m = FastDivmod(tiled_shape.m());
    div_mod_tiled_shape_n = FastDivmod(tiled_shape.n());
    div_mod_tiled_cohort_shape_n = FastDivmod(tiled_cohort_shape.n());
    div_mod_iters_per_tile = FastDivmod(iters_per_tile);

  }

  /// Number of blocks performing useful work
  int get_num_active_blocks() const
  {
    return (sk_waves * avail_sms) + dp_blocks + reduction_blocks;
  }

  /// Obtains number of threadblocks per GEMM
  int get_num_blocks() const
  {
    int active_blocks = get_num_active_blocks();
    if (remap_block_indices)
    {
      // Add padding blocks if we are performing remapping in order to dispatch a grid of at least four waves
      return fast_max(active_blocks, avail_sms * 4);
    }

    return active_blocks;
  }


  /// Obtains grid extents in CTAs
  dim3 get_grid_dims() const
  {
    return dim3(get_num_blocks(), 1, batch_count);
  }


  //
  // Device-side interface
  //

  /// Obtains number of threadblocks per GEMM
  CUTLASS_DEVICE
  int device_num_blocks() const
  {
    return gridDim.x;
  }

  /// Obtains tile index for the given sk iteration
  CUTLASS_DEVICE
  int get_sk_tile_idx(int iter) const
  {
    int tile_idx = div_mod_iters_per_tile.div(iter);
    return tile_idx;
  }

  /// Obtains the batch index
  CUTLASS_DEVICE
  int get_batch_idx() const
  {
    return RematerializeBlockIdxZ();
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int tile_idx) const
  {
    int m, n;

    // row-major raster
    div_mod_tiled_shape_n(m, n, tile_idx);

    if (tiled_shape().m() < tiled_shape().n())
    {
      // column-major raster
      div_mod_tiled_shape_m(n, m, tile_idx);
    }

    if (cohort_raster)
    {
      // tiled cohort raster
      int cohort_tile_idx = tile_idx / kCtasPerCohort;
      int cohort_grid_m, cohort_grid_n;
      div_mod_tiled_cohort_shape_n(cohort_grid_m, cohort_grid_n, cohort_tile_idx);

      int block_idx_cohort = tile_idx % kCtasPerCohort;
      int block_cohort_m = block_idx_cohort / kCohortCtasN;
      int block_cohort_n = block_idx_cohort % kCohortCtasN;

      m = (cohort_grid_m * kCohortCtasM) + block_cohort_m;
      n = (cohort_grid_n * kCohortCtasN) + block_cohort_n;
    }

    return GemmCoord(m, n, get_batch_idx());
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index (row-major rasterization)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset_row_major(int tile_idx) const
  {
    // row-major raster
    int m, n;
    div_mod_tiled_shape_n(m, n, tile_idx);
    return GemmCoord(m, n, get_batch_idx());
  }

  /// Obtains calling threadblock's linear threadblock index
  CUTLASS_DEVICE
  int get_block_idx() const
  {
    int block_idx = RematerializeBlockIdxX();

    // Remap the block indices for the first two waves of thread blocks if
    // we have multi-occupancy and the grid constitutes four or more waves
    if (remap_block_indices && (block_idx < avail_sms * 2))
    {
      int dest_sm = block_idx / 2;
      int dest_wave = block_idx % 2;
      int remapped_block_idx = dest_sm + (dest_wave * avail_sms);
      block_idx = remapped_block_idx;
    }

    // Remap block indices to interleave SK regions to limit intra-region waiting
    if (block_idx < sk_regions() * sk_blocks_per_region())
    {
      int block_in_region;
      int region;
      div_mod_sk_regions(block_in_region, region, block_idx);
      block_idx = (region * sk_blocks_per_region()) + block_in_region;
    }

    return block_idx;
  }


  /// Obtains calling linear threadblock index of the first block to work on the given tile
  CUTLASS_DEVICE
  int get_sk_block_idx(int iter) const
  {
    int region_idx;
    int iter_in_region;
    div_mod_sk_iters_per_region(region_idx, iter_in_region, iter);

    int big_block_iters = (sk_big_blocks_per_region * sk_iters_per_normal_block()) + sk_big_blocks_per_region;   // number of iterations in the region's big blocks
    int normal_block_iters = iter_in_region - big_block_iters;                                                 // number of iterations in the region's normal blocks

    int big_block_idx_in_region = div_mod_sk_iters_per_big_block.div(iter_in_region);
    int normal_block_idx_in_region = sk_big_blocks_per_region + div_mod_sk_iters_per_normal_block.div(normal_block_iters);

    int block_idx_in_region = (big_block_idx_in_region < sk_big_blocks_per_region) ?
        big_block_idx_in_region :
        normal_block_idx_in_region;

    int owning_block_idx = (sk_blocks_per_region() * region_idx) + block_idx_in_region;

    return owning_block_idx;
  }

  /// Obtains iteration extends for the given SK block index
  CUTLASS_DEVICE
  void get_iter_extents(
      int sk_block_idx,
      int &block_iter_begin,
      int &block_iter_end) const
  {
    int region_idx;
    int block_idx_in_region;
    div_mod_sk_blocks_per_region(region_idx, block_idx_in_region, sk_block_idx);

    block_iter_begin = (region_idx * sk_iters_per_region) + (block_idx_in_region * sk_iters_per_normal_block());

    // Adjust extents for the first "num_big_blocks" blocks that get one extra iteration
    int block_iters = sk_iters_per_normal_block();
    if (block_idx_in_region < sk_big_blocks_per_region) {
      // This is a +1 iteration block
      block_iter_begin += block_idx_in_region;
      block_iters++;
    } else {
      // This is a regular block
      block_iter_begin += sk_big_blocks_per_region;
    }
    block_iter_end = block_iter_begin + block_iters;
  }


  /// Obtains calling linear threadblock index of the first block to work on the given tile
  CUTLASS_DEVICE
  int get_first_block_idx(int tile_idx, int block_idx) const
  {
    if (tile_idx >= sk_tiles) {
      // DP tile
      return block_idx;
    }

    int iter = tile_idx * iters_per_tile();
    return get_sk_block_idx(iter);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

