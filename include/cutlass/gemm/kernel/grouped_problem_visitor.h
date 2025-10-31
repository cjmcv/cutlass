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
    \brief Base scheduler for grouped problems
*/
// <NT> GroupedProblemVisitor 是sm80的分组gemm中调度引擎, 核心函数是next_tile()。
// next_tile: 让 当前CTA 在任意大小、任意数量的GEMM problem序列里，快速、无锁地 找到 “下一个该我算的tile” 属于 哪个问题、哪个坐标，并把所有必要信息填进基类字段；
// 返回 false 时表示所有问题所有 tile 都已排完，内核可以退出。
// 因主循环就变成这样：GroupedProblemVisitor visitor(...);
//                   while (visitor.next_tile()) {   // ← 就是它在干活
//                     GemmCoord threadblock_offset = visitor.get_threadblock_offset(...);
//                     // 真正算 GEMM
//                   }
// 定义和使用位置: include/cutlass/gemm/kernel/gemm_grouped.h: GemmGrouped
// 
//  这里GroupedProblemVisitor主要划分了两种模式，分别对应: cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly  / kHostPrecompute
// kDeviceOnly: 所有调度都在device端执行。搜 GroupedProblemVisitor->kDeviceOnly
// kHostPrecompute： 在主机上预先计算待访问的完整问题序列。搜 GroupedProblemVisitor->kHostPrecompute
//
//
// <NT> warp shuffle指令: __shfl_sync / __shfl_up_sync /  __shfl_down_sync / __shfl_xor_sync
// 1）int __shfl_sync(unsigned mask, int var, int srcLane, int width=warpSize): 指令单周期完成，是 warp内 通信的最快方式。
//  基于掩码 0xffffffff, 可以把第 srcLane 号线程的 var 广播给本 warp 所有线程，通过函数返回值拿到该广播值。
// 2）int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width=warpSize)：指令单周期完成，往前看 delta 位，拿得到就拿，拿不到就用自己的。
//  掩码仍固定为 0xffffffff，delta是序号偏移量，以delta = 1为例，
//  如lane0输入var0为10, 由delta=1得到目标lane是-1，越界用自己的，则返回值为10；
//  lane1输入var1为20，由delta=1得到目标lane是0，则返回值为var0的10；
//  lane2输入var2为30，由delta=1得到目标lane是1，则返回值为var1的20；
//   典型用法是前缀和：
//             for (int delta = 1; delta < warpSize; delta <<= 1) {
//               int val = cc(0xffffffff, problem_ending_tile, i);
//               if (lane_idx >= i) problem_ending_tile += val;
//             }
//             如一个warp的初始值为 11111111 11111111 11111111 11111111
//             第一次 delta=1：    12222222 22222222 22222222 22222222
//             第二次 delta=2：    12344444 44444444 44444444 44444444
//             第三次 delta=4：    12345678 88888888 88888888 88888888
//             第四次 delta=8：    12345678 9.....16 16xxxxxx xxxxxx16
//             第五次 delta=16：   12345678 9.....16 17....24 ......32
// 3）int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width=warpSize) 
//  与 __shfl_up_sync 对应，方向相反，从高位到低位。加上广播可用于规约（__shfl_up_sync也一样，只是一个广播最后一位，一个广播第一位）。
//             __device__ int warpReduceSum(int val) {
//               #pragma unroll
//               for (int offset = warpSize/2; offset > 0; offset >>= 1)
//                 val += __shfl_down_sync(0xffffffff, val, offset);
//               return val;          // lane-0 为总和
//             }
//             return val;
//             对于reduce需要再使用广播 __shfl_sync(0xffffffff, val, 0)
// 4）int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width=warpSize)，warp内以 异或模式 交换数据，而异或就是“无进位二进制加法”
//   var是需要交换的变量；laneMask决定“和谁交换”，是2的幂(1/2/4/8/16)，可理解成距离，1为相邻两个线程交换，2为间距2的线程交换。
//             __device__ int warpReduceSum_xor(int val) {
//               #pragma unroll
//               for (int i = 1; i < warpSize; i <<= 1)
//                 val += __shfl_xor_sync(0xffffffff, val, i);
//               return val;          // **所有 lane** 都等于总和，不需要广播，warp reudce一般都用该方法
//             }
//           如一个warp的初始值为    1 1 1 1 1 1 1 1   1 1 1 1 1 1 1 1   1 1 1 1 1 1 1 1   1 1 1 1 1 1 1 1
//           第一次 laneMask=1：    22 22 22 22   22 22 22 22   22 22 22 22   22 22 22 22    (相邻两两做异或，即相加)
//           第二次 laneMask=2：    4444 4444   4444 4444   4444 4444   4444 4444            (间距2，两两做异或)
//           第三次 laneMask=4：    88888888 88888888 88888888 88888888                      (间距4，两两做异或)
//           第四次 laneMask=8：    16...............................16                      (间距8，两两做异或)
//           第五次 laneMask=16：   32...............................32                      (间距16，两两做异或，至此所有线程都拿到了reduce结果，不需要再广播)
//
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type describing the type of scheduling to perform for the ProblemVisitor
enum class GroupScheduleMode {
  // Perform all scheduling on device
  kDeviceOnly,
  // Precompute on the host the full sequence of problems to access
  kHostPrecompute
};

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ProblemSizeHelper,
          typename ThreadblockShape_>
struct BaseGroupedProblemVisitor {
  using ThreadblockShape = ThreadblockShape_;

  struct ProblemInfo {
    static int32_t const kNoPrefetchEntry = -1;
    int32_t problem_idx;
    int32_t problem_start;

    CUTLASS_HOST_DEVICE
    ProblemInfo() : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

    CUTLASS_HOST_DEVICE
    ProblemInfo(int32_t problem_idx_, int32_t problem_start_) :
      problem_idx(problem_idx_), problem_start(problem_start_) {}
  };

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;
    void const                     *workspace;
    int32_t                         tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes(nullptr), problem_count(0), workspace(nullptr), tile_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes,
      int32_t                         problem_count,
      void const                     *workspace = nullptr,
      int32_t                         tile_count = 0
    ):
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      workspace(workspace),
      tile_count(tile_count)
    {}

  };

  Params params;
  int32_t tile_idx;
  int32_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaseGroupedProblemVisitor(
    Params const &params_,
    int32_t block_idx
  ):
  params(params_),
  tile_idx(block_idx),
  problem_tile_start(0),
  problem_idx(0)
  {}

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return ProblemSizeHelper::grid_shape(problem);
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {
    ProblemSizeHelper::possibly_transpose_problem(problem);
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    GemmCoord problem = params.problem_sizes[problem_idx];
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    return ProblemSizeHelper::tile_count(grid);
  }

  static int32_t group_tile_count(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count) {
    int32_t total_tiles = 0;
    for (int32_t i = 0; i < problem_count; ++i) {
      auto problem = host_problem_sizes_ptr[i];
      possibly_transpose_problem(problem);
      auto grid = grid_shape(problem);
      total_tiles += tile_count(grid);
    }

    return total_tiles;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ProblemSizeHelper,
  typename ThreadblockShape,
  GroupScheduleMode GroupScheduleMode_,
  int PrefetchTileCount,
  int ThreadCount
>
struct GroupedProblemVisitor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kDeviceOnly,
                             PrefetchTileCount,
                             ThreadCount>: public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  static int const kThreadCount = ThreadCount;
  static bool const kRequiresPrecomputation = false;
  static int const kThreadsPerWarp = 32;

  struct SharedStorage {};

  // Final tile of the problem loaded by this thread. Each thread will hold
  // a separate value.
  int32_t problem_ending_tile;

  SharedStorage &shared_storage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  problem_ending_tile(0),
  shared_storage(shared_storage_)
  {
    this->problem_idx = -1 * kThreadsPerWarp;
    this->problem_tile_start = 0;
  }


  // <NT> GroupedProblemVisitor->kDeviceOnly:
  // 作用概述: 给“当前线程”分配一个 tile，并告诉它 这个 tile 属于哪个 problem、在该 problem 内的局部编号是多少。
  //     即需要计算得到以下三个变量：
  //        this->problem_idx        // 要算的 problem 编号
  //        this->problem_tile_start // 该 problem 的第一个全局 tile 序号
  //        this->tile_idx           // 当前线程接下来要算的全局 tile 序号
  //     一个线程只处理一个 problem，一个warp 32个线程负责一组的32个problem。
  //     函数返回true表示“已领到 tile”；返回 false 表示“没有 tile 可领”。领完后在next_tile函数外做实际gemm的计算。
  // 大致流程：1) 前缀和产生全局边界
  //          2) 线程每次 next_tile() 用 __ballot+__popc 二分查找边界
  //          3) 把 global_tile_idx 翻译成 (problem, 局部 tile)
  //      
  // next_tile 函数步骤。。。(TODO)：
  //  1) 广播：使所有线程 都知道“当前问题”结束边界 problem_tile_end，如果当前tile在这个范围内，则直接退出去执行gemm计算，否则需要给该tile换一组problem（32个problem为一组）。
  //         注意 ProblemVisitor 在operator中创建，每个线程创建一份，但构建以block_idx为基础，所有一个block内所有线程的该类对象是一样的。
  //  2）换一组problem：lane-31 总是保存“这组最后一个问题”的结束 tile（前缀和结果）。所以基于lane-31去广播，拿到group_tile_end就能判断 整组 32 个问题 是否全部排完。
  //
  CUTLASS_DEVICE
  bool next_tile() {
    // Check whether the tile to compute is within the range of the current problem.
    int32_t problem_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, this->problem_idx % kThreadsPerWarp);
    if (this->tile_idx < problem_tile_end) {
      return true;
    }

    // Check whether the tile to compute is within the current group of problems fetched by the warp.
    // The last tile for this group is the final tile of the problem held by the final thread in the warp.
    int32_t group_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

    // Keep the starting problem for this group in `problem_idx`. This is done to reduce
    // register pressure. The starting problem for this group is simply the first problem
    // in the group most recently fetched by the warp.
    int32_t &group_problem_start = this->problem_idx;
    group_problem_start = (this->problem_idx / kThreadsPerWarp) * kThreadsPerWarp;

    // Keep the starting tile for this group in `problem_tile_start`. This is done to reduce
    // register pressure.
    int32_t &group_tile_start = this->problem_tile_start;

    // Each thread in the warp processes a separate problem to advance until
    // reaching a problem whose starting tile is less less than tile_idx.
    while (group_tile_end <= this->tile_idx) {
      group_problem_start += kThreadsPerWarp;
      if (group_problem_start > this->params.problem_count) {
        return false;
      }

      // Since `group_tile_start` is a reference to `this->problem_tile_start`, this
      // also sets `this->problem_tile_start`. The fact that `this->problem_tile_start`
      // is also set here is used later in `next_tile`.
      group_tile_start = group_tile_end;

      int lane_idx = threadIdx.x % kThreadsPerWarp;
      int32_t lane_problem = group_problem_start + lane_idx;

      // Compute the number of tiles in the problem assigned to each thread.
      problem_ending_tile = 0;
      if (lane_problem < this->params.problem_count) {
        cutlass::gemm::GemmCoord problem = this->params.problem_sizes[lane_problem];
        this->possibly_transpose_problem(problem);
        cutlass::gemm::GemmCoord grid = this->grid_shape(problem);
        problem_ending_tile = this->tile_count(grid);
      }

      // Compute a warp-wide inclusive prefix sum to compute the ending tile index of
      // each thread's problem.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
        int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
        if (lane_idx >= i) {
          problem_ending_tile += val;
        }
      }

      // The total tile count for this group is now in the final position of the prefix sum
      int32_t tiles_in_group = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

      problem_ending_tile += group_tile_start;
      group_tile_end += tiles_in_group;
    }

    // The next problem to process is the first one that does not have ending tile position
    // that is greater than or equal to tile index.
    int32_t problem_idx_in_group =
        __popc(__ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));

    this->problem_idx = group_problem_start + problem_idx_in_group;

    // The starting tile for this problem is the ending tile of the previous problem. In cases
    // where `problem_idx_in_group` is the first problem in the group, we do not need to reset
    // `problem_tile_start`, because it is set to the previous group's ending tile in the while
    // loop above.
    if (problem_idx_in_group > 0) {
      this->problem_tile_start = __shfl_sync(0xffffffff, problem_ending_tile, problem_idx_in_group - 1);
    }

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    return 0;
  }

  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Precomputes schedule on host and prefetches into shared memory
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kHostPrecompute,
                             PrefetchTileCount,
                             ThreadCount> : public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  static_assert(PrefetchTileCount > 0,
                "GroupedProblemVisitor with GroupScheduleMode `kHostPrecompute` currently requires prefetching to shared memory");

  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  using ProblemInfo = typename Base::ProblemInfo;
  static bool const kRequiresPrecomputation = true;

  static int const kPrefetchTileCount = PrefetchTileCount;
  static int const kThreadCount = ThreadCount;

  struct SharedStorage {
    // Sequence of problem IDs and starting tiles to compute
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
  };

  int32_t tiles_computed;
  int32_t iterations_per_block;
  int32_t block_load_start;
  SharedStorage &shared_storage;
  ProblemInfo const *problem_info_ptr;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  tiles_computed(0),
  shared_storage(shared_storage_),
  problem_info_ptr(reinterpret_cast<ProblemInfo const*>(params_.workspace))
  {
    iterations_per_block = (params_.tile_count - 1 + gridDim.x) / gridDim.x;
    block_load_start = iterations_per_block * block_idx;
    // Start prefetching the first set of tiles to compute
    prefetch_tiles();
  }

  CUTLASS_DEVICE
  bool next_tile() {
    if (this->tile_idx >= this->params.tile_count) {
      return false;
    }

    int32_t prefetch_idx = (tiles_computed % kPrefetchTileCount);
    if (prefetch_idx == 0) {
      // Ensure all previous stores to shared memory have been completed
      __syncthreads();
    }

    auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
    ++tiles_computed;

    if ((tiles_computed % kPrefetchTileCount) == 0) {
      // Begin prefetching next set of tiles. Synchronize first to ensure that
      // we don't overwrite the current buffer while someone else is using it.
      __syncthreads();
      prefetch_tiles();
    }

    this->problem_idx = problem_info.problem_idx;
    this->problem_tile_start = problem_info.problem_start;

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = ((total_tiles - 1 + block_count) / block_count);
    return sizeof(ProblemInfo) * entries_per_block * block_count;
  }
#if !defined(__CUDACC_RTC__)
  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {
    ProblemInfo* host_problem_info_ptr = reinterpret_cast<ProblemInfo*>(host_workspace_ptr);
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = (total_tiles - 1 + block_count) / block_count;

    int tile = 0;
    int start_tile = 0;
    for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
      auto problem = host_problem_sizes_ptr[p_idx];
      Base::possibly_transpose_problem(problem);
      auto grid = Base::grid_shape(problem);
      int tiles = Base::tile_count(grid);
      ProblemInfo problem_info(p_idx, start_tile);
      for (int i = 0; i < tiles; ++i, ++tile) {
        host_problem_info_ptr[(entries_per_block * (tile % block_count)) + (tile / block_count)] = problem_info;
      }
      start_tile += tiles;
    }
  }
#endif
private:
  CUTLASS_DEVICE
  void prefetch_tiles() {
    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < kPrefetchTileCount; i += kThreadCount) {
      int32_t offset = threadIdx.x + i;
      if (offset < kPrefetchTileCount && (tiles_computed + offset < iterations_per_block)) {
        shared_storage.prefetched_problems[offset] = problem_info_ptr[block_load_start + tiles_computed + offset];
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
