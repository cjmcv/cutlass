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
    \brief Basic include for CUTLASS.
*/

#pragma once

#include "cutlass/arch/synclog.hpp"
#include "cutlass/detail/helper_macros.hpp"

// <NT>M 显卡架构演进和重大更新
// sm70 volta: 首次提出tensor core，开始支持mma，仅支持fp16。
// sm75 turing: 一个warp的mma处理数据量增多，扩展int8和int4计算
// sm80 ampere：一个warp的mma处理数据量进一步增多, 新增cp.async异步拷贝指令（用于multi-stage，之前的只能是同步的2-stage，即double-buffer）
// sm89 ada lovelace: mma开始支持fp8. (L40)
// sm90 hopper: mma开始支持fp64，提出tma张量拷贝指令，warp-specialize的概念和wgmma(wg=warp_group，定义为连续4个warp，在hopper中首次被提出)
//              wgmma.mma_async 是异步操作，对比之前的同步指令mma.sync, 颗粒度更大。
// sm100 blackwell, 
// => vtahb
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/// Status code returned by CUTLASS operations
enum class Status {
  kSuccess,                    ///< Operation was successful.
  kErrorMisalignedOperand,     ///< operands fail alignment requirements.
  kErrorInvalidDataType,       ///< DataType fails requirement.
  kErrorInvalidLayout,         ///< Layout fails alignment requirement.
  kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
  kErrorNotSupported,          ///< Operation is not supported on current device.
  kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
  kErrorInternal,              ///< An error within CUTLASS occurred.
  kErrorArchMismatch,          ///< CUTLASS runs on a device that it was not compiled for.
  kErrorInsufficientDriver,    ///< CUTLASS runs with a driver that is too old.
  kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
  kInvalid                     ///< Status is unspecified.
};

/// Convert cutlass status to status strings
CUTLASS_HOST_DEVICE
static char const* cutlassGetStatusString(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "Success";
    case cutlass::Status::kErrorMisalignedOperand:
      return "Error Misaligned Operand";
    case cutlass::Status::kErrorInvalidDataType:
      return "Error Invalid Data Type";
    case cutlass::Status::kErrorInvalidLayout:
      return "Error Invalid Layout";
    case cutlass::Status::kErrorInvalidProblem:
      return "Error Invalid Problem";
    case cutlass::Status::kErrorNotSupported:
      return "Error Not Supported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "Error Workspace Null";
    case cutlass::Status::kErrorInternal:
      return "Error Internal";
    case cutlass::Status::kErrorInsufficientDriver:
      return "Error Insufficient Driver";
    case cutlass::Status::kErrorArchMismatch:
      return "Error Architecture Mismatch";
    case cutlass::Status::kErrorMemoryAllocation:
      return "Error Memory Allocation failed";
    case cutlass::Status::kInvalid: break;
  }

  return "Invalid status";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// <NT> warp是32个线程，太老的显卡不做考虑。1 warpgroup = 4 warp = 128 thread。
//      quad是4线程组，一种线程组织的单元 （20250227-代码中未被使用到）
static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpGroup = 128;
static const int NumWarpsPerWarpGroup = NumThreadsPerWarpGroup / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;
static const int NumThreadsPerQuad = 4;
static const int NumThreadsPerQuadPair = NumThreadsPerQuad * 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to return true when called by thread 0 of threadblock 0.
CUTLASS_HOST_DEVICE bool thread0() {
  #if defined(__CUDA_ARCH__)
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
  #else
    return false;
  #endif
}

// <NT> warp内线程id号(0-31)。调用此函数时，warp 内的线程可能并未收敛（即可能处于warp分支发散状态）。
/// Returns a lane index in the warp. The threads in warp may not be convergent
CUTLASS_DEVICE
int canonical_lane_idx() { 
  #if defined(__CUDA_ARCH__)
    return threadIdx.x % NumThreadsPerWarp;
  #else
    return 0;
  #endif
}

// <NT> __shfl_sync 用于在warp内进行数据交换，原型：typename T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
// mask: 32位无符号整数，指定哪些线程会参与当前的数据交换操作，0xffffffff 意味着线程束内的所有 32 个线程都会参与此次操作。
// var：要进行交换的数据. srcLane: 对应warp里线程号（0-31），填0，表示由warp里0号线程获取数据结果，并广播给其他参与此次操作的线程。
// 返回一个warp索引
/// Returns a warp-uniform value indicating the canonical warp index of the calling threads.
/// Threads within the warp must be converged.
CUTLASS_DEVICE
int canonical_warp_idx_sync() { 
  #if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
  #else
    return 0;
  #endif
}

// <NT> block内warp id号, threadIdx.x的范围是(0 - block内线程数), cutlass的线程组织大多以一维布局，下面写法只针对一维。
/// Returns a warp index in the CTA. The threads in warp may not be convergent
/// As it doesn't sync the warp, it faster and allows forward progress
CUTLASS_DEVICE
int canonical_warp_idx() { 
  #if defined(__CUDA_ARCH__)
    return threadIdx.x / NumThreadsPerWarp;
  #else
    return 0;
  #endif
}

/// Returns a warp-uniform value indicating the canonical warp group index of the calling threads.
/// Threads within the warp must be converged.
CUTLASS_DEVICE
int canonical_warp_group_idx() {
  #if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarpGroup, 0);
  #else
    return 0;
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
