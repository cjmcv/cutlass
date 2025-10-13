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

// <NT>M �Կ��ܹ��ݽ����ش����
// sm70 volta: �״����tensor core����ʼ֧��mma����֧��fp16��
// sm75 turing: һ��warp��mma�������������࣬��չint8��int4����
// sm80 ampere��һ��warp��mma������������һ������, ����cp.async�첽����ָ�����multi-stage��֮ǰ��ֻ����ͬ����2-stage����double-buffer��
// sm89 ada lovelace: mma��ʼ֧��fp8. (L40)
// sm90 hopper: mma��ʼ֧��fp64�����tma��������ָ�warp-specialize�ĸ����wgmma(wg=warp_group������Ϊ����4��warp����hopper���״α����)
//              wgmma.mma_async ���첽�������Ա�֮ǰ��ͬ��ָ��mma.sync, �����ȸ���
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
// <NT> warp��32���̣߳�̫�ϵ��Կ��������ǡ�1 warpgroup = 4 warp = 128 thread��
//      quad��4�߳��飬һ���߳���֯�ĵ�Ԫ ��20250227-������δ��ʹ�õ���
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

// <NT> warp���߳�id��(0-31)�����ô˺���ʱ��warp �ڵ��߳̿��ܲ�δ�����������ܴ���warp��֧��ɢ״̬����
/// Returns a lane index in the warp. The threads in warp may not be convergent
CUTLASS_DEVICE
int canonical_lane_idx() { 
  #if defined(__CUDA_ARCH__)
    return threadIdx.x % NumThreadsPerWarp;
  #else
    return 0;
  #endif
}

// <NT> __shfl_sync ������warp�ڽ������ݽ�����ԭ�ͣ�typename T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
// mask: 32λ�޷���������ָ����Щ�̻߳���뵱ǰ�����ݽ���������0xffffffff ��ζ���߳����ڵ����� 32 ���̶߳������˴β�����
// var��Ҫ���н���������. srcLane: ��Ӧwarp���̺߳ţ�0-31������0����ʾ��warp��0���̻߳�ȡ���ݽ�������㲥����������˴β������̡߳�
// ����һ��warp����
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

// <NT> block��warp id��, threadIdx.x�ķ�Χ��(0 - block���߳���), cutlass���߳���֯�����һά���֣�����д��ֻ���һά��
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
