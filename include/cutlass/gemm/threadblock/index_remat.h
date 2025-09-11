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
    \brief Helpers for rematerializing indices/dimensions in the thread hierarchy from special registers
*/

#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
// <NT> rematerialize辅助函数：重新计算 threadIdx/blockIdx，用来减少寄存器生命周期（从而省寄存器）
// CUDA 内核里如果直接保存 threadIdx.x 到局部变量：如 int tidx = threadIdx.x; 编译器会 把 bx 留在寄存器里一直不释放，直到最后一次使用。
// 而Rematerialize的做法是 用内联函数 int bx = RematerializeBlockIdxX(); 每次用到时都重新去读特殊寄存器 SR_BLOCKIDX_X，不再用局部变量缓存。
// 代价：多读一次特殊寄存器，1 cycle 延迟，几乎免费。  收益：省 1 个寄存器/线程，1024 线程就能省 1k 寄存器，
//
// 疑问分析：int bx = RematerializeBlockIdxX() 中bx也会放在寄存器，还有何区别。
// 答：区别在于编译器的动作，int bx = blockIdx.x 强制产生“mov rX, SR_BLOCKIDX_X”。
// 而 RematerializeBlockIdxX() 让编译器 “把 SR_BLOCKIDX_X 直接当立即数”，从而真正省掉一条 MOV 和一个 live 寄存器。

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX() {
  return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY() {
  return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
  return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX() {
  return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY() {
  return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
  return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY() {
  return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ() {
  return blockDim.z;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass


