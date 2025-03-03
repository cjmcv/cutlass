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
    \brief Architecture-specific operators on memory added for SM75
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/layout/matrix.h"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/util.hpp"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount
>
CUTLASS_DEVICE void ldsm(Array<unsigned, MatrixCount> & D, void const* ptr);

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Determine the appropriate way to target PTX's "ldmatrix" instruction.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// CUTLASS helper to get SMEM pointer
CUTLASS_DEVICE unsigned cutlass_get_smem_pointer(void *ptr) {
  return cute::cast_smem_ptr_to_uint(ptr);
}

/// CUTLASS helper to get SMEM pointer
CUTLASS_DEVICE unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y, z, w;
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Transpose on 16b granularity
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// <NT> ldmatrix.sync.aligned.m8n8.x4.shared.b16 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
// 
//   warp级别的加载矩阵指令，有同步指令和内存对齐的特点，加载的是16位8x8的矩阵，x4表示加载4份，即4个16位8x8的矩阵。
//   4个出参为4个32位寄存器（FragA是4个16x2的Vec，即有4个fp32的寄存器）
// 对于其中一个出参而言，一个warp有32个线程，共读取8x8=64个元素，则分到这个线程的数据就刚好2个fp16，
// 即刚好是一个fp32的寄存器，所以当前线程提供一个fp32寄存器用作出参即可。x4则提供4个fp32寄存器。
// x1时: 8行32个线程，t0-t3获得t0提供的输入地址的8个fp16数据，1个线程2个数据；t4-t7获得t1输入地址的8个数据。。。
//       (t0将拿到[0,0][0,1]，t1([0,2][0,3])...t3([0,6][0,7]), t4([1,0][1,1])... t31([7,6][7,7])) 
//       与 mma中行优先的A和C矩阵一个线程读取的8x8子块内的顺序是一样的！所以可以用完ldmatrix后直接进入mma指令。
//       对于B矩阵只需要用 ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
//       需要注意bank冲突，t0读取0号32位数据, 属于bank0, 假设共享内存大小为32x32，则t4访问的也是bank0.
// x4时: t0获得t0提供的输入地址的8个数据，t1获得t1输入地址的8个数据。。。
//       (假设4个8x8矩阵被排列成32x8，t0-t31提供每行首地址) 则t0将拿到a0a1..a7, t1拿到a8-a15, ...每个寄存器的数据顺序同x1，
//       (假设4个8x8矩阵被排列成16x16，t0-t16提供每行首地址，t16-t31提供每行右半边首地址) 则t0将拿到a0a1..a7，t16拿到a8-a15，t1拿到a16-a23了。
//
//   1个入参用于设置读取数据的首地址，因为读取的矩阵数据列是连续的，但行不是连续的，所以需要提供每一行的首地址。
// 这里读取的矩阵是4个8x8，即需要提供4x8=32个首地址，一个warp有32个线程，每个线程都会调用到这个指令，
// 指令约定t0-t31来分别提供这32个首地址，通过每个线程的%4输入
//（x1指令：则约定t0-t7来提供8个首地址；x2则约定t0-t15来提供16个首地址）。
// 
template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y, z, w;
    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct shared_load_op {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    D = *reinterpret_cast<AccessType const *>(ptr);  
  }
};

template <typename AccessType>
CUTLASS_DEVICE void shared_load(AccessType &D, void const *ptr) {
  shared_load_op<AccessType, int(sizeof(AccessType))>(D, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 16> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint4 v;
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : 
      "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 8> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint2 v;
    asm volatile ("ld.shared.v2.b32 {%0, %1}, [%2];" : 
      "=r"(v.x), "=r"(v.y) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
