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
//   warp����ļ��ؾ���ָ���ͬ��ָ����ڴ������ص㣬���ص���16λ8x8�ľ���x4��ʾ����4�ݣ���4��16λ8x8�ľ���
//   4������Ϊ4��32λ�Ĵ�����FragA��4��16x2��Vec������4��fp32�ļĴ�����
// ��������һ�����ζ��ԣ�һ��warp��32���̣߳�����ȡ8x8=64��Ԫ�أ���ֵ�����̵߳����ݾ͸պ�2��fp16��
// ���պ���һ��fp32�ļĴ��������Ե�ǰ�߳��ṩһ��fp32�Ĵ����������μ��ɡ�x4���ṩ4��fp32�Ĵ�����
// x1ʱ: 8��32���̣߳�t0-t3���t0�ṩ�������ַ��8��fp16���ݣ�1���߳�2�����ݣ�t4-t7���t1�����ַ��8�����ݡ�����
//       (t0���õ�[0,0][0,1]��t1([0,2][0,3])...t3([0,6][0,7]), t4([1,0][1,1])... t31([7,6][7,7])) 
//       �� mma�������ȵ�A��C����һ���̶߳�ȡ��8x8�ӿ��ڵ�˳����һ���ģ����Կ�������ldmatrix��ֱ�ӽ���mmaָ�
//       ����B����ֻ��Ҫ�� ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
//       ��Ҫע��bank��ͻ��t0��ȡ0��32λ����, ����bank0, ���蹲���ڴ��СΪ32x32����t4���ʵ�Ҳ��bank0.
// x4ʱ: t0���t0�ṩ�������ַ��8�����ݣ�t1���t1�����ַ��8�����ݡ�����
//       (����4��8x8�������г�32x8��t0-t31�ṩÿ���׵�ַ) ��t0���õ�a0a1..a7, t1�õ�a8-a15, ...ÿ���Ĵ���������˳��ͬx1��
//       (����4��8x8�������г�16x16��t0-t16�ṩÿ���׵�ַ��t16-t31�ṩÿ���Ұ���׵�ַ) ��t0���õ�a0a1..a7��t16�õ�a8-a15��t1�õ�a16-a23�ˡ�
//
//   1������������ö�ȡ���ݵ��׵�ַ����Ϊ��ȡ�ľ����������������ģ����в��������ģ�������Ҫ�ṩÿһ�е��׵�ַ��
// �����ȡ�ľ�����4��8x8������Ҫ�ṩ4x8=32���׵�ַ��һ��warp��32���̣߳�ÿ���̶߳�����õ����ָ�
// ָ��Լ��t0-t31���ֱ��ṩ��32���׵�ַ��ͨ��ÿ���̵߳�%4����
//��x1ָ���Լ��t0-t7���ṩ8���׵�ַ��x2��Լ��t0-t15���ṩ16���׵�ַ����
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
