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
// <NT> Cluster �߳̿鼯Ⱥ ����
// Cluster��sm90������ĸ����ԭ�����ļ�ģ�� thread �� warp �� CTA �� grid��
// ��չΪ�弶��thread �� warp �� CTA �� cluster �� grid����Ϊ����ָ����洢��ϵͳ�͵������϶�����������ơ�
// 1) һ��cluster��1-16��CTA��Ӳ�����壨H20���8����H100���16��������ClusterShape��ָ������
//    һ���Ա��ɷ�/ռ��ͬһ�� GPC �ڵ����� SM (H20��8����H100��16��)
// 2��cluster �����/���ģ�Ͳ���ĳ���GPC ��Ӳ��ʵ�壻
//    ��� cluster���������Բ�ͬ kernel�����Ա����λ򲢷����ɷ���ͬһ�� GPC �ϣ�ֻҪ�� GPC �ﻹ���㹻���� SM ���ɡ�
//    cluster ��֤������N��CTA����ͬһ GPC������GPC������ռ��ĳ��cluster��SM��Դ����ͻ���һ�������á�
// 3��һ��cluster�ڵĶ��cta ���԰ѱ˴˵� shared memory ����һ�顮��˵ķֲ�ʽ�����ڴ桯��ֱ�Ӷ�д���� DSM (Distributed Shared Memory)��
//    ÿ�� SM ��Ȼ���Լ��� 256 kB Shared Memory���� GPC �ڲ�������һ�����ٽ��濪�أ�crossbar������ͬ GPC ������ SM �� SMEM �˿�����һ��
//    A-SM �� CUDA �߳̿���ͨ������ crossbar ֱ�ӷ��� B-SM �� SMEM���ӳ� �� 40 cycles������ӽ� L1��
//
// gemmʵ�������ֳ����ĺô� (sm90��cluster������Ĭ�ϵ��ȵ�Ԫ)��
// 1����cluster��֧��TMA�㲥���������TMA multicast���ܣ�����gmem�Ķ�ȡ��
// 2��split-k/stream-k��global reduction���ڴ�ͳд���У�ÿ��CTA����Ҫ���Լ���k�ν��д�ص�gmem��Ȼ���ٻ���gmem��k�δ���õ����ս����
//   ��cluster�£�һ��cluster�ж��cta�������k�ν����cluster�ڿ���ͨ��DSMֱ����ɹ�Լ��������Ҫдgmem��
//   ֻ�е�k�α���ֵ����clusterʱ����cluster�Ĳ��ֲ���Ҫ����gmem���л��ܡ���������gmem�Ķ�д������
// 3��С�����£�����cluster�ܰ�ʹ���ʴ�������sm80�Ĳ��У�����ʹ����stream-k��
//     һ�� wave = ͬʱ�ܱ� GPU ���ɲ���������� CTA �������� SM ��������sm80�£�ֻҪʣ��CTA����SM����driver�Ϳ���һ�ΰ�����(full wave)����ȥ,
//   ���Ƕ���β��(tail wave)����ʣ��cta��С��sm��ʱ��Ӳ�����������ϲ����ڡ�partial wave��ԭ���ɷ����ƣ��������ڡ�ͬһʱ������һ���䡱����Ӳ�����ƣ�
//   ����ֻ��һ��һ��CTA�������ɷ������ǽ������ȷ����Ǽ��� CTA ���ܶ������ˣ�����Ļ�û��أ��մ��ڵ��Ӻ��ƽ�� occupancy ���塣
//     ������cluster�£���cluste����8��cta����tail wave��ԭ������8��cta 8��cta���ɷ�����û��wave�Ĵ�����ȴ󣬵�Ҳ�ȵ���CTAҪ��ܶࡣ
// 4���Ĵ���ѹ����С��1. sm80��ÿ���̼߳ȼ����ּ��㣬���Լ�Ҫ��ַ�Ĵ�������Ҫ�ۼӼĴ�������sm90��ws�£����߳�ֻ��һ���£����ԼĴ�����Ҫ��С��
//                  2. sm80����TMA���� ld.global��SMEM��REG��REG ������ tile ���棻��sm90��TMA ֱ�� async-fence��SMEM��consumer warp �������� REG��ռ��ʱ��̡�
//
// ����䣺
// 1��GPC (Graphics Processing Cluster), ���ڰѶ�� SM ��̶����ܵ�Ԫ����ɿɶ��������ġ���оƬ����
//    ����keplerʱ�����У�����hopper���״θ� GPC �ڲ�����ר�ŵĽ��滥�����磬��ͬһ��GPC�ڵ�����SM
//    ����ֱ�ӷ��ʱ˴˵� shared memory���Ӷ�֧���� thread-block cluster ��һ�±�̳���
//    * GPC��Ӳ��λ�ã�thread��Сִ�е�Ԫ -> warpΪ32���߳� -> cta/block Ϊ���warp -> sm ΪӲ����ִ�� CTA �ĵ�Ԫ
//                   -> GPU ������� SM �͹̶����ܵ�Ԫ -> GPU оƬ �������GPC
//    * cluster ������ȫ����һ�� GPC �ڣ���� GPC ��Сֱ�Ӿ��� cluster �� CTA ���ޡ�
// 2) cluster-barrier��cluster�ڶ��blockͬ�������Ĵ��������֣�Hopper�¼�CLUSTER_ARV,CLUSTER_WAITָ�
//        ÿ�� SM ֻ���Լ������� д 1 �� bit �Ĵ��� �� Ӳ���㲥���ռ� �� һ����bit ��=CTA ������ͬʱ���С�
//        ȫ�̲��� SRAM/DRAM�������ǡ����ں��ġ�����ʮ cycles �㶨��
//    * __syncthread(): ��cta��ͬ����Ӳ��ʵ���� SMEM �� bank arbiter ������ÿ�� warp �Ѽ�����д��
//        SMEM �� �ٲ��߼���ѯ �� ֱ����������=�� warp ���� �� �ٹ㲥����
// 3) TMA multicast��Hopper �ܹ��� cluster ������ Ӳ������һд����� ����ֻ��һ�� TMA load��
//   ���ܰ�ͬһ�� GMEM ����ͬʱ��� cluster ������ CTA �� SMEM��ʡȥ�ظ���������֤ L2 ���С�
//    �� 4 �� CTA ��Ҫͬһ�� A �Ӿ��󣬰�������������Ҫÿ��CTA���� ld.global -> SMEM����4��������4��L2��ѯ������4��miss��
//   ����TMA multicast����һ�μ��ع㲥��1~16��Ŀ��smem�ϡ�
// 4) cluster�����ʼǣ� include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp

 #pragma once

#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
  ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#  define CUTE_ARCH_CLUSTER_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#  define CUTE_ARCH_ELECT_ONE_SM90_ENABLED
#endif

namespace cute {

CUTE_DEVICE void cluster_arrive_relaxed()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_arrive()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_wait()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_sync()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  cluster_arrive();
  cluster_wait();
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

// Returns the dim3 grid size in terms of number of clusters.
CUTE_DEVICE dim3 cluster_grid_dims()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of gridDim with __CUDA_ARCH__.
  return gridDim;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_grid_dims() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the dim3 cluster rank in the grid.
CUTE_DEVICE dim3 cluster_id_in_grid()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of blockIdx with __CUDA_ARCH__.
  return blockIdx;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_id_in_grid() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the relative dim3 block rank local to the cluster.
CUTE_DEVICE dim3 block_id_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {0,0,0};
#endif
}

// Returns the dim3 cluster shape.
CUTE_DEVICE dim3 cluster_shape()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {1,1,1};
#endif
}

// <NT> ���һ��cluster�е�cta id�š�
// Get 1D ctaid in a cluster.
CUTE_DEVICE uint32_t block_rank_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0;
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
CUTE_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
              : "=r"(result)
              : "r"(smemAddr), "r"(rank));
  return result;
#else
  return smemAddr;
#endif
}

// <NT> ��warp��ѡ�ٳ�һ���̣߳��̵߳���cute::elect_one_sync()����lane_predicateΪtrue�����ʾ���Ǳ�ѡ�ٳ������̣߳�����Ϊfalse���ʾδ��ѡ�С�
// Elect one thread in the warp. The elected thread gets its predicate set to true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x % 32) == 0;
#else
  return true;
#endif
}

struct ElectOneLaneIdReturnType {
  uint32_t is_leader;
  uint32_t leader_lane_id;
};

CUTE_HOST_DEVICE
ElectOneLaneIdReturnType
elect_one_leader_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return {pred, laneid};
#elif defined(__CUDA_ARCH__)
  return {(threadIdx.x % 32) == 0, 0};
#else
  return {true, 0};
#endif
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote(uint32_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
#endif
}

} // end namespace cute
