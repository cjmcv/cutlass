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
// <NT> Cluster 线程块集群 介绍
// Cluster是sm90后提出的概念，将原来的四级模型 thread → warp → CTA → grid，
// 扩展为五级：thread → warp → CTA → cluster → grid，并为此在指令集、存储器系统和调度器上都做了配套设计。
// 1) 一个cluster是1-16个CTA的硬件绑定体（H20最多8个，H100最多16个，可由ClusterShape来指定），
//    一次性被派发/占满同一个 GPC 内的若干 SM (H20有8个，H100有16个)
// 2）cluster 是软件/编程模型层面的抽象，GPC 是硬件实体；
//    多个 cluster（甚至来自不同 kernel）可以被依次或并发地派发到同一个 GPC 上，只要该 GPC 里还有足够空闲 SM 即可。
//    cluster 保证“我这N个CTA必落同一 GPC”，但GPC并不独占给某个cluster；SM资源用完就换下一批，灵活复用。
// 3）一个cluster内的多个cta 可以把彼此的 shared memory 当作一块‘跨核的分布式共享内存’来直接读写，即 DSM (Distributed Shared Memory)。
//    每块 SM 仍然有自己的 256 kB Shared Memory，在 GPC 内部新增了一条高速交叉开关（crossbar），把同 GPC 里所有 SM 的 SMEM 端口连在一起。
//    A-SM 的 CUDA 线程可以通过这条 crossbar 直接访问 B-SM 的 SMEM，延迟 ≈ 40 cycles，带宽接近 L1。
//
// gemm实现上体现出来的好处 (sm90后，cluster都会是默认调度单元)：
// 1）在cluster可支持TMA广播，见下面的TMA multicast介绍，减少gmem的读取。
// 2）split-k/stream-k的global reduction：在传统写法中，每个CTA都需要将自己的k段结果写回到gmem，然后再基于gmem将k段凑齐得到最终结果。
//   在cluster下，一个cluster有多个cta，里面的k段结果在cluster内可以通过DSM直接完成规约，不再需要写gmem。
//   只有当k段被拆分到多个cluster时，跨cluster的部分才需要经过gmem进行汇总。大大减少了gmem的读写次数。
// 3）小矩阵下，基于cluster能把使用率打满，而sm80的不行，尽管使用了stream-k。
//     一个 wave = 同时能被 GPU 接纳并启动的最大 CTA 数量（≤ SM 数），在sm80下，只要剩余CTA数≥SM数，driver就可以一次把整波(full wave)推下去,
//   但是对于尾波(tail wave)，即剩余cta数小于sm数时，硬件驱动层面上不存在“partial wave”原子派发机制，即不存在“同一时钟周期一起发射”这种硬件机制，
//   所以只能一个一个CTA地零星派发、零星结束，先发的那几个 CTA 可能都跑完了，后面的还没落地，空窗期叠加后把平均 occupancy 拉垮。
//     而基于cluster下，如cluste包含8个cta，则tail wave的原子则是8个cta 8个cta地派发，虽没有wave的打包粒度大，但也比单个CTA要大很多。
// 4）寄存器压力变小：1. sm80中每个线程既加载又计算，所以既要地址寄存器，又要累加寄存器；而sm90的ws下，单线程只干一件事，所以寄存器需要变小。
//                  2. sm80中无TMA，先 ld.global→SMEM→REG，REG 必须整 tile 缓存；而sm90中TMA 直接 async-fence→SMEM，consumer warp 逐条吸入 REG；占用时间短。
//
// 概念补充：
// 1）GPC (Graphics Processing Cluster), 用于把多个 SM 与固定功能单元打包成可独立工作的“子芯片”。
//    早在kepler时代就有，但在hopper上首次给 GPC 内部加了专门的交叉互连网络，让同一个GPC内的所有SM
//    可以直接访问彼此的 shared memory，从而支撑起 thread-block cluster 这一新编程抽象。
//    * GPC的硬件位置：thread最小执行单元 -> warp为32个线程 -> cta/block 为多个warp -> sm 为硬件上执行 CTA 的单元
//                   -> GPU 包含多个 SM 和固定功能单元 -> GPU 芯片 包含多个GPC
//    * cluster 必须完全落在一个 GPC 内，因此 GPC 大小直接决定 cluster 的 CTA 上限。
// 2) cluster-barrier：cluster内多个block同步，纯寄存器级握手（Hopper新加CLUSTER_ARV,CLUSTER_WAIT指令）
//        每个 SM 只在自己核心里 写 1 个 bit 寄存器 → 硬件广播网收集 → 一旦“bit 数=CTA 数”就同时放行。
//        全程不走 SRAM/DRAM，距离是“隔壁核心”，几十 cycles 搞定。
//    * __syncthread(): 是cta内同步，硬件实现是 SMEM 的 bank arbiter 计数：每个 warp 把计数器写到
//        SMEM → 仲裁逻辑轮询 → 直到“到达数=总 warp 数” → 再广播放行
// 3) TMA multicast：Hopper 架构给 cluster 新增的 硬件级“一写多读” 能力只用一次 TMA load，
//   就能把同一块 GMEM 数据同时搬进 cluster 内所有 CTA 的 SMEM，省去重复流量、保证 L2 命中。
//    如 4 个 CTA 需要同一份 A 子矩阵，按以往的做法需要每个CTA各自 ld.global -> SMEM，即4份流量、4份L2查询、可能4份miss。
//   而用TMA multicast可以一次加载广播到1~16个目标smem上。
// 4) cluster其他笔记： include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp

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

// <NT> 获得一个cluster中的cta id号。
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

// <NT> 从warp里选举出一个线程，线程调用cute::elect_one_sync()返回lane_predicate为true，则表示该是被选举出来的线程，返回为false则表示未被选中。
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
