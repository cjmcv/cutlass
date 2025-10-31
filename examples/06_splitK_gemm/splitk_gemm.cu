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

/**
<NT> split-k 注解翻译
什么是split-k：考虑一个问题规模为 M = 128、N = 128、K = 4096 的情况。在这种情况下，如果thread-block tile是 128x128x4096，那么我们将启动一个thread block，
它会占用 V100 上 84 个流式多处理器（SM）中的一个。因此，计算效率非常低。那么，可以通过split-k来处理。它是一种对矩阵乘法的 K 维度进行划分，并将其分布到多个 SM 上的方法，
从而比使用单个 SM 获得更高的效率。在上述示例中，我们可以使用分块 K 因子 16 对 K 维度进行划分，即block tile大小将变为 128x128x256，并且会在 16 个 SM 上启动。
一旦每个block计算出它们的部分内积（输出的 1/16），它们会将结果累加到单个输出矩阵中。

CUTLASS 将一个kernel划分为层次化的可组合部分。thread tile => warp tile => block tile。分别对应一个线程，一个warp和一个block负责计算的tile大小。

=====================================================

在这个示例中，我们将变量初始化分为以下两部分：
设置数据属性：描述矩阵在内存中的布局方式，以及内核如何看待它们（逻辑到物理的映射）。
设置计算属性：描述上述设置的矩阵将如何用于计算矩阵乘法的输出。

首先，我们设置矩阵 A、B、C 和 D 的数据类型，以及 α 和 β，因为 GEMM 的方程是 D = α * A * B + β * C。
在 CUTLASS 中，内核首先计算 A * B，然后将其余的计算留到内核末尾，因为 α * X + β * C 是对 X（A * B）和 C 进行的简单逐元素操作。
我们将此称为内核的尾声（epilogue）。因此，我们将 α 和 β 的数据类型设置为与 ElementComputeEpilogue = float 相等。
由于我们希望在 Volta 上使用矩阵乘法累加（MMA）指令，而这些指令仅支持半精度浮点数（fp16 或 half），
所以我们将输入矩阵 A 和 B 中的元素的数据类型设置为 cutlass::half_t。Volta 还支持将部分点积累加到 fp32，fp32 可以存储更大范围的数字，
我们将其用作输出矩阵元素和累加的数据类型。我们通过初始化模板变量 ElementAccumulator（float）、ElementComputeEpilogue（float）、
ElementInputA（cutlass::half_t）、ElementInputB（cutlass::half_t）、ElementOutput（float）将这些信息传达给 CUTLASS 内核。
仅仅传达数据类型是不够的。由于数据在内存中是线性布局的，我们还必须传达矩阵的布局。

我们通过将模板变量 LayoutInputA 初始化为列主序的 CUTLASS 变量、LayoutInputB 初始化为行主序以及 LayoutOutput 初始化为行主序来实现这一点。
接下来，我们设置计算 α * X + β * C 的规则，这被称为内核的尾声。我们初始化模板变量 EpilogueOp，它接受输出的数据类型 ElementOutput（float）、
每个向量内存访问的元素数量（16）、累加器的数据类型（float）以及线性组合（α * X + β * C）的计算数据类型。

现在我们已经设置了数据的属性，接下来必须设置计算的属性。
其次，我们分别将线程块、线程束和 MMA 操作的瓦片大小的模板变量设置为 128x128x32、64x64x4、8x8x4（M x N x K）。
当将这些变量传递给实例化 CUTLASS GEMM 内核时，它会在内部推导出每个线程块所需的线程数量、共享内存的大小、以无存储体冲突的方式存储数据，
以及组合、初始化和启动高性能 GEMM 内核所需的大量其他变量。这就是 CUTLASS 的美妙之处，它让开发人员无需理解和编写复杂的硬件优化代码，因为这些代码很容易出错。
还有一些其他的模板变量也会被初始化，例如，在某个 SM 上启动的线程块将处理输出矩阵的哪个线程块瓦片，以及你希望运行程序的 GPU 的 CUDA SM 架构。
所有这些设置组合在一起，使用 cutlass::gemm::device::GemmSplitKParallel 模板创建一个描述 CUTLASS GEMM 内核的模板变量。
下一步是初始化物理数据、实例化并初始化 CUTLASS 内核，然后运行它。我们使用 CUTLASS 实用工具来初始化、填充和比较矩阵，因为这些工具简单易用，不会影响我们学习 CUTLASS。
一旦所有矩阵都被初始化并填充了数据，就创建一个参数元组来启动 CUTLASS 内核，该元组包含问题规模（M = 5120、N = 4096 和 K = 4096）、
矩阵、α、β 以及重要的分块 K 维度因子。此外，我们还会查询 CUTLASS，了解我们实例化的内核是否需要任何临时存储空间。如果需要，我们会创建它，
并将其与其他创建的参数一起传递给初始化 CUTLASS 内核，然后启动内核。
在这个示例中，我们随后会启动一个参考 GEMM 内核（来自 CUTLASS 实用工具），以比较 CUTLASS 内核的输出是否与参考 GEMM 内核的输出相同。

<NT>M threadblock tile / warp tile / mma tile 的选择
下面例子分别设为(128,128,32), (64,64,32), (8,8,4). 
1）三者需要能依次被整除；
2）每个SM都有4个warp调度器，应每个调度器有多于1个warp可以让它有更多的调度空间，也不能设置太多，会减少每个warp可用的寄存器数量。
   所以一般一个block里设定256(256/32=8个warp)，有时也因整除需要或寄存器不足的需求，会设为128个线程即4个warp。
   如当前splitk里面，threadblock_tile里可分成4个warp_tile。
   又如hopper架构后(sm90)，一个warp group为4个warp。
3）block_tile的大小需要根据problem_shape和sm数量来考量，一个block由一个sm调度。如block_tile太大则sm用不完(寄存器也需要考虑)，sm太小则切换频繁。
   如L40的sm数量为108，H20有132个。
4）mma_tile可以查询 笔记 "mma.sync 随架构演进"，架构可支持下，一般越大越好，除非problem shape有限制。
   或者网页 https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-shape

This example shows how to use split-k version of matrix multiplication using functions and data
structures provided by CUTLASS; which we run on a NVIDIA Volta GPU.

What is split-k?
Consider a problem size of M = 128, N = 128, K = 4096. In this case, if my thread-block tile size (a
tile can be viewed as a 2d matrix) is 128x128x4096, then we launch a singled a thread-block taking
up a single SM of 84 SMs present on V100. Hence the efficiency of computation is really low. So, how
to solve it? This is where split-k comes in. It is a way of partitioning K-dimension of matrix
multiplication and distribute across multiple SMs and get better efficiency than single SM. In the
above example, we can partition K-dimension with split-k factor of 16 i.e., thread-block tile size
will be 128x128x256 and will be launching on 16 SMs. Once each thread-block computes their partial
inner product (1/16th of output), they accumulate to single output matrix.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In this example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
Volta and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. Volta also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to compute alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (float), the number of elements per vector memory access (16),
data type of accumulator (float) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x4, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::GemmSplitKParallel template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run() {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  //
  // Define problem size
  //

  const int length_m = 5120;
  const int length_n = 4096;
  const int length_k = 4096;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 16 partitions
  int split_k_slices = 16;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0  : -1);
}

int main() {

  //
  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  //
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero, so this test passes when built with older CUDA Toolkits. Its action are no-op.
    return 0;
  }
  else {
    return run();
  }
}

