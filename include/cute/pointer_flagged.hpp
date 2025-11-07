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
#pragma once

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/layout_composed.hpp>            // cute::ComposedLayout
#include <cute/pointer.hpp>                    // cute::make_smem_ptr
#include <cute/pointer_sparse.hpp>             // cute::is_sparse
#include <cute/pointer_swizzle.hpp>            // cute::make_swizzle_ptr
#include <cute/arch/util.hpp>                  // cute::cast_smem_ptr_to_uint
#include <cute/numeric/integral_constant.hpp>  // cute::Int

namespace cute
{

//
// Stand-in Swizzle Layout
//   A model of a nullptr smem_ptr<T> with B == sizeof_bits<T>::value
//   That represents an unset pointer. This is a placeholder type that is waiting for an smem_ptr
//

template <int Bits>
struct smem_ptr_flag_bits : Int<0> {};

using smem_ptr_flag = smem_ptr_flag_bits<1>;

// A flagged construction method to transform ComposedLayout
// Make a swizzle pointer tensor and check that the intended type size matches
template <class Iterator, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& ptr,
            ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  static_assert(is_smem<Iterator>::value, "Expected smem.");
  static_assert(B == sizeof_bits<iter_value_t<Iterator>>::value, "Expected a B-bit pointer type.");
  return make_tensor(make_smem_ptr(ptr.get(), layout.layout_a()),
                     layout.layout_b());
}

// NOTE: To preserve smem_ptr_flag_bits under recast ops
template <int N, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(layout.layout_a(), smem_ptr_flag_bits<B*N>{}, upcast<N>(layout.layout_b()));
}

template <int N, class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
downcast(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(layout.layout_a(), smem_ptr_flag_bits<B/N>{}, downcast<N>(layout.layout_b()));
}

//
// Conversion with swizzle_layout
//

template <class SwizzleFn, int B, class Layout>
CUTE_HOST_DEVICE
auto
as_position_independent_swizzle_layout(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  return composition(recast_layout<uint8_t,uint_bit_t<B>>(layout.layout_a()), Int<0>{}, layout.layout_b());
}

// <NT> 把 已经由硬件实现的 swizzle 规则 从 “字节地址视角” 转换成 “元素类型视角”，好让 C++ 代码能用类型安全的 smem_ptr<T> 去访问；
// 它本身并不实现 swizzle，也不给 TMA 增加 swizzle 能力。因此：
// 对于 ≥8-bit 类型，如果 TMA 描述符里开了硬件 swizzle，调用这个函数后得到的张量视图就会让后续 ld.matrix / wgmma 自动享受硬件重排。
// 对于 4-bit 类型，TMA 根本没有硬件 swizzle，函数同样会返回「无 swizzle」的视图——此时要靠软件在寄存器里手动解包，与函数无关。
// 
// 补充：只要 swizzle 函数对象是在 uint8_t*（或 void）上计算偏移，就是“字节地址视角”；
//      当它被 recast 到 T* 并按 sizeof(T) 倍数去计算偏移时，就成了“元素类型视角”。
//      如：
//      | cute::make_swizzle<Swizzle<B,M,S>>{} 的默认模板参数              | 字节地址视角 | 内部所有偏移量以 byte 为单位                                 |
//      | recast_layout<uint8_t, T>(swizzle)                              | 元素类型视角 | 把原 byte 偏移 ÷ `sizeof(T)`，得到元素粒度的偏移                |
//      | as_position_independent_swizzle_tensor(smem_tensor) 返回的新张量 | 元素类型视角 | 里面已经做了上面的 recast，迭代器按 T 步进                        |
//      | TMA 描述符里的 swizzle 模式（32 B/64 B/128 B）                   | 字节地址视角 | 硬件在 byte 地址线上做 XOR 重排，与类型无关                       |
//      | 后续 `ld.matrix`/ `wgmma.desc` 真正访问内存                      | 元素类型视角 | 指令一次性读 16×FP16 或 8×BF16，但硬件仍按 byte-swizzle 去算物理地址 |

template <class Tensor>
CUTE_HOST_DEVICE
auto
as_position_independent_swizzle_tensor(Tensor&& tensor)
{
  static_assert(is_smem<remove_cvref_t<Tensor>>::value, "Expected smem tensor.");
  using SwizzleFn = get_swizzle_t<remove_cvref_t<Tensor>>;
  if constexpr (SwizzleFn::num_bits == 0) {
    return tensor;
  } else {
#if !defined(NDEBUG)
    {
    uint32_t address = cast_smem_ptr_to_uint(raw_pointer_cast(static_cast<Tensor&&>(tensor).data()));
    uint32_t mask    = ((uint32_t(1) << SwizzleFn::num_base) - 1) | SwizzleFn::swizzle_code;
    assert((address & mask) == 0);  // Alignment to the Base, Z, and Y of Swizzle
    }
#endif
    using T = typename remove_cvref_t<Tensor>::value_type;
    // Recast swizzle from acting on byte-addressed pointers to elements of type-T
    auto new_swizzle = recast_layout<uint8_t, T>(SwizzleFn{});
    // Strip off everything and create a new smem_ptr for type-T
    auto new_ptr = make_smem_ptr<T>(raw_pointer_cast(static_cast<Tensor&&>(tensor).data()));
    return make_tensor(new_ptr, composition(new_swizzle, Int<0>{}, tensor.layout()));
  }
  CUTE_GCC_UNREACHABLE;
}

// A model of a nullptr sparse_ptr<S, smem_ptr<T>> with B == sizeof_bits<T>::value
// That represents an unset pointer. This is a placeholder type that is waiting for an smem_ptr
template <int Sparsity, int Bits>
struct smem_sparse_ptr_flag_bits : Int<0> {};

template <int Sparsity>
using smem_sparse_ptr_flag = smem_sparse_ptr_flag_bits<Sparsity, 1>;

// A flagged construction method to transform ComposedLayout
// Make a swizzle pointer tensor and check that the intended type size matches
template <class Iterator, class SwizzleFn, int S, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& ptr,
            ComposedLayout<SwizzleFn,smem_sparse_ptr_flag_bits<S,B>,Layout> const& layout)
{
  static_assert(is_smem<Iterator>::value, "Expected smem.");
  static_assert(is_sparse_ptr<Iterator>::value, "Expected sparse iter");
  static_assert(is_sparse<iter_value_t<Iterator>>::value, "Expected sparse elem");
  static_assert(S == iter_value_t<Iterator>::sparsity, "Expected sparsity S");
  static_assert(B == sizeof_bits<typename iter_value_t<Iterator>::raw_type>::value, "Expected B-bit pointer type");
  return make_tensor(make_swizzle_ptr(ptr, layout.layout_a()), layout.layout_b());
}

// NOTE: To preserve smem_ptr_flag_bits under recast ops
template <int N, class SwizzleFn, int S, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<SwizzleFn,smem_sparse_ptr_flag_bits<S,B>,Layout> const& layout)
{
  static_assert(dependent_false<SwizzleFn>, "Not implemented for safety");
}

template <int N, class SwizzleFn, int S, int B, class Layout>
CUTE_HOST_DEVICE constexpr
auto
downcast(ComposedLayout<SwizzleFn,smem_sparse_ptr_flag_bits<S,B>,Layout> const& layout)
{
  static_assert(dependent_false<SwizzleFn>, "Not implemented for safety");
}

//
// Display utilities
//

template <int B>
CUTE_HOST_DEVICE void print(smem_ptr_flag_bits<B> ptr)
{
  printf("smem_ptr[%db](unset)", B);
}

template <int S, int B>
CUTE_HOST_DEVICE void print(smem_sparse_ptr_flag_bits<S,B>)
{
  printf("smem_sparse<%d>_ptr[%db](unset)", S, B);
}

} // end namespace cute
