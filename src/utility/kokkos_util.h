/*******************************************************************************
 * MIT License
 *
 * This file is part of GPU-HeiPa.
 *
 * Copyright (C) 2025 Henning Woydt <henning.woydt@informatik.uni-heidelberg.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#ifndef GPU_HEIPA_KOKKOS_UTIL_H
#define GPU_HEIPA_KOKKOS_UTIL_H

#include <Kokkos_Core_fwd.hpp>

#include "definitions.h"

namespace GPU_HeiPa {
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    constexpr T min_sentinel() {
        if constexpr (std::is_same<T, f32>::value) {
            return -1e30f;
        } else if constexpr (std::is_same<T, f64>::value) {
            return -1e300;
        } else if constexpr (std::is_same<T, s8>::value) {
            return -128;
        } else if constexpr (std::is_same<T, s16>::value) {
            return -32768;
        } else if constexpr (std::is_same<T, s32>::value) {
            return -2147483647 - 1;
        } else if constexpr (std::is_same<T, s64>::value) {
            return static_cast<s64>(-9223372036854775807LL - 1);
        } else if constexpr (std::is_same<T, u8>::value ||
                             std::is_same<T, u16>::value ||
                             std::is_same<T, u32>::value ||
                             std::is_same<T, u64>::value) {
            return 0; // No negative sentinel for unsigned
        } else {
            return static_cast<T>(-1); // fallback for unknown types
        }
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    constexpr T max_sentinel() {
        if constexpr (std::is_same<T, f32>::value) {
            return 1e30f;
        } else if constexpr (std::is_same<T, f64>::value) {
            return 1e300;
        } else if constexpr (std::is_same<T, s8>::value) {
            return 127;
        } else if constexpr (std::is_same<T, s16>::value) {
            return 32767;
        } else if constexpr (std::is_same<T, s32>::value) {
            return 2147483647;
        } else if constexpr (std::is_same<T, s64>::value) {
            return static_cast<s64>(9223372036854775807LL);
        } else if constexpr (
            std::is_same<T, u8>::value ||
            std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value ||
            std::is_same<T, u64>::value
        ) {
            // all bits 1 for unsigned gives the maximum
            return static_cast<T>(-1);
        } else {
            // fallback for other types
            return std::numeric_limits<T>::max();
        }
    }

    std::string get_kokkos_execution_space_as_str() {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
#if defined(KOKKOS_ENABLE_CUDA)
        if (std::is_same<ExecSpace, Kokkos::Cuda>::value) { return "Cuda"; }
#endif

#if defined(KOKKOS_ENABLE_HIP)
        if (std::is_same<ExecSpace, Kokkos::HIP>::value) { return "HIP"; }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
        if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) { return "OpenMP"; }
#endif

#if defined(KOKKOS_ENABLE_THREADS)
        if (std::is_same<ExecSpace, Kokkos::Threads>::value) { return "Threads"; }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
        if (std::is_same<ExecSpace, Kokkos::Serial>::value) { return "Serial"; }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    u32 xs32(u32 x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    /**
     * Uniformly pick a random element from a rank-1 Kokkos View.
     * Assumes `size > 0`. Returns the element by value.
     *
     * @tparam View1D A rank-1 Kokkos::View (or subview) with operator()(u32)
     * @param v       The view
     * @param size    Number of items to sample from (usually v.extent(0))
     * @param seed    Run seed (same across threads if you want determinism)
     */
    template<class View1D>
    KOKKOS_INLINE_FUNCTION
    auto draw_random(const View1D &v, u64 size, u32 seed) {
        // simple xorshift-based PRN; fine for selection
        const u32 r = xs32(seed ^ 0x9e3779b9u);
        const u64 idx = r % size;

        return v(idx);
    }
}

#endif //GPU_HEIPA_KOKKOS_UTIL_H
