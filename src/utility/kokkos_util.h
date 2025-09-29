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

namespace Hei_Pa {
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
}

#endif //GPU_HEIPA_KOKKOS_UTIL_H
