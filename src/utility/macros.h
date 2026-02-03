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

#ifndef GPU_HEIPA_MACROS_H
#define GPU_HEIPA_MACROS_H

#include <Kokkos_Core.hpp>

namespace GPU_HeiPa {
    #ifndef ENABLE_PROFILER
    #define ENABLE_PROFILER 0
    #endif

    #ifndef ASSERT_ENABLED
    #define ASSERT_ENABLED false
    #endif

    #if (ASSERT_ENABLED)
    #define ASSERT(condition) if(!(condition)) {std::cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " at line " << __LINE__ << "!" << std::endl; abort(); } ((void)0)
    #else
    #define ASSERT(condition) if(!(false)) {((void)0); } ((void)0)
    #endif

    #if (ENABLE_PROFILER)
    #define KOKKOS_PROFILE_FENCE() do { Kokkos::fence(); } while(0)
    #else
    #define KOKKOS_PROFILE_FENCE() do {} while(0)
    #endif

    #if (ASSERT_ENABLED)

    // Basic assert
    #define MY_KOKKOS_ASSERT(cond) \
        do { \
        if (!(cond)) { \
        printf("\n[ASSERT FAILED]\n" \
        "  Condition : %s\n" \
        "  File      : %s\n" \
        "  Function  : %s\n" \
        "  Line      : %d\n", \
        #cond, __FILE__, __func__, __LINE__); \
        Kokkos::abort("[ASSERT FAILED]"); \
        } \
        } while (0)

    #else  // ASSERT_ENABLED

    #define MY_KOKKOS_ASSERT(cond)         do { (void)sizeof(cond); } while (0)

    #endif
}

#endif //GPU_HEIPA_MACROS_H
