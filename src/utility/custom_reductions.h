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

#ifndef GPU_HEIPA_REDUCTIONS_H
#define GPU_HEIPA_REDUCTIONS_H

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>

#include "../definitions.h"

namespace GPU_HeiPa {
    struct Accumulators {
        u32 partial_0s = 0;
        u32 partial_1s = 0;

        KOKKOS_INLINE_FUNCTION
        void operator+=(const Accumulators &rhs) {
            partial_0s += rhs.partial_0s;
            partial_1s += rhs.partial_1s;
        }
    };

    struct WeightAccumulators {
        weight_t partial_0s = 0;
        weight_t partial_1s = 0;

        KOKKOS_INLINE_FUNCTION
        void operator+=(const WeightAccumulators &rhs) {
            partial_0s += rhs.partial_0s;
            partial_1s += rhs.partial_1s;
        }
    };

    struct bigAccumulator {
        u32 num_edges_0s = 0;
        u32 num_edges_1s = 0;

        weight_t weight_0s = 0;
        weight_t weight_1s = 0;

        KOKKOS_INLINE_FUNCTION
        void operator +=(const bigAccumulator &rhs) {
            num_edges_0s += rhs.num_edges_0s;
            num_edges_1s += rhs.num_edges_1s;
            weight_0s += rhs.weight_0s;
            weight_1s += rhs.weight_1s;
        }
    };
}


#endif
