#ifndef GPU_HEIPA_REDUCTIONS_H
#define GPU_HEIPA_REDUCTIONS_H

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>

#include "definitions.h"

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
