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

#ifndef GPU_HEIPA_DISTANCE_ORACLE_HELPERS_H
#define GPU_HEIPA_DISTANCE_ORACLE_HELPERS_H

#include "../utility/definitions.h"
#include "../datastructures/kokkos_memory_stack.h"
#include "distance_oracle_matrix.h"


namespace GPU_HeiPa {
    // Generic declaration (never defined)
    template<typename DOracle>
    DOracle initialize_distance_oracle(partition_t k,
                                       std::vector<partition_t> &hierarchy,
                                       std::vector<weight_t> &distances,
                                       KokkosMemoryStack &mem_stack);

    // Specialization for matrix
    template<>
    inline DistanceOracleMatrix initialize_distance_oracle<DistanceOracleMatrix>(partition_t k,
                                                                                 std::vector<partition_t> &hierarchy,
                                                                                 std::vector<weight_t> &distances,
                                                                                 KokkosMemoryStack &mem_stack) {
        return initialize_distance_oracle_matrix(k, hierarchy, distances, mem_stack);
    }

    // Declaration
    template<typename DOracle>
    void free_distance_oracle(DOracle &oracle, KokkosMemoryStack &mem_stack);

    // Specialization for matrix
    template<>
    inline void free_distance_oracle<DistanceOracleMatrix>(DistanceOracleMatrix &oracle, KokkosMemoryStack &mem_stack) {
        free_distance_oracle_matrix(oracle, mem_stack);
    }
}

#endif //GPU_HEIPA_DISTANCE_ORACLE_HELPERS_H
