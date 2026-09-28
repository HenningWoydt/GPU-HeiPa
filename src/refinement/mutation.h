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

#ifndef GPU_HEIPA_MUTATION_H
#define GPU_HEIPA_MUTATION_H

#include <Kokkos_Core.hpp>

#include "../definitions.h"
#include "../datastructures/partition.h"
#include "../solver/GPU_HeiPa_solver.h"



namespace GPU_HeiPa {

    
    void mutate_individual(
        Partition &individual,
        Graph &graph,
        partition_t t_k,
        f64 imbalance,
        u64 seed,
        bool t_use_ultra,
        KokkosMemoryStack &mem_stack,
        DeviceExecutionSpace &exec_space
    ){

        Solver(
            graph,
            t_k,
            imbalance,
            seed,
            t_use_ultra,
            individual,
            mem_stack,
            exec_space
        );

        return;
    }


}

#endif