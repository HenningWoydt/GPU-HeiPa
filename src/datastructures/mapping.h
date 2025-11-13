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

#ifndef GPU_HEIPA_MAPPING_H
#define GPU_HEIPA_MAPPING_H
#include "../utility/definitions.h"

namespace GPU_HeiPa {
    /**
     * Maps each vertex of a graph to a new vertex of its coarser representation.
     */
    struct Mapping {
        vertex_t old_n = 0;
        vertex_t coarse_n = 0;
        UnmanagedDeviceVertex mapping;
    };

    inline Mapping initialize_mapping(vertex_t t_old_n,
                                      vertex_t t_coarse_n,
                                      KokkosMemoryStack &mem_stack) {
        Mapping mapping;

        mapping.old_n = t_old_n;
        mapping.coarse_n = t_coarse_n;

        auto *mapping_ptr = (vertex_t *) get_chunk_front(mem_stack, sizeof(vertex_t) * t_old_n);
        mapping.mapping = UnmanagedDeviceVertex(mapping_ptr, t_old_n);

        return mapping;
    }

    inline void free_mapping(Mapping &mapping,
                             KokkosMemoryStack &mem_stack) {
        pop_front(mem_stack);
    }
}

#endif //GPU_HEIPA_MAPPING_H
