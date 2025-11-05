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

#ifndef GPU_HEIPA_KOKKOS_MEMORY_STACK_H
#define GPU_HEIPA_KOKKOS_MEMORY_STACK_H

#include <iostream>

#include "../utility/definitions.h"
#include "../utility/macros.h"

namespace GPU_HeiPa {
    struct KokkosMemoryStack {
        std::string name;
        size_t n_bytes_allocated = 0; // total bytes in the pool
        size_t n_bytes_in_use = 0; // top of stack (offset)
        std::vector<size_t> reserved_chunks; // LIFO sizes (aligned)
        void *base = nullptr; // base pointer to Kokkos-allocated memory
    };

    // Helper: round up to 64-byte alignment
    inline size_t align64(size_t n) {
        return (n + 63u) & ~size_t(63u);
    }

    inline KokkosMemoryStack initialize_kokkos_memory_stack(size_t n_bytes, std::string t_name) {
        ScopedTimer _t("io", "KokkosMemoryStack", "allocate");

        std::cout << t_name << std::endl;

        KokkosMemoryStack stack;
        stack.name = t_name;
        stack.n_bytes_allocated = align64(n_bytes);
        stack.n_bytes_in_use = 0;
        stack.base = Kokkos::kokkos_malloc<DeviceMemorySpace>(stack.n_bytes_allocated);
        if (!stack.base) {
            std::cerr << "ERROR: Failed to allocate Kokkos memory stack '" << t_name << "'\n";
            std::cerr << "       Requested: " << ((double)n_bytes / (1024.0 * 1024.0)) << " MB (" << n_bytes << " bytes)\n";
            std::cerr << "       Aligned:   " << ((double)stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_allocated << " bytes)\n";
            exit(EXIT_FAILURE);
        }

        return stack;
    }

    inline void *get_chunk(KokkosMemoryStack &stack, size_t n_bytes) {
        const size_t need = align64(n_bytes);

        if (need == 0) return nullptr;

        if (stack.n_bytes_in_use + need > stack.n_bytes_allocated) {
            std::cerr << "ERROR: Memory stack '" << stack.name << "' out of memory\n";
            std::cerr << "       Stack size:    " << ((double)stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_allocated << " bytes)\n";
            std::cerr << "       Already used:  " << ((double)stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_in_use << " bytes)\n";
            std::cerr << "       Requested:     " << ((double)n_bytes / (1024.0 * 1024.0)) << " MB (" << n_bytes << " bytes)\n";
            std::cerr << "       Aligned need:  " << ((double)need / (1024.0 * 1024.0)) << " MB (" << need << " bytes)\n";
            std::cerr << "       Available:     " << ((double)(stack.n_bytes_allocated - stack.n_bytes_in_use) / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Active chunks: " << stack.reserved_chunks.size() << "\n";
            abort();
        }

        // Compute pointer to current top, then bump
        char *ptr = static_cast<char *>(stack.base) + stack.n_bytes_in_use;
        stack.n_bytes_in_use += need;
        stack.reserved_chunks.push_back(need);

        return static_cast<void *>(ptr);
    }

    inline void pop(KokkosMemoryStack &stack) {
        if (stack.reserved_chunks.empty()) {
            std::cerr << "ERROR: Attempting to pop from empty memory stack '" << stack.name << "'\n";
            std::cerr << "       Stack size: " << ((double)stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Bytes in use: " << ((double)stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB\n";
            exit(EXIT_FAILURE);
        }
        const size_t sz = stack.reserved_chunks.back();
        stack.reserved_chunks.pop_back();

        // Safe underflow check (defensive)
        if (sz > stack.n_bytes_in_use) {
            std::cerr << "WARNING: Memory stack '" << stack.name << "' corruption detected\n";
            std::cerr << "         Trying to free " << ((double)sz / (1024.0 * 1024.0)) << " MB but only " 
                      << ((double)stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB in use\n";
            stack.n_bytes_in_use = 0;
        } else {
            stack.n_bytes_in_use -= sz;
        }
    }

    inline void destroy(KokkosMemoryStack& stack) {
        ScopedTimer _t("io", "KokkosMemoryStack", "free");

        if (stack.base) {
            Kokkos::kokkos_free<DeviceMemorySpace>(stack.base);
            stack.base = nullptr;
        }
        stack.n_bytes_allocated = 0;
        stack.n_bytes_in_use = 0;
        stack.reserved_chunks.clear();
    }

    inline void assert_is_empty(KokkosMemoryStack& mem_stack) {
#if ASSERT_ENABLED
        ASSERT(mem_stack.n_bytes_in_use == 0);
#endif
    }
}

#endif //GPU_HEIPA_KOKKOS_MEMORY_STACK_H
