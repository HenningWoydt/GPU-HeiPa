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

        // FRONT side (grows from the beginning of the buffer upwards)
        size_t n_bytes_in_use = 0; // bytes used from the front (kept for backward compat)
        std::vector<size_t> reserved_chunks; // LIFO sizes from the front (aligned)

        // BACK side (grows from the end of the buffer downwards)
        size_t n_bytes_in_use_back = 0; // bytes used from the back
        std::vector<size_t> reserved_chunks_back; // LIFO sizes from the back (aligned)

        void *base = nullptr; // base pointer to Kokkos-allocated memory
    };

    // Helper: round up to 64-byte alignment
    inline size_t align64(size_t n) {
        return (n + 63u) & ~size_t(63u);
    }

    inline KokkosMemoryStack initialize_kokkos_memory_stack(size_t n_bytes, const std::string &t_name) {
        ScopedTimer _t("misc", "KokkosMemoryStack", "allocate");

        KokkosMemoryStack stack;
        stack.name = t_name;
        stack.n_bytes_allocated = align64(n_bytes);
        stack.n_bytes_in_use = 0;
        stack.n_bytes_in_use_back = 0;
        stack.base = Kokkos::kokkos_malloc<DeviceMemorySpace>(stack.n_bytes_allocated);
        if (!stack.base) {
            std::cerr << "ERROR: Failed to allocate Kokkos memory stack '" << t_name << "'\n";
            std::cerr << "       Requested: " << ((double) n_bytes / (1024.0 * 1024.0)) << " MB (" << n_bytes << " bytes)\n";
            std::cerr << "       Aligned:   " << ((double) stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_allocated << " bytes)\n";
            exit(EXIT_FAILURE);
        }

        return stack;
    }

    inline void *get_chunk_front(KokkosMemoryStack &stack, size_t n_bytes) {
        const size_t need = align64(n_bytes);
        if (need == 0) return nullptr;

        // total used = front + back; must not exceed allocated
        const size_t total_used = stack.n_bytes_in_use + stack.n_bytes_in_use_back;

        if (total_used + need > stack.n_bytes_allocated) {
            std::cerr << "ERROR: Memory stack '" << stack.name << "' out of memory (front allocation)\n";
            std::cerr << "       Stack size:        " << ((double) stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_allocated << " bytes)\n";
            std::cerr << "       Used front:        " << ((double) stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_in_use << " bytes)\n";
            std::cerr << "       Used back:         " << ((double) stack.n_bytes_in_use_back / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_in_use_back << " bytes)\n";
            std::cerr << "       Requested:         " << ((double) n_bytes / (1024.0 * 1024.0)) << " MB (" << n_bytes << " bytes)\n";
            std::cerr << "       Aligned need:      " << ((double) need / (1024.0 * 1024.0)) << " MB (" << need << " bytes)\n";
            std::cerr << "       Available (total): " << ((double) (stack.n_bytes_allocated - total_used) / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Active front chunks: " << stack.reserved_chunks.size() << "\n";
            std::cerr << "       Active back chunks:  " << stack.reserved_chunks_back.size() << "\n";
            abort();
        }

        // pointer at current front top
        char *ptr = static_cast<char *>(stack.base) + stack.n_bytes_in_use;
        stack.n_bytes_in_use += need;
        stack.reserved_chunks.push_back(need);

        return static_cast<void *>(ptr);
    }

    inline void pop_front(KokkosMemoryStack &stack) {
        if (stack.reserved_chunks.empty()) {
            std::cerr << "ERROR: Attempting to pop_front from empty memory stack '" << stack.name << "'\n";
            std::cerr << "       Stack size:   " << ((double) stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Bytes front:  " << ((double) stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Bytes back:   " << ((double) stack.n_bytes_in_use_back / (1024.0 * 1024.0)) << " MB\n";
            exit(EXIT_FAILURE);
        }

        const size_t sz = stack.reserved_chunks.back();
        stack.reserved_chunks.pop_back();

        if (sz > stack.n_bytes_in_use) {
            std::cerr << "WARNING: Memory stack '" << stack.name << "' corruption detected (front)\n";
            std::cerr << "         Trying to free " << ((double) sz / (1024.0 * 1024.0)) << " MB but only "
                    << ((double) stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB in use at front\n";
            stack.n_bytes_in_use = 0;
        } else {
            stack.n_bytes_in_use -= sz;
        }
    }

    inline void *get_chunk_back(KokkosMemoryStack &stack, size_t n_bytes) {
        const size_t need = align64(n_bytes);
        if (need == 0) return nullptr;

        const size_t total_used = stack.n_bytes_in_use + stack.n_bytes_in_use_back;

        if (total_used + need > stack.n_bytes_allocated) {
            std::cerr << "ERROR: Memory stack '" << stack.name << "' out of memory (back allocation)\n";
            std::cerr << "       Stack size:        " << ((double) stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_allocated << " bytes)\n";
            std::cerr << "       Used front:        " << ((double) stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_in_use << " bytes)\n";
            std::cerr << "       Used back:         " << ((double) stack.n_bytes_in_use_back / (1024.0 * 1024.0)) << " MB (" << stack.n_bytes_in_use_back << " bytes)\n";
            std::cerr << "       Requested:         " << ((double) n_bytes / (1024.0 * 1024.0)) << " MB (" << n_bytes << " bytes)\n";
            std::cerr << "       Aligned need:      " << ((double) need / (1024.0 * 1024.0)) << " MB (" << need << " bytes)\n";
            std::cerr << "       Available (total): " << ((double) (stack.n_bytes_allocated - total_used) / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Active front chunks: " << stack.reserved_chunks.size() << "\n";
            std::cerr << "       Active back chunks:  " << stack.reserved_chunks_back.size() << "\n";
            abort();
        }

        // back grows from the end of the buffer downwards
        const size_t back_offset = stack.n_bytes_in_use_back + need;
        char *ptr = static_cast<char *>(stack.base) + (stack.n_bytes_allocated - back_offset);

        stack.n_bytes_in_use_back += need;
        stack.reserved_chunks_back.push_back(need);

        return static_cast<void *>(ptr);
    }

    inline void pop_back(KokkosMemoryStack &stack) {
        if (stack.reserved_chunks_back.empty()) {
            std::cerr << "ERROR: Attempting to pop_back from empty memory stack '" << stack.name << "'\n";
            std::cerr << "       Stack size:   " << ((double) stack.n_bytes_allocated / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Bytes front:  " << ((double) stack.n_bytes_in_use / (1024.0 * 1024.0)) << " MB\n";
            std::cerr << "       Bytes back:   " << ((double) stack.n_bytes_in_use_back / (1024.0 * 1024.0)) << " MB\n";
            exit(EXIT_FAILURE);
        }

        const size_t sz = stack.reserved_chunks_back.back();
        stack.reserved_chunks_back.pop_back();

        if (sz > stack.n_bytes_in_use_back) {
            std::cerr << "WARNING: Memory stack '" << stack.name << "' corruption detected (back)\n";
            std::cerr << "         Trying to free " << ((double) sz / (1024.0 * 1024.0)) << " MB but only "
                    << ((double) stack.n_bytes_in_use_back / (1024.0 * 1024.0)) << " MB in use at back\n";
            stack.n_bytes_in_use_back = 0;
        } else {
            stack.n_bytes_in_use_back -= sz;
        }
    }

    inline void destroy(KokkosMemoryStack &stack) {
        if (stack.base) {
            Kokkos::kokkos_free<DeviceMemorySpace>(stack.base);
            stack.base = nullptr;
        }
        stack.n_bytes_allocated = 0;
        stack.n_bytes_in_use = 0;
        stack.n_bytes_in_use_back = 0;
        stack.reserved_chunks.clear();
        stack.reserved_chunks_back.clear();
    }

    inline void assert_front_is_empty(KokkosMemoryStack &mem_stack) {
        #if ASSERT_ENABLED
        ASSERT(mem_stack.n_bytes_in_use == 0);
        ASSERT(mem_stack.reserved_chunks.empty());
        #endif
    }

    inline void assert_back_is_empty(KokkosMemoryStack &mem_stack) {
        #if ASSERT_ENABLED
        ASSERT(mem_stack.n_bytes_in_use_back == 0);
        ASSERT(mem_stack.reserved_chunks_back.empty());
        #endif
    }

    inline void assert_is_empty(KokkosMemoryStack &mem_stack) {
        assert_front_is_empty(mem_stack);
        assert_back_is_empty(mem_stack);
    }
}

#endif //GPU_HEIPA_KOKKOS_MEMORY_STACK_H
