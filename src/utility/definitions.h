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

#ifndef GPU_HEIPA_DEFINITIONS_H
#define GPU_HEIPA_DEFINITIONS_H

#include <Kokkos_Core.hpp>

namespace GPU_HeiPa {
    typedef int8_t s8;
    typedef int16_t s16;
    typedef int32_t s32;
    typedef int64_t s64;

    typedef uint8_t u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;

    typedef float f32;
    typedef double f64;

    typedef u32 vertex_t;
    typedef s32 weight_t;
    typedef u32 partition_t;

    static constexpr partition_t NULL_PART = std::numeric_limits<partition_t>::max();
    static constexpr partition_t NO_MOVE = std::numeric_limits<partition_t>::max() - 1;

    using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

    using HostVertex = Kokkos::View<vertex_t *, Kokkos::HostSpace>;
    using HostWeight = Kokkos::View<weight_t *, Kokkos::HostSpace>;
    using HostPartition = Kokkos::View<partition_t *, Kokkos::HostSpace>;
    using HostU8 = Kokkos::View<u8 *, Kokkos::HostSpace>;
    using HostU16 = Kokkos::View<u16 *, Kokkos::HostSpace>;
    using HostU32 = Kokkos::View<u32 *, Kokkos::HostSpace>;
    using HostU64 = Kokkos::View<u64 *, Kokkos::HostSpace>;
    using HostS8 = Kokkos::View<s8 *, Kokkos::HostSpace>;
    using HostS16 = Kokkos::View<s16 *, Kokkos::HostSpace>;
    using HostS32 = Kokkos::View<s32 *, Kokkos::HostSpace>;
    using HostS64 = Kokkos::View<s64 *, Kokkos::HostSpace>;
    using HostF32 = Kokkos::View<f32 *, Kokkos::HostSpace>;
    using HostF64 = Kokkos::View<f64 *, Kokkos::HostSpace>;

    using DeviceVertex = Kokkos::View<vertex_t *, DeviceMemorySpace>;
    using DeviceWeight = Kokkos::View<weight_t *, DeviceMemorySpace>;
    using DevicePartition = Kokkos::View<partition_t *, DeviceMemorySpace>;
    using DeviceU8 = Kokkos::View<u8 *, DeviceMemorySpace>;
    using DeviceU16 = Kokkos::View<u16 *, DeviceMemorySpace>;
    using DeviceU32 = Kokkos::View<u32 *, DeviceMemorySpace>;
    using DeviceU64 = Kokkos::View<u64 *, DeviceMemorySpace>;
    using DeviceS8 = Kokkos::View<s8 *, DeviceMemorySpace>;
    using DeviceS16 = Kokkos::View<s16 *, DeviceMemorySpace>;
    using DeviceS32 = Kokkos::View<s32 *, DeviceMemorySpace>;
    using DeviceS64 = Kokkos::View<s64 *, DeviceMemorySpace>;
    using DeviceF32 = Kokkos::View<f32 *, DeviceMemorySpace>;
    using DeviceF64 = Kokkos::View<f64 *, DeviceMemorySpace>;

    using UnmanagedDeviceVertex = Kokkos::View<vertex_t *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceWeight = Kokkos::View<weight_t *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDevicePartition = Kokkos::View<partition_t *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceU8 = Kokkos::View<u8 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceU16 = Kokkos::View<u16 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceU32 = Kokkos::View<u32 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceU64 = Kokkos::View<u64 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceS8 = Kokkos::View<s8 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceS16 = Kokkos::View<s16 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceS32 = Kokkos::View<s32 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceS64 = Kokkos::View<s64 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceF32 = Kokkos::View<f32 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using UnmanagedDeviceF64 = Kokkos::View<f64 *, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
}

#endif //GPU_HEIPA_DEFINITIONS_H
