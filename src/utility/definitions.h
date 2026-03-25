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
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>

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

    constexpr weight_t GAIN_MIN = std::numeric_limits<weight_t>::lowest();
    constexpr partition_t NULL_PART = std::numeric_limits<u32>::max() - 1;
    constexpr partition_t HASH_RECLAIM = std::numeric_limits<u32>::max() - 2;
    constexpr partition_t NO_MOVE = std::numeric_limits<u32>::max() - 3;
    constexpr u32 NO_BLOCK_ID = std::numeric_limits<u32>::max();

    using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
    using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using HostMemory = Kokkos::CudaHostPinnedSpace;

    using UnmanagedHostVertex = Kokkos::View<vertex_t *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using HostVertex = Kokkos::View<vertex_t *, HostMemory>;
    using UnmanagedHostWeight = Kokkos::View<weight_t *, HostMemory, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using HostWeight = Kokkos::View<weight_t *, HostMemory>;
    using HostPartition = Kokkos::View<partition_t *, HostMemory>;
    using HostU8 = Kokkos::View<u8 *, HostMemory>;
    using HostU16 = Kokkos::View<u16 *, HostMemory>;
    using HostU32 = Kokkos::View<u32 *, HostMemory>;
    using HostU64 = Kokkos::View<u64 *, HostMemory>;
    using HostS8 = Kokkos::View<s8 *, HostMemory>;
    using HostS16 = Kokkos::View<s16 *, HostMemory>;
    using HostS32 = Kokkos::View<s32 *, HostMemory>;
    using HostS64 = Kokkos::View<s64 *, HostMemory>;
    using HostF32 = Kokkos::View<f32 *, HostMemory>;
    using HostF64 = Kokkos::View<f64 *, HostMemory>;

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
    using DeviceScalarVertex = Kokkos::View<vertex_t, DeviceMemorySpace>;
    using DeviceScalarWeight = Kokkos::View<weight_t, DeviceMemorySpace>;
    using DeviceScalarPartition = Kokkos::View<partition_t, DeviceMemorySpace>;
    using DeviceScalarU8 = Kokkos::View<u8, DeviceMemorySpace>;
    using DeviceScalarU16 = Kokkos::View<u16, DeviceMemorySpace>;
    using DeviceScalarU32 = Kokkos::View<u32, DeviceMemorySpace>;
    using DeviceScalarU64 = Kokkos::View<u64, DeviceMemorySpace>;
    using DeviceScalarS8 = Kokkos::View<s8, DeviceMemorySpace>;
    using DeviceScalarS16 = Kokkos::View<s16, DeviceMemorySpace>;
    using DeviceScalarS32 = Kokkos::View<s32, DeviceMemorySpace>;
    using DeviceScalarS64 = Kokkos::View<s64, DeviceMemorySpace>;
    using DeviceScalarF32 = Kokkos::View<f32, DeviceMemorySpace>;
    using DeviceScalarF64 = Kokkos::View<f64, DeviceMemorySpace>;

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

    using Device = Kokkos::Device<DeviceExecutionSpace, DeviceMemorySpace>;
    using GraphType = KokkosSparse::StaticCrsGraph<s32, Kokkos::LayoutLeft, Device, void, s32>;
    using MatrixType = KokkosSparse::CrsMatrix<weight_t, s32, Device, void, s32>;

    using HostScalarPinnedVertex = Kokkos::View<vertex_t, Kokkos::SharedHostPinnedSpace>;
    using HostScalarPinnedU32 = Kokkos::View<u32, Kokkos::SharedHostPinnedSpace>;
    using HostPinnedWeight = Kokkos::View<weight_t *, Kokkos::SharedHostPinnedSpace>;
    using HostScalarPinnedWeight = Kokkos::View<weight_t, Kokkos::SharedHostPinnedSpace>;
    using Policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
}

#endif //GPU_HEIPA_DEFINITIONS_H
