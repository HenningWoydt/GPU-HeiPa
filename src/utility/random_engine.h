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

#ifndef GPU_HEIPA_RANDOMENGINE_H
#define GPU_HEIPA_RANDOMENGINE_H

#include <random>

#include "definitions.h"

namespace GPU_HeiPa {
    class RandomEngine {
    public:
        std::uniform_int_distribution<s32> dis_s32;
        std::uniform_int_distribution<u32> dis_u32;
        std::uniform_int_distribution<u64> dis_u64;
        std::uniform_real_distribution<f32> dis_f32;
        std::uniform_real_distribution<f64> dis_f64;
        std::mt19937 generator;

        RandomEngine() = default;

        explicit RandomEngine(const u64 t_seed) {
            generator.seed(t_seed);
            dis_f32 = std::uniform_real_distribution<f32>(0.0f, 1.0f);
            dis_f64 = std::uniform_real_distribution<f64>(0.0, 1.0);
            dis_s32 = std::uniform_int_distribution<s32>(std::numeric_limits<s32>::min(), std::numeric_limits<s32>::max());
            dis_u32 = std::uniform_int_distribution<u32>(std::numeric_limits<u32>::min(), std::numeric_limits<u32>::max());
            dis_u64 = std::uniform_int_distribution<u64>(std::numeric_limits<u64>::min(), std::numeric_limits<u64>::max());
        }

        f32 get_f32() { return dis_f32(generator); }

        f32 get_f32(const f32 low, const f32 high) { return low + dis_f32(generator) * (high - low); }

        f64 get_f64() { return dis_f64(generator); }

        f64 get_f64(const f64 low, const f64 high) { return low + dis_f64(generator) * (high - low); }

        s32 get_s32() { return dis_s32(generator); }

        u32 get_u32() { return dis_u32(generator); }

        u64 get_u64() { return dis_u64(generator); }
    };
}

#endif //GPU_HEIPA_RANDOMENGINE_H
