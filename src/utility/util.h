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

#ifndef GPU_HEIPA_UTIL_H
#define GPU_HEIPA_UTIL_H

#include <charconv>
#include <fstream>
#include <iomanip>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "definitions.h"

namespace GPU_HeiPa {
    inline std::vector<std::string> split(const std::string &str,
                                          char c) {
        std::vector<std::string> splits;

        std::istringstream iss(str);
        std::string token;

        while (std::getline(iss, token, c)) {
            splits.push_back(token);
        }

        return splits;
    }

    inline std::vector<std::string> split_ws(const std::string &str) {
        std::vector<std::string> result;
        std::istringstream iss(str);
        std::string token;
        while (iss >> token) {
            // skips all whitespace automatically
            result.push_back(token);
        }
        return result;
    }

    inline bool file_exists(const std::string &path) {
        std::ifstream f(path.c_str());
        return f.good();
    }

    inline std::vector<char> load_file_to_buffer(const std::string &file_path) {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate); // open at end
        if (!file) {
            std::cerr << "Could not open file " << file_path << std::endl;
            exit(EXIT_FAILURE);
        }

        std::streamsize size = file.tellg(); // get size
        file.seekg(0, std::ios::beg); // rewind

        std::vector<char> buffer((size_t) size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read file into buffer" << std::endl;
            exit(EXIT_FAILURE);
        }

        return buffer;
    }

    inline std::vector<std::string> read_header(const std::string &str) {
        std::vector<std::string> header;

        size_t i = 0;
        while (i < str.size()) {
            // Skip leading spaces
            while (i < str.size() && str[i] == ' ') { ++i; }

            size_t start = i;

            while (i < str.size() && str[i] != ' ') { ++i; }

            if (start < i) { header.emplace_back(str.substr(start, i - start)); }
        }

        return header;
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<u64> &ints) {
        ints.resize(str.size());

        size_t idx = 0;
        u64 curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (u64) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline void str_to_ints(const std::string &str,
                            std::vector<int> &ints) {
        ints.resize(str.size());

        size_t idx = 0;
        int curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                curr_number = curr_number * 10 + (int) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;
        ints.resize(idx);
    }

    inline size_t str_to_ints(const std::string &str,
                            std::vector<vertex_t> &ints) noexcept {
        size_t idx = 0;
        vertex_t curr_number = 0;

        for (const char c: str) {
            if (c == ' ') {
                ints[idx] = curr_number;
                idx += curr_number != 0;
                curr_number = 0;
            } else {
                // curr_number = curr_number * 10 + (vertex_t) (c - '0');
                curr_number = (curr_number << 3) + (curr_number << 1) + (vertex_t) (c - '0');
            }
        }

        ints[idx] = curr_number;
        idx += curr_number != 0;

        return idx;
    }

    inline auto get_time_point() {
        return std::chrono::high_resolution_clock::now();
    }

    inline f64 get_seconds(std::chrono::high_resolution_clock::time_point sp,
                           std::chrono::high_resolution_clock::time_point ep) {
        return (f64) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e9;
    }

    inline f64 get_milli_seconds(std::chrono::high_resolution_clock::time_point sp,
                                 std::chrono::high_resolution_clock::time_point ep) {
        return (f64) std::chrono::duration_cast<std::chrono::nanoseconds>(ep - sp).count() / 1e6;
    }

    template<typename T>
    T convert_to(const std::string &str) {
        T result;
        std::istringstream iss(str);
        iss >> result;
        return result;
    }

    template<typename T>
    std::vector<T> convert(const std::vector<std::string> &vec) {
        std::vector<T> v;

        for (auto &s: vec) {
            v.push_back(convert_to<T>(s));
        }

        return v;
    }

    template<typename T1, typename T2>
    T1 prod(const std::vector<T2> &vec) {
        T1 p = (T1) 1;

        for (auto &x: vec) {
            p *= (T1) x;
        }

        return p;
    }

    template<typename T>
    T max(const std::vector<T> &vec) {
        T p = vec[0];

        for (auto &x: vec) {
            p = std::max(p, x);
        }

        return p;
    }

    inline void write_partition(const std::vector<partition_t> &partition,
                                const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary); // Open file in binary mode for faster writing
        if (!out.is_open()) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!" << std::endl;
            return;
        }

        // Write each partition element directly to the file, separating them by newlines
        for (const auto &i: partition) {
            out << i << '\n'; // Write each element followed by a newline
        }

        out.close();
    }

    inline void write_partition(const HostPartition &partition,
                                vertex_t n,
                                const std::string &file_path) {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Could not open " << file_path << " to write partition!\n";
            return;
        }

        // 1) Enlarge the stream's internal buffer (optional but helps).
        std::vector<char> stream_buf(1 << 20); // 1 MB
        out.rdbuf()->pubsetbuf(stream_buf.data(), (long int) stream_buf.size());

        // 2) Our own aggregation buffer for batched writes.
        std::string buf;
        buf.reserve(1 << 20); // 1 MB; tune as needed

        for (vertex_t u = 0; u < n; ++u) {
            // Convert integer to text without iostream overhead.
            char tmp[32]; // enough for signed 64-bit
            auto val = partition(u);
            auto res = std::to_chars(std::begin(tmp), std::end(tmp), val);
            if (res.ec != std::errc{}) {
                std::cerr << "Error: to_chars failed while writing.\n";
                return;
            }
            const size_t len = static_cast<size_t>(res.ptr - tmp);

            // Flush our buffer if adding this line would overflow capacity.
            if (buf.size() + len + 1 > buf.capacity()) {
                out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                buf.clear();
            }
            buf.append(tmp, len);
            buf.push_back('\n');
        }

        if (!buf.empty())
            out.write(buf.data(), static_cast<std::streamsize>(buf.size()));

        out.flush(); // optional
    }
}

#endif //GPU_HEIPA_UTIL_H
