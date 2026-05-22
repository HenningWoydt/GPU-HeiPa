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

#ifndef GPU_HEIPA_PROFILER_H
#define GPU_HEIPA_PROFILER_H

#include <unordered_map>
#include <vector>
#include <string>
#include <string_view>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <shared_mutex>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "macros.h"
#include "util.h"

namespace GPU_HeiPa {
    inline std::string pad_cell(const std::string &text, int width) {
        if (width <= 0) return "";
        if (static_cast<int>(text.size()) <= width) {
            return text + std::string(width - static_cast<int>(text.size()), ' ');
        }
        if (width <= 1) {
            return text.substr(0, width);
        }
        return text.substr(0, width - 1) + "…";
    }

    inline bool stream_is_tty(std::ostream &stream) {
#if defined(_WIN32)
        DWORD mode;
        HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
        return handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &mode);
#else
        return &stream == &std::cout || &stream == &std::cerr;
#endif
    }

    struct ZebraTheme {
        const char *even_background = "\x1b[48;5;236m";
        const char *odd_background = "\x1b[48;5;235m";
        const char *header_background = "\x1b[48;5;238m";
        const char *rule_foreground = "\x1b[38;5;240m";
        const char *text_foreground = "\x1b[38;5;252m";
        const char *bold_on = "\x1b[1m";
        const char *bold_off = "\x1b[22m";
        const char *reset = "\x1b[0m";
    };

    inline ZebraTheme basic_theme() {
        return ZebraTheme{
            "\x1b[47m",
            "\x1b[107m",
            "\x1b[47m",
            "\x1b[90m",
            "\x1b[30m",
            "\x1b[1m",
            "\x1b[22m",
            "\x1b[0m"
        };
    }

    struct TimingStats {
        double total_ms = 0.0;
        unsigned long calls = 0;

        void add(double milliseconds) {
            total_ms += milliseconds;
            ++calls;
        }

        double average_ms() const {
            return calls ? total_ms / static_cast<double>(calls) : 0.0;
        }
    };

    struct FunctionProfile {
        std::unordered_map<std::string, TimingStats> kernels;
        TimingStats aggregate;
    };

    struct GroupProfile {
        std::unordered_map<std::string, FunctionProfile> functions;
        TimingStats aggregate;
    };

    struct ProfileKey {
        std::string_view group;
        std::string_view function;
        std::string_view kernel;

        bool operator==(const ProfileKey &other) const {
            return group == other.group && function == other.function && kernel == other.kernel;
        }
    };

    struct ProfileKeyHash {
        std::size_t operator()(const ProfileKey &key) const {
            auto h1 = std::hash<std::string_view>{}(key.group);
            auto h2 = std::hash<std::string_view>{}(key.function);
            auto h3 = std::hash<std::string_view>{}(key.kernel);
            std::size_t seed = 0;
            auto hash_combine = [](std::size_t &s, std::size_t v) {
                s ^= v + 0x9e3779b9 + (s << 6) + (s >> 2);
            };
            hash_combine(seed, h1);
            hash_combine(seed, h2);
            hash_combine(seed, h3);
            return seed;
        }
    };

    class Profiler {
    public:
        static Profiler &instance() {
            static Profiler profiler;
            return profiler;
        }

        void add(const char *group_name,
                 const char *function_name,
                 const char *kernel_name,
                 double milliseconds) {
#if ENABLE_PROFILER
            std::unique_lock<std::shared_mutex> lock(mutex_);
            stats_[{group_name, function_name, kernel_name}].add(milliseconds);
            total_.add(milliseconds);
#else
            (void) group_name;
            (void) function_name;
            (void) kernel_name;
            (void) milliseconds;
#endif
        }

        std::string to_JSON() const {
#if !ENABLE_PROFILER
            return "{}";
#else
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto groups = get_hierarchy();

            auto escape_json = [](const std::string &text) {
                std::ostringstream escaped;
                for (char ch: text) {
                    if (ch == '"' || ch == '\\') {
                        escaped << '\\' << ch;
                    } else if (ch == '\n') {
                        escaped << "\\n";
                    } else {
                        escaped << ch;
                    }
                }
                return escaped.str();
            };

            auto write_indent = [](std::ostringstream &output, int level) {
                for (int i = 0; i < level; ++i) {
                    output << '\t';
                }
            };

            std::ostringstream output;
            output.setf(std::ios::fixed);
            output.precision(6);

            int indent_level = 0;
            output << "{\n";

            write_indent(output, ++indent_level);
            output << "\"total\": {\n";
            write_indent(output, ++indent_level);
            output << "\"total_ms\": " << total_.total_ms << "\n";
            write_indent(output, --indent_level);
            output << "},\n";

            write_indent(output, indent_level);
            output << "\"groups\": {\n";
            ++indent_level;

            std::vector<std::pair<std::string, const GroupProfile *> > sorted_groups;
            sorted_groups.reserve(groups.size());
            for (const auto &group_entry: groups) {
                sorted_groups.emplace_back(group_entry.first, &group_entry.second);
            }

            std::sort(
                sorted_groups.begin(),
                sorted_groups.end(),
                [](const auto &left, const auto &right) {
                    return left.second->aggregate.total_ms > right.second->aggregate.total_ms;
                }
            );

            bool first_group = true;
            for (const auto &group_entry: sorted_groups) {
                const std::string &group_name = group_entry.first;
                const GroupProfile *group_profile = group_entry.second;

                if (!first_group) {
                    output << ",\n";
                }
                first_group = false;

                write_indent(output, indent_level);
                output << "\"" << escape_json(group_name) << "\": {\n";
                ++indent_level;

                write_indent(output, indent_level);
                output << "\"total_ms\": " << group_profile->aggregate.total_ms << ",\n";

                write_indent(output, indent_level);
                output << "\"functions\": {\n";
                ++indent_level;

                std::vector<std::pair<std::string, const FunctionProfile *> > sorted_functions;
                sorted_functions.reserve(group_profile->functions.size());
                for (const auto &function_entry: group_profile->functions) {
                    sorted_functions.emplace_back(function_entry.first, &function_entry.second);
                }

                std::sort(
                    sorted_functions.begin(),
                    sorted_functions.end(),
                    [](const auto &left, const auto &right) {
                        return left.second->aggregate.total_ms > right.second->aggregate.total_ms;
                    }
                );

                bool first_function = true;
                for (const auto &function_entry: sorted_functions) {
                    const std::string &function_name = function_entry.first;
                    const FunctionProfile *function_profile = function_entry.second;

                    if (!first_function) {
                        output << ",\n";
                    }
                    first_function = false;

                    write_indent(output, indent_level);
                    output << "\"" << escape_json(function_name) << "\": {\n";
                    ++indent_level;

                    write_indent(output, indent_level);
                    output << "\"total_ms\": " << function_profile->aggregate.total_ms << ",\n";

                    write_indent(output, indent_level);
                    output << "\"kernels\": {\n";
                    ++indent_level;

                    std::vector<std::pair<std::string, const TimingStats *> > sorted_kernels;
                    sorted_kernels.reserve(function_profile->kernels.size());
                    for (const auto &kernel_entry: function_profile->kernels) {
                        sorted_kernels.emplace_back(kernel_entry.first, &kernel_entry.second);
                    }

                    std::sort(
                        sorted_kernels.begin(),
                        sorted_kernels.end(),
                        [](const auto &left, const auto &right) {
                            return left.second->total_ms > right.second->total_ms;
                        }
                    );

                    bool first_kernel = true;
                    for (const auto &kernel_entry: sorted_kernels) {
                        const std::string &kernel_name = kernel_entry.first;
                        const TimingStats *kernel_stats = kernel_entry.second;

                        if (!first_kernel) {
                            output << ",\n";
                        }
                        first_kernel = false;

                        write_indent(output, indent_level);
                        output << "\"" << escape_json(kernel_name) << "\": {\n";
                        ++indent_level;

                        write_indent(output, indent_level);
                        output << "\"calls\": " << kernel_stats->calls << ",\n";

                        write_indent(output, indent_level);
                        output << "\"total_ms\": " << kernel_stats->total_ms << ",\n";

                        write_indent(output, indent_level);
                        output << "\"avg_ms\": " << kernel_stats->average_ms() << "\n";

                        --indent_level;
                        write_indent(output, indent_level);
                        output << "}";
                    }

                    output << "\n";
                    --indent_level;
                    write_indent(output, indent_level);
                    output << "}\n";

                    --indent_level;
                    write_indent(output, indent_level);
                    output << "}";
                }

                output << "\n";
                --indent_level;
                write_indent(output, indent_level);
                output << "}\n";

                --indent_level;
                write_indent(output, indent_level);
                output << "}";
            }

            output << "\n";
            --indent_level;
            write_indent(output, indent_level);
            output << "}\n";

            --indent_level;
            output << "}";

            return output.str();
#endif
        }

        void print_table_ascii_colored(std::ostream &output_stream = std::cout,
                                       int max_functions_per_group = -1,
                                       int max_kernels_per_function = -1,
                                       int name_column_width = 48,
                                       bool force_color = false,
                                       bool use_basic_colors = false) const {
#if !ENABLE_PROFILER
            return;
#else
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto groups = get_hierarchy();

            if (total_.total_ms <= 0.0) {
                output_stream << "Profiler: no samples recorded.\n";
                return;
            }

            const bool no_color_requested = std::getenv("NO_COLOR") != nullptr;
            bool use_colors = !no_color_requested && (force_color || stream_is_tty(output_stream));

#ifdef _WIN32
            if (use_colors) {
                // enable_ansi_on_windows(); // Assuming user handles this if needed or omitted for simple ANSI stream support
            }
#endif

            ZebraTheme theme = use_basic_colors ? basic_theme() : ZebraTheme{};

            auto colorize_row = [&](const std::string &line, bool is_header, bool is_even_row) -> std::string {
                if (!use_colors) {
                    return line + '\n';
                }

                const char *background =
                        is_header
                            ? theme.header_background
                            : (is_even_row ? theme.even_background : theme.odd_background);

                return std::string(background) + theme.text_foreground + line + theme.reset + '\n';
            };

            auto colorize_rule = [&](const std::string &line) -> std::string {
                if (!use_colors) {
                    return line + '\n';
                }
                return std::string(theme.rule_foreground) + line + theme.reset + '\n';
            };

            if (name_column_width < 24) name_column_width = 24;
            if (name_column_width > 96) name_column_width = 96;

            auto format_milliseconds = [](double value) {
                std::ostringstream stream;
                stream.setf(std::ios::fixed);
                stream << std::setprecision(3) << value;
                return stream.str();
            };

            auto format_percent = [](double value) {
                std::ostringstream stream;
                stream.setf(std::ios::fixed);
                stream << std::setprecision(1) << value;
                return stream.str();
            };

            auto percent_of_total = [&](double milliseconds) {
                return total_.total_ms > 0.0 ? (milliseconds * 100.0 / total_.total_ms) : 0.0;
            };

            std::vector<std::pair<std::string, const GroupProfile *> > sorted_groups;
            sorted_groups.reserve(groups.size());
            for (const auto &group_entry: groups) {
                sorted_groups.emplace_back(group_entry.first, &group_entry.second);
            }

            std::sort(
                sorted_groups.begin(),
                sorted_groups.end(),
                [](const auto &left, const auto &right) {
                    return left.second->aggregate.total_ms > right.second->aggregate.total_ms;
                }
            );

            const int calls_column_width = 8;
            const int total_column_width = 12;
            const int average_column_width = 10;
            const int percent_column_width = 7;

            auto make_rule_line = [&]() {
                return std::string(
                    name_column_width + 3 +
                    calls_column_width + 3 +
                    total_column_width + 3 +
                    average_column_width + 3 +
                    percent_column_width,
                    '-'
                );
            };

            size_t row_number = 0;

            output_stream << colorize_rule(make_rule_line());

            {
                std::string header_line;
                header_line.reserve(120);

                if (use_colors) {
                    header_line += theme.bold_on;
                }

                header_line += pad_cell("Scope", name_column_width);
                header_line += "   " + pad_cell("Calls", calls_column_width);
                header_line += "   " + pad_cell("Total ms", total_column_width);
                header_line += "   " + pad_cell("Avg ms", average_column_width);
                header_line += "   " + pad_cell("%Tot", percent_column_width);

                if (use_colors) {
                    header_line += theme.bold_off;
                }

                output_stream << colorize_row(header_line, true, true);
            }

            output_stream << colorize_rule(make_rule_line());

            auto emit_row = [&](const std::string &scope,
                                const std::string &calls,
                                const std::string &total,
                                const std::string &average,
                                const std::string &percent) {
                std::string row;
                row.reserve(128 + scope.size());

                row += pad_cell(scope, name_column_width);
                row += "   " + pad_cell(calls, calls_column_width);
                row += "   " + pad_cell(total, total_column_width);
                row += "   " + pad_cell(average, average_column_width);
                row += "   " + pad_cell(percent, percent_column_width);

                bool is_even_row = (row_number++ % 2 == 0);
                output_stream << colorize_row(row, false, is_even_row);
            };

            emit_row("TOTAL", "-", format_milliseconds(total_.total_ms), "-", format_percent(100.0));

            for (const auto &group_entry: sorted_groups) {
                const std::string &group_name = group_entry.first;
                const GroupProfile *group_profile = group_entry.second;

                std::vector<std::pair<std::string, const FunctionProfile *> > sorted_functions;
                sorted_functions.reserve(group_profile->functions.size());
                for (const auto &function_entry: group_profile->functions) {
                    sorted_functions.emplace_back(function_entry.first, &function_entry.second);
                }

                std::sort(
                    sorted_functions.begin(),
                    sorted_functions.end(),
                    [](const auto &left, const auto &right) {
                        return left.second->aggregate.total_ms > right.second->aggregate.total_ms;
                    }
                );

                if (max_functions_per_group >= 0 &&
                    static_cast<int>(sorted_functions.size()) > max_functions_per_group) {
                    sorted_functions.resize(static_cast<size_t>(max_functions_per_group));
                }

                emit_row(
                    "+-- [G] " + group_name,
                    "-",
                    format_milliseconds(group_profile->aggregate.total_ms),
                    "-",
                    format_percent(percent_of_total(group_profile->aggregate.total_ms))
                );

                for (const auto &function_entry: sorted_functions) {
                    const std::string &function_name = function_entry.first;
                    const FunctionProfile *function_profile = function_entry.second;

                    std::vector<std::pair<std::string, const TimingStats *> > sorted_kernels;
                    sorted_kernels.reserve(function_profile->kernels.size());
                    for (const auto &kernel_entry: function_profile->kernels) {
                        sorted_kernels.emplace_back(kernel_entry.first, &kernel_entry.second);
                    }

                    std::sort(
                        sorted_kernels.begin(),
                        sorted_kernels.end(),
                        [](const auto &left, const auto &right) {
                            return left.second->total_ms > right.second->total_ms;
                        }
                    );

                    if (max_kernels_per_function >= 0 &&
                        static_cast<int>(sorted_kernels.size()) > max_kernels_per_function) {
                        sorted_kernels.resize(static_cast<size_t>(max_kernels_per_function));
                    }

                    emit_row(
                        "|   +-- [F] " + function_name,
                        "-",
                        format_milliseconds(function_profile->aggregate.total_ms),
                        "-",
                        format_percent(percent_of_total(function_profile->aggregate.total_ms))
                    );

                    for (const auto &kernel_entry: sorted_kernels) {
                        const std::string &kernel_name = kernel_entry.first;
                        const TimingStats *kernel_stats = kernel_entry.second;

                        emit_row(
                            "|   |   +-- [K] " + kernel_name,
                            std::to_string(kernel_stats->calls),
                            format_milliseconds(kernel_stats->total_ms),
                            format_milliseconds(kernel_stats->average_ms()),
                            format_percent(percent_of_total(kernel_stats->total_ms))
                        );
                    }
                }
            }

            output_stream << colorize_rule(make_rule_line());
#endif
        }
        
        // Ensure old name print_table works if called
        void print_table(std::ostream &output_stream = std::cout,
                         int max_functions_per_group = -1,
                         int max_kernels_per_function = -1,
                         int name_column_width = 48,
                         bool force_color = false,
                         bool use_basic_colors = false) const {
            print_table_ascii_colored(output_stream, max_functions_per_group, max_kernels_per_function, name_column_width, force_color, use_basic_colors);
        }

    private:
        Profiler() = default;

        std::unordered_map<std::string, GroupProfile> get_hierarchy() const {
            std::unordered_map<std::string, GroupProfile> groups;
            for (const auto &entry: stats_) {
                const ProfileKey &key = entry.first;
                const TimingStats &stats = entry.second;

                GroupProfile &gp = groups[std::string(key.group)];
                FunctionProfile &fp = gp.functions[std::string(key.function)];
                fp.kernels[std::string(key.kernel)] = stats;

                fp.aggregate.total_ms += stats.total_ms;
                fp.aggregate.calls += stats.calls;
                gp.aggregate.total_ms += stats.total_ms;
                gp.aggregate.calls += stats.calls;
            }
            return groups;
        }

        mutable std::shared_mutex mutex_;
        std::unordered_map<ProfileKey, TimingStats, ProfileKeyHash> stats_;
        TimingStats total_;
    };

    struct ScopedTimer;
    inline thread_local ScopedTimer *current_active_timer = nullptr;

    struct ScopedTimer {
#if ENABLE_PROFILER
        const char *group = nullptr;
        const char *function = nullptr;
        const char *kernel = nullptr;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        bool stopped = false;
        double child_ms = 0.0;
        ScopedTimer *parent = nullptr;
#endif

        ScopedTimer(const char *group_name, const char *function_name, const char *kernel_name) {
#if ENABLE_PROFILER
            group = group_name;
            function = function_name;
            kernel = kernel_name;
            // Assuming util.h has get_time_point() or we use steady_clock::now()
            start_time = std::chrono::steady_clock::now();
            stopped = false;
            child_ms = 0.0;
            parent = current_active_timer;
            current_active_timer = this;
#else
            (void) group_name;
            (void) function_name;
            (void) kernel_name;
#endif
        }

        void stop() noexcept {
#if ENABLE_PROFILER
            if (!stopped) {
                // Assuming util.h has get_milli_seconds(start, end) or we compute it
                const auto end_time = std::chrono::steady_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                double exclusive_ms = elapsed_ms - child_ms;
                Profiler::instance().add(group, function, kernel, exclusive_ms);

                if (parent) {
                    parent->child_ms += elapsed_ms;
                }
                current_active_timer = parent;
                stopped = true;
            }
#endif
        }

        ~ScopedTimer() {
            stop();
        }
    };
} // namespace GPU_HeiPa

#endif //GPU_HEIPA_PROFILER_H