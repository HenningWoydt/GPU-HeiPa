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
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#include "macros.h"
#include "util.h"

namespace GPU_HeiPa {
    // ===== Helpers (keep with your class or in an anon namespace) =====
    static std::string pad_cell(const std::string &s, unsigned int w) {
        if (w <= 0) return "";
        if (s.size() <= w) return s + std::string(w - s.size(), ' ');
        if (w <= 1) return s.substr(0, w);
        return s.substr(0, w - 1) + "…";
    }

    struct ZebraTheme {
        // Use 256-color backgrounds by default; fall back to simple if needed
        // You can tweak these. 236/235 are subtle dark grays; 22 is a dark green.
        const char *even_bg = "\x1b[48;5;236m";
        const char *odd_bg = "\x1b[48;5;235m";
        const char *header_bg = "\x1b[48;5;238m";
        const char *rule_fg = "\x1b[38;5;240m";
        const char *text_fg = "\x1b[38;5;252m";
        const char *bold_on = "\x1b[1m";
        const char *bold_off = "\x1b[22m";
        const char *reset = "\x1b[0m";
    };

    // Simpler theme (basic 8 colors) if 256-color looks bad
    static ZebraTheme basic_theme() {
        return ZebraTheme{
            /*even_bg*/ "\x1b[47m", // white bg
            /*odd_bg*/ "\x1b[107m", // bright white bg
            /*header_bg*/"\x1b[47m",
            /*rule_fg*/ "\x1b[90m", // bright black (gray)
            /*text_fg*/ "\x1b[30m", // black text
            /*bold_on*/ "\x1b[1m",
            /*bold_off*/ "\x1b[22m",
            /*reset*/ "\x1b[0m"
        };
    }

    // ---------- simple stats -----------
    struct KTStat {
        double total_ms = 0.0;
        unsigned long calls = 0;

        inline void add(double ms) {
            total_ms += ms;
            ++calls;
        }

        inline double avg() const { return calls ? total_ms / (f64) calls : 0.0; }
    };

    struct KTKernels {
        std::unordered_map<std::string, KTStat> kernels; // kernel name -> stat
        KTStat agg;                                      // aggregate of kernels
    };

    struct KTGroup {
        std::unordered_map<std::string, KTKernels> functions; // function name -> kernels
        KTStat agg;                                           // aggregate of functions
    };

    // ========== Profiler (hierarchical) ==========
    class Profiler {
    public:
        static Profiler &instance() {
            static Profiler R;
            return R;
        }

        // Add a timing sample
        void add(const char *group, const char *function, const char *kernel, double ms) {
            auto &g = groups_[group];
            auto &f = g.functions[function];
            f.kernels[kernel].add(ms);
            f.agg.add(ms);
            g.agg.add(ms);
            total_.add(ms);
        }

        // ===== Colored printer =====
        void print_table_ascii_colored(std::ostream &os = std::cout,
                                       int max_funcs_per_group = -1,
                                       int max_kernels_per_func = -1,
                                       unsigned int name_width = 48,
                                       bool force_color = false,
                                       bool use_basic_colors = false) const {
            #if !ENABLE_PROFILER
            os << "Profiler disabled (ENABLE_PROFILER=0).\n";
            return;
            #endif
            if (total_.total_ms <= 0.0) {
                os << "Profiler: no samples recorded.\n";
                return;
            }

            // Color gating (unchanged) ...
            const bool no_color_env = std::getenv("NO_COLOR") != nullptr;
            bool color_ok = !no_color_env && (force_color || true);

            ZebraTheme theme = use_basic_colors ? basic_theme() : ZebraTheme{};
            auto apply_bg = [&](const std::string &s, bool header, bool even)-> std::string {
                if (!color_ok) return s + '\n';
                const char *bg = header ? theme.header_bg : (even ? theme.even_bg : theme.odd_bg);
                return std::string(bg) + theme.text_fg + s + theme.reset + '\n';
            };
            auto apply_rule = [&](const std::string &s)-> std::string {
                if (!color_ok) return s + '\n';
                return std::string(theme.rule_fg) + s + theme.reset + '\n';
            };

            // Clamp name column
            if (name_width < 24) name_width = 24;
            if (name_width > 96) name_width = 96;

            auto fmt_ms = [](double x) {
                std::ostringstream s;
                s.setf(std::ios::fixed);
                s << std::setprecision(3) << x;
                return s.str();
            };
            auto fmt_pct = [](double x) {
                std::ostringstream s;
                s.setf(std::ios::fixed);
                s << std::setprecision(1) << x;
                return s.str();
            };
            auto pct_of_total = [&](double ms) { return total_.total_ms > 0.0 ? (ms * 100.0 / total_.total_ms) : 0.0; };

            // --- New: IO-less baseline
            double io_ms = 0.0; {
                auto it = groups_.find("io"); // your group is named "io"
                if (it != groups_.end()) io_ms = it->second.agg.total_ms;
            }
            const double denom_no_io = std::max(0.0, total_.total_ms - io_ms);
            auto pct_no_io = [&](double ms) {
                return denom_no_io > 0.0 ? (ms * 100.0 / denom_no_io) : 0.0;
            };

            // Sort groups
            std::vector<std::pair<std::string, KTGroup const *> > gs;
            gs.reserve(groups_.size());
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            const int W_CALL = 8, W_TOT = 12, W_AVG = 10, W_PCT = 7, W_PCT2 = 7; // new width
            auto make_rule = [&]() {
                return std::string(name_width + 3u + W_CALL + 3 + W_TOT + 3 + W_AVG + 3 + W_PCT + 3 + W_PCT2, '-');
            };

            size_t row_index = 0;

            // Header
            os << apply_rule(make_rule()); {
                std::string hdr;
                if (color_ok) hdr += theme.bold_on;
                hdr += pad_cell("Scope", name_width);
                hdr += "   " + pad_cell("Calls", W_CALL);
                hdr += "   " + pad_cell("Total ms", W_TOT);
                hdr += "   " + pad_cell("Avg ms", W_AVG);
                hdr += "   " + pad_cell("%Tot", W_PCT);
                hdr += "   " + pad_cell("%NoIO", W_PCT2); // new column
                if (color_ok) hdr += theme.bold_off;
                os << apply_bg(hdr, /*header=*/true, /*even=*/true);
            }
            os << apply_rule(make_rule());

            // Row emitter (now takes two pct strings)
            auto emit_row = [&](const std::string &scope, const std::string &calls,
                                const std::string &tot, const std::string &avg,
                                const std::string &pct, const std::string &pct_noio) {
                std::string line;
                line.reserve(128 + scope.size());
                line += pad_cell(scope, name_width);
                line += "   " + pad_cell(calls, W_CALL);
                line += "   " + pad_cell(tot, W_TOT);
                line += "   " + pad_cell(avg, W_AVG);
                line += "   " + pad_cell(pct, W_PCT);
                line += "   " + pad_cell(pct_noio, W_PCT2);
                const bool even = (row_index++ % 2 == 0);
                os << apply_bg(line, /*header=*/false, even);
            };

            // TOTAL row: %NoIO is 100.0 if we have any non-IO time
            emit_row("TOTAL", "-", fmt_ms(total_.total_ms), "-", fmt_pct(100.0),
                     fmt_pct(denom_no_io > 0.0 ? 100.0 : 0.0));

            // Groups
            for (size_t gi = 0; gi < gs.size(); ++gi) {
                const auto &gname = gs[gi].first;
                const auto *gptr = gs[gi].second;
                const bool is_io_group = (gname == "io");

                // Functions sort
                std::vector<std::pair<std::string, KTKernels const *> > fs;
                for (auto &fk: gptr->functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });
                if (max_funcs_per_group >= 0 && (int) fs.size() > max_funcs_per_group)
                    fs.resize((size_t) max_funcs_per_group);

                emit_row("+-- [G] " + gname, "-",
                         fmt_ms(gptr->agg.total_ms), "-",
                         fmt_pct(pct_of_total(gptr->agg.total_ms)),
                         fmt_pct(is_io_group ? 0.0 : pct_no_io(gptr->agg.total_ms)));

                // Functions
                for (size_t fi = 0; fi < fs.size(); ++fi) {
                    const auto &fname = fs[fi].first;
                    const auto *fptr = fs[fi].second;

                    // Kernels sort
                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    for (auto &kk: fptr->kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });
                    if (max_kernels_per_func >= 0 && (int) ks.size() > max_kernels_per_func)
                        ks.resize((size_t) max_kernels_per_func);

                    emit_row("|   +-- [F] " + fname, "-",
                             fmt_ms(fptr->agg.total_ms), "-",
                             fmt_pct(pct_of_total(fptr->agg.total_ms)),
                             fmt_pct(is_io_group ? 0.0 : pct_no_io(fptr->agg.total_ms)));

                    for (size_t ki = 0; ki < ks.size(); ++ki) {
                        const auto &kname = ks[ki].first;
                        const auto *kstat = ks[ki].second;
                        emit_row("|   |   +-- [K] " + kname,
                                 std::to_string(kstat->calls),
                                 fmt_ms(kstat->total_ms),
                                 fmt_ms(kstat->avg()),
                                 fmt_pct(pct_of_total(kstat->total_ms)),
                                 fmt_pct(is_io_group ? 0.0 : pct_no_io(kstat->total_ms)));
                    }
                }
            }

            // Footer
            os << apply_rule(make_rule());

            // Optional legend line (non-colored):
            os << "(* %NoIO = share of time if IO group were removed from the total)\n";
        }


        // --------- Pretty print (sorted by total time desc at each level) ----------
        void print(FILE *out = stderr) const {
            #if !ENABLE_PROFILER
            return;
            #endif
            auto pct = [](double part, double whole) -> double {
                return (whole > 0.0) ? (part * 100.0 / whole) : 0.0;
            };

            std::fprintf(out, "\n=== Kernel Timing (group → function → kernel) ===\n");
            std::fprintf(out, "%-50s %10s %12s %12s %8s %8s\n",
                         "Name", "Calls", "Total(ms)", "Avg(ms)", "%Func", "%Group");

            // Sort groups by total time
            std::vector<std::pair<std::string, KTGroup const *> > gs;
            gs.reserve(groups_.size());
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            for (auto &[gname, gptr]: gs) {
                const auto &g = *gptr;
                // Group header
                std::fprintf(out, "%-50s %10lu %12.3f %12.3f %8s %8.1f\n",
                             gname.c_str(), g.agg.calls, g.agg.total_ms, g.agg.avg(),
                             "", pct(g.agg.total_ms, total_.total_ms));

                // Sort functions by total time
                std::vector<std::pair<std::string, KTKernels const *> > fs;
                fs.reserve(g.functions.size());
                for (auto &fk: g.functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

                for (auto &[fname, fptr]: fs) {
                    const auto &f = *fptr;
                    // Function row (1-tab indent)
                    std::fprintf(out, "    %-46s %10lu %12.3f %12.3f %8s %8.1f\n",
                                 fname.c_str(), f.agg.calls, f.agg.total_ms, f.agg.avg(),
                                 "", pct(f.agg.total_ms, g.agg.total_ms));

                    // Sort kernels by total time
                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    ks.reserve(f.kernels.size());
                    for (auto &kk: f.kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });

                    for (auto &[kname, kstat]: ks) {
                        // Kernel row (2-tab indent)
                        std::fprintf(out, "        %-42s %10lu %12.3f %12.3f %8.1f %8.1f\n",
                                     kname.c_str(), kstat->calls, kstat->total_ms, kstat->avg(),
                                     pct(kstat->total_ms, f.agg.total_ms),
                                     pct(kstat->total_ms, g.agg.total_ms));
                    }
                }
                std::fprintf(out, "\n"); // blank line after each group
            }

            // Total row at end
            std::fprintf(out, "-------------------------------------------------\n");
            std::fprintf(out, "%-50s %10lu %12.3f %12.3f %8s %8s\n",
                         "TOTAL", total_.calls, total_.total_ms, total_.avg(), "", "100.0");
            std::fprintf(out, "=================================================\n");
        }

        // --------- JSON export (nested, pretty-printed) ----------
        std::string to_JSON() const {
            #if !ENABLE_PROFILER
            return "{}";
            #endif

            auto esc = [](const std::string &s) {
                std::ostringstream e;
                for (char c: s) {
                    if (c == '\"' || c == '\\') e << '\\' << c;
                    else if (c == '\n') e << "\\n";
                    else e << c;
                }
                return e.str();
            };

            auto indent = [](std::ostringstream &oss, int level) {
                for (int i = 0; i < level; ++i) oss << '\t'; // tabs for indentation
            };

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(6);

            int lvl = 0;
            oss << "{\n";

            // overall total (only total_ms)
            indent(oss, ++lvl);
            oss << "\"total\": {\n";
            indent(oss, ++lvl);
            oss << "\"total_ms\": " << total_.total_ms << "\n";
            indent(oss, --lvl);
            oss << "},\n";

            // groups (sorted by total time desc)
            indent(oss, lvl);
            oss << "\"groups\": {\n";
            ++lvl;

            std::vector<std::pair<std::string, KTGroup const *> > gs;
            for (auto &kv: groups_) gs.emplace_back(kv.first, &kv.second);
            std::sort(gs.begin(), gs.end(),
                      [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

            bool first_g = true;
            for (auto &[gname, gptr]: gs) {
                if (!first_g) oss << ",\n";
                first_g = false;

                indent(oss, lvl);
                oss << "\"" << esc(gname) << "\": {\n";
                ++lvl;

                // group total
                indent(oss, lvl);
                oss << "\"total_ms\": " << gptr->agg.total_ms << ",\n";

                // functions
                indent(oss, lvl);
                oss << "\"functions\": {\n";
                ++lvl;

                std::vector<std::pair<std::string, KTKernels const *> > fs;
                for (auto &fk: gptr->functions) fs.emplace_back(fk.first, &fk.second);
                std::sort(fs.begin(), fs.end(),
                          [](auto &a, auto &b) { return a.second->agg.total_ms > b.second->agg.total_ms; });

                bool first_f = true;
                for (auto &[fname, fptr]: fs) {
                    if (!first_f) oss << ",\n";
                    first_f = false;

                    indent(oss, lvl);
                    oss << "\"" << esc(fname) << "\": {\n";
                    ++lvl;

                    // function total
                    indent(oss, lvl);
                    oss << "\"total_ms\": " << fptr->agg.total_ms << ",\n";

                    // kernels
                    indent(oss, lvl);
                    oss << "\"kernels\": {\n";
                    ++lvl;

                    std::vector<std::pair<std::string, KTStat const *> > ks;
                    for (auto &kk: fptr->kernels) ks.emplace_back(kk.first, &kk.second);
                    std::sort(ks.begin(), ks.end(),
                              [](auto &a, auto &b) { return a.second->total_ms > b.second->total_ms; });

                    bool first_k = true;
                    for (auto &[kname, kstat]: ks) {
                        if (!first_k) oss << ",\n";
                        first_k = false;

                        indent(oss, lvl);
                        oss << "\"" << esc(kname) << "\": {\n";
                        ++lvl;
                        indent(oss, lvl);
                        oss << "\"calls\": " << kstat->calls << ",\n";
                        indent(oss, lvl);
                        oss << "\"total_ms\": " << kstat->total_ms << ",\n";
                        indent(oss, lvl);
                        oss << "\"avg_ms\": " << kstat->avg() << "\n";
                        --lvl;
                        indent(oss, lvl);
                        oss << "}";
                    }
                    oss << "\n";
                    --lvl;
                    indent(oss, lvl);
                    oss << "}\n"; // end kernels

                    --lvl;
                    indent(oss, lvl);
                    oss << "}";
                }
                oss << "\n";
                --lvl;
                indent(oss, lvl);
                oss << "}\n"; // end functions

                --lvl;
                indent(oss, lvl);
                oss << "}";
            }
            oss << "\n";
            --lvl;
            indent(oss, lvl);
            oss << "}\n"; // end groups

            --lvl;
            oss << "}";
            return oss.str();
        }

    private:
        Profiler() = default;

        std::unordered_map<std::string, KTGroup> groups_;
        KTStat total_;
    };

    struct ScopedTimer {
        #if ENABLE_PROFILER
        const char *group;
        const char *function;
        const char *kernel;
        std::chrono::time_point<std::chrono::system_clock> t0;
        bool stopped = false;
        #endif

        #if ENABLE_PROFILER
        ScopedTimer(const char *g, const char *f, const char *k)
            : group(g), function(f), kernel(k), t0(get_time_point()) {
            // std::cout << group << " " << function << " " << kernel << " " << std::flush;
        }
        #else
        ScopedTimer(const char *g, const char *f, const char *k) {
        }
        #endif

        void stop() {
            #if !ENABLE_PROFILER
            return;
            #else
            if (!stopped) {
                Profiler::instance().add(group, function, kernel, get_milli_seconds(t0, get_time_point()));
                // std::cout << "took " << get_milli_seconds(t0, get_time_point()) << " ms" << std::endl;
                stopped = true;
            }
            #endif
        }

        ~ScopedTimer() { stop(); }
    };
}

#endif //GPU_HEIPA_PROFILER_H
