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
#include <cstring>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#include "macros.h"
#include "util.h"

namespace GPU_HeiPa {
    namespace detail {
        // ---------- tiny helpers ----------
        inline std::string pad_cell(std::string s, size_t w) {
            if (w == 0) return {};
            if (s.size() <= w) {
                s.append(w - s.size(), ' ');
                return s;
            }
            if (w == 1) return s.substr(0, 1);
            return s.substr(0, w - 1) + "…";
        }

        inline std::string fmt_fixed(double x, int prec) {
            std::ostringstream os;
            os.setf(std::ios::fixed);
            os << std::setprecision(prec) << x;
            return os.str();
        }

        inline double pct(double part, double whole) {
            return (whole > 0.0) ? (part * 100.0 / whole) : 0.0;
        }

        template<class MapT, class GetScoreFn>
        auto sorted_items(const MapT &m, GetScoreFn score_desc) {
            using K = typename MapT::key_type;
            using V = typename MapT::mapped_type;
            std::vector<std::pair<const K *, const V *> > out;
            out.reserve(m.size());
            for (auto &kv: m) out.push_back({&kv.first, &kv.second});
            std::sort(out.begin(), out.end(),
                      [&](auto a, auto b) { return score_desc(*a.second) > score_desc(*b.second); });
            return out;
        }
    } // namespace detail

    // ---------- ANSI theming ----------
    struct ZebraTheme {
        const char *even_bg = "\x1b[48;5;236m";
        const char *odd_bg = "\x1b[48;5;235m";
        const char *header_bg = "\x1b[48;5;238m";
        const char *rule_fg = "\x1b[38;5;240m";
        const char *text_fg = "\x1b[38;5;252m";
        const char *bold_on = "\x1b[1m";
        const char *bold_off = "\x1b[22m";
        const char *reset = "\x1b[0m";
    };

    inline ZebraTheme basic_theme() {
        return ZebraTheme{
            "\x1b[47m", // even
            "\x1b[107m", // odd
            "\x1b[47m", // header
            "\x1b[90m", // rule
            "\x1b[30m", // text
            "\x1b[1m",
            "\x1b[22m",
            "\x1b[0m"
        };
    }

    class AnsiOut {
    public:
        AnsiOut(bool enabled, ZebraTheme theme) : enabled_(enabled), t_(theme) {
        }

        void rule(std::ostream &os, const std::string &s) const {
            if (!enabled_) {
                os << s << "\n";
                return;
            }
            os << t_.rule_fg << s << t_.reset << "\n";
        }

        void row(std::ostream &os, const std::string &s, bool header, bool even, bool bold = false) const {
            if (!enabled_) {
                os << s << "\n";
                return;
            }
            const char *bg = header ? t_.header_bg : (even ? t_.even_bg : t_.odd_bg);
            os << bg << t_.text_fg;
            if (bold) os << t_.bold_on;
            os << s;
            if (bold) os << t_.bold_off;
            os << t_.reset << "\n";
        }

    private:
        bool enabled_;
        ZebraTheme t_;
    };

    struct KTStat {
        double total_ms = 0.0;
        unsigned long calls = 0;

        inline void add(double ms) {
            total_ms += ms;
            ++calls;
        }

        inline double avg() const { return calls ? total_ms / (double) calls : 0.0; }
    };

    struct FlatKey {
        const char *g;
        const char *f;
        const char *k;
    };

    struct FlatRec {
        FlatKey key;
        KTStat stat;
    };

    class Profiler {
    public:
        static Profiler &instance() {
            static Profiler P;
            return P;
        }

        void add(const char *g, const char *f, const char *k, double ms) {
            #if !ENABLE_PROFILER
            (void) g;
            (void) f;
            (void) k;
            (void) ms;
            return;
            #else
            const uint64_t h = hash_triple_ptr(g, f, k);
            auto it = flat_.find(h);
            if (it == flat_.end()) {
                FlatRec r;
                r.key = {g, f, k};
                r.stat.add(ms);
                flat_.emplace(h, r);
            } else {
                // ultra-rare collision safety (optional but cheap)
                if (it->second.key.g != g || it->second.key.f != f || it->second.key.k != k) {
                    const uint64_t h2 = hash_triple_ptr_salted(g, f, k);
                    auto &r = flat_[h2];
                    if (r.key.g == nullptr) r.key = {g, f, k};
                    r.stat.add(ms);
                } else {
                    it->second.stat.add(ms);
                }
            }
            total_.add(ms);
            #endif
        }

        // Your existing print_table_ascii_colored can now:
        //  - iterate flat_
        //  - build hierarchy in local structures
        //  - sort and print
        void print_table(std::ostream &os = std::cout,
                         int max_funcs_per_group = -1,
                         int max_kernels_per_func = -1,
                         unsigned int name_width = 48,
                         bool force_color = false,
                         bool use_basic_colors = false) const;

    private:
        Profiler() = default;

        static inline uint64_t splitmix64(uint64_t x) {
            x += 0x9e3779b97f4a7c15ull;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            return x ^ (x >> 31);
        }

        static inline uint64_t hash_triple_ptr(const void *a, const void *b, const void *c) {
            uint64_t x = splitmix64((uint64_t) (uintptr_t) a);
            x ^= splitmix64((uint64_t) (uintptr_t) b + 0x9e3779b97f4a7c15ull);
            x ^= splitmix64((uint64_t) (uintptr_t) c + 0xbf58476d1ce4e5b9ull);
            return x;
        }

        static inline uint64_t hash_triple_ptr_salted(const void *a, const void *b, const void *c) {
            // fallback if a collision is detected (practically never)
            uint64_t x = splitmix64((uint64_t) (uintptr_t) a ^ 0xA5A5A5A5A5A5A5A5ull);
            x ^= splitmix64((uint64_t) (uintptr_t) b ^ 0x0123456789ABCDEFull);
            x ^= splitmix64((uint64_t) (uintptr_t) c ^ 0xF0E1D2C3B4A59687ull);
            return x;
        }

        std::unordered_map<uint64_t, FlatRec> flat_;
        KTStat total_;
    };


    struct ScopedTimer {
        #if ENABLE_PROFILER
        using clock = std::chrono::steady_clock;

        const char *group;
        const char *function;
        const char *kernel;
        clock::time_point t0;
        bool stopped = false;

        ScopedTimer(const char *g, const char *f, const char *k)
            : group(g), function(f), kernel(k), t0(clock::now()) {
        }

        void stop() {
            if (stopped) return;
            const auto t1 = clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            Profiler::instance().add(group, function, kernel, ms);
            stopped = true;
        }

        ~ScopedTimer() { stop(); }
        #else
        ScopedTimer(const char *, const char *, const char *) {
        }

        void stop() {
        }
        #endif
    };

    static inline bool streq(const char *a, const char *b) {
        return (a == b) || (a && b && std::strcmp(a, b) == 0);
    }

    static inline void merge_stat(KTStat &dst, const KTStat &src) {
        dst.total_ms += src.total_ms;
        dst.calls += src.calls;
    }

    // local hierarchy nodes (only used during printing)
    struct PrintFunc {
        std::unordered_map<const char *, KTStat> kernels; // kernel name -> stat
        KTStat agg;
    };

    struct PrintGroup {
        std::unordered_map<const char *, PrintFunc> funcs; // function name -> func
        KTStat agg;
    };

    void Profiler::print_table(std::ostream &os,
                               int max_funcs_per_group,
                               int max_kernels_per_func,
                               unsigned int name_width,
                               bool force_color,
                               bool use_basic_colors) const {
        #if !ENABLE_PROFILER
        (void) os;
        (void) max_funcs_per_group;
        (void) max_kernels_per_func;
        (void) name_width;
        (void) force_color;
        (void) use_basic_colors;
        return;
        #endif

        if (total_.total_ms <= 0.0) {
            os << "Profiler: no samples recorded.\n";
            return;
        }

        // clamp
        name_width = std::min(std::max(name_width, 24u), 96u);

        // color gating
        const bool no_color_env = std::getenv("NO_COLOR") != nullptr;
        const bool color_ok = !no_color_env && (force_color || true); // keep your behavior
        AnsiOut out(color_ok, use_basic_colors ? basic_theme() : ZebraTheme{});

        // ---- Build hierarchy from flat_ (print-time only) ----
        std::unordered_map<const char *, PrintGroup> groups;
        groups.reserve(flat_.size());

        for (const auto &kv: flat_) {
            const FlatRec &r = kv.second;
            auto &g = groups[r.key.g];
            auto &f = g.funcs[r.key.f];

            // merge kernel stat
            auto &kstat = f.kernels[r.key.k];
            merge_stat(kstat, r.stat);

            // roll-ups
            merge_stat(f.agg, r.stat);
            merge_stat(g.agg, r.stat);
        }

        // ---- IO-less denominator ----
        double io_ms = 0.0;
        for (const auto &gkv: groups) {
            if (streq(gkv.first, "io")) {
                io_ms = gkv.second.agg.total_ms;
                break;
            }
        }
        const double denom_no_io = std::max(0.0, total_.total_ms - io_ms);

        auto pct_tot = [&](double ms) { return detail::pct(ms, total_.total_ms); };
        auto pct_noio = [&](double ms) { return detail::pct(ms, denom_no_io); };

        // column widths
        constexpr int W_CALL = 8, W_TOT = 12, W_AVG = 10, W_PCT = 7, W_PCT2 = 7;
        const auto rule_len = name_width + 3u + W_CALL + 3 + W_TOT + 3 + W_AVG + 3 + W_PCT + 3 + W_PCT2;
        const std::string rule(rule_len, '-');

        auto make_line = [&](const std::string &scope,
                             const std::string &calls,
                             const std::string &tot,
                             const std::string &avg,
                             const std::string &p1,
                             const std::string &p2) {
            std::string s;
            s.reserve(128 + scope.size());
            s += detail::pad_cell(scope, name_width);
            s += "   " + detail::pad_cell(calls, W_CALL);
            s += "   " + detail::pad_cell(tot, W_TOT);
            s += "   " + detail::pad_cell(avg, W_AVG);
            s += "   " + detail::pad_cell(p1, W_PCT);
            s += "   " + detail::pad_cell(p2, W_PCT2);
            return s;
        };

        size_t row_i = 0;
        auto emit = [&](const std::string &line, bool header = false, bool bold = false) {
            out.row(os, line, header, /*even*/ (row_i++ % 2 == 0), bold);
        };

        // ---- Header ----
        out.rule(os, rule);
        emit(make_line("Scope", "Calls", "Total ms", "Avg ms", "%Tot", "%NoIO"),
             /*header*/true, /*bold*/true);
        out.rule(os, rule);

        // ---- TOTAL ----
        emit(make_line("TOTAL", "-",
                       detail::fmt_fixed(total_.total_ms, 3), "-",
                       detail::fmt_fixed(100.0, 1),
                       detail::fmt_fixed(denom_no_io > 0.0 ? 100.0 : 0.0, 1)));

        // ---- Groups sorted by total desc ----
        auto gs = detail::sorted_items(groups, [](const PrintGroup &g) { return g.agg.total_ms; });

        for (auto [gname_ptr, gptr]: gs) {
            const char *gname = *gname_ptr;
            const bool is_io = streq(gname, "io");

            emit(make_line(std::string("+-- [G] ") + gname, "-",
                           detail::fmt_fixed(gptr->agg.total_ms, 3), "-",
                           detail::fmt_fixed(pct_tot(gptr->agg.total_ms), 1),
                           detail::fmt_fixed(is_io ? 0.0 : pct_noio(gptr->agg.total_ms), 1)));

            // functions sorted
            auto fs = detail::sorted_items(gptr->funcs, [](const PrintFunc &f) { return f.agg.total_ms; });
            if (max_funcs_per_group >= 0 && (int) fs.size() > max_funcs_per_group)
                fs.resize((size_t) max_funcs_per_group);

            for (auto [fname_ptr, fptr]: fs) {
                const char *fname = *fname_ptr;

                emit(make_line(std::string("|   +-- [F] ") + fname, "-",
                               detail::fmt_fixed(fptr->agg.total_ms, 3), "-",
                               detail::fmt_fixed(pct_tot(fptr->agg.total_ms), 1),
                               detail::fmt_fixed(is_io ? 0.0 : pct_noio(fptr->agg.total_ms), 1)));

                // kernels sorted
                auto ks = detail::sorted_items(fptr->kernels, [](const KTStat &k) { return k.total_ms; });
                if (max_kernels_per_func >= 0 && (int) ks.size() > max_kernels_per_func)
                    ks.resize((size_t) max_kernels_per_func);

                for (auto [kname_ptr, kstat]: ks) {
                    const char *kname = *kname_ptr;

                    emit(make_line(std::string("|   |   +-- [K] ") + kname,
                                   std::to_string(kstat->calls),
                                   detail::fmt_fixed(kstat->total_ms, 3),
                                   detail::fmt_fixed(kstat->avg(), 3),
                                   detail::fmt_fixed(pct_tot(kstat->total_ms), 1),
                                   detail::fmt_fixed(is_io ? 0.0 : pct_noio(kstat->total_ms), 1)));
                }
            }
        }

        // ---- Footer ----
        out.rule(os, rule);
        os << "(* %NoIO = share of time if IO group were removed from the total)\n";
    }
} // namespace GPU_HeiPa


#endif //GPU_HEIPA_PROFILER_H
