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

#include <iostream>

#include <Kokkos_Core.hpp>

#include "../src/datastructures/solver.h"
#include "../src/datastructures/solverRecursiveBisection.h"
#include "../src/utility/configuration.h"

using namespace GPU_HeiPa;

int main(int argc, char *argv[]) {
    auto sp = get_time_point();
    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);
    int verbose_level = 1;
    //
    {
        ScopedTimer _t("io", "main", "Kokkos::initialize");
        Kokkos::initialize();
    }

    Configuration config;
    if (argc == 1) {
        // Configuration config;
        // config.print_help_message();
        // return 0;
        {
            ScopedTimer _t("io", "main", "parse_args");
            std::vector<std::pair<std::string, std::string> > input = {
                {"--graph", "./res/graphs/144.graph"},    // 100.054 in 334ms
                // {"--graph", "../../ProMapRepo/data/mapping/cfd2.mtx.graph"}, // 92.920 in 40ms
                {"--k", "32"},
                {"--imbalance", "0.03"},
                {"--config", "default"},
                {"--seed", "1"},
                {"--verbose-level", "2"}
            };

            std::vector<std::string> args = {"GPU-HeiPa"};
            for (const auto &[key, val]: input) {
                args.push_back(key);
                args.push_back(val);
            }

            // Step 3: Prepare argc and argv.
            int argc_temp = (int) args.size();
            if (argc_temp < 0) {
                std::cerr << "Error: Invalid argc size" << std::endl;
                exit(EXIT_FAILURE);
            }

            // Allocate an array of char* for argv.
            char **argv_temp = new char *[(size_t) argc_temp];

            for (size_t i = 0; i < (size_t) argc_temp; ++i) {
                // Allocate enough space for the string plus the null terminator.
                argv_temp[i] = new char[args[i].size() + 1];
                std::strcpy(argv_temp[i], args[i].c_str());
            }

            config = Configuration(argc_temp, argv_temp);

            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        ScopedTimer _t("io", "main", "parse_args");
        config = Configuration(argc, argv);
    }
    verbose_level = config.verbose_level;
    //
    {
        HostGraph host_g = from_file(config.graph_in);

        f64 io_ms = get_milli_seconds(sp, get_time_point());

        if (verbose_level >= 1) {
            std::cout << "Read graph in     : " << io_ms << std::endl;
        }


        auto sp_solver = get_time_point();
        
        //! Only change this line to use the new solver!
        // HostPartition host_partition = Solver(config).solve(host_g);
        HostPartition host_partition = SolverRecursiveBisection(config).solve(host_g);

        if (verbose_level >= 1) {
            std::cout << "Solved in         : " << get_milli_seconds(sp_solver, get_time_point()) << std::endl;
        }

        //if (config.is_set("--mapping")) {
        //    ScopedTimer _t("io", "main", "write_partition");
        //    auto p = get_time_point();
//
        //    write_partition(host_partition, host_g.n, config.mapping_out);
//
        //    if (verbose_level >= 1) {
        //        io_ms = get_milli_seconds(p, get_time_point());
        //        std::cout << "Write partition in: " << io_ms << std::endl;
        //    }
        //}
    }
    Kokkos::fence();

    //
    {
        ScopedTimer _t("io", "main", "Kokkos::finalize");
        Kokkos::finalize();
    }

    if (verbose_level >= 1) {
        Profiler::instance().print_table(std::cout);
    }

    auto ep = get_time_point();
    if (verbose_level >= 2) {
        std::cout << "Total Time in GPU-HeiPa.cpp : " << get_seconds(sp, ep) << " seconds." << std::endl;
    }

    return 0;
}
