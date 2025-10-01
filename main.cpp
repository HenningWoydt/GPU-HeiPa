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

#include "src/datastructures/solver.h"
#include "src/utility/configuration.h"

using namespace GPU_HeiPa;

int main(int argc, char *argv[]) {
    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);

    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::fence();

    if (argc == 1) {
        // Configuration config;
        // config.print_help_message();
        // return 0;
        {
            std::vector<std::pair<std::string, std::string> > input = {
                {"--graph", "../../graph_collection/mapping/rgg24.graph"},
                {"--mapping", "../data/out/partition/rgg24.txt"},
                {"--statistics", "../data/out/statistics/rgg24.JSON"},
                {"--k", "32"},
                {"--imbalance", "0.03"},
                {"--config", "IM"},
                {"--seed", "0"},
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

            Configuration config(argc_temp, argv_temp);
            Solver(config).solve();

            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        Configuration config(argc, argv);
        Solver(config).solve();
    }

    return 0;
}