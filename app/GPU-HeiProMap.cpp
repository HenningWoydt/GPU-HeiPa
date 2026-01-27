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

#include "../src/datastructures/promap_solver.h"
#include "../src/utility/promap_configuration.h"

using namespace GPU_HeiPa;

int main(int argc, char *argv[]) {
    auto sp = get_time_point();
    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);
    int verbose_level = 1;

    ScopedTimer _t_guard("io", "main", "Kokkos::initialize");
    Kokkos::initialize();
    _t_guard.stop();

    if (argc == 1) {
        // ProMapConfiguration config;
        // config.print_help_message();
        // return 0;
        //
        {
            ScopedTimer _t_parse("io", "main", "parse_args");
            std::vector<std::pair<std::string, std::string> > input = {
                // {"--graph", "../../graph_collection/mapping/rgg24.graph"},
                // {"--mapping", "../data/out/partition/rgg24.txt"},
                // {"--statistics", "../data/out/statistics/rgg24.JSON"},
                // {"--graph", "../../graph_collection/mapping/rgg23.graph"},
                // {"--mapping", "../data/out/partition/rgg23.txt"},
                // {"--statistics", "../data/out/statistics/rgg23.JSON"},
                {"--graph", "../../ProMapRepo/data/mapping/rgg23.graph"}, // comm cost 9543754, 1098 ms
                {"--mapping", "../data/out/partition/rgg23.txt"},
                {"--statistics", "../data/out/statistics/rgg23.JSON"},
                // {"--graph", "../../graph_collection/mapping/GAP-road.graph"},
                // {"--mapping", "../data/out/partition/GAP-road.txt"},
                // {"--statistics", "../data/out/statistics/GAP-road.JSON"},
                // {"--graph", "../../graph_collection/mapping/2cubes_sphere.mtx.graph"},
                // {"--mapping", "../data/out/partition/2cubes_sphere.mtx.txt"},
                // {"--statistics", "../data/out/statistics/2cubes_sphere.mtx.JSON"},
                // {"--graph", "../../graph_collection/mapping/cop20k_A.mtx.graph"},
                // {"--mapping", "../data/out/partition/cop20k_A.mtx.txt"},
                // {"--statistics", "../data/out/statistics/cop20k_A.mtx.JSON"},
                // {"--graph", "../../graph_collection/mapping/cfd2.mtx.graph"},
                // {"--mapping", "../data/out/partition/cfd2.mtx.txt"},
                // {"--statistics", "../data/out/statistics/cfd2.mtx.JSON"},
                {"--hierarchy", "4:8:6"},
                {"--distance", "1:10:100"},
                {"--imbalance", "0.03"},
                {"--config", "IM"},
                {"--seed", "1"},
                {"--distance-oracle", "matrix"},
                {"--verbose-level", "2"}
            };

            std::vector<std::string> args = {"GPU-HeiProMap"};
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

            ProMapConfiguration config(argc_temp, argv_temp);
            verbose_level = config.verbose_level;
            _t_parse.stop();

            HostGraph host_g = from_file(config.graph_in);
            HostPartition host_partition;

            f64 add_io_ms = get_milli_seconds(sp, get_time_point());
            if (config.distance_oracle_string == "matrix") {
                host_partition = ProMapSolver<DistanceOracleMatrix>(config).solve(host_g, verbose_level, add_io_ms);
            } else if (config.distance_oracle_string == "binary") {
                host_partition = ProMapSolver<DistanceOracleBinary>(config).solve(host_g, verbose_level, add_io_ms);
            } else {
                std::cerr << "Error: Invalid distance oracle string: " << config.distance_oracle_string << std::endl;
            }

            // write_partition(host_partition, host_g.n, config.mapping_out);

            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        ProMapConfiguration config(argc, argv);
        verbose_level = config.verbose_level;

        HostGraph host_g = from_file(config.graph_in);
        HostPartition host_partition;

        f64 add_io_ms = get_milli_seconds(sp, get_time_point());
        if (config.distance_oracle_string == "matrix") {
            host_partition = ProMapSolver<DistanceOracleMatrix>(config).solve(host_g, verbose_level, add_io_ms);
        } else if (config.distance_oracle_string == "binary") {
            host_partition = ProMapSolver<DistanceOracleBinary>(config).solve(host_g, verbose_level, add_io_ms);
        } else {
            std::cerr << "Error: Invalid distance oracle string: " << config.distance_oracle_string << std::endl;
        }

        write_partition(host_partition, host_g.n, config.mapping_out);
    }
    Kokkos::fence();
    //
    {
        ScopedTimer _t("io", "main", "Kokkos::finalize");
        Kokkos::finalize();
    }

    if (verbose_level >= 2) {
        Profiler::instance().print_table_ascii_colored(std::cout);
    }

    auto ep = get_time_point();
    if (verbose_level >= 2) {
        std::cout << "Total Time spent in GPU-HeiProMap.cpp: " << get_seconds(sp, ep) << " seconds." << std::endl;
    }

    return 0;
}
