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
    //
    {
        ScopedTimer _t("io", "main", "Kokkos::initialize");
        Kokkos::initialize();
    }

    ProMapConfiguration config;
    if (argc == 1) {
        config.print_help_message();
        Kokkos::finalize();
        return 0;
        //
        {
            ScopedTimer _t("io", "main", "parse_args");
            std::vector<std::pair<std::string, std::string> > input = {
                // {"--graph", "../../ProMapRepo/data/mapping/rgg23.graph"}, // comm cost 9543754, 1098 ms
                // {"--graph", "../../ProMapRepo/data/mapping/shipsec5.mtx.graph"},     // 1.778114 s
                // {"--graph", "../../ProMapRepo/data/mapping/2cubes_sphere.mtx.graph"},
                // {"--graph", "../../ProMapRepo/data/mapping/bmwcra_1.mtx.graph"}, // 5.71 s
                // {"--graph", "../../ProMapRepo/data/mapping/europe_osm.graph"},
                {"--graph", "../../ProMapRepo/data/mapping/cop20k_A.mtx.graph"},
                {"--hierarchy", "4:8:6"},
                {"--distance", "1:10:100"},
                {"--imbalance", "0.03"},
                {"--config", "IM"},
                {"--seed", "1"},
                {"--distance-oracle", "matrix"},
                {"--verbose-level", "1"}
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

            config = ProMapConfiguration(argc_temp, argv_temp);

            for (int i = 0; i < argc_temp; ++i) { delete[] argv_temp[i]; }
            delete[] argv_temp;
        }
    } else {
        ScopedTimer _t_parse("io", "main", "parse_args");
        config = ProMapConfiguration(argc, argv);
    }
    verbose_level = config.verbose_level;

    auto t_before_dtors = get_time_point();
    //
    {
        HostGraph host_g = from_file(config.graph_in);
        HostPartition host_partition;

        f64 io_ms = get_milli_seconds(sp, get_time_point());

        if (verbose_level >= 1) {
            std::cout << "Read graph in     : " << io_ms << std::endl;
        }

        auto sp_solver = get_time_point();
        if (config.config == "IM") {
            if (config.distance_oracle_string == "matrix") {
                host_partition = ProMapSolver<DistanceOracleMatrix>(config).solve(host_g);
            } else if (config.distance_oracle_string == "binary") {
                host_partition = ProMapSolver<DistanceOracleBinary>(config).solve(host_g);
            } else {
                std::cerr << "Error: Invalid distance oracle string: " << config.distance_oracle_string << std::endl;
            }
        } else if (config.config == "HM" || config.config == "HM-ultra") {
            host_partition = ProMapSolver<DistanceOracleMatrix>(config).solve_multisection(host_g);
        } else {
            std::cerr << "Error: Invalid config: " << config.config << std::endl;
        }

        if (verbose_level >= 1) {
            std::cout << "Solved in         : " << get_milli_seconds(sp_solver, get_time_point()) << std::endl;
        }

        if (config.is_set("--mapping")) {
            ScopedTimer _t("io", "main", "write_partition");
            auto p = get_time_point();

            write_partition(host_partition, host_g.n, config.mapping_out);

            if (verbose_level >= 1) {
                io_ms = get_milli_seconds(p, get_time_point());
                std::cout << "Write partition in: " << io_ms << std::endl;
            }
        }

        t_before_dtors = get_time_point();
    }
    if (verbose_level >= 1) {
        std::cout << "Destructed in     : " << get_milli_seconds(t_before_dtors, get_time_point()) << std::endl;
    }

    //
    {
        ScopedTimer _t("io", "main", "Kokkos::finalize");
        Kokkos::finalize();
    }

    if (verbose_level >= 1) {
        Profiler::instance().print_table(std::cout);
    }

    auto ep = get_time_point();
    if (verbose_level >= 1) {
        std::cout << "Total Time spent in GPU-HeiProMap.cpp: " << get_seconds(sp, ep) << " seconds." << std::endl;
    }

    return 0;
}
