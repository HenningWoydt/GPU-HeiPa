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

#ifndef GPU_HEIPA_MEMETIC_CONFIGURATION_H
#define GPU_HEIPA_MEMETIC_CONFIGURATION_H

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cctype>

#include "definitions.h"
#include "JSON_util.h"
#include "kokkos_util.h"

namespace GPU_HeiPa {
    struct MemeticCommandLineOption {
        std::string large_key;
        std::string small_key;
        std::string description;
        std::string default_val;
        std::string input;
        bool is_set;
    };

    class MemeticConfiguration {
        std::vector<MemeticCommandLineOption> options = {
            {"--help", "", "Produces the help message", "", "", false},
            {"--graph", "-g", "Filepath to the graph.", "", "", false},
            {"--mapping", "-m", "Output filepath to the generated mapping.", "GPU-HeiPa_par.txt", "", false},
            {"--k", "-k", "Number of blocks k", "", "", false},
            {"--imbalance", "-e", "Allowed imbalance (for example 0.03).", "0.03", "", false},
            {"--config", "-c", "Broad Config.", "", "", false},
            {"--distance", "", "Distance computation mode: exact or sampled.", "exact", "", false},
            {"--leftover-strategy", "", "Leftover distribution strategy: random, balanced, gain, mixed.", "mixed", "", false},
            {"--alpha", "", "Alpha parameter for mixed leftover strategy.", "100.0", "", false},
            {"--extent", "", "Extent parameter for backbone crossover in range [1, k].", "1", "", false},
            {"--statistics", "", "Output filepath to the statistics file.", "GPU-HeiPa_stats.JSON", "", false},
            {"--seed", "-s", "Seed for more randomness.", "0", "", false},
            {"--verbose-level", "", "Whether to print.", "2", "", false},
            {"--num-cpu-threads", "", "Number of CPU threads for memetic algorithm.", "4", "", false},
            {"--num-individuals", "", "Population size for memetic algorithm.", "20", "", false},
            {"--population-management", "", "Population management mode: shrinking or steadystate.", "shrinking", "", false},
            {"--reduction-factor", "", "Population shrinking factor used by shrinking population management.", "1", "", false},
            {"--num-crossovers", "", "Number of crossovers per generation.", "1", "", false},
            {"--num-parents", "", "Number of parents for crossover.", "2", "", false},
            {"--tournament-size", "", "Tournament size for selection.", "2", "", false},
            {"--perform-memetic-refinement", "", "Enable memetic refinement (true/false).", "true", "", false},
        };

    public:
        std::string graph_in;
        std::string mapping_out;
        std::string statistics_out;

        partition_t k = 0;
        f64 imbalance = 0.0;

        std::string config;
        std::string distance = "exact";
        std::string leftover_strategy = "mixed";
        f64 alpha = 100.0;
        partition_t extent = 1;

        u64 seed = 0;

        int verbose_level = 1;

        std::string device_space;

        size_t num_cpu_threads = 4;
        size_t num_individuals = 20;
        std::string population_management = "shrinking";
        size_t reduction_factor = 1;
        u32 num_crossovers = 1;
        u32 num_parents = 2;
        u32 tournament_size = 2;
        bool perform_memetic_refinement = true;

        MemeticConfiguration() = default;

        MemeticConfiguration(int argc, char *argv[]) {
            std::vector<std::string> args(argv, argv + argc);

            for (size_t i = 1; i < (size_t) argc; ++i) {
                if (args[i] == "--help") {
                    print_help_message();
                    exit(EXIT_SUCCESS);
                }
            }

            for (size_t i = 1; i < (size_t) argc; ++i) {
                for (auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                    if (large_key == args[i] || small_key == args[i]) {
                        input = args[i + 1];
                        is_set = true;
                        i += 1;
                        break;
                    }
                }
            }

            graph_in = get("--graph");
            mapping_out = get("--mapping");
            statistics_out = get("--statistics");

            k = (partition_t) std::stoul(get("--k"));
            imbalance = std::stod(get("--imbalance"));
            config = get("--config");
            distance = get("--distance");
            leftover_strategy = get("--leftover-strategy");
            alpha = std::stod(get("--alpha"));
            extent = (partition_t) std::stoul(get("--extent"));

            if (is_set("--seed")) {
                seed = std::stoull(get("--seed"));
            } else {
                seed = std::random_device{}();
            }

            verbose_level = 1;
            if (is_set("--verbose-level")) {
                verbose_level = std::stoi(get("--verbose-level"));
            }

            device_space = get_kokkos_execution_space_as_str();

            if (is_set("--num-cpu-threads")) {
                num_cpu_threads = std::stoul(get("--num-cpu-threads"));
            }
            if (is_set("--num-individuals")) {
                num_individuals = std::stoul(get("--num-individuals"));
            }
            if (is_set("--population-management")) {
                population_management = get("--population-management");
            }
            if (is_set("--reduction-factor")) {
                reduction_factor = std::stoul(get("--reduction-factor"));
            }
            if (is_set("--num-crossovers")) {
                num_crossovers = (u32) std::stoul(get("--num-crossovers"));
            }
            if (is_set("--num-parents")) {
                num_parents = (u32) std::stoul(get("--num-parents"));
            }
            if (is_set("--tournament-size")) {
                tournament_size = (u32) std::stoul(get("--tournament-size"));
            }

            {
                std::string refinement = get("--perform-memetic-refinement");
                for (char &c: refinement) {
                    c = (char) std::tolower((unsigned char) c);
                }

                if (refinement == "1" || refinement == "true" || refinement == "yes" || refinement == "on") {
                    perform_memetic_refinement = true;
                } else if (refinement == "0" || refinement == "false" || refinement == "no" || refinement == "off") {
                    perform_memetic_refinement = false;
                } else {
                    std::cerr << "Warning: perform memetic refinement value \"" << refinement
                              << "\" is invalid. Falling back to \"true\"." << std::endl;
                    perform_memetic_refinement = true;
                }
            }

            validate_memetic_parameters();
        }

        bool is_set(const std::string &var) {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (large_key == var || small_key == var) {
                    return is_set;
                }
            }
            std::cout << "Command Line \"" << var << "\" is not an allowed name!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string get(const std::string &var) {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (large_key == var || small_key == var) {
                    if (input.empty() && default_val.empty()) {
                        std::cout << "Command Line \"" << var << "\" not set!" << std::endl;
                        exit(EXIT_FAILURE);
                    } else if (input.empty()) {
                        return default_val;
                    }
                    return input;
                }
            }
            std::cout << "Command Line \"" << var << "\" is not an allowed name!" << std::endl;
            exit(EXIT_FAILURE);
        }

        void print_help_message() {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (small_key.empty()) {
                    std::cout << "[ " << large_key << "] - " << description << std::endl;
                } else {
                    std::cout << "[ " << large_key << ", " << small_key << "] - " << description << std::endl;
                }
            }
        }

        void validate_memetic_parameters() {
            for (char &c: distance) {
                c = (char) std::tolower((unsigned char) c);
            }
            if (distance != "exact" && distance != "sampled") {
                std::cerr << "Warning: distance mode \"" << distance
                          << "\" is invalid. Falling back to \"exact\"." << std::endl;
                distance = "exact";
            }

            for (char &c: leftover_strategy) {
                c = (char) std::tolower((unsigned char) c);
            }

            for (char &c: population_management) {
                c = (char) std::tolower((unsigned char) c);
            }
            if (
                population_management == "steady" ||
                population_management == "steady_state" ||
                population_management == "steady-state"
            ) {
                population_management = "steadystate";
            }
            if (population_management != "shrinking" && population_management != "steadystate") {
                std::cerr << "Warning: population management mode \"" << population_management
                          << "\" is invalid. Falling back to \"shrinking\"." << std::endl;
                population_management = "shrinking";
            }

            if (leftover_strategy == "gain and weight" || leftover_strategy == "gainandweight") {
                leftover_strategy = "mixed";
            } else if (
                leftover_strategy == "favorunderloadedblocks" ||
                leftover_strategy == "favorunderloadedblock"
            ) {
                leftover_strategy = "balanced";
            }

            if (
                leftover_strategy != "random" &&
                leftover_strategy != "balanced" &&
                leftover_strategy != "gain" &&
                leftover_strategy != "mixed"
            ) {
                std::cerr << "Warning: leftover strategy \"" << leftover_strategy
                          << "\" is invalid. Falling back to \"mixed\"." << std::endl;
                leftover_strategy = "mixed";
            }

            if (alpha < 0.0) {
                std::cerr << "Warning: alpha is negative, setting to 0.0." << std::endl;
                alpha = 0.0;
            }

            if (num_cpu_threads == 0) {
                std::cerr << "Warning: num_cpu_threads is 0, setting to 1." << std::endl;
                num_cpu_threads = 1;
            }
            if (num_individuals == 0) {
                std::cerr << "Warning: num_individuals is 0, setting to 1." << std::endl;
                num_individuals = 1;
            }
            if (reduction_factor == 0) {
                std::cerr << "Warning: reduction_factor is 0, setting to 1." << std::endl;
                reduction_factor = 1;
            }
            if (num_crossovers == 0) {
                std::cerr << "Warning: num_crossovers is 0, setting to 1." << std::endl;
                num_crossovers = 1;
            }
            if (num_parents == 0) {
                std::cerr << "Warning: num_parents is 0, setting to 1." << std::endl;
                num_parents = 1;
            }
            if (num_parents > num_individuals) {
                std::cerr << "Warning: num_parents (" << num_parents << ") > num_individuals ("
                          << num_individuals << "), capping num_parents to num_individuals." << std::endl;
                num_parents = (u32) num_individuals;
            }
            if (tournament_size == 0) {
                std::cerr << "Warning: tournament_size is 0, setting to 1." << std::endl;
                tournament_size = 1;
            }
            if (tournament_size > num_individuals) {
                std::cerr << "Warning: tournament_size (" << tournament_size << ") > num_individuals ("
                          << num_individuals << "), capping tournament_size to num_individuals." << std::endl;
                tournament_size = (u32) num_individuals;
            }

            if (extent < 1) {
                std::cerr << "Warning: extent is < 1, setting to 1." << std::endl;
                extent = 1;
            }
            if (k > 0 && extent > k) {
                std::cerr << "Warning: extent (" << extent << ") > k (" << k
                          << "), capping extent to k." << std::endl;
                extent = k;
            }
        }

        std::string to_JSON(const int n_tabs = 0) const {
            std::string tabs;
            for (int i = 0; i < n_tabs; ++i) { tabs.push_back('\t'); }

            std::string s = "{\n";

            s += tabs + to_JSON_MACRO(graph_in);
            s += tabs + to_JSON_MACRO(mapping_out);
            s += tabs + to_JSON_MACRO(statistics_out);
            s += tabs + to_JSON_MACRO(k);
            s += tabs + to_JSON_MACRO(imbalance);
            s += tabs + to_JSON_MACRO(config);
            s += tabs + to_JSON_MACRO(distance);
            s += tabs + to_JSON_MACRO(leftover_strategy);
            s += tabs + to_JSON_MACRO(alpha);
            s += tabs + to_JSON_MACRO(extent);
            s += tabs + to_JSON_MACRO(seed);
            s += tabs + to_JSON_MACRO(device_space);
            s += tabs + to_JSON_MACRO(num_cpu_threads);
            s += tabs + to_JSON_MACRO(num_individuals);
            s += tabs + to_JSON_MACRO(population_management);
            s += tabs + to_JSON_MACRO(reduction_factor);
            s += tabs + to_JSON_MACRO(num_crossovers);
            s += tabs + to_JSON_MACRO(num_parents);
            s += tabs + to_JSON_MACRO(tournament_size);
            s += tabs + to_JSON_MACRO(perform_memetic_refinement);

            s.pop_back();
            s.pop_back();
            s += "\n" + tabs + "}";
            return s;
        }
    };
}

#endif //GPU_HEIPA_MEMETIC_CONFIGURATION_H
