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

#ifndef GPU_HEIPA_PROMAP_CONFIGURATION_H
#define GPU_HEIPA_PROMAP_CONFIGURATION_H

#include <string>
#include <vector>

#include "configuration.h"
#include "util.h"

namespace GPU_HeiPa {
    class ProMapConfiguration {
        std::vector<CommandLineOption> options = {
            {"--help", "", "Produces the help message", "", "", false},
            {"--graph", "-g", "Filepath to the graph.", "", "", false},
            {"--mapping", "-m", "Output filepath to the generated mapping.", "GPU-HeiProMap_par.txt", "", false},
            {"--hierarchy", "-h", "Hierarchy in the form a1:a2:...:al .", "", "", false},
            {"--distance", "-d", "Distance in the form d1:d2:...:dl .", "", "", false},
            {"--imbalance", "-e", "Allowed imbalance (for example 0.03).", "0.03", "", false},
            {"--config", "-c", "Algorithm Config {IM, HM, HM-ultra}.", "", "", false},
            {"--verbose-level", "", "Whether to print.", "1", "", false},
        };

    public:
        // graph information
        std::string graph_in;
        std::string mapping_out;

        // partition information
        std::string hierarchy_string;
        std::vector<partition_t> hierarchy;
        partition_t k = 0;

        // distance information
        std::string distance_string;
        std::vector<weight_t> distance;

        // balancing information
        f64 imbalance = 0.0;

        // partitioning algorithm
        std::string config;

        // random initialization
        u64 seed = 0;

        int verbose_level = 1;

        // distance oracle
        std::string distance_oracle_string = "matrix";

        // device space info
        std::string device_space;

        ProMapConfiguration() = default;

        ProMapConfiguration(int argc, char *argv[]) {
            // read command lines into vector
            std::vector<std::string> args(argv, argv + argc);

            // check for a help message
            for (size_t i = 1; i < (size_t) argc; ++i) {
                if (args[i] == "--help") {
                    print_help_message();
                    exit(EXIT_SUCCESS);
                }
            }

            // read all command line args
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

            // extract info
            graph_in = get("--graph");
            mapping_out = get("--mapping");

            hierarchy_string = get("--hierarchy");
            hierarchy = convert<partition_t>(split(hierarchy_string, ':'));
            k = prod<partition_t>(hierarchy);

            distance_string = get("--distance");
            distance = convert<weight_t>(split(distance_string, ':'));

            imbalance = std::stod(get("--imbalance"));

            // partitioning algorithm
            config = get("--config");

            // random initialization
            seed = std::random_device{}();

            if (is_set("--verbose-level")) {
                verbose_level = std::stoi(get("--verbose-level"));
            }

            // distance_oracle_string = get("--distance-oracle");

            device_space = get_kokkos_execution_space_as_str();
        }

        /**
         * Returns whether the option was entered.
         *
         * @param var The option in interest.
         * @return True if the option was entered, false else.
         */
        bool is_set(const std::string &var) {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (large_key == var || small_key == var) {
                    return is_set;
                }
            }
            std::cout << "Command Line \"" << var << "\" is not an allowed name!" << std::endl;
            exit(EXIT_FAILURE);
        }

        /**
         * Gets the entered input as a string.
         *
         * @param var The option in interest.
         * @return The input.
         */
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

        /**
         * Prints the help message.
         */
        void print_help_message() {
            for (const auto &[large_key, small_key, description, default_val, input, is_set]: options) {
                if (small_key.empty()) {
                    std::cout << "[ " << large_key << "] - " << description << std::endl;
                } else {
                    std::cout << "[ " << large_key << ", " << small_key << "] - " << description << std::endl;
                }
            }
        }

        /**
         * Converts algorithm configuration into a string in JSON format.
         *
         * @param n_tabs Number of tabs appended in front of each line (for visual purposes).
         * @return String in JSON format.
         */
        std::string to_JSON(const int n_tabs = 0) const {
            std::string tabs;
            for (int i = 0; i < n_tabs; ++i) { tabs.push_back('\t'); }

            std::string s = "{\n";

            s += tabs + to_JSON_MACRO(graph_in);
            s += tabs + to_JSON_MACRO(mapping_out);
            s += tabs + to_JSON_MACRO(k);
            s += tabs + to_JSON_MACRO(imbalance);
            s += tabs + to_JSON_MACRO(config);
            s += tabs + to_JSON_MACRO(seed);
            s += tabs + to_JSON_MACRO(device_space);

            s.pop_back();
            s.pop_back();
            s += "\n" + tabs + "}";
            return s;
        }
    };
}

#endif //GPU_HEIPA_PROMAP_CONFIGURATION_H
