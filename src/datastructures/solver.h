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

#ifndef GPU_HEIPA_SOLVER_H
#define GPU_HEIPA_SOLVER_H

#include <vector>

#include "graph.h"
#include "host_graph.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"

namespace GPU_HeiPa {
    class Solver {
    public:
        Configuration config;
        weight_t lmax = 0;

        HostGraph host_g;

        std::vector<Graph> graphs;

        explicit Solver(Configuration t_config) : config(std::move(t_config)) {
        }

        std::vector<partition_t> solve() {
            load_graph();

            std::string config_JSON = config.to_JSON();
            std::string profile_JSON = Profiler::instance().to_JSON();

            // Combine manually into a single JSON string
            std::string combined_JSON = "{\n";
            combined_JSON += "  \"config\": " + config_JSON + ",\n";
            combined_JSON += "  \"profile\": " + profile_JSON + "\n";
            combined_JSON += "}";

            std::cout << combined_JSON << std::endl;

            return {};
        }

    private:
        void load_graph() {
            host_g = from_file(config.graph_in);
            graphs.emplace_back(from_HostGraph(host_g));
        }
    };
}

#endif //GPU_HEIPA_SOLVER_H
