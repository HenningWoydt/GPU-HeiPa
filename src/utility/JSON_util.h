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

#ifndef GPU_HEIPA_JSON_UTIL_H
#define GPU_HEIPA_JSON_UTIL_H

#include <iomanip>

#include "definitions.h"

namespace GPU_HeiPa {
    #define to_JSON_MACRO(x) (std::string("\"") + (#x) + "\" : " + to_JSON_value(x) + ",\n")

    inline std::string escape_json_string(const std::string &in) {
        std::string out;
        out.reserve(in.size() + 8);
        for (const char c: in) {
            switch (c) {
                case '\"': out += "\\\"";
                    break;
                case '\\': out += "\\\\";
                    break;
                case '\b': out += "\\b";
                    break;
                case '\f': out += "\\f";
                    break;
                case '\n': out += "\\n";
                    break;
                case '\r': out += "\\r";
                    break;
                case '\t': out += "\\t";
                    break;
                default:
                    if (c < 0x20) {
                        // control chars -> \u00XX
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04X", c);
                        out += buf;
                    } else {
                        out += static_cast<char>(c);
                    }
            }
        }
        return out;
    }

    template<class T>
    inline std::enable_if_t<std::is_integral_v<T>, std::string>
    to_JSON_value(T x) {
        return std::to_string(static_cast<long long>(x));
    }

    template<class T>
    inline std::enable_if_t<std::is_floating_point_v<T>, std::string>
    to_JSON_value(T x) {
        if (!std::isfinite(x)) return "null";
        std::ostringstream oss;
        oss << std::setprecision(15) << std::defaultfloat << x;
        return oss.str();
    }

    inline std::string to_JSON_value(const std::string &s) {
        return "\"" + escape_json_string(s) + "\"";
    }

    template<class T>
    inline std::string to_JSON_value(const std::vector<T> &vec) {
        if (vec.empty()) return "[]";
        std::string s;
        s += '[';
        for (size_t i = 0; i < vec.size(); ++i) {
            s += to_JSON_value(vec[i]);
            if (i + 1 < vec.size()) s += ", ";
        }
        s += ']';
        return s;
    }
}

#endif //GPU_HEIPA_JSON_UTIL_H
