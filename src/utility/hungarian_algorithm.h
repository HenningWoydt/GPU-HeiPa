#ifndef GPU_HEIPA_HUNGARIAN_ALGORITHM_H
#define GPU_HEIPA_HUNGARIAN_ALGORITHM_H

#include <vector>
#include <algorithm>
#include <limits>
#include <queue>
#include "definitions.h"

namespace GPU_HeiPa {

/*
    This was claudecoded, because i didnt find a library which implements it the way i want
    + i want to minimize dependencies
*/
class HungarianAlgorithm {
public:
    /**
     * Solves the maximum weight bipartite matching problem
     * 
     * @param weight_matrix: weight[i][j] is the weight of assigning row i to column j
     *                       stored as a flattened array in row-major order
     * @param n: dimension of the square matrix (n x n)
     * @return: maximum total weight of the optimal matching
     */
    static u32 solve(const u32* weight_matrix, u32 n) {
        if (n == 0) return 0;
        if (n == 1) return weight_matrix[0];

        std::vector<u32> matching(n);
        u32 total = solve_with_matching(weight_matrix, n, matching);
        
        return total;
    }

    /**
     * Solves maximum weight matching and returns the actual assignment
     * 
     * @param weight_matrix: weight[i][j] is the weight of assigning row i to column j
     * @param n: dimension of the square matrix
     * @param matching: output array where matching[i] = j means row i is matched to column j
     * @return: maximum total weight
     */
    static u32 solve_with_matching(
        const u32* weight_matrix, 
        u32 n, 
        std::vector<u32>& matching
    ) {
        if (n == 0) return 0;
        if (n == 1) {
            matching.assign(1, 0);
            return weight_matrix[0];
        }

        // Create cost matrix: cost[i][j] = max_val - weight[i][j]
        // This way, minimizing cost = maximizing weight
        s64 max_val = 0;
        for (u32 i = 0; i < n * n; ++i) {
            if (weight_matrix[i] > max_val) max_val = weight_matrix[i];
            max_val = std::max(max_val, static_cast<s64>(weight_matrix[i]));
        }

        std::vector<s64> cost(n * n);
        for (u32 i = 0; i < n * n; ++i) {
            cost[i] = max_val + 1 - static_cast<s64>(weight_matrix[i]);
        }

        // Hungarian algorithm implementation
        std::vector<s64> u(n + 1, 0), v(n + 1, 0);
        std::vector<u32> p(n + 1, 0), way(n + 1, 0);

        for (u32 i = 1; i <= n; ++i) {
            p[0] = i;
            u32 j0 = 0;  // dummy column
            std::vector<s64> minv(n + 1, std::numeric_limits<s64>::max());
            std::vector<bool> used(n + 1, false);

            do {
                used[j0] = true;
                u32 i0 = p[j0];
                s64 delta = std::numeric_limits<s64>::max();
                u32 j1 = 0;

                for (u32 j = 1; j <= n; ++j) {
                    if (!used[j]) {
                        s64 cur = cost[index_2d(i0 - 1, j - 1, n)] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (u32 j = 0; j <= n; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            // Augment
            do {
                u32 j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }

        // Extract matching
        matching.assign(n, 0);
        for (u32 j = 1; j <= n; ++j) {
            if (p[j] != 0) {
                matching[p[j] - 1] = j - 1;
            }
        }

        u32 total = 0;
        for (u32 i = 0; i < n; ++i) {
            total += weight_matrix[i * n + matching[i]];
        }

        return total;
    }

private:
    static inline u32 index_2d(u32 i, u32 j, u32 n) {
        return i * n + j;
    }
};

} 

#endif // GPU_HEIPA_HUNGARIAN_ALGORITHM_H
