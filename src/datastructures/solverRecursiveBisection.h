#ifndef GPU_HEIPA_SOLVER_REC_BISEC_H
#define GPU_HEIPA_SOLVER_REC_BISEC_H

#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>

#include "graph.h"
#include "host_graph.h"
#include "kokkos_memory_stack.h"
#include "mapping.h"
#include "partition.h"
#include "../coarsening/two_hop_matching.h"
#include "../refinement/jet_label_propagation.h"
#include "../initial_partitioning/metis_partitioning.h"
#include "../utility/definitions.h"
#include "../utility/configuration.h"
#include "../utility/profiler.h"
#include "../utility/asserts.h"
#include "../utility/edge_cut.h"
#include "solver.h"


namespace GPU_HeiPa {

    class SolverRecursiveBisection {


        public:
            Configuration config; 
            //? what does this hold?
            //? -> basically all the config info that you can supply to the algo upon calling the executable

            // constructor:
            explicit SolverRecursiveBisection(Configuration t_config) : config(std::move(t_config)) {
        }

            HostPartition solve(HostGraph &host_g) {
                
                //other stuff...
                
                internal_solve(host_g);
            
                //other stuff...

                HostPartition host_partition; 
                //? how does this look like? what do i actually need to return?
                //? -> is just a an array of u32 (but defined as a kokkos view)

                // retrieve the actual host partition from the solver

                return host_partition;
            }


        private:

            void internal_solve(HostGraph &host_g) {
                // do recursive bisection on the inputgraph

                //Step 1: Just partition once:
                
                Configuration conf1; // use settings of original config but k != 2

                HostPartition first_partition = Solver(conf1).solve(host_g);
            
                // then i can go over the partition and check where the vertices lie!
                // from thus i can create two subgraphs!

            }
        
        
    };

}


#endif