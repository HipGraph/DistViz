/**
 * This class re distribute the nodes based on distribution policy
 */
#pragma once
#include "../common/common.h"
#include "sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>

using namespace std;
using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::net;


namespace hipgraph::distviz::embedding {

class Partitioner {

public:
  virtual int block_owner(int row_block, int col_block) = 0;
};

class GlobalAdjacency1DPartitioner : public Partitioner {

public:
  Process3DGrid *process_3D_grid;

  GlobalAdjacency1DPartitioner(Process3DGrid *process_3D_grid);

  ~GlobalAdjacency1DPartitioner();

  int block_owner(int row_block, int col_block);

  int get_owner_Process(uint64_t row, uint64_t column, uint64_t  proc_row_width,
                        uint64_t  proc_col_width, uint64_t gCols,bool transpose);

  template <typename INDEX_TYPE,typename VALUE_TYPE>
  void partition_data(SpMat<INDEX_TYPE,VALUE_TYPE> *sp_mat) {

    int world_size = process_3D_grid->col_world_size;
    int my_rank = process_3D_grid->rank_in_col;

    Tuple<VALUE_TYPE> *sendbuf = new Tuple<VALUE_TYPE>[sp_mat->coords->size()];

    if (world_size > 1) {
      vector<int> sendcounts(world_size, 0);
      vector<int> recvcounts(world_size, 0);

      vector<int> offsets, bufindices;

      vector<Tuple<VALUE_TYPE>>* coords = sp_mat->coords;



#pragma omp parallel for
      for (int i = 0; i < (*coords).size(); i++) {
        int owner = get_owner_Process((*coords)[i].row, (*coords)[i].col,
                                      sp_mat->proc_row_width,
                                      sp_mat->proc_col_width,
                                      sp_mat->gCols,sp_mat->col_partitioned);
#pragma omp atomic update
        sendcounts[owner]++;
      }
      prefix_sum(sendcounts, offsets);
      bufindices = offsets;

#pragma omp parallel for
      for (int i = 0; i < (*coords).size(); i++) {
        int owner = get_owner_Process((*coords)[i].row, (*coords)[i].col,
                                      sp_mat->proc_row_width,
                                      sp_mat->proc_col_width,
                                      sp_mat->gCols,sp_mat->col_partitioned);

        int idx;
#pragma omp atomic capture
        idx = bufindices[owner]++;

        //        sendbuf[idx].row = transpose ? coords[i].col : coords[i].row;
        //        sendbuf[idx].col = transpose ? coords[i].row : coords[i].col;
        sendbuf[idx].row = (*coords)[i].row;
        sendbuf[idx].col = (*coords)[i].col;
        sendbuf[idx].value = (*coords)[i].value;
      }

      // Broadcast the number of nonzeros that each processor is going to
      // receive
      MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                   process_3D_grid->col_world);

      vector<int> recvoffsets;
      prefix_sum(recvcounts, recvoffsets);

      // Use the sizing information to execute an AlltoAll
      int total_received_coords =
          std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

      (*(sp_mat->coords)).resize(total_received_coords);

      MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(), SPTUPLE,
                    (*(sp_mat->coords)).data(), recvcounts.data(),
                    recvoffsets.data(), SPTUPLE, process_3D_grid->col_world);

      // TODO: Parallelize the sort routine?
      if (sp_mat->transpose){
        std::sort((*(sp_mat->coords)).begin(), (*(sp_mat->coords)).end(),column_major<VALUE_TYPE>);
      } else {
        std::sort((*(sp_mat->coords)).begin(), (*(sp_mat->coords)).end(),row_major<VALUE_TYPE>);
      }
    // This helps to speed up CSR creation
    }
//    __gnu_parallel::sort((sp_mat->coords).begin(), (sp_mat->coords).end(),
//                         column_major<T>);

    delete[] sendbuf;
  }
};

} // namespace distblas::partition
