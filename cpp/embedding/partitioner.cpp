#include "sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "partitioner.hpp"

using namespace  hipgraph::distviz::embedding;

GlobalAdjacency1DPartitioner::GlobalAdjacency1DPartitioner(Process3DGrid *process_3D_grid){
  this->process_3D_grid =  process_3D_grid;

}

GlobalAdjacency1DPartitioner::~GlobalAdjacency1DPartitioner(){

}

int GlobalAdjacency1DPartitioner::block_owner(int row_block, int col_block) {
  int rowRank = row_block;
  return process_3D_grid->get_global_rank(rowRank, 0, 0);

}

int GlobalAdjacency1DPartitioner::get_owner_Process(uint64_t row, uint64_t column, uint64_t  proc_row_width,
                                                    uint64_t  proc_col_width, uint64_t gCols,bool transpose) {
  if(!transpose) {
    return block_owner(static_cast<int>(row / proc_row_width),  static_cast<int>(column/gCols));
  }
  else {
    return block_owner(static_cast<int>(column / proc_row_width),  static_cast<int>(row/gCols));
  }
}
