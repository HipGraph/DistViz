#pragma once
#include "../net/process_3D_grid.hpp"
#include "global_tree_handler.hpp"
#include <algorithm>
#include <cblas.h>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <string>
#include <vector>
#include "common.h"
#include "math_operations.hpp"
#include <memory>

namespace hipgraph::distviz::knng {

template <typename INDEX_TYPE, typename VALUE_TYPE>
class KNNGHandler {
private:
  int ntrees;
  int tree_depth;
  double tree_depth_ratio;
  int data_dimension;
  int starting_data_index;
  Process3DGrid *grid;
  int global_data_set_size;
  int local_data_set_size;
  vector<vector<vector<DataNode<INDEX_TYPE, VALUE_TYPE>>>> trees_leaf_all;
  vector<set<int>> index_distribution;
  int local_tree_offset;
  int total_leaf_size;
  int leafs_per_node;
  int my_leaf_start_index;
  int my_leaf_end_index;
  std::map<int, vector<VALUE_TYPE>> datamap;

  int *receive_random_seeds(int seed) {
    int* receive = new int[grid->col_world_size]();
    if (grid->rank_in_col== 0) {
      for (int i = 0; i < grid->col_world_size; i++)
      {
        std::random_device rd;
        int seed = rd();
        receive[i] = seed;
      }
      MPI_Bcast(receive, grid->col_world_size, MPI_INT, grid->rank_in_col, grid->col_world);
    } else {
      MPI_Bcast(receive, grid->col_world_size, MPI_INT, NULL, grid->col_world);
    }
    return receive;
  }


public:
  KNNGHandler(int ntrees, int tree_depth, double tree_depth_ratio,
        int local_tree_offset, int total_data_set_size, int local_data_set_size,
        int dimension, Process3DGrid* grid) {

    this->data_dimension = dimension;
    this->tree_depth = tree_depth;
    this->global_data_set_size = total_data_set_size;
    this->local_data_set_size = local_data_set_size;
    this->grid = grid;
    this->ntrees = ntrees;
    this->tree_depth_ratio = tree_depth_ratio;

    this->trees_leaf_all = vector<vector<vector<DataNode<INDEX_TYPE,VALUE_TYPE>>>>(ntrees);

    this->index_distribution = vector<set<int>>(grid->col_world_size);

    this->local_tree_offset = local_tree_offset;
  }

  void grow_trees(vector<vector<VALUE_TYPE>>* original_data, float density, bool use_locality_optimization, int nn) {
    unique_ptr<MathOp<VALUE_TYPE>> mathOp_ptr; //class uses for math operations
    VALUE_TYPE* row_data_array = mathOp_ptr.get()->convert_to_row_major_format(original_data); // this algorithm assumes row major format for operations
    int global_tree_depth = this->tree_depth * this->tree_depth_ratio;

    // generate random seed at process 0 and broadcast it to multiple processes.
    int* receive = this->receive_random_seeds(1);

    // build global sparse random project matrix for all trees
    VALUE_TYPE* B = mathOp_ptr.get()->build_sparse_projection_matrix(this->data_dimension,global_tree_depth * this->ntrees, density, receive[0]);

    // get the matrix projection
    // P= X.R
    VALUE_TYPE* P = mathOp_ptr.get()->multiply_mat(row_data_array, B, this->data_dimension,
                                        global_tree_depth * this->ntrees,
                                        this->local_data_set_size,
                                        1.0);

    // creating DRPTGlobal class
    GlobalTreeHandler<INDEX_TYPE,VALUE_TYPE> drpt_global = GlobalTreeHandler<INDEX_TYPE,VALUE_TYPE>(P,
                                                    B,this->grid,
                                                    this->local_data_set_size,
                                                    this->data_dimension,
                                                    global_tree_depth,
                                                    this->ntrees,
                                                    this->global_data_set_size);


    cout << " rank " << grid->rank_in_col << " starting growing trees" << endl;
    // start growing global tree
    drpt_global.grow_global_tree(original_data);
    cout << " rank " <<  grid->rank_in_col << " completing growing trees" << endl;

    //
    //	//calculate locality optimization to improve data locality
    if (use_locality_optimization)
    {
      cout << " rank " << grid->rank_in_col << " starting tree leaf correlation " << endl;
      drpt_global.calculate_tree_leaf_correlation();
      cout << " rank " << grid->rank_in_col << "  tree leaf correlation completed " << endl;
    }

    shared_ptr<DataNode3DVector<INDEX_TYPE,VALUE_TYPE>> leaf_nodes_of_trees_ptr =  make_shared<DataNode3DVector<INDEX_TYPE, VALUE_TYPE>>(ntrees);

    cout << " rank " << grid->rank_in_col << " running  datapoint collection  "<< endl;
    // running the similar datapoint collection
    for (int i = 0; i < ntrees; i++)
    {
      drpt_global.collect_similar_data_points(i, use_locality_optimization,
                                                                       this->index_distribution,this->datamap,&leaf_nodes_of_trees_ptr->at(i));
    }


    // get the global minimum value of a leaf
//    int global_minimum = this->get_global_minimum_leaf_size(leaf_nodes_of_trees);

    //        cout << " rank " << rank << " global_minimum  "<<global_minimum<< endl;

    delete[] receive;
  }
};
} // namespace hipgraph::distviz::knng


