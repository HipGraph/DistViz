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
#include "../common/common.h"
#include "math_operations.hpp"
#include <memory>
#include "../lib/Mrpt.h"

using namespace hipgraph::distviz::common;
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
  shared_ptr<vector<set<INDEX_TYPE>>> process_to_index_set_ptr;
  int local_tree_offset;
  int total_leaf_size;
  int leafs_per_node;
  int my_leaf_start_index;
  int my_leaf_end_index;
  std::shared_ptr<std::map<INDEX_TYPE, INDEX_TYPE>> datamap_ptr = std::make_shared<std::map<INDEX_TYPE, INDEX_TYPE>>();
  std::shared_ptr<map<INDEX_TYPE,vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>> local_nn_map_ptr= std::make_shared<map<INDEX_TYPE,vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>>();


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

    this->process_to_index_set_ptr = make_shared<vector<set<INDEX_TYPE>>>(grid->col_world_size);

    this->local_tree_offset = local_tree_offset;
  }

  void build_index(ValueType2DVector<VALUE_TYPE>* original_data, float density,
                   bool use_locality_optimization, int nn, float target_recall) {
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
                                        this->local_data_set_size,1.0);

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

    Eigen::MatrixXf data_matrix =  drpt_global.collect_similar_data_points_of_all_trees(use_locality_optimization,process_to_index_set_ptr.get(),datamap_ptr.get(),local_nn_map_ptr.get(),nn);

    cout<<"rank "<<grid->rank_in_col<<" rows "<<data_matrix.rows()<<" cols "<<data_matrix.cols()<<endl;

    Mrpt mrpt(data_matrix);
    mrpt.grow_autotune(target_recall, nn);

    Eigen::MatrixXi neighbours(data_matrix.cols(),nn);
    Eigen::MatrixXf distances(data_matrix.cols(),nn);

    #pragma omp parallel for schedule (static)
    for(int i=0;i<data_matrix.cols();i++){
      Eigen::VectorXi tempRow(nn);
      Eigen::VectorXf tempDis(nn);
      mrpt.query(data_matrix.col(i), tempRow.data(),tempDis.data());
      neighbours.row(i)=tempRow;
      distances.row(i)=tempDis;
      INDEX_TYPE  global_index =  (*datamap_ptr)[i];
      EdgeNode<INDEX_TYPE,VALUE_TYPE> edge;
      edge.src_index=global_index;
      for(int k=0;k<nn;k++){
       edge.dst_index = (*datamap_ptr)[tempRow[k]];
       edge.distance = tempDis[k];
       (*local_nn_map_ptr)[global_index][k]=edge;
      }
    }

  }

  map<INDEX_TYPE, vector<EdgeNode<INDEX<TYPE,VALUE_TYPE>>>* communicate_nns(map<INDEX_TYPE, vector<EdgeNode<INDEX<TYPE,VALUE_TYPE>>>* local_nns,int nn) {

    int* receiving_indices_count = new int[grid->col_world_size]();
    int* disps_receiving_indices = new int[grid->col_world_size]();
    int send_count = 0;
    int total_receving = 0;

    //send distance threshold to original data owner
    index_distance_pair *out_index_dis = send_min_max_distance_to_data_owner(local_nns,receiving_indices_count,disps_receiving_indices,send_count,total_receving,nn);


    vector<index_distance_pair> final_sent_indices_to_rank_map(local_data_set_size);
    //
    //	//finalize data owners based on data owner having minimum distance threshold.
    this->finalize_final_dataowner(receiving_indices_count,disps_receiving_indices,out_index_dis,final_sent_indices_to_rank_map);
    //
    //	//announce the selected dataowner to all interesting data holders
    vector<vector<index_distance_pair>> final_indices_allocation =  announce_final_dataowner(total_receving,
                                                                                                  receiving_indices_count, disps_receiving_indices,out_index_dis,final_sent_indices_to_rank_map);
    //
    //
    std::map<int, vector<DataPoint>>final_nn_sending_map;
    std::map<int, vector<DataPoint>>final_nn_map;
    //
    int* sending_selected_indices_count = new int[this->world_size]();
    int* sending_selected_indices_nn_count = new int[this->world_size]();
    //
    int* receiving_selected_indices_count = new int[this->world_size]();
    int* receiving_selected_indices_nn_count = new int[this->world_size]();
    //
    //	//select final nns to be forwared to dataowners
    this->select_final_forwarding_nns(final_indices_allocation,
                                      local_nns,
                                      final_nn_sending_map,final_nn_map,
                                      sending_selected_indices_count,
                                      sending_selected_indices_nn_count);
    //
    //
    this->send_nns(sending_selected_indices_count,sending_selected_indices_nn_count,
                   receiving_selected_indices_count,final_nn_map,final_nn_sending_map,final_indices_allocation);


    return final_nn_map;
  }
////
////
  index_distance_pair<INDEX_TYPE>* send_min_max_distance_to_data_owner(map<INDEX_TYPE, vector<>>* local_nns,
                                                                                     int* receiving_indices_count,
                                                                                     int* disps_receiving_indices,
                                                                                     int &send_count,int &total_receiving, int nn) {
    int* sending_indices_count = new int[grid->col_world_size]();
    int* disps_sending_indices = new int[grid->col_world_size]();

    for (int i = 0;i < grid->col_world_size;i++)
    {
      sending_indices_count[i] = (*process_to_index_set_ptr)[i].size();
      send_count += sending_indices_count[i];
      disps_sending_indices[i] = (i > 0) ? (disps_sending_indices[i - 1] + sending_indices_count[i - 1]) : 0;
    }

    //sending back received data during collect similar data points to original process
    MPI_Alltoall(sending_indices_count,1, MPI_INT, receiving_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0;i < this->world_size;i++)
    {
      total_receiving += receiving_indices_count[i];
      disps_receiving_indices[i] = (i > 0) ? (disps_receiving_indices[i - 1] + receiving_indices_count[i - 1]) : 0;
    }

    index_distance_pair *in_index_dis = new index_distance_pair[send_count];
    index_distance_pair *out_index_dis =  new index_distance_pair[total_receiving];
    int co_process = 0;
    for (int i = 0;i < this->world_size;i++)
    {
      set<int> process_se_indexes = (*process_to_index_set_ptr)[i];
      for (set<int>::iterator it = process_se_indexes.begin();it != process_se_indexes.end();it++)
      {
        in_index_dis[co_process].index = (*it);
        in_index_dis[co_process].distance = (*local_nns)[(*it)][nn - 1].distance;
        co_process++;
      }
    }

    //distribute minimum maximum distance threshold (for k=nn)
    MPI_Alltoallv(in_index_dis, sending_indices_count, disps_sending_indices, MPI_FLOAT_INT,out_index_dis,
                  receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT, MPI_COMM_WORLD);

    delete [] sending_indices_count;
    delete [] disps_sending_indices;
    return out_index_dis;
  }
//
//
  void finalize_final_dataowner(int *receiving_indices_count,int *disps_receiving_indices,
                                             index_distance_pair *out_index_dis,vector<index_distance_pair> &final_sent_indices_to_rank_map) {

    int my_end_index = starting_data_index + local_data_set_size;

#pragma omp parallel for
    for (int i = this->starting_data_index;i < my_end_index;i++)
    {
      int selected_rank = -1;
      int search_index = i;
      float minium_distance = std::numeric_limits<float>::max();

      for (int j = 0;j < grid->col_world_size;j++)
      {
        int amount = receiving_indices_count[j];
        int offset = disps_receiving_indices[j];

        for (int k = offset;k < (offset + amount); k++)
        {
          if (search_index == out_index_dis[k].index)
          {
            if (minium_distance > out_index_dis[k].distance)
            {
              minium_distance = out_index_dis[k].distance;
              selected_rank = j;
            }
            break;
          }
        }
      }
      index_distance_pair rank_distance;
      rank_distance.index = selected_rank;  //TODO: replace with rank
      rank_distance.distance = minium_distance;
      final_sent_indices_to_rank_map[search_index - starting_data_index] = rank_distance;
    }
  }
//
  vector<vector<index_distance_pair>> announce_final_dataowner(int total_receving,
                                                               int *receiving_indices_count,
                                                               int *disps_receiving_indices,
                                                               index_distance_pair *out_index_dis,
                                                               vector<index_distance_pair> &final_sent_indices_to_rank_map) {

    int* minimal_selected_rank_sending = new int[total_receving]();
    index_distance_pair minimal_index_distance[total_receving];

#pragma omp parallel for
    for (int i = 0;i < total_receving;i++)
    {
      minimal_index_distance[i].index = out_index_dis[i].index;
      minimal_index_distance[i].
          distance = final_sent_indices_to_rank_map[out_index_dis[i].index - this->starting_data_index].distance;
      minimal_selected_rank_sending[i] = final_sent_indices_to_rank_map[out_index_dis[i].index - this->starting_data_index].index; //TODO: replace
    }

    int* receiving_indices_count_back = new int[grid->col_world_size]();
    int* disps_receiving_indices_count_back = new int[grid->col_world_size]();

    // we recalculate how much we are receiving for minimal dst distribution
    MPI_Alltoall(receiving_indices_count,1, MPI_INT, receiving_indices_count_back, 1, MPI_INT, MPI_COMM_WORLD);

    int total_receivce_back = 0;
    for (int i = 0;i < grid->col_world_size;i++)
    {
      total_receivce_back += receiving_indices_count_back[i];
      disps_receiving_indices_count_back[i] = (i > 0) ?
                                                      (disps_receiving_indices_count_back[i - 1] + receiving_indices_count_back[i - 1]) : 0;
    }

    int* minimal_selected_rank_reciving = new int[total_receivce_back]();
    index_distance_pair minimal_index_distance_receiv[total_receivce_back];


    MPI_Alltoallv(minimal_index_distance, receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT,
                  minimal_index_distance_receiv,
                  receiving_indices_count_back, disps_receiving_indices_count_back, MPI_FLOAT_INT, MPI_COMM_WORLD);


    MPI_Alltoallv(minimal_selected_rank_sending, receiving_indices_count, disps_receiving_indices, MPI_INT,
                  minimal_selected_rank_reciving,
                  receiving_indices_count_back, disps_receiving_indices_count_back, MPI_INT, MPI_COMM_WORLD);


    vector<vector<index_distance_pair>> final_indices_allocation(grid->col_world_size);

#pragma omp parallel
    {
      vector<vector<index_distance_pair>> final_indices_allocation_local(grid->col_world_size);

#pragma omp  for nowait
      for (int i = 0;i < total_receivce_back;i++)
      {
        index_distance_pair distance_pair;
        distance_pair.
            index = minimal_index_distance_receiv[i].index;
        distance_pair.
            distance = minimal_index_distance_receiv[i].distance;
        final_indices_allocation_local[minimal_selected_rank_reciving[i]].
            push_back(distance_pair);

      }

#pragma omp critical
      {
        for (int i = 0;i < grid->col_world_size;i++)
        {
          final_indices_allocation[i].insert(final_indices_allocation[i].end(),
                                             final_indices_allocation_local[i].begin(),
                                             final_indices_allocation_local[i].end());
        }
      }
    }

    return final_indices_allocation;

  }
//
  void select_final_forwarding_nns(vector<vector<index_distance_pair>> &final_indices_allocation,
                                                map<INDEX_TYPE, vector<>>* local_nns,
                                                map<INDEX_TYPE, vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>* final_nn_sending_map,
                                                map<INDEX_TYPE, vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>*  final_nn_map,
                                                int* sending_selected_indices_count,
                                                int* sending_selected_indices_nn_count){

    for (int i = 0;i < grid->col_world_size;i++)
    {

#pragma omp parallel for
      for (int j = 0;j < final_indices_allocation[i].size();j++)
      {
        index_distance_pair in_dis = final_indices_allocation[i][j];
        int selected_index = in_dis.index;
        float dst_th = in_dis.distance;
        if (i != this->rank)
        {
          if ((*local_nns).find(selected_index)!= local_nns.end())
          {
            vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>> target;
            std::copy_if((*local_nns)[selected_index].begin(),
                         (*local_nns)[selected_index].end(),
                         std::back_inserter(target),
                         [dst_th](
                             EdgeNode<INDEX_TYPE,VALUE_TYPE> dataPoint
                         )
                         {
                           return dataPoint.distance < dst_th;
                         });
            if (target.size()> 0)
            {
#pragma omp critical
              {
                if ((*final_nn_sending_map).find(selected_index)== (*final_nn_sending_map).end())
                {
                  (*final_nn_sending_map).insert(pair < INDEX_TYPE, vector <EdgeNode<INDEX_TYPE,VALUE_TYPE>  >>
                                              (selected_index, target));
                  sending_selected_indices_nn_count[i] += target.size();
                  sending_selected_indices_count[i] += 1;
                }
              }
            }
          }
        }
        else
        {
#pragma omp critical
          (*final_nn_map).insert(pair < INDEX_TYPE, vector < EdgeNode<INDEX_TYPE,VALUE_TYPE> >>(selected_index,
                                                           (*local_nns)[selected_index]));
        }
      }
    }

  }
//
  void send_nns(int *sending_selected_indices_count,int *sending_selected_indices_nn_count,int *receiving_selected_indices_count,
                             std::map<INDEX_TYPE, vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>* final_nn_map,
                             std::map<INDEX_TYPE, vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>>>* final_nn_sending_map,
                             vector<vector<index_distance_pair>> &final_indices_allocation) {

    int total_receiving_count = 0;

    int total_receiving_nn_count = 0;

    MPI_Alltoall(sending_selected_indices_count,
                 1, MPI_INT, receiving_selected_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

    int* disps_receiving_selected_indices = new int[grid->col_world_size]();
    int* disps_sending_selected_indices = new int[grid->col_world_size]();
    int* disps_sending_selected_nn_indices = new int[grid->col_world_size]();
    int* disps_receiving_selected_nn_indices = new int[grid->col_world_size]();


    int total_selected_indices_count=0;

    int total_selected_indices_nn_count=0;

    for (int i = 0;i < grid->col_world_size;i++)
    {
      disps_receiving_selected_indices[i] = (i > 0) ? (disps_receiving_selected_indices[i - 1] +
                                                       receiving_selected_indices_count[i - 1]) : 0;
      disps_sending_selected_indices[i] = (i > 0) ? (disps_sending_selected_indices[i - 1] +
                                                     sending_selected_indices_count[i - 1]) : 0;
      disps_sending_selected_nn_indices[i] = (i > 0) ? (disps_sending_selected_nn_indices[i - 1] +
                                                        sending_selected_indices_nn_count[i - 1]) : 0;

      total_selected_indices_count += sending_selected_indices_count[i];
      total_selected_indices_nn_count += sending_selected_indices_nn_count[i];
    }

    int* sending_selected_indices = new int[total_selected_indices_count]();

    int* sending_selected_nn_count_for_each_index = new int[total_selected_indices_count]();

    index_distance_pair* sending_selected_nn = new index_distance_pair[total_selected_indices_nn_count];

    int inc = 0;
    int selected_nn = 0;
    for (int i = 0;i < grid->col_world_size;i++)
    {
      total_receiving_count += receiving_selected_indices_count[i];
      if (i != grid->rank_in_col)
      {
        vector<index_distance_pair> final_indices = final_indices_allocation[i];
        for (int j = 0;j < final_indices.size();j++)
        {
          if ((*final_nn_sending_map).find(final_indices[j].index) != (*final_nn_sending_map).end())
          {
            vector<EdgeNode<INDEX_TYPE,VALUE_TYPE>> nn_sending = (*final_nn_sending_map)[final_indices[j].index];
            if (nn_sending.size()> 0)
            {
              sending_selected_indices[inc] = final_indices[j].index;
              for (int k = 0;k < nn_sending.size();k++)
              {
                sending_selected_nn[selected_nn].
                    index = nn_sending[k].index;
                sending_selected_nn[selected_nn].
                    distance = nn_sending[k].distance;
                selected_nn++;
              }
              sending_selected_nn_count_for_each_index[inc] = nn_sending.
                                                              size();
              inc++;
            }
          }
        }
      }
    }

    int* receiving_selected_nn_indices_count = new int[total_receiving_count]();

    int* receiving_selected_indices = new int[total_receiving_count]();

    MPI_Alltoallv(sending_selected_nn_count_for_each_index, sending_selected_indices_count,
                  disps_sending_selected_indices, MPI_INT, receiving_selected_nn_indices_count,
                  receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD
    );

    MPI_Alltoallv(sending_selected_indices, sending_selected_indices_count, disps_sending_selected_indices, MPI_INT,
                  receiving_selected_indices,
                  receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD
    );

    int* receiving_selected_nn_indices_count_process = new int[grid->col_world_size]();

    for (int i = 0;i < grid->col_world_size;i++)
    {
      int co = receiving_selected_indices_count[i];
      int offset = disps_receiving_selected_indices[i];

      for (int k = offset;k < (co + offset); k++)
      {
        receiving_selected_nn_indices_count_process[i] += receiving_selected_nn_indices_count[k];
      }
      total_receiving_nn_count += receiving_selected_nn_indices_count_process[i];

      disps_receiving_selected_nn_indices[i] = (i > 0) ? (disps_receiving_selected_nn_indices[i - 1] +
                                                          receiving_selected_nn_indices_count_process[i - 1]) : 0;
    }

    index_distance_pair* receving_selected_nn = new index_distance_pair[total_receiving_nn_count];




    MPI_Alltoallv(sending_selected_nn, sending_selected_indices_nn_count, disps_sending_selected_nn_indices,
                  MPI_FLOAT_INT,
                  receving_selected_nn,
                  receiving_selected_nn_indices_count_process, disps_receiving_selected_nn_indices, MPI_FLOAT_INT,
                  MPI_COMM_WORLD);

    int nn_index = 0;
    for (int i = 0;i < total_receiving_count;i++)
    {
      int src_index = receiving_selected_indices[i];
      int nn_count = receiving_selected_nn_indices_count[i];
      vector<EdgeNode> vec;
      for (int j = 0;j < nn_count;j++)
      {
        int nn_indi = receving_selected_nn[nn_index].index;
        VALUE_TYPE distance = receving_selected_nn[nn_index].distance;
        EdgeNode<INDEX_TYPE,VALUE_TYPE> dataPoint;
        dataPoint.src_index = src_index;
        dataPoint.dst_index = nn_indi;
        dataPoint.distance = distance;
        vec.push_back(dataPoint);
        nn_index++;
      }

      auto its = (*final_nn_map).find(src_index);
      if (its == (*final_nn_map).end())
      {
        (*final_nn_map).insert(pair < int, vector < EdgeNode<INDEX_TYPE,VALUE_TYPE> >>(src_index, vec));
      }
      else
      {
        vector<EdgeNode> dst;
        vector<EdgeNode> ex_vec = its->second;
        sort(vec.begin(), vec.end(),
             [](const EdgeNode& lhs,const EdgeNode& rhs)
             {
               return lhs.distance < rhs.distance;
             });
        std::merge(ex_vec.begin(), ex_vec.end(), vec.begin(),
                   vec.end(), std::back_inserter(dst),
                   [](const EdgeNode& lhs,const EdgeNode& rhs
                   )
                   {
                     return lhs.distance < rhs.distance;
                   });
        dst.
            erase(unique(dst.begin(), dst.end(), [](const EdgeNode& lhs,
                                                    const EdgeNode& rhs)
                         {
                           return lhs.dst_index == rhs.dst_index;
                         }),
                  dst.end()
            );
        (its->second) =dst;
      }
    }
  }

};
} // namespace hipgraph::distviz::knng


