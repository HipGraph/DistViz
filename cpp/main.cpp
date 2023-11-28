#pragma once
#include "lib/Mrpt.h"
#include <iostream>
#include "Eigen/Dense"
#include "io/file_reader.hpp"
#include <chrono>
#include "knng/global_tree_handler.hpp"
#include "net/process_3D_grid.hpp"
#include "knng/knng_handler.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>
#include <fstream>
#include <math.h>
#include <chrono>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include "common/common.h"
#include "embedding/embedding_handler.hpp"
#include "embedding/dense_mat.hpp"

using namespace std;
using namespace std::chrono;
using namespace hipgraph::distviz::net;
using namespace hipgraph::distviz::knng;
using namespace hipgraph::distviz::embedding;

using namespace hipgraph::distviz::io;
int main(int argc, char* argv[]) {

  string input_path;
  string output_path;
  long data_set_size = 0;
  int dimension = 0;
  int ntrees = 10;
  int tree_depth = 0;
  double density = 0;
  int nn = 0;
  double tree_depth_ratio = 0.6;
  bool use_locality_optimization = true;
  int local_tree_offset = 2;
  int file_format = 0;
  int data_starting_index = 8;

  const int embedding_dimension = 2;

  int bytes_for_data_type = 4;

  for (int p = 0; p < argc; p++)
  {
    if (strcmp(argv[p], "-input") == 0)
    {
      input_path = argv[p + 1];
    }
    else if (strcmp(argv[p], "-output") == 0)
    {
      output_path = argv[p + 1];
    }
    else if (strcmp(argv[p], "-data-set-size") == 0)
    {
      data_set_size = atol(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-dimension") == 0)
    {
      dimension = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-ntrees") == 0)
    {
      ntrees = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-tree-depth-ratio") == 0)
    {
      tree_depth_ratio = stof(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-density") == 0)
    {
      density = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-nn") == 0)
    {
      nn = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-locality") == 0)
    {
      use_locality_optimization = atoi(argv[p + 1]) == 1 ? true : false;
    }
    else if (strcmp(argv[p], "-local-tree-offset") == 0)
    {
      local_tree_offset = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-data-file-format") == 0)
    {
      file_format = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-bytes-for-data-type") == 0)
    {
      bytes_for_data_type = atoi(argv[p + 1]);
    }
    else if (strcmp(argv[p], "-data-starting-index") == 0)
    {
      data_starting_index = atoi(argv[p + 1]);
    }
  }

  if (input_path.size() == 0)
  {
    printf("Valid input path needed!...\n");
    exit(1);
  }

  if (output_path.size() == 0)
  {
    printf("Valid out path needed!...\n");
    exit(1);
  }

  if (data_set_size == 0)
  {
    printf("Dataset size should be greater than 0\n");
    exit(1);
  }

  if (dimension == 0)
  {
    printf("Dimension size should be greater than 0\n");
    exit(1);
  }

  if (nn == 0)
  {
    printf("Nearest neighbours size should be greater than 0\n");
    exit(1);
  }

  if (density == 0)
  {
    density = 1.0 / sqrt(dimension);
  }


  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if (tree_depth == 0)
  {
    tree_depth = static_cast<int>(log2((data_set_size / size)) - 2);
    cout << " tree depth " << tree_depth << endl;
  }


  int k = 10;
  double target_recall = 0.9;
//  int n = 10000, d = 200, k = 10;
//  double target_recall = 0.9;
//  Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
//  Eigen::MatrixXf q = Eigen::VectorXf::Random(d);


  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(size, 1, 1, 1));



  Eigen::VectorXi indices(k),  indices_exact(k);

//  Eigen::VectorXf distances(k);
  auto start_io_index = high_resolution_clock::now();
  std::cout << "calling data loading"<< rank<< " "<<std::endl;
  unique_ptr<ValueType2DVector<float>> data_matrix_ptr= make_unique<ValueType2DVector<float>>();;
   FileReader<float>::load_data_into_2D_vector(input_path,data_matrix_ptr.get(),data_set_size,dimension,grid.get()->rank_in_col,grid.get()->col_world_size);
   cout<<"rank "<<grid->rank_in_col<<" data_matrix_ptr size "<<data_matrix_ptr.get()->size()<<"* "<<(*data_matrix_ptr.get())[7499].size()<<endl;
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "calling data loading completed "<<rank<<" "<<std::endl;
  auto stop_io_index = high_resolution_clock::now();
  auto io_time = duration_cast<microseconds>(stop_io_index - start_io_index);
  data_matrix_ptr.get()->size();
  std::cout << "calling KNNGHandler "<<rank<<" size  "<<data_matrix_ptr->size()<<std::endl;
  auto knng_handler = unique_ptr<KNNGHandler<int,float>>(new KNNGHandler<int,float>(ntrees,  tree_depth,  tree_depth_ratio,
                                                                                       local_tree_offset,  data_set_size,
                                                                                      data_matrix_ptr.get()->size(),
                                                                                      dimension,  grid.get()));

  std::cout << "calling grow trees"<< rank<< " "<<std::endl;

  shared_ptr<vector<Tuple<float>>> knng_graph_ptr = make_shared<vector<Tuple<float>>>();
  knng_handler.get()->build_distributed_KNNG(data_matrix_ptr.get(),knng_graph_ptr.get(),density,use_locality_optimization,nn,0.9);


  cout<<" rank "<<rank<<" output size: "<<knng_graph_ptr.get()->size()<<endl;

  initialize_mpi_datatypes<int, float, embedding_dimension>();

  auto localARows = divide_and_round_up(data_set_size,grid.get()->col_world_size);

  auto dense_mat = shared_ptr<DenseMat<int, float, embedding_dimension>>(
      new DenseMat<int, float, embedding_dimension>(grid.get(), localARows));

   auto embedding_handler = unique_ptr<EmbeddingHandler<int,float,embedding_dimension>>(new EmbeddingHandler<int, float,embedding_dimension>(grid.get()));
   auto gNNZ = data_set_size* (nn-1);

  embedding_handler->generate_embedding(knng_graph_ptr.get(),dense_mat.get(),data_set_size,data_set_size,gNNZ,
                                         localARows,1200,0.2,5,0.5,0.5,false,false);

  auto stop_index_building = high_resolution_clock::now();

  auto duration_index_building = duration_cast<microseconds>(stop_index_building - stop_io_index);





  double* execution_times = new double[2];

  double* execution_times_global = new double[2];

  execution_times[0] = (io_time.count()) / 1000;
  execution_times[1] = duration_index_building.count() / 1000;
//  execution_times[2] = duration_query.count() / 1000;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Allreduce(execution_times, execution_times_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(rank==0)
  {
//    cout << "IO Time (s)" << execution_times_global[0]/(size*1000) << " Index building (s) "
//         << execution_times_global[1]/(size*1000) <<" Querying Time (s) "<< execution_times_global[2]/(size*1000)<< endl;
//
    cout << "IO Time (s)" << execution_times_global[0]/(size*1000) << " Index building (s) "<< execution_times_global[1]/(size*1000) << endl;
  }

//  delete[] execution_times;
//  delete[] execution_times_global;

  MPI_Finalize();

}
