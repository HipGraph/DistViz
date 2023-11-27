#pragma once

#include "../common/common.h"
#include "dense_mat.hpp"
#include "sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "partitioner.hpp"
#include "algo.hpp"
namespace hipgraph::distviz::embedding {

template<typename INDEX_TYPE, typename  VALUE_TYPE,size_t dimension>
class EmbeddingHandler {



public:

  Process3DGrid* grid;
  EmbeddingHandler<INDEX_TYPE,VALUE_TYPE,dimension>(Process3DGrid* grid){
    this->grid = grid;
  }




  void generate_embedding(vector<Tuple<VALUE_TYPE>>* input_graph,DenseMat<INDEX_TYPE, VALUE_TYPE, dimension>* dense_output,
                          int gRows, int gCols, int gNNZ, int batch_size,
                          int iterations, float lr, int nsamples, float alpha,float beta,
                          bool col_major, bool sync_comm){

    auto localBRows = divide_and_round_up(gCols,
                                          grid->col_world_size);
    auto localARows = divide_and_round_up(gRows,
                                          grid->col_world_size);


    auto shared_sparseMat = make_shared<SpMat<INDEX_TYPE>>(grid,input_graph, gRows,gCols, gNNZ, batch_size,
                                                                          localARows, localBRows, false, false);

    auto shared_sparseMat_sender = make_shared<SpMat<INDEX_TYPE>>(grid,input_graph, gRows,gCols, gNNZ, batch_size,
                                                                           localARows, localBRows, false, true);

    auto shared_sparseMat_receiver = make_shared<SpMat<INDEX_TYPE>>(grid,input_graph, gRows,gCols, gNNZ, batch_size,
                                                                             localARows, localBRows, true, false);


    auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(grid));

    partitioner.get()->partition_data(shared_sparseMat_sender.get());
    partitioner.get()->partition_data(shared_sparseMat_receiver.get());
    partitioner.get()->partition_data(shared_sparseMat.get());

    unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>

        embedding_algo = unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>(
                new EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_output, grid,
                    alpha, beta, 5, -5,col_major,sync_comm));

  }





  };
}

