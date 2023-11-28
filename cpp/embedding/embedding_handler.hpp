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
                          uint64_t gRows, uint64_t gCols, uint64_t gNNZ, int batch_size,
                          int iterations, float lr, int nsamples, float alpha,float beta,
                          bool col_major=false, bool sync_comm=false,bool self_converge=true){

    auto localBRows = divide_and_round_up(gCols,
                                          grid->col_world_size);
    auto localARows = divide_and_round_up(gRows,
                                          grid->col_world_size);



    vector<Tuple<VALUE_TYPE>>* shared_sparseMat_sender_coords = new vector<Tuple<VALUE_TYPE>>(*input_graph);
    vector<Tuple<VALUE_TYPE>>* shared_sparseMat_receiver_coords = new vector<Tuple<VALUE_TYPE>>(*input_graph);

    auto shared_sparseMat = make_shared<SpMat<VALUE_TYPE>>(grid,input_graph, gRows,gCols, gNNZ, batch_size,
                                                                          localARows, localBRows, false, false);

    auto shared_sparseMat_sender = make_shared<SpMat<VALUE_TYPE>>(grid,shared_sparseMat_sender_coords, gRows,gCols, gNNZ, batch_size,
                                                                           localARows, localBRows, false, true);

    auto shared_sparseMat_receiver = make_shared<SpMat<VALUE_TYPE>>(grid,shared_sparseMat_receiver_coords, gRows,gCols, gNNZ, batch_size,
                                                                             localARows, localBRows, true, false);


    auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(grid));

    partitioner.get()->partition_data(shared_sparseMat_sender.get());
    partitioner.get()->partition_data(shared_sparseMat_receiver.get());
    partitioner.get()->partition_data(shared_sparseMat.get());

    cout<<" rank "<<grid->rank_in_col<<" size "<<shared_sparseMat.get()->coords->size()<<endl;


    shared_sparseMat.get()->initialize_CSR_blocks();
    shared_sparseMat_sender.get()->initialize_CSR_blocks();
    shared_sparseMat_receiver.get()->initialize_CSR_blocks();

    unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>

        embedding_algo = unique_ptr<EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>(
                new EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_output, grid,
                    alpha, beta, 5, -5,col_major,sync_comm));

   vector<VALUE_TYPE> error_convergence = embedding_algo.get()->algo_force2_vec_ns(iterations, batch_size, nsamples, lr,self_converge);

   cout<<" rank  "<<grid->rank_in_col<<"erros #####"<<endl;
   for(int i=0;i<error_convergence.size();i++){
     cout<<error_convergence<<" "
   }
   cout<<endl;
   cout<<" rank  "<<grid->rank_in_col<<"erros  completed #####"<<endl;
  }





  };
}

