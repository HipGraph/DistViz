/**
 * This class represents the implementation of the distributed Embedding
 * Algorithm. This is tightly coupled with 1D partitioning of Sparse Matrix and
 * Dense Matrix.
 */
#pragma once
#include "../common/common.h"
#include "../common/json.hpp"
#include "../lib/Mrpt.h"
#include "../lib/Mrpt_Sparse.h"
#include "../net/data_comm.hpp"
#include "../net/process_3D_grid.hpp"
#include "csr_local.hpp"
#include "dense_mat.hpp"
#include "sparse_mat.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <math.h>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>
#include "../knng/math_operations.hpp"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_errno.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace std;
using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::net;
using namespace Eigen;

namespace hipgraph::distviz::embedding {
template <typename SPT, typename DENT, size_t embedding_dim>

class EmbeddingAlgo {

protected:
  DenseMat<SPT, DENT, embedding_dim> *dense_local;
  SpMat<SPT, DENT> *sp_local_receiver;
  SpMat<SPT, DENT> *sp_local_sender;
  SpMat<SPT, DENT> *sp_local_native;
  Process3DGrid *grid;
  DENT MAX_BOUND, MIN_BOUND;
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>>
      data_comm_cache;

  std::vector<DENT> sigma_cache;
  std::vector<DENT> minimum_dis_cache;
  std::vector<vector<DENT>> samples_per_epoch;
  std::vector<vector<DENT>> samples_per_epoch_next;
  std::vector<vector<DENT>> samples_per_epoch_negative;
  std::vector<vector<DENT>> samples_per_epoch_negative_next;

  // cache size controlling hyper parameter
  double alpha = 1.0;

  // hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  // hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  // hyper parameter controls the col major or row major  data access
  bool col_major = true;

public:
  EmbeddingAlgo(SpMat<SPT, DENT> *sp_local_native,
                SpMat<SPT, DENT> *sp_local_receiver,
                SpMat<SPT, DENT> *sp_local_sender,
                DenseMat<SPT, DENT, embedding_dim> *dense_local,
                Process3DGrid *grid, double alpha, double beta, DENT MAX_BOUND,
                DENT MIN_BOUND, bool col_major, bool sync_comm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), dense_local(dense_local), grid(grid),
        alpha(alpha), beta(beta), MAX_BOUND(MAX_BOUND), MIN_BOUND(MIN_BOUND),
        col_major(col_major), sync(sync_comm) {}

  DENT scale(DENT v) {
    if (v > MAX_BOUND)
      return MAX_BOUND;
    else if (v < -MAX_BOUND)
      return -MAX_BOUND;
    else
      return v;
  }

  void algo_force2_vec_ns(int iterations, int batch_size, int ns, DENT lr,
      double drop_out_error_threshold = 0) {
    int batches = 0;
    int last_batch_size = batch_size;
    sigma_cache.resize(sp_local_receiver->proc_row_width, -1);
    minimum_dis_cache.resize(sp_local_receiver->proc_row_width, -1);

    samples_per_epoch.resize(sp_local_receiver->proc_row_width, vector<DENT>());
    samples_per_epoch_next.resize(sp_local_receiver->proc_row_width, vector<DENT>());
    samples_per_epoch_negative.resize(sp_local_receiver->proc_row_width, vector<DENT>());
    samples_per_epoch_negative_next.resize(sp_local_receiver->proc_row_width, vector<DENT>());

    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size = sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << grid->rank_in_col << " total batches " << batches<< endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto full_comm = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            sp_local_receiver, sp_local_sender, dense_local, grid, -1, alpha));
    full_comm.get()->onboard_data();

    cout << " rank " << grid->rank_in_col<<
        " onboard data sucessfully for initial iteration" << batches<< endl;

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    // Buffer used for send MPI operations data
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, dense_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    cout << " rank " << grid->rank_in_col << " onboard_data completed "<< batches << endl;

    //    DENT *prevCoordinates = static_cast<DENT *>(::operator
    //    new(sizeof(DENT[batch_size * embedding_dim])));

    shared_ptr<vector<DENT>> prevCoordinates_ptr =
        make_shared<vector<DENT>>(batch_size * embedding_dim, DENT{});

    size_t total_memory = 0;

    CSRLocal<SPT, DENT> *csr_block =
        (col_major) ? sp_local_receiver->csr_local_data.get()
                    : sp_local_native->csr_local_data.get();

    int considering_batch_size = batch_size;
    vector<DENT> error_convergence;

    if (csr_block->handler != nullptr) {
      CSRHandle<SPT, DENT> *csr_handle = csr_block->handler.get();
      calculate_membership_strength(csr_handle);
      apply_set_operations(true,1.0,
                           csr_handle->rowStart,csr_handle->col_idx,csr_handle->values);
      cout<<"apply_set_operations"<<endl;
      make_epochs_per_sample(csr_handle,iterations,ns);
      cout<<"make_epochs_per_sample completed"<<endl;
      computeTopKEigenvectorsFromLaplacian(csr_handle->rowStart,csr_handle->col_idx,csr_handle->values,10);

    }
    double min_dist=0.1;
    double spread=1.0;
//    pair<double,double> ab = find_ab_params(spread,min_dist);
    double a = 42.73;
    double b = 1.33;
    cout<<"a "<<a<<"b "<<b<<endl;
    int seed =0;
    DENT alpha = lr;
    for (int i = 0; i < iterations; i++) {
      DENT batch_error = 0;
      for (int j = 0; j < batches; j++) {
//        int seed = j + i;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }

        unique_ptr<vector<vector<vector<SPT>>>> negative_samples_ptr =
            make_unique<vector<vector<vector<SPT>>>>(considering_batch_size, vector<vector<SPT>>(ns, vector<SPT>()));
        CSRHandle<SPT, DENT> *csr_handle = csr_block->handler.get();



//            // negative samples generation
//        vector<SPT> random_number_vec = generate_random_numbers<SPT>(
//            0, (this->sp_local_receiver)->gRows, seed, ns);

        // One process computations without MPI operations
        if (grid->col_world_size == 1) {
          for (int k = 0; k < batch_size; k++) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              (*prevCoordinates_ptr)[IDIM + d] = 0;
            }
          }
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(
              csr_block, prevCoordinates_ptr.get(), alpha, i,j, batch_size,
              considering_batch_size, true, false, 0, 0, false, a, b);

          generate_negative_samples(negative_samples_ptr.get(),csr_handle,i,j,batch_size,
                                    considering_batch_size,seed);
          this->calc_t_dist_replus_rowptr_new(
              prevCoordinates_ptr.get(), negative_samples_ptr.get(),
              csr_handle,alpha, j, batch_size,
              considering_batch_size, a, b);

          batch_error += this->update_data_matrix_rowptr(
              prevCoordinates_ptr.get(), j, batch_size);
          alpha = lr * (1.0 - (float(i) / float(iterations)));
        } else {
          // These operations are for more than one processes.
//          full_comm.get()->transfer_data(random_number_vec, i, j);
//          this->calc_t_dist_replus_rowptr(prevCoordinates_ptr.get(),
//                                          random_number_vec, lr, j, batch_size,
//                                          considering_batch_size);

            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr, prevCoordinates_ptr.get(), 1, true,
                0, true);

            batch_error += this->update_data_matrix_rowptr(
                prevCoordinates_ptr.get(), j, batch_size);

            for (int k = 0; k < batch_size; k++) {
              int IDIM = k * embedding_dim;
              for (int d = 0; d < embedding_dim; d++) {
                (*prevCoordinates_ptr)[IDIM + d] = 0;
              }
            }
        }
      }
    }
  }

  inline void execute_pull_model_computations(
      std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
      std::vector<DataTuple<DENT, embedding_dim>> *receivebuf, int iteration,
      int batch, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT, DENT> *csr_block, int batch_size,
      int considering_batch_size, double lr, vector<DENT> *prevCoordinates,
      int comm_initial_start, bool local_execution, int first_execution_proc,
      bool communication) {

    int proc_length = get_proc_length(beta, grid->col_world_size);
    int prev_start = comm_initial_start;
    //    cout << " rank " << grid->rank_in_col << " run pull model algorithm "
    //    << endl;
    for (int k = prev_start; k < grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, grid->col_world_size);

      MPI_Request req;

      if (communication) {
        data_comm->transfer_data(sendbuf, receivebuf, sync, &req, iteration,
                                 batch, k, end_process, true);
      }

      if (!sync and communication) {
        MPI_Ialltoallv(
            (*sendbuf).data(), (*(data_comm->send_counts_cyclic)).data(),
            (*(data_comm->sdispls_cyclic)).data(), DENSETUPLE,
            (*receivebuf).data(), (*(data_comm->receive_counts_cyclic)).data(),
            (*(data_comm->rdispls_cyclic)).data(), DENSETUPLE, grid->col_world,
            &req);
      }

      if (k == comm_initial_start) {
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, iteration,batch, batch_size,
            considering_batch_size, local_execution, col_major,
            first_execution_proc, prev_start, local_execution);

      } else if (k > comm_initial_start) {
        int prev_end_process =
            get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, iteration,batch,
                                      batch_size, considering_batch_size, false,
                                      col_major, prev_start, prev_end_process,
                                      true);
      }

      if (!sync and communication) {
        data_comm->populate_cache(sendbuf, receivebuf, &req, sync, iteration,
                                  batch, true);
      }

      prev_start = k;
    }

    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                  iteration,batch_size, considering_batch_size, false,
                                  col_major, prev_start, prev_end_process,
                                  true);

    // dense_local->invalidate_cache(i, j, true);
  }



  inline void calc_t_dist_grad_rowptr(
      CSRLocal<SPT, DENT> *csr_block, vector<DENT> *prevCoordinates, DENT lr,
      int iteration,int batch_id, int batch_size, int block_size, bool local, bool col_major,
      int start_process, int end_process, bool fetch_from_temp_cache, double a=1.0, double b=1.0) {

    auto source_start_index = batch_id * batch_size;
    auto source_end_index = std::min((batch_id + 1) * batch_size,
                                     this->sp_local_receiver->proc_row_width) -
                            1;

    auto dst_start_index =
        this->sp_local_receiver->proc_col_width * grid->rank_in_col;
    auto dst_end_index =
        std::min(static_cast<uint64_t>(this->sp_local_receiver->proc_col_width *
                                       (grid->rank_in_col + 1)),
                 this->sp_local_receiver->gCols) -
        1;

    if (local) {
      if (col_major) {
        calc_embedding(source_start_index, source_end_index, dst_start_index,
                       dst_end_index, csr_block, prevCoordinates, lr, batch_id,
                       batch_size, block_size, fetch_from_temp_cache);
      } else {
        calc_embedding_row_major_new(iteration,source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                 prevCoordinates, lr, batch_id, batch_size,
                                 block_size, fetch_from_temp_cache,a,b);
      }
    } else {
      for (int r = start_process; r < end_process; r++) {

        if (r != grid->rank_in_col) {

          int computing_rank =
              (grid->rank_in_col >= r)
                  ? (grid->rank_in_col - r) % grid->col_world_size
                  : (grid->col_world_size - r + grid->rank_in_col) %
                        grid->col_world_size;

          dst_start_index =
              this->sp_local_receiver->proc_row_width * computing_rank;
          dst_end_index = std::min(static_cast<uint64_t>(
                                       this->sp_local_receiver->proc_row_width *
                                       (computing_rank + 1)),
                                   this->sp_local_receiver->gCols) -
                          1;

//          if (col_major) {
//            calc_embedding(source_start_index, source_end_index,
//                           dst_start_index, dst_end_index, csr_block,
//                           prevCoordinates, lr, batch_id, batch_size,
//                           block_size, fetch_from_temp_cache);
//          } else {
            calc_embedding_row_major(iteration,source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                     prevCoordinates, lr, batch_id, batch_size,
                                     block_size, fetch_from_temp_cache);
//          }
        }
      }
    }
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT, DENT> *csr_block,
                             vector<DENT> *prevCoordinates, DENT lr,
                             int batch_id, int batch_size, int block_size,
                             bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle<SPT, DENT> *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        uint64_t local_dst =
            i - (grid)->rank_in_col * (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (grid)->rank_in_col ? false : true;

        bool matched = false;
        std::array<DENT, embedding_dim> array_ptr;
        bool col_inserted = false;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          if (csr_handle->col_idx[j] >= source_start_index and
              csr_handle->col_idx[j] <= source_end_index) {
            DENT forceDiff[embedding_dim];
            auto source_id = csr_handle->col_idx[j];
            auto index = source_id - batch_id * batch_size;

            if (!matched) {
              if (fetch_from_cache) {
                unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
                    &arrayMap =
                        (temp_cache)
                            ? (*this->dense_local->tempCachePtr)[target_rank]
                            : (*this->dense_local->cachePtr)[target_rank];
                array_ptr = arrayMap[i].value;
              }
              matched = true;
            }
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              if (!fetch_from_cache) {
                forceDiff[d] =
                    (this->dense_local)
                        ->nCoordinates[source_id * embedding_dim + d] -
                    (this->dense_local)
                        ->nCoordinates[local_dst * embedding_dim + d];
              } else {
                forceDiff[d] =
                    (this->dense_local)
                        ->nCoordinates[source_id * embedding_dim + d] -
                    (array_ptr[d]);
              }
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);

            for (int d = 0; d < embedding_dim; d++) {
              DENT l = scale(forceDiff[d] * d1);
              (*prevCoordinates)[index * embedding_dim + d] =
                  (*prevCoordinates)[index * embedding_dim + d] + (lr)*l;
            }
          }
        }
      }
    }
  }

  inline void calc_embedding_row_major(int iteration,
      uint64_t source_start_index, uint64_t source_end_index,
      uint64_t dst_start_index, uint64_t dst_end_index,
      CSRLocal<SPT, DENT> *csr_block, vector<DENT> *prevCoordinates, DENT lr,
      int batch_id, int batch_size, int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle<SPT, DENT> *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or
                                          // // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;

        int nn_size = csr_handle->rowStart[i + 1] - csr_handle->rowStart[i];
        unordered_map<int64_t, float> distance_map;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          int dst_index = j - static_cast<uint64_t>(csr_handle->rowStart[i]);
//          cout<<" i "<<i<<" j"<<dst_index<<" itr"<<iteration<<" val "<<samples_per_epoch_next[i][dst_index]<<endl;
          if (samples_per_epoch_next[i][dst_index] <= iteration+1) {
            auto dst_id = csr_handle->col_idx[j];
            auto distance = csr_handle->values[j];
            if (dst_id >= dst_start_index and dst_id < dst_end_index) {
              uint64_t local_dst =
                  dst_id - (grid)->rank_in_col *
                               (this->sp_local_receiver)->proc_col_width;
              int target_rank =
                  (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
              bool fetch_from_cache =
                  target_rank == (grid)->rank_in_col ? false : true;

              DENT forceDiff[embedding_dim];
              std::array<DENT, embedding_dim> array_ptr;
              if (fetch_from_cache) {
                unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
                    &arrayMap =
                        (temp_cache)
                            ? (*this->dense_local->tempCachePtr)[target_rank]
                            : (*this->dense_local->cachePtr)[target_rank];
                array_ptr = arrayMap[dst_id].value;
              }

              DENT attrc = 0;

              for (int d = 0; d < embedding_dim; d++) {
                if (!fetch_from_cache) {
                  forceDiff[d] =
                      (this->dense_local)->nCoordinates[i * embedding_dim + d] -
                      (this->dense_local)
                          ->nCoordinates[local_dst * embedding_dim + d];

                } else {
                  forceDiff[d] =
                      (this->dense_local)->nCoordinates[i * embedding_dim + d] -
                      array_ptr[d];
                }
                //              forceDiff[d] = forceDiff[d] * exp(-1 * distance / smoothe_factor);
                attrc += forceDiff[d] * forceDiff[d];
              }

              DENT d1 = -2.0 / (1.0 + attrc);
              for (int d = 0; d < embedding_dim; d++) {
                DENT l = scale(forceDiff[d] * d1);
                //              DENT l = (forceDiff[d] * d1 );
                //              l=l*exp(-1*distance/smoothe_factor);
                (*prevCoordinates)[index * embedding_dim + d] =
                    (*prevCoordinates)[index * embedding_dim + d] + (lr)*l;
              }
            }
          }

        }
      }
    }
  }


  inline void calc_embedding_row_major_new(int iteration,
                                       uint64_t source_start_index, uint64_t source_end_index,
                                       uint64_t dst_start_index, uint64_t dst_end_index,
                                       CSRLocal<SPT, DENT> *csr_block, vector<DENT> *prevCoordinates, DENT lr,
                                       int batch_id, int batch_size, int block_size, bool temp_cache,double a=1.0, double b=1.0) {
    if (csr_block->handler != nullptr) {
      CSRHandle<SPT, DENT> *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or
                                          // // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;

        int nn_size = csr_handle->rowStart[i + 1] - csr_handle->rowStart[i];
        unordered_map<int64_t, float> distance_map;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          int dst_index = j - static_cast<uint64_t>(csr_handle->rowStart[i]);
          //          cout<<" i "<<i<<" j"<<dst_index<<" itr"<<iteration<<" val "<<samples_per_epoch_next[i][dst_index]<<endl;
          if (samples_per_epoch_next[i][dst_index] <= iteration+1) {
            auto dst_id = csr_handle->col_idx[j];
            auto distance = csr_handle->values[j];
            if (dst_id >= dst_start_index and dst_id < dst_end_index) {
              uint64_t local_dst =
                  dst_id - (grid)->rank_in_col *
                               (this->sp_local_receiver)->proc_col_width;
              int target_rank =
                  (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
              bool fetch_from_cache =
                  target_rank == (grid)->rank_in_col ? false : true;

              DENT forceDiff[embedding_dim];
              std::array<DENT, embedding_dim> array_ptr;
              if (fetch_from_cache) {
                unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
                    &arrayMap =
                        (temp_cache)
                            ? (*this->dense_local->tempCachePtr)[target_rank]
                            : (*this->dense_local->cachePtr)[target_rank];
                array_ptr = arrayMap[dst_id].value;
              }

              DENT attrc = 0;
             hipgraph::distviz::knng::MathOp<SPT,DENT> mapOp;
             pair<DENT,vector<DENT>> distance_gradient =  mapOp.euclidean_grad((this->dense_local)->nCoordinates+(i * embedding_dim),
                             (this->dense_local)->nCoordinates+(local_dst * embedding_dim),embedding_dim);
             DENT low_dim_distance = distance_gradient.first;
             vector<DENT> grd  = distance_gradient.second;

//              for (int d = 0; d < embedding_dim; d++) {
//                if (!fetch_from_cache) {
//                  forceDiff[d] =
//                      (this->dense_local)->nCoordinates[i * embedding_dim + d] -
//                      (this->dense_local)
//                          ->nCoordinates[local_dst * embedding_dim + d];
//
//                } else {
//                  forceDiff[d] =
//                      (this->dense_local)->nCoordinates[i * embedding_dim + d] -
//                      array_ptr[d];
//                }
//                //              forceDiff[d] = forceDiff[d] * exp(-1 * distance / smoothe_factor);
//                attrc += forceDiff[d] * forceDiff[d];
//              }

//              DENT d1 = -2.0 / (1.0 + attrc);
              DENT w_l=1.0;
//              DENT a = 1.0;
//              DENT b = 1.0;
              if (low_dim_distance > 0.0)
              {
                      w_l = pow((1 + a * pow(low_dim_distance, 2 * b)), -1);
              }
              DENT grad_coeff = 2 * b * (w_l - 1) / (low_dim_distance + 1e-6);
              for (int d = 0; d < embedding_dim; d++) {
//                DENT l = scale(forceDiff[d] * d1);
                DENT grad_d = scale(grad_coeff * grd[d]);
                //              DENT l = (forceDiff[d] * d1 );
                //              l=l*exp(-1*distance/smoothe_factor);
                (*prevCoordinates)[index * embedding_dim + d] =
                    (*prevCoordinates)[index * embedding_dim + d] + (lr)*grad_d;
              }
            }
          }

        }
      }
    }
  }

  inline void calc_t_dist_replus_rowptr(
      vector<DENT> *prevCoordinates, vector<SPT> &col_ids, DENT lr,
      int batch_id, int batch_size, int block_size) {

    int row_base_index = batch_id * batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < block_size; i++) {
      uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
      DENT forceDiff[embedding_dim];
      for (int j = 0; j < col_ids.size(); j++) {
        uint64_t global_col_id = col_ids[j];
        uint64_t local_col_id =
            global_col_id -
            static_cast<uint64_t>(((grid)->rank_in_col *
                                   (this->sp_local_receiver)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank = static_cast<int>(
            global_col_id / (this->sp_local_receiver)->proc_row_width);

        if (owner_rank != (grid)->rank_in_col) {
          fetch_from_cache = true;
        }

        DENT repuls = 0;
        if (fetch_from_cache) {
          unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>> &arrayMap =
              (*this->dense_local->tempCachePtr)[owner_rank];
          std::array<DENT, embedding_dim> &colvec =
              arrayMap[global_col_id].value;

          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                colvec[d];
            repuls += forceDiff[d] * forceDiff[d];
          }
        } else {
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                (this->dense_local)
                    ->nCoordinates[local_col_id * embedding_dim + d];

            repuls += forceDiff[d] * forceDiff[d];
          }
        }
        DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
        for (int d = 0; d < embedding_dim; d++) {
          forceDiff[d] = scale(forceDiff[d] * d1);
          (*prevCoordinates)[i * embedding_dim + d] += (lr)*forceDiff[d];
        }
      }
    }
  }


  inline void calc_t_dist_replus_rowptr_new(
      vector<DENT> *prevCoordinates, vector<vector<vector<SPT>>> *negative_samples_ptr,
      CSRHandle<SPT,DENT> *csr_handle,DENT lr,
      int batch_id, int batch_size, int block_size,double a=1.0, double b=1.0) {

    int row_base_index = batch_id * batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < block_size; i++) {
      uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
      DENT forceDiff[embedding_dim];
      for(int k=0;k<(*negative_samples_ptr)[i].size();k++) {
        for (int j = 0; j < (*negative_samples_ptr)[i][k].size(); j++) {
          SPT global_col_id = (*negative_samples_ptr)[i][k][j];
          SPT local_col_id =
              global_col_id - static_cast<SPT>(
                                  ((grid)->rank_in_col *
                                   (this->sp_local_receiver)->proc_row_width));
          bool fetch_from_cache = false;

          int owner_rank = static_cast<int>(
              global_col_id / (this->sp_local_receiver)->proc_row_width);

          if (owner_rank != (grid)->rank_in_col) {
            fetch_from_cache = true;
          }

          DENT repuls = 0;

          hipgraph::distviz::knng::MathOp<SPT,DENT> mapOp;
          pair<DENT,vector<DENT>> distance_gradient =  mapOp.euclidean_grad((this->dense_local)->nCoordinates+(row_id * embedding_dim),
                                                                            (this->dense_local)->nCoordinates+(local_col_id * embedding_dim),embedding_dim);
          DENT low_dim_distance = distance_gradient.first;
          vector<DENT> grd  = distance_gradient.second;

//          if (fetch_from_cache) {
//            unordered_map<uint64_t , CacheEntry<DENT, embedding_dim>> &arrayMap =
//                (*this->dense_local->tempCachePtr)[owner_rank];
//            std::array<DENT, embedding_dim> &colvec =
//                arrayMap[global_col_id].value;
//
//            for (int d = 0; d < embedding_dim; d++) {
//              forceDiff[d] = (this->dense_local)
//                                 ->nCoordinates[row_id * embedding_dim + d] -
//                             colvec[d];
//              repuls += forceDiff[d] * forceDiff[d];
//            }
//          } else {
//            for (int d = 0; d < embedding_dim; d++) {
//              forceDiff[d] =
//                  (this->dense_local)
//                      ->nCoordinates[row_id * embedding_dim + d] -
//                  (this->dense_local)
//                      ->nCoordinates[local_col_id * embedding_dim + d];
//
//              repuls += forceDiff[d] * forceDiff[d];
//            }
//          }
          DENT w_l=1.0;
          DENT gamma = 1.0;
          if (low_dim_distance > 0.0)
          {
            w_l = pow((1 + a * pow(low_dim_distance, 2 * b)), -1);
          }
          DENT grad_coeff = gamma* 2 * b * w_l / (low_dim_distance + 1e-6);
          for (int d = 0; d < embedding_dim; d++) {
            DENT grad_d = scale(grad_coeff * grd[d]);
            (*prevCoordinates)[i * embedding_dim + d] += (lr)*grad_d;
          }
        }
      }
    }
  }

  inline DENT update_data_matrix_rowptr(vector<DENT> *prevCoordinates,
                                        int batch_id, int batch_size) {

    int row_base_index = batch_id * batch_size;
    int end_row = std::min((batch_id + 1) * batch_size,
                           ((this->sp_local_receiver)->proc_row_width));
    DENT total_error = 0;
#pragma omp parallel for schedule(static) reduction(+ : total_error)
    for (int i = 0; i < (end_row - row_base_index); i++) {
      DENT error = 0;
      for (int d = 0; d < embedding_dim; d++) {
        DENT val = (*prevCoordinates)[i * embedding_dim + d];
        (dense_local)->nCoordinates[(row_base_index + i) * embedding_dim + d] +=
            (*prevCoordinates)[i * embedding_dim + d];

        error += ((val) * (val));
      }
      total_error += sqrt(error);
    }
    return total_error;
  }

  double smooth_knn_distance(int node_index, int nn,CSRHandle<SPT, DENT> *csr_handle = nullptr) {
    double lo = 0.0;
    double hi = 1.0 * INT_MAX;
    double tolerance = 0.001;
    double target = log2(nn);
    double mid = 1.0;
    double value = 0;
    DENT max_value=std::numeric_limits<DENT>::min();
    DENT distance_sum=0;
    DENT mean_distance =0;
    DENT MIN_K_DIST_SCALE=1e-3;
    if (sigma_cache[node_index] > -1) {
      return sigma_cache[node_index];
    }
    do {
      value = 0;
      if (csr_handle != nullptr) {

        for (uint64_t j =static_cast<uint64_t>(csr_handle->rowStart[node_index]);
             j < static_cast<uint64_t>(csr_handle->rowStart[node_index + 1]);j++) {
          if (max_value< csr_handle->values[j]){
            distance_sum += csr_handle->values[j];
          }
        }
        mean_distance = distance_sum/( static_cast<uint64_t>(csr_handle->rowStart[node_index + 1]) - static_cast<uint64_t>(csr_handle->rowStart[node_index]));
        minimum_dis_cache[node_index]=max_value;
        for (uint64_t j =static_cast<uint64_t>(csr_handle->rowStart[node_index]);
             j < static_cast<uint64_t>(csr_handle->rowStart[node_index + 1]);j++) {
//         auto distance = max(static_cast<double>(0.0), (static_cast<double>(csr_handle->values[j])-static_cast<double>(max_value)));
         auto distance = static_cast<double>(csr_handle->values[j])-static_cast<double>(max_value);
         if (distance>0) {
              value += exp(-1 * distance / mid);
           } else {
              value +=1;
           }
        }
      }
      if (abs(target - value) <= tolerance) {
        sigma_cache[node_index] = mid;
          if (sigma_cache[node_index]<MIN_K_DIST_SCALE*mean_distance){
               sigma_cache[node_index] = MIN_K_DIST_SCALE * mean_distance;
          }
        return mid;
      }
      if (value > target) {
        hi = mid;
        mid = (lo + hi) / 2;
      } else {
        lo = mid;
        if (hi == INT_MAX) {
          mid *= 2;
        } else {
          mid = (lo + hi) / 2;
        }
      }
    } while (true);
  }


  void calculate_membership_strength(CSRHandle<SPT, DENT> *csr_handle){
    auto source_start_index = 0;
    auto source_end_index =this->sp_local_receiver->proc_row_width;

    #pragma omp parallel for schedule(static)
    for(SPT i=source_start_index;i<source_end_index;i++){
        int nn = csr_handle->rowStart[i+1]- csr_handle->rowStart[i];
        double sigma = smooth_knn_distance(i,nn,csr_handle);
        double value=1.0;
        for(uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++){
          if ((csr_handle->values[j]-minimum_dis_cache[j])<=0 or sigma==0){
            value = 1.0;
          }else {
            value = exp(-1*(csr_handle->values[j]-minimum_dis_cache[j])/sigma);

          }
          csr_handle->values[j]=value;
        }
    }
  }

  void make_epochs_per_sample(CSRHandle<SPT, DENT> *csr_handle, int iterations, int ns){
    auto source_start_index = 0;
    auto source_end_index =this->sp_local_receiver->proc_row_width;
    DENT maxElement = *std::max_element(csr_handle->values.begin(), csr_handle->values.end());
    #pragma omp parallel for schedule(static)
    for(SPT i=source_start_index;i<source_end_index;i++) {
      int nn = csr_handle->rowStart[i + 1] - csr_handle->rowStart[i];
      if (nn > 0) {
        samples_per_epoch[i].resize(nn, -1.0);
        samples_per_epoch_next[i].resize(nn, -1.0);
        samples_per_epoch_negative[i].resize(nn, -1.0 / ns);
        samples_per_epoch_negative_next[i].resize(nn, -1.0 / ns);
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          DENT value = csr_handle->values[j];
          DENT n_samples = static_cast<DENT>(iterations) * (value / maxElement);
          if (n_samples > 0) {
            int index = j - static_cast<int>(csr_handle->rowStart[i]);
            samples_per_epoch[i][index] =
                static_cast<DENT>(iterations) / n_samples;
            samples_per_epoch_next[i][index] =
                static_cast<DENT>(samples_per_epoch[i][index]);
            samples_per_epoch_negative[i][index] =
                static_cast<DENT>(samples_per_epoch[i][index] / ns);
            samples_per_epoch_negative_next[i][index] =
                static_cast<DENT>(samples_per_epoch_negative[i][index]);
          }
        }
      }else {
        cout<<" i "<<i<<" zero nn"<<nn<<endl;
      }
    }
  }


  void generate_negative_samples(vector<vector<vector<SPT>>> *negative_samples_ptr,CSRHandle<SPT, DENT> *csr_handle, int iteration,
                                 int batch_id, int batch_size, int block_size, int seed) {
    auto source_start_index = batch_id * batch_size;
    auto source_end_index = std::min((batch_id + 1) * batch_size,
                                     this->sp_local_receiver->proc_row_width);
    #pragma omp parallel for schedule(static)
    for(int i=source_start_index;i<source_end_index;i++){
      int nn = csr_handle->rowStart[i+1]- csr_handle->rowStart[i];
      int access_index = i-source_start_index;
      (*negative_samples_ptr)[access_index].resize(nn,vector<SPT>());
      for(uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
           j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
        int index = j - static_cast<int>(csr_handle->rowStart[i]);
        if (samples_per_epoch_next[i][index] <= iteration+1) {

          int ns = (iteration - samples_per_epoch_negative_next[i][index]) /samples_per_epoch_negative[i][index];
          if (ns > 0) {
            seed += i+index+rand();
            vector<SPT> random_number_vec = generate_random_numbers<SPT>(
                0, (this->sp_local_receiver)->gRows, seed, ns);
            (*negative_samples_ptr)[access_index][index] = random_number_vec;
            samples_per_epoch_negative_next[i][index] += ns * samples_per_epoch_negative[i][index];
          }
          samples_per_epoch_next[i][index] += samples_per_epoch[i][index];
        }
      }
    }
  }


    void apply_set_operations(bool apply_set_operations, float set_op_mix_ratio,
                              std::vector<int>& row_offsets, std::vector<int>& col_indices,
                              std::vector<float>& values) {
      if (apply_set_operations) {
        int numRows = row_offsets.size() - 1;

        // Create sparse matrix in CSR format
        Eigen::SparseMatrix<float> csrMatrix(numRows, numRows);
        for (int i = 0; i < numRows+1; ++i) {
          for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
            csrMatrix.coeffRef(i, col_indices[j]) = values[j];
          }
        }

        // Transpose the CSR matrix
        Eigen::SparseMatrix<float> csrTranspose = csrMatrix.transpose();

        // Multiply csrMatrix with its transpose
        Eigen::SparseMatrix<float> prodMatrix = csrMatrix.cwiseProduct(csrTranspose);

        // Compute result = set_op_mix_ratio * (result + transpose - prod_matrix) + (1.0 - set_op_mix_ratio) * prod_matrix
        Eigen::SparseMatrix<float> tempMatrix = csrMatrix + csrTranspose - prodMatrix;
//        Eigen::SparseMatrix<float> result = set_op_mix_ratio * tempMatrix + (1.0 - set_op_mix_ratio) * prodMatrix;

        int rows = tempMatrix.rows();
        int cols = tempMatrix.cols();
        int nnz = tempMatrix.nonZeros();

        col_indices.resize(nnz);
        values.resize(nnz);

        std::copy(tempMatrix.outerIndexPtr(), tempMatrix.outerIndexPtr() + rows + 1, row_offsets.begin());
        std::copy(tempMatrix.innerIndexPtr(), tempMatrix.innerIndexPtr() + nnz, col_indices.begin());
        std::copy(tempMatrix.valuePtr(), tempMatrix.valuePtr() + nnz, values.begin());

      }
    }

    Eigen::SparseMatrix<float> computeTopKEigenvectorsFromLaplacian(
        const std::vector<int>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<float>& values,
        int k) {
      int num_nodes = row_offsets.size()-1;
     // Step 1: Compute the normalized Laplacian matrix
      Eigen::SparseMatrix<float> normalized_laplacian = computeNormalizedLaplacianFromCSR(row_offsets, col_indices, values, num_nodes);

//      // Step 2: Compute the eigenvectors of the Laplacian matrix
//      Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<float>> solver(normalized_laplacian);
//      Eigen::MatrixXf laplacian_eigenvectors_dense = solver.eigenvectors();
//      Eigen::VectorXf eigenvalues = solver.eigenvalues();
//
//      // Step 3: Sort eigenvalues and eigenvectors
//      std::vector<int> order(num_nodes);
//      std::iota(order.begin(), order.end(), 0);
//      std::sort(order.begin(), order.end(), [&](int i, int j) { return eigenvalues(i) < eigenvalues(j); });
//
//      // Step 4: Select the top k eigenvectors
      Eigen::SparseMatrix<float> top_k_eigenvectors(num_nodes, k);
//      for (int i = 0; i < k; ++i) {
//        Eigen::VectorXf eigenvector = laplacian_eigenvectors_dense.col(order[i]);
//        Eigen::SparseVector<float> sparse_eigenvector(num_nodes);
//        sparse_eigenvector.resize(num_nodes);
//        for (int j = 0; j < num_nodes; ++j) {
//          sparse_eigenvector.coeffRef(j) = eigenvector(j);
//        }
//        top_k_eigenvectors.col(i) = sparse_eigenvector;
//      }

      return top_k_eigenvectors;
    }


    Eigen::SparseMatrix<float> computeNormalizedLaplacianFromCSR(
        const std::vector<int>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<float>& values,
        int num_nodes) {

      // Step 1: Compute the degree vector
      Eigen::VectorXd degree_vector(num_nodes);
//      #pragma  omp parallel for
      for (int i = 0; i < num_nodes; ++i) {
        int degree = row_offsets[i + 1] - row_offsets[i];
        degree_vector(i) = degree;
      }

      // Step 2: Compute the degree matrix
      Eigen::SparseMatrix<float> degree_matrix(num_nodes, num_nodes);
      degree_matrix.reserve(Eigen::VectorXi::Constant(num_nodes, 1));
//      #pragma  omp parallel for
      for (int i = 0; i < num_nodes; ++i) {
        degree_matrix.insert(i, i) = degree_vector(i);
      }
      degree_matrix.makeCompressed();

      // Step 3: Compute the unnormalized Laplacian matrix
      Eigen::SparseMatrix<float> adjacency_matrix(num_nodes, num_nodes);
      adjacency_matrix.reserve(Eigen::VectorXi::Constant(num_nodes, 1));
//      #pragma  omp parallel for
      for (int i = 0; i < num_nodes; ++i) {
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
          int col_index = col_indices[j];
          adjacency_matrix.insert(i, col_index) = values[j];
        }
      }
      adjacency_matrix.makeCompressed();

      Eigen::SparseMatrix<float> unnormalized_laplacian = degree_matrix - adjacency_matrix;

      // Step 4: Compute the normalized Laplacian matrix
      Eigen::SparseMatrix<float> degree_sqrt_inv = degree_matrix;
//      #pragma  omp parallel for
      for (int i = 0; i < num_nodes; ++i) {
        float degree_sqrt_inv_value = 1.0 / std::sqrt(degree_vector(i));
        degree_sqrt_inv.coeffRef(i, i) = degree_sqrt_inv_value;
      }
      Eigen::SparseMatrix<float> normalized_laplacian = degree_sqrt_inv * unnormalized_laplacian * degree_sqrt_inv;

      return normalized_laplacian;
    }


};
} // namespace hipgraph::distviz::embedding
