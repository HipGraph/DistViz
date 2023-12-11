/**
 * This class represents the implementation of the distributed Embedding Algorithm.
 * This is tightly coupled with 1D partitioning of Sparse Matrix and Dense Matrix.
 */
#pragma once
#include "../common/common.h"
#include "csr_local.hpp"
#include "dense_mat.hpp"
#include "../common/json.hpp"
#include "sparse_mat.hpp"
#include "../net/data_comm.hpp"
#include "../net/process_3D_grid.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>

using namespace std;
using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::net;
using namespace Eigen;

namespace hipgraph::distviz::embedding {
template <typename SPT, typename DENT, size_t embedding_dim>

class EmbeddingAlgo {

protected:
  DenseMat<SPT, DENT, embedding_dim> *dense_local;
  SpMat<SPT,DENT> *sp_local_receiver;
  SpMat<SPT,DENT> *sp_local_sender;
  SpMat<SPT,DENT> *sp_local_native;
  Process3DGrid *grid;
  DENT MAX_BOUND, MIN_BOUND;
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>>
      data_comm_cache;

  //cache size controlling hyper parameter
  double alpha = 1.0;

 //hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  //hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  //hyper parameter controls the col major or row major  data access
  bool col_major = true;

public:
  EmbeddingAlgo(SpMat<SPT,DENT> *sp_local_native,
                SpMat<SPT,DENT> *sp_local_receiver,
                SpMat<SPT,DENT> *sp_local_sender,
                DenseMat<SPT, DENT, embedding_dim> *dense_local,
                Process3DGrid *grid, double alpha, double beta, DENT MAX_BOUND,
                DENT MIN_BOUND, bool col_major, bool sync_comm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), dense_local(dense_local), grid(grid),
        alpha(alpha), beta(beta), MAX_BOUND(MAX_BOUND), MIN_BOUND(MIN_BOUND),
        col_major(col_major),sync(sync_comm) {}

  DENT scale(DENT v) {
    if (v > MAX_BOUND)
      return MAX_BOUND;
    else if (v < -MAX_BOUND)
      return -MAX_BOUND;
    else
      return v;
  }

  vector<DENT> algo_force2_vec_ns(int iterations, int batch_size, int ns, DENT lr, double drop_out_error_threshold=0) {
    int batches = 0;
    int last_batch_size = batch_size;

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

    cout << " rank " << grid->rank_in_col << " onboard data sucessfully for initial iteration" << batches<< endl;

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    //Buffer used for send MPI operations data
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    if (alpha > 0 and grid->col_world_size > 1) {
      MPI_Request fetch_batch_full;
      int alpha_proc_end = get_end_proc(1, alpha, grid->col_world_size);
      full_comm.get()->transfer_data(sendbuf_ptr.get(), update_ptr.get(), true,&fetch_batch_full, 0, 0, 1, alpha_proc_end,false);
      cout << " rank " << grid->rank_in_col << " transfer data completed for initial iteration" << batches<< endl;
    }

    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, dense_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    cout << " rank " << grid->rank_in_col << " onboard_data completed " << batches << endl;

//    DENT *prevCoordinates = static_cast<DENT *>(::operator new(sizeof(DENT[batch_size * embedding_dim])));

    shared_ptr<vector<DENT>> prevCoordinates_ptr = make_shared<vector<DENT>>(batch_size * embedding_dim, DENT{});

    size_t total_memory = 0;

    CSRLocal<SPT,DENT> *csr_block = (col_major) ? sp_local_receiver->csr_local_data.get(): sp_local_native->csr_local_data.get();

    int considering_batch_size = batch_size;

    vector<DENT> error_convergence;
    for (int i = 0; i < iterations; i++) {

      if (alpha > 0 and grid->col_world_size > 1 and i == 0) {

        for (int k = 0; k < batch_size; k += 1) {
          int IDIM = k * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            (*prevCoordinates_ptr)[IDIM + d] = 0;
          }
        }

        int last_proc = this->execute_push_model_computations(
            sendbuf_ptr.get(), update_ptr.get(), 0, 0, batches,
            this->data_comm_cache[0].get(), csr_block, batch_size,
            last_batch_size, considering_batch_size, lr, prevCoordinates_ptr.get(), 0, 0,
            considering_batch_size, false);

        if (alpha < 1.0) {
          int prev_start = get_end_proc(1, alpha, grid->col_world_size);
          this->execute_pull_model_computations(
              sendbuf_ptr.get(), update_ptr.get(), 0, 0,
              this->data_comm_cache[0].get(), csr_block, batch_size,
              considering_batch_size, lr, prevCoordinates_ptr.get(), prev_start, false,
              last_proc, true);
        }
      }

      DENT batch_error =0;
      for (int j = 0; j < batches; j++) {
        int seed = j + i;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }

        // negative samples generation
        vector<uint64_t> random_number_vec = generate_random_numbers(
            0, (this->sp_local_receiver)->gRows, seed, ns);

        // One process computations without MPI operations
        if (grid->col_world_size == 1) {
          for (int k = 0; k < batch_size; k ++) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              (*prevCoordinates_ptr)[IDIM + d] = 0;
            }
          }
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates_ptr.get(), lr, j,
                                        batch_size, considering_batch_size,
                                        true, false, 0, 0, false);

          this->calc_t_dist_replus_rowptr(prevCoordinates_ptr.get(), random_number_vec,
                                          lr, j, batch_size,
                                          considering_batch_size);


          batch_error += this->update_data_matrix_rowptr(prevCoordinates_ptr.get(), j, batch_size);


        } else {
          //These operations are for more than one processes.
          full_comm.get()->transfer_data(random_number_vec, i, j);
          this->calc_t_dist_replus_rowptr(prevCoordinates_ptr.get(), random_number_vec,
                                          lr, j, batch_size,
                                          considering_batch_size);

          //  pull model code
          if (alpha == 0) {
//            cout << " rank " << grid->rank_in_col << " executing pull model " << batches << endl;
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr, prevCoordinates_ptr.get(), 1,
                true, 0, true);

            batch_error += this->update_data_matrix_rowptr(prevCoordinates_ptr.get(), j, batch_size);

            for (int k = 0; k < batch_size; k++) {
              int IDIM = k * embedding_dim;
              for (int d = 0; d < embedding_dim; d++) {
                (*prevCoordinates_ptr)[IDIM + d] = 0;
              }
            }

          } else {
            cout << " rank " << grid->rank_in_col << " executing push model " << batches << endl;
            batch_error += this->update_data_matrix_rowptr(prevCoordinates_ptr.get(), j, batch_size);

            if (!(i == iterations - 1 and j == batches - 1)) {
              // clear up data
              for (int k = 0; k < batch_size; k += 1) {
                int IDIM = k * embedding_dim;
                for (int d = 0; d < embedding_dim; d++) {
                  (*prevCoordinates_ptr)[IDIM + d] = 0;
                }
              }
              int next_batch_id = (j + 1) % batches;
              int next_iteration = (next_batch_id == 0) ? i + 1 : i;
              int next_considering_batch_size = (next_batch_id == batches - 1)
                                                    ? last_batch_size
                                                    : considering_batch_size;

              int last_proc = this->execute_push_model_computations(
                  sendbuf_ptr.get(), update_ptr.get(), i, j, batches,
                  this->data_comm_cache[j].get(), csr_block, batch_size,
                  last_batch_size, considering_batch_size, lr, prevCoordinates_ptr.get(),
                  next_batch_id, next_iteration, next_considering_batch_size,
                  true);

              if (alpha < 1.0) {
                int prev_start = get_end_proc(1, alpha, grid->col_world_size);
                this->execute_pull_model_computations(
                    sendbuf_ptr.get(), update_ptr.get(), next_iteration,
                    next_batch_id, this->data_comm_cache[next_batch_id].get(),
                    csr_block, batch_size, considering_batch_size, lr,
                    prevCoordinates_ptr.get(), prev_start,
                    false, last_proc, true);
              }
            }
          }
        }
      }
      batch_error = batch_error/batches;
      if(drop_out_error_threshold>0){
        if (grid->col_world_size>1){
          DENT global_error =0;
          MPI_Allreduce(&batch_error, &global_error, 1, MPI_VALUE_TYPE, MPI_SUM, grid->col_world);
          global_error = global_error/grid->col_world_size;
          error_convergence.push_back(global_error);
//          if (global_error<=drop_out_error_threshold){
//            cout<<"rank "<<grid->rank_in_col<<" dropping out at"<<i<<endl;
//            break;
//          }
        }else {
//          if (batch_error<=drop_out_error_threshold){
//            cout<<"rank "<<grid->rank_in_col<<" dropping out at"<<i<<endl;
//            break;
//          }
        }
      }
    }
    return error_convergence;
  }

  inline void execute_pull_model_computations(
      std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
      std::vector<DataTuple<DENT, embedding_dim>> *receivebuf, int iteration,
      int batch, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT,DENT> *csr_block, int batch_size, int considering_batch_size,
      double lr, vector<DENT> *prevCoordinates, int comm_initial_start, bool local_execution,
      int first_execution_proc, bool communication) {

    int proc_length = get_proc_length(beta, grid->col_world_size);
    int prev_start = comm_initial_start;
//    cout << " rank " << grid->rank_in_col << " run pull model algorithm "  << endl;
    for (int k = prev_start; k < grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, grid->col_world_size);

      MPI_Request req;

      if (communication) {
        data_comm->transfer_data(sendbuf, receivebuf, sync, &req, iteration,batch, k, end_process, true);
      }

      if (!sync and communication) {
        MPI_Ialltoallv(
            (*sendbuf).data(), data_comm->send_counts_cyclic.data(),
            data_comm->sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
            data_comm->receive_counts_cyclic.data(),
            data_comm->rdispls_cyclic.data(), DENSETUPLE, grid->col_world, &req);
      }

      if (k == comm_initial_start) {
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, batch, batch_size,
            considering_batch_size, local_execution, col_major,
            first_execution_proc, prev_start, local_execution);

      } else if (k > comm_initial_start) {
        int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                      batch_size, considering_batch_size, false,
                                      col_major, prev_start, prev_end_process,
                                      true);
      }

      if (!sync and communication) {
        data_comm->populate_cache(sendbuf, receivebuf, &req, sync, iteration, batch, true);
      }

      prev_start = k;
    }

    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                  batch_size, considering_batch_size, false,
                                  col_major, prev_start, prev_end_process,
                                  true);

    // dense_local->invalidate_cache(i, j, true);
  }

  inline int execute_push_model_computations(
      std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
      std::vector<DataTuple<DENT, embedding_dim>> *receivebuf, int iteration,
      int batch, int batches, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT,DENT> *csr_block, int batch_size, int last_batch_size,
      int considering_batch_size, double lr, vector<DENT> *prevCoordinates,
      int next_batch_id, int next_iteration, int next_considering_batch_size,
      bool communication) {

    int prev_start_proc = 0;
    int alpha_proc_length = get_proc_length(alpha, grid->col_world_size);

    int alpha_cyc_len = get_proc_length(beta, alpha_proc_length);
    int alpha_cyc_end = get_end_proc(1, beta, alpha_proc_length);

    for (int k = 1; k < alpha_proc_length; k += alpha_cyc_len) {
      MPI_Request request_batch_update;

      if (communication) {
        data_comm->transfer_data(sendbuf, receivebuf, sync,
                                 &request_batch_update, iteration, batch, k,
                                 (k + alpha_cyc_len), false);
      }
      if (!sync and communication) {
        MPI_Ialltoallv((*sendbuf).data(), data_comm->send_counts_cyclic.data(),
                       data_comm->sdispls_cyclic.data(), DENSETUPLE,
                       (*receivebuf).data(),
                       data_comm->receive_counts_cyclic.data(),
                       data_comm->rdispls_cyclic.data(), DENSETUPLE,
                       MPI_COMM_WORLD, &request_batch_update);
      }

      if (k == 1) {
        // local computation for first batch
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, next_batch_id, batch_size,
            next_considering_batch_size, true, col_major, 0, 0, false);
      } else if (k > 1) {
        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr,
                                      next_batch_id, batch_size,
                                      next_considering_batch_size, false,
                                      col_major, prev_start_proc, k, false);
      }

      if (!sync and communication) {
        data_comm->populate_cache(sendbuf, receivebuf, &request_batch_update,
                                  sync, iteration, batch, false);
      }

      prev_start_proc = k;
    }
    if (alpha == 1.0) {
      this->calc_t_dist_grad_rowptr(
          csr_block, prevCoordinates, lr, next_batch_id, batch_size,
          next_considering_batch_size, false, col_major, prev_start_proc,
          alpha_cyc_end, false);
      return alpha_cyc_end;
    }
    return prev_start_proc;
    // dense_local->invalidate_cache(i, j, false);
  }

  inline void
  calc_t_dist_grad_rowptr(CSRLocal<SPT,DENT> *csr_block, vector<DENT> *prevCoordinates,
                          DENT lr, int batch_id, int batch_size, int block_size,
                          bool local, bool col_major, int start_process,
                          int end_process, bool fetch_from_temp_cache) {

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
        calc_embedding_row_major(source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                 prevCoordinates, lr, batch_id, batch_size,
                                 block_size, fetch_from_temp_cache);
      }
    } else {
      for (int r = start_process; r < end_process; r++) {

        if (r != grid->rank_in_col) {

          int computing_rank = (grid->rank_in_col >= r)
                                   ? (grid->rank_in_col - r) % grid->col_world_size
                                   : (grid->col_world_size - r + grid->rank_in_col) % grid->col_world_size;

          dst_start_index = this->sp_local_receiver->proc_row_width * computing_rank;
          dst_end_index =
              std::min(static_cast<uint64_t>(
                           this->sp_local_receiver->proc_row_width * (computing_rank + 1)),
                       this->sp_local_receiver->gCols) -
              1;

          if (col_major) {
            calc_embedding(source_start_index, source_end_index,
                           dst_start_index, dst_end_index, csr_block,
                           prevCoordinates, lr, batch_id, batch_size,
                           block_size, fetch_from_temp_cache);
          } else {
            calc_embedding_row_major(source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                     prevCoordinates, lr, batch_id, batch_size,
                                     block_size, fetch_from_temp_cache);
          }
        }
      }
    }
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT,DENT> *csr_block, vector<DENT> *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle<SPT,DENT> *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        uint64_t local_dst = i - (grid)->rank_in_col *
                                     (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (grid)->rank_in_col ? false : true;


        bool matched = false;
        std::optional<std::array<DENT, embedding_dim>> array_ptr;
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
                    (array_ptr.value()[d]);
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

  inline void
  calc_embedding_row_major(uint64_t source_start_index,
                           uint64_t source_end_index, uint64_t dst_start_index,
                           uint64_t dst_end_index, CSRLocal<SPT,DENT> *csr_block,
                           vector<DENT> *prevCoordinates, DENT lr, int batch_id,
                           int batch_size, int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle<SPT,DENT> *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;

        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
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
              attrc += forceDiff[d] * forceDiff[d];
            }
            attrc  = attrc * (1/(0.0000001+ distance)); //TODO change this UMAP approach later

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

  inline void calc_t_dist_replus_rowptr(vector<DENT> *prevCoordinates,
                                        vector<uint64_t> &col_ids, DENT lr,
                                        int batch_id, int batch_size,
                                        int block_size) {

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
//        repuls = repuls*(1/(1-(distance+0.0000001)));
        DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
        for (int d = 0; d < embedding_dim; d++) {
          forceDiff[d] = scale(forceDiff[d] * d1);
          (*prevCoordinates)[i * embedding_dim + d] += (lr)*forceDiff[d];
        }
      }
    }
  }

  inline DENT update_data_matrix_rowptr(vector<DENT> *prevCoordinates, int batch_id,
                                        int batch_size) {

    int row_base_index = batch_id * batch_size;
    int end_row = std::min((batch_id + 1) * batch_size,
                           ((this->sp_local_receiver)->proc_row_width));
    DENT total_error=0;
#pragma omp parallel for schedule(static) reduction(+:total_error)
    for (int i = 0; i < (end_row - row_base_index); i++) {
      DENT error = 0;
      for (int d = 0; d < embedding_dim; d++) {
       DENT val =  (*prevCoordinates)[i * embedding_dim + d];
        (dense_local)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] += (*prevCoordinates)[i * embedding_dim + d];

        error += ((val)*(val));
      }
      total_error +=sqrt(error);
    }
    return total_error;
  }
};
} // namespace distblas::algo
