#pragma once
#include "../common//common.h"
#include "../embedding/dense_mat.hpp"
#include "../embedding/sparse_mat.hpp"
#include "process_3D_grid.hpp"
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>
#include <thread>
#include <chrono>

using namespace hipgraph::distviz::common;
using namespace hipgraph::distviz::embedding;

namespace hipgraph::distviz::net {

/**
 * This class represents the data transfer related operations across processes
 * based on internal data connectivity patterns. This implementation is tightly coupled
 * with 1D partitioning of Sparse Matrix and 1D partitioning of Dense Matrix.
 */

template <typename SPT, typename DENT, size_t embedding_dim> class DataComm {

private:
  SpMat<SPT,DENT> *sp_local_receiver;
  SpMat<SPT,DENT> *sp_local_sender;
  DenseMat<SPT, DENT, embedding_dim> *dense_local;
  Process3DGrid *grid;
  vector<int> sdispls;
  vector<int> sendcounts;
  vector<int> rdispls;
  vector<int> receivecounts;
  vector<unordered_set<uint64_t>> receive_col_ids_list;
  vector<unordered_set<uint64_t>> send_col_ids_list;
  unordered_map<uint64_t, unordered_map<int,bool>> send_indices_to_proc_map;
  unordered_map<uint64_t, unordered_map<int,bool>> receive_indices_to_proc_map;

  int batch_id;

  double alpha;

public:
  DataComm(SpMat<SPT,DENT> *sp_local_receiver,
           SpMat<SPT,DENT> *sp_local_sender,
           DenseMat<SPT, DENT, embedding_dim> *dense_local, Process3DGrid *grid,
           int batch_id, double alpha) {
    this->sp_local_receiver = sp_local_receiver;
    this->sp_local_sender = sp_local_sender;
    this->dense_local = dense_local;
    this->grid = grid;
    this->sdispls = vector<int>(grid->world_size, 0);
    this->sendcounts = vector<int>(grid->world_size, 0);
    this->rdispls = vector<int>(grid->world_size, 0);
    this->receivecounts = vector<int>(grid->world_size, 0);
    this->send_counts_cyclic = vector<int>(grid->world_size, 0);
    this->receive_counts_cyclic = vector<int>(grid->world_size, 0);
    this->sdispls_cyclic = vector<int>(grid->world_size, 0);
    this->rdispls_cyclic = vector<int>(grid->world_size, 0);
    this->receive_col_ids_list = vector<unordered_set<uint64_t>>(grid->world_size);
    this->send_col_ids_list = vector<unordered_set<uint64_t>>(grid->world_size);
    this->batch_id = batch_id;
    this->alpha = alpha;

  }

  vector<int> receive_counts_cyclic;
  vector<int> rdispls_cyclic;
  vector<int> send_counts_cyclic;
  vector<int> sdispls_cyclic;

   MPI_Request request = MPI_REQUEST_NULL;

  ~DataComm() {}

  void onboard_data() {

    int total_send_count = 0;
    // processing chunks
    // calculating receiving data cols

    if (alpha==0) {
      // This represents the case for pulling
         this->sp_local_receiver->fill_col_ids(batch_id, 0, grid->col_world_size, receive_col_ids_list,receive_indices_to_proc_map, 0);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,0,grid->col_world_size, send_col_ids_list,send_indices_to_proc_map, 0);
    } else if (alpha == 1.0) {
      // This represents the case for pushing
         this->sp_local_receiver->fill_col_ids(batch_id, 0, grid->col_world_size, receive_col_ids_list,receive_indices_to_proc_map, 1);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,0,grid->col_world_size, send_col_ids_list,send_indices_to_proc_map, 1);
    } else if (alpha> 0 and alpha < 1.0){

      // This represents the case for pull and pushing
          int end_process = get_end_proc(1,alpha, grid->col_world_size);

         this->sp_local_receiver->fill_col_ids(batch_id, 1, end_process, receive_col_ids_list,receive_indices_to_proc_map, 1);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id,1,end_process, send_col_ids_list,send_indices_to_proc_map, 1);

      if (batch_id>=0) {
         this->sp_local_receiver->fill_col_ids(batch_id, end_process, grid->col_world_size, receive_col_ids_list,receive_indices_to_proc_map, 0);

         // calculating sending data cols
         this->sp_local_sender->fill_col_ids(batch_id, end_process,grid->col_world_size, send_col_ids_list,send_indices_to_proc_map, 0);
      }

    } else {
      cout<<"  alpha needs to be in the range of [0,1]"<<endl;
    }

    for (int i = 0; i < grid->world_size; i++) {
        receivecounts[i] = receive_col_ids_list[i].size();
        sendcounts[i] = send_col_ids_list[i].size();
    }
  }

 inline void transfer_data(std::vector<DataTuple<DENT, embedding_dim>> *sendbuf_cyclic,
                            std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                            bool synchronous,MPI_Request* req, int iteration,
                            int batch_id, int starting_proc, int end_proc,
                            bool temp_cache) {

    int total_receive_count = 0;
    vector<int> offset_vector(grid->col_world_size, 0);

    int total_send_count = 0;
    send_counts_cyclic = vector<int>(grid->col_world_size, 0);
    receive_counts_cyclic = vector<int>(grid->col_world_size, 0);
    sdispls_cyclic = vector<int>(grid->col_world_size, 0);
    rdispls_cyclic = vector<int>(grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank = (grid->rank_in_col + i) % grid->col_world_size;
      int receiving_rank =
          (grid->rank_in_col >= i)
              ? (grid->rank_in_col - i) % grid->col_world_size
              : (grid->col_world_size - i + grid->rank_in_col) % grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);
    }

    for (int i = 0; i < sending_procs.size(); i++) {
      send_counts_cyclic[sending_procs[i]] = sendcounts[sending_procs[i]];
      receive_counts_cyclic[receiving_procs[i]] =
          receivecounts[receiving_procs[i]];
      total_send_count += send_counts_cyclic[sending_procs[i]];
      total_receive_count += receive_counts_cyclic[receiving_procs[i]];
    }

    for (int i = 0; i < grid->col_world_size; i++) {
      sdispls_cyclic[i] =
          (i > 0) ? sdispls_cyclic[i - 1] + send_counts_cyclic[i - 1]
                  : sdispls_cyclic[i];
      rdispls_cyclic[i] =
          (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                  : rdispls_cyclic[i];
    }


    if (total_send_count > 0) {
      sendbuf_cyclic->resize(total_send_count);
      for (const auto &pair : DataComm<SPT,DENT,embedding_dim>::send_indices_to_proc_map) {
        auto col_id = pair.first;
        bool already_fetched = false;
        std::array<DENT, embedding_dim> dense_vector;
        for (int i = 0; i < sending_procs.size(); i++) {
          if (pair.second.count(sending_procs[i]) > 0) {
            if (!already_fetched) {
              dense_vector = (this->dense_local)->fetch_local_data(col_id);
              already_fetched = true;
            }
            int offset = sdispls_cyclic[sending_procs[i]];
            int index = offset_vector[sending_procs[i]] + offset;
            (*sendbuf_cyclic)[index].col =
                col_id + (this->sp_local_sender->proc_col_width *
                          this->grid->global_rank);
            (*sendbuf_cyclic)[index].value = dense_vector;
            offset_vector[sending_procs[i]]++;
          }
        }
      }
    }

    if (total_receive_count>0) {
      receivebuf->resize(total_receive_count);
    }

    if (synchronous) {
      auto t = start_clock();
      MPI_Alltoallv((*sendbuf_cyclic).data(), send_counts_cyclic.data(),
                    sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
                    receive_counts_cyclic.data(), rdispls_cyclic.data(),
                    DENSETUPLE, grid->col_world);
      MPI_Request dumy;
      this->populate_cache(sendbuf_cyclic, receivebuf, &dumy, true, iteration, batch_id,temp_cache);
      stop_clock_and_add(t, "Embedding Communication Time");
    }
  }

  void transfer_data(vector<uint64_t> &col_ids, int iteration, int batch_id) {

    vector<vector<uint64_t>> receive_col_ids_list(grid->col_world_size);
    vector<uint64_t> send_col_ids_list;

    int total_send_count = 0;
    int total_receive_count = 0;

    for (int i = 0; i < col_ids.size(); i++) {
      int owner_rank = col_ids[i] / (this->sp_local_receiver)->proc_row_width;
      if (owner_rank == grid->rank_in_col) {
        send_col_ids_list.push_back(col_ids[i]);
      } else {
        receive_col_ids_list[owner_rank].push_back(col_ids[i]);
      }
    }

    for (int i = 0; i < grid->col_world_size; i++) {
       total_send_count = send_col_ids_list.size();
      if (i != grid->rank_in_col) {
        sendcounts[i] = total_send_count;
      } else {
        sendcounts[i] = 0;
      }
      receive_counts_cyclic[i] = receive_col_ids_list[i].size();
    }

    sdispls[0] = 0;
    rdispls_cyclic[0] = 0;
    for (int i = 0; i < grid->col_world_size; i++) {

      sdispls[i] = 0;
      rdispls_cyclic[i] =
          (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                  : rdispls_cyclic[i];
      total_receive_count = total_receive_count + receive_counts_cyclic[i];
    }

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    sendbuf->resize(total_send_count);

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> receivebuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    receivebuf_ptr.get()->resize(total_receive_count);

    for (int j = 0; j < send_col_ids_list.size(); j++) {
      int local_key =
          send_col_ids_list[j] -
          (grid->rank_in_col) * (this->sp_local_receiver)->proc_row_width;
      std::array<DENT, embedding_dim> val_arr =
          (this->dense_local)->fetch_local_data(local_key);
        int index = j;
        (*sendbuf)[index].col = send_col_ids_list[j];
        (*sendbuf)[index].value = val_arr;
    }

    auto t = start_clock();
    MPI_Alltoallv((*sendbuf).data(), sendcounts.data(), sdispls.data(),
                  DENSETUPLE, (*receivebuf_ptr.get()).data(),
                  receive_counts_cyclic.data(), rdispls_cyclic.data(),
                  DENSETUPLE, grid->col_world);
    stop_clock_and_add(t, "Embedding Communication Time");
    MPI_Request dumy;
    this->populate_cache(sendbuf.get(),receivebuf_ptr.get(), &dumy, true, iteration, batch_id,true); // we should not do this

    //    delete[] sendbuf;
  }


  inline void populate_cache(std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
                             std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                      MPI_Request *req, bool synchronous, int iteration,
                      int batch_id, bool temp) {
    if (!synchronous) {
      MPI_Status status;
      auto t = start_clock();
      MPI_Wait(req, &status);
      stop_clock_and_add(t, "Embedding Communication Time");
    }

    for (int i = 0; i < this->grid->col_world_size; i++) {
      int base_index = this->rdispls_cyclic[i];
      int count = this->receive_counts_cyclic[i];

      for (int j = base_index; j < base_index + count; j++) {
        DataTuple<DENT, embedding_dim> t = (*receivebuf)[j];
        (this->dense_local)->insert_cache(i, t.col, batch_id, iteration, t.value, temp);
      }
    }
    receivebuf->clear();
    receivebuf->shrink_to_fit();
    sendbuf->clear();
    sendbuf->shrink_to_fit();

  }


};

} // namespace distblas::net
