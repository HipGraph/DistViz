/**
 * This implements the CSR data structure. We use MKL routines to create CSR
 * from COO.
 */
#pragma once
#include "../common/common.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>
#include <string.h>
#include <set>

using namespace std;
using namespace hipgraph::distviz::common;

namespace hipgraph::distviz::embedding {

template <typename INDEX_TYPE,typename VALUE_TYPE>
class CSRLocal {

public:
  INDEX_TYPE rows, cols;

  INDEX_TYPE max_nnz, num_coords = 0;

  bool transpose;

  unique_ptr<CSRHandle<INDEX_TYPE,VALUE_TYPE>> handler = nullptr;

  CSRLocal() {}

  CSRLocal(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, Tuple<VALUE_TYPE> *coords,
           int num_coords, bool transpose) {
    if (num_coords > 0) {

      this->transpose = transpose;
      this->num_coords = num_coords;
      this->rows = rows;
      this->cols = cols;
      this->handler = unique_ptr<CSRHandle<INDEX_TYPE, VALUE_TYPE>>(
          new CSRHandle<INDEX_TYPE, VALUE_TYPE>());

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      vector<MKL_INT> rArray(num_coords, 0);
      vector<MKL_INT> cArray(num_coords, 0);
      vector<double> vArray(num_coords, 0.0);

      //    cout << " number of coordinates " << num_coords << endl;
      //      #pragma omp parallel for schedule (static)
      for (int i = 0; i < num_coords; i++) {
        rArray[i] = coords[i].row;
        cArray[i] = coords[i].col;
        vArray[i] = static_cast<double>(coords[i].value);
      }

      sparse_operation_t op;

      if (transpose) {
        op = SPARSE_OPERATION_TRANSPOSE;
      } else {
        op = SPARSE_OPERATION_NON_TRANSPOSE;
      }

      sparse_matrix_t tempCOO, tempCSR;

      sparse_status_t status_coo = mkl_sparse_d_create_coo(
          &tempCOO, SPARSE_INDEX_BASE_ZERO, rows, cols, max(num_coords, 1),
          rArray.data(), cArray.data(), vArray.data());

//      if (status_coo != SPARSE_STATUS_SUCCESS) {
//        // Handle the error
//        cout << "Error in coo conversion:" << endl;
//        // Additional error-handling code here
//      }

      cout << " processing transpose before conversion " << transpose << endl;
      sparse_status_t status_csr =
          mkl_sparse_convert_csr(tempCOO, op, &tempCSR);

//      if (status_csr != SPARSE_STATUS_SUCCESS) {
//        // Handle the error
//        cout << "Error in matrix conversion:" << endl;
//        // Additional error-handling code here
//      }

      cout << " processing transpose after conversion " << transpose << endl;
      mkl_sparse_destroy(tempCOO);

      vector<MKL_INT>().swap(rArray);
      vector<MKL_INT>().swap(cArray);
      vector<double>().swap(vArray);

      sparse_index_base_t indexing;
      MKL_INT *rows_start, *rows_end, *col_idx;
      double *values;
      cout << " processing transpose before export " << transpose << endl;
      mkl_sparse_d_export_csr(tempCSR, &indexing, &(this->rows), &(this->cols),
                              &rows_start, &rows_end, &col_idx, &values);

      cout << " processing transpose after export " << transpose <<" size of rows_start"<<(*rows_start).size()<< endl;

      int rv = 0;
      for (int i = 0; i < num_coords; i++) {

        while (rv < this->rows and i >= rows_start[rv + 1]) {
          rv++;
        }
        coords[i].row = rv;
        coords[i].col = col_idx[i];
        coords[i].value = static_cast<VALUE_TYPE>(values[i]);
      }

      cout << " processing coords " << transpose << " max nns" << max_nnz
           << endl;
      //      assert(num_coords <= max_nnz);

      (handler.get())->values.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->row_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->rowStart.resize(this->rows + 1);

      // Copy over row indices
#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_coords; i++) {
        (handler.get())->row_idx[i] = coords[i].row;
      }
      cout << " processing transpose before memcopy export " << transpose
           << endl;
      memcpy((handler.get())->values.data(), values,
             sizeof(double) * max(num_coords, 1));
      memcpy((handler.get())->col_idx.data(), col_idx,
             sizeof(MKL_INT) * max(num_coords, 1));
      memcpy((handler.get())->rowStart.data(), rows_start,
             sizeof(MKL_INT) * this->rows);
      cout << " processing transpose after memcopy export " << transpose
           << endl;
      (handler.get())->rowStart[this->rows] = max(num_coords, 1);

      cout << " processing transpose csr  create " << transpose << endl;
      mkl_sparse_d_create_csr(
          &((handler.get())->mkl_handle), SPARSE_INDEX_BASE_ZERO, this->rows,
          this->cols, (handler.get())->rowStart.data(),
          (handler.get())->rowStart.data() + 1, (handler.get())->col_idx.data(),
          (handler.get())->values.data());
      cout << " processing transpose after csr  create " << transpose << endl;
      mkl_sparse_destroy(tempCSR);
    }
  }


//
//  CSRLocal(INDEX_TYPE rows, INDEX_TYPE cols, INDEX_TYPE max_nnz, Tuple<VALUE_TYPE> *coords,
//           int num_coords, bool transpose) {
//    if (num_coords > 0) {
//
//      this->transpose = transpose;
//      this->num_coords = num_coords;
//      this->rows = rows;
//      this->cols = cols;
//      cout << "rows " << this->rows << " cols " << this->cols << " transpose "
//           << transpose << " nnz " << max_nnz << endl;
//
//      this->handler = unique_ptr<CSRHandle<INDEX_TYPE, VALUE_TYPE>>(
//          new CSRHandle<INDEX_TYPE, VALUE_TYPE>());
//
//      (handler.get())->values.resize(max_nnz == 0 ? 1 : max_nnz);
//      (handler.get())->col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
//      (handler.get())->rowStart.resize((transpose)?this->cols+1:this->rows + 1);
//
//      //    cout << " number of coordinates " << num_coords << endl;
//      //      #pragma omp parallel for schedule (static)
//
//      if (!transpose) {
//        int expected_col=0;
//        int current_row_value=0;
//        set<INDEX_TYPE> dups;
//        int index = 0;
//        for (INDEX_TYPE i = 0; i < num_coords; i++) {
//          if (dups.insert(coords[i].row).second) {
//            if (expected_col==coords[i].row) {
//              (handler.get())->rowStart[index] = current_row_value;
//              index++;
//              expected_col++;
//            }else if (expected_col<coords[i].row){
//              while(expected_col<=coords[i].row){
//                (handler.get())->rowStart[index] = current_row_value;
//                index++;
//                expected_col++;
//              }
//            }
//          }
//          (handler.get())->col_idx[i] = coords[i].col;
//          (handler.get())->values[i] = coords[i].value;
//          current_row_value++;
//        }
//
//        if (expected_col<=this->rows){
//          while(expected_col<=this->rows){
//            (handler.get())->rowStart[index] = current_row_value;
//            index++;
//            expected_col++;
//          }
//        }
//      } else {
//        int expected_col=0;
//        int current_row_value=0;
//        set<INDEX_TYPE> dups;
//        int index = 0;
//        for (INDEX_TYPE i = 0; i < num_coords; i++) {
//          if (dups.insert(coords[i].col).second) {
//            if (expected_col==coords[i].col){
//              (handler.get())->rowStart[index] = current_row_value;
//              index++;
//              expected_col++;
//            }else if (expected_col<coords[i].col){
//              while(expected_col<=coords[i].col){
//                (handler.get())->rowStart[index] = current_row_value;
//                index++;
//                expected_col++;
//              }
//            }
//          }
//          (handler.get())->col_idx[i] = coords[i].row;
//          (handler.get())->values[i] = coords[i].value;
//          current_row_value++;
//        }
//        if (expected_col<=this->cols){
//          while(expected_col<=this->cols){
//            (handler.get())->rowStart[index] = current_row_value;
//            index++;
//            expected_col++;
//          }
//        }
//      }
//    }
//  }

  ~CSRLocal() {
    //    mkl_sparse_destroy((handler.get())->mkl_handle);
  }
};

} // namespace hipgraph::distviz::embedding
