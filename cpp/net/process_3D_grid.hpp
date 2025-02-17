#pragma once
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

namespace hipgraph::distviz::net {
/**
 * This class provides 3D processor grid for different processor layouts.
 */
class Process3DGrid {
private:
  int adjacency;
  int dim_list[3];
  int permutation[3];

  void get_ijk_indices(int global_rank, int *i, int *j, int *k);

  template <typename T> void prettyPrint(vector<T> &input) {
    for (int k = 0; k < nl; k++) {
      cout << "========= Layer " << k << " ==========" << endl;

      for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
          cout << input[get_global_rank(i, j, k)] << "\t";
        }
        cout << endl;
      }

      cout << "============================" << endl;
    }
  }

public:
  int global_rank;
  int i, j, k;
  int nr, nc, nl;
  MPI_Comm row_world, col_world, fiber_world, global;
  MPI_Comm rowcol_slice, rowfiber_slice, colfiber_slice;
  int rank_in_row, rank_in_col, rank_in_fiber;
  int world_size, row_world_size, col_world_size;

  /**
   *
   * @param nr Number of rows in processor grid
   * @param nc Number of columns in processor grid
   * @param nl Number of layers in processor grid
   * @param adjacency determine how to consider (row,col,layer), 6  combinations
   * are considered
   */
  Process3DGrid(int nr, int nc, int nl, int adjacency);

  ~Process3DGrid();

  /**
   * Give processor indices in Grid
   * @param i
   * @param j
   * @param k
   */
  void get_ijk_indices(int *i, int *j, int *k);

  /**
   * Get the global rank from processor coordinates
   * @param i
   * @param j
   * @param k
   * @return
   */
  int get_global_rank(int i, int j, int k);

  /**
   * Print the rank information of the calling processor respective the grid
   * positions
   */
  void print_rank_information();

  /**
   *
   * @param title
   * @param msg
   */
  void gather_and_pretty_print(string title, int msg);
};
} // namespace distblas::core
