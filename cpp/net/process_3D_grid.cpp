#include "process_3D_grid.hpp"
#include <cassert>

using namespace hipgraph::distviz::net;

// This clas is taken from: https://github.com/PASSIONLab/distributed_sddmm
Process3DGrid::Process3DGrid(int nr, int nc, int nl, int adjacency) {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  global = MPI_COMM_WORLD;

  assert(nr * nc * nl == world_size);

  dim_list[0] = nr;
  dim_list[1] = nc;
  dim_list[2] = nl;
  this->nr = nr;
  this->nc = nc;
  this->nl = nl;
  this->adjacency = adjacency;

  switch (adjacency) {
  case 1:
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;
    break;
  case 2:
    permutation[0] = 0;
    permutation[1] = 2;
    permutation[2] = 1;
    break;
  case 3:
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;
    break;
  case 4:
    permutation[0] = 1;
    permutation[1] = 2;
    permutation[2] = 0;
    break;
  case 5:
    permutation[0] = 2;
    permutation[1] = 0;
    permutation[2] = 1;
    break;
  case 6:
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;
    break;
  }

  get_ijk_indices(&i, &j, &k);

  assert(global_rank == get_global_rank(i, j, k));

  // Create subcommunicators for row, column, fiber worlds; we can
  // ignore the permutation here since we're just chunking up the grid.
  MPI_Comm_split(MPI_COMM_WORLD, i + k * nr, j, &row_world);
  MPI_Comm_split(MPI_COMM_WORLD, j + k * nc, i, &col_world);
  MPI_Comm_split(MPI_COMM_WORLD, i + j * nr, k, &fiber_world);

  // Create subcommunicators for all slices. TODO: We should really use
  // the permutation to order the processes within a slice.
  MPI_Comm_split(MPI_COMM_WORLD, k, i + j * nr, &rowcol_slice);
  MPI_Comm_split(MPI_COMM_WORLD, j, i + k * nr, &rowfiber_slice);
  MPI_Comm_split(MPI_COMM_WORLD, i, j + k * nc, &colfiber_slice);

  MPI_Comm_rank(row_world, &rank_in_row);
  MPI_Comm_rank(col_world, &rank_in_col);
  MPI_Comm_rank(fiber_world, &rank_in_fiber);

  MPI_Comm_size(row_world, &row_world_size);
  MPI_Comm_size(col_world, &col_world_size);
}

Process3DGrid::~Process3DGrid() {
  int mpi_finalized;

  MPI_Finalized(&mpi_finalized);

  if (!mpi_finalized) {
    MPI_Comm_free(&row_world);
    MPI_Comm_free(&col_world);
    MPI_Comm_free(&fiber_world);
    MPI_Comm_free(&rowcol_slice);
    MPI_Comm_free(&rowfiber_slice);
    MPI_Comm_free(&colfiber_slice);
  }
}

void Process3DGrid::gather_and_pretty_print(string title, int msg) {
  vector<int> buff(world_size, 0.0);
  MPI_Gather(&msg, 1, MPI_INT, buff.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (global_rank == 0) {
    cout << title << endl;
    prettyPrint(buff);
  }
}

int Process3DGrid::get_global_rank(int i, int j, int k) {
  int global_rank = 0;

  int ijk_temp[3];
  ijk_temp[0] = i;
  ijk_temp[1] = j;
  ijk_temp[2] = k;

  global_rank += ijk_temp[permutation[0]];
  global_rank += ijk_temp[permutation[1]] * dim_list[permutation[0]];
  global_rank += ijk_temp[permutation[2]] * dim_list[permutation[0]] *
                 dim_list[permutation[1]];

  return global_rank;
}

void Process3DGrid::get_ijk_indices(int *i, int *j, int *k) {
  get_ijk_indices(global_rank, i, j, k);
}

void Process3DGrid::get_ijk_indices(int global_rank, int *i, int *j, int *k) {
  int ijk_temp[3];
  ijk_temp[permutation[0]] = global_rank % dim_list[permutation[0]];
  ijk_temp[permutation[1]] =
      (global_rank / dim_list[permutation[0]]) % dim_list[permutation[1]];
  ijk_temp[permutation[2]] =
      (global_rank / (dim_list[permutation[0]] * dim_list[permutation[1]])) %
      dim_list[permutation[2]];

  *i = ijk_temp[0];
  *j = ijk_temp[1];
  *k = ijk_temp[2];
}

void Process3DGrid::print_rank_information() {
  cout << "Global Rank: " << global_rank << "i, j, k: (" << i << ", " << j
       << ", " << k << ")" << endl;
}