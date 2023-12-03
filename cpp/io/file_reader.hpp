#pragma once
#include "Eigen/Dense"
#include <string>
#include <fstream>
#include <iostream>
#include "../common/common.h"

using namespace std;
using namespace hipgraph::distviz::common;
namespace hipgraph::distviz::io {
template <typename VALUE_TYPE>
class FileReader {

public:
 static Eigen::MatrixXf load_data(string file_path, int no_of_images,
                                   int dimension, int rank, int world_size) {
   // Read the file
   std::ifstream file(file_path, std::ios::binary);

   if (!file.is_open()) {
     std::cout << "Error: Could not open the file." << std::endl;
     return Eigen::MatrixXf();
   }

   int magic_number = 0;
   int number_of_images = 0;
   int n_rows = 0;
   int n_cols = 0;
   file.read((char *)&magic_number, sizeof(magic_number));
   magic_number = reverse_int(magic_number);
   file.read((char *)&number_of_images, sizeof(number_of_images));
   number_of_images = reverse_int(number_of_images);
   file.read((char *)&n_rows, sizeof(n_rows));
   n_rows = reverse_int(n_rows);
   file.read((char *)&n_cols, sizeof(n_cols));
   n_cols = reverse_int(n_cols);


   int chunk_size = number_of_images / world_size;
   if (rank == world_size - 1) {
     chunk_size = no_of_images - chunk_size * (world_size - 1);
   }
   Eigen::MatrixXf matrix(chunk_size, dimension);

   for (int i = 0; i < number_of_images; ++i) {
     for (int r = 0; r < n_rows; ++r) {
       for (int c = 0; c < n_cols; ++c) {
         unsigned char temp = 0;
         file.read((char *)&temp, sizeof(temp));
         if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1) {
           matrix(i - rank * chunk_size,(n_rows * r) + c) = (float)temp;
         } else if (rank == world_size - 1 && i >= (rank)*chunk_size) {
           matrix(i - rank * chunk_size,(n_rows * r) + c) = (float)temp;
         }
       }
     }
   }
   file.close();

   // Create Eigen::MatrixXf
   return matrix;
 }

static void ubyte_read(string file_path,ValueType2DVector<VALUE_TYPE>* datamatrix,
                                      int no_of_datapoints,int dimension, int rank, int world_size){

  ifstream file (file_path, ios::binary);
  if (file.is_open ())
  {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read ((char *) &magic_number, sizeof (magic_number));
    magic_number = reverse_int (magic_number);
    file.read ((char *) &number_of_images, sizeof (number_of_images));
    number_of_images = reverse_int (number_of_images);
    file.read ((char *) &n_rows, sizeof (n_rows));
    n_rows = reverse_int (n_rows);
    file.read ((char *) &n_cols, sizeof (n_cols));
    n_cols = reverse_int (n_cols);

    int chunk_size = number_of_images / world_size;
    if (rank < world_size - 1)
    {
      datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    }
    else if (rank == world_size - 1)
    {
      chunk_size = no_of_datapoints - chunk_size * (world_size - 1);
      datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    }

    for (int i = 0; i < number_of_images; ++i)
    {
      for (int r = 0; r < n_rows; ++r)
      {
        for (int c = 0; c < n_cols; ++c)
        {
          unsigned char temp = 0;
          file.read ((char *) &temp, sizeof (temp));
          if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1)
          {
            (*datamatrix)[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
          }
          else if (rank == world_size - 1 && i >= (rank) * chunk_size)
          {
            (*datamatrix)[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
          }
        }
      }
    }
  }
  file.close ();
}

static void fvecs_read(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                int no_of_datapoints,int dimension, int rank, int world_size) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "I/O error: Unable to open the file " << filename << "\n";
    exit(EXIT_FAILURE);
  }

  // Read the vector size
  int d;
  file.read(reinterpret_cast<char*>(&d), sizeof(int));
  int vecsizeof = 1 * 4 + d * 4;

  int chunk_size = no_of_datapoints / world_size;

  vector<int>bounds = vector<int>(2);
  if (rank < world_size - 1){
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    bounds[0]= rank * chunk_size;
    bounds[1] = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    chunk_size = no_of_datapoints - chunk_size * (world_size - 1);
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dimension));
    bounds[0]= rank * chunk_size;
    bounds[1] = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
  }

  cout<<" rank "<<rank<<" a "<<bounds[0]<<" b "<<bounds[1]<<"d"<<d<<endl;

  // Get the number of vectors
  file.seekg(0, std::ios::end);
  int a = 0;  // Start from 0-based index
  int bmax = file.tellg() / vecsizeof;
  int b = bmax - 1;  // Convert to 0-based index

  if (!bounds.empty()) {
    if (bounds.size() == 1) {
      b = bounds[0] ;  // Convert to 0-based index
    } else if (bounds.size() == 2) {
      a = bounds[0];
      b = bounds[1];  // Convert to 0-based index
    }
  }

  assert(a >= 0);

  if (b == -1 || b < a) {
    return;  // Return an empty vector
  }

  // Compute the number of vectors that are really read and go to starting positions
  int n = b - a + 1;
  file.seekg(a * vecsizeof, std::ios::beg);  // Adjust to 0-based index

  // Read n vectors
  std::vector<float> rawVectors((d + 1) * n);
  file.read(reinterpret_cast<char*>(rawVectors.data()), sizeof(float) * (d + 1) * n);

  // Check if the first column (dimension of the vectors) is correct
//  assert(std::count(rawVectors.begin() + 1, rawVectors.end(), rawVectors[0]) == n - 1);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      (*datamatrix)[i][j] = rawVectors[(d + 1) * i + j + 1];
    }
  }
}

static void  read_fbin(string filename, ValueType2DVector<VALUE_TYPE>* datamatrix,
                      int no_of_datapoints,int dim, int rank, int world_size) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    // Handle file opening error
    std::cerr << "Error: Unable to open the file " << filename << std::endl;
    return {};
  }

  int nvecs, dim;
  file.read(reinterpret_cast<char*>(&nvecs), sizeof(int));
  file.read(reinterpret_cast<char*>(&dim), sizeof(int));

  int chunk_size = no_of_datapoints / world_size;
  int start_idx =0;

  if (rank < world_size - 1){
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dim));
    start_idx = rank * chunk_size;
    chunk_size = (rank+1) * chunk_size -1;
  }else if (rank == world_size - 1){
    chunk_size = no_of_datapoints - chunk_size * (world_size - 1);
    datamatrix->resize (chunk_size, vector<VALUE_TYPE> (dim));
    start_idx = rank * chunk_size;
    chunk_size = std::min((rank+1) * chunk_size -1,no_of_datapoints-1);
  }

  if (chunk_size == -1) {
    chunk_size = nvecs - start_idx;
  }

  std::vector<float> data(chunk_size * dim);

  file.seekg(start_idx * 4 * dim, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * chunk_size * dim);


  for (int i = 0; i < chunk_size; ++i) {
    std::vector<float> vec(dim);
    std::copy(data.begin() + i * dim, data.begin() + (i + 1) * dim, vec.begin());
    (*datamatrix)[i]=vec;
  }

}


static int reverse_int (int i){
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

};
}

