#include "file_reader.hpp"
#include <fstream>
#include <iostream>

Eigen::MatrixXf hipgraph::distviz::FileReader::load_data(string file_path, int no_of_images,
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
  magic_number = this->reverse_int(magic_number);
  file.read((char *)&number_of_images, sizeof(number_of_images));
  number_of_images = this->reverse_int(number_of_images);
  file.read((char *)&n_rows, sizeof(n_rows));
  n_rows = this->reverse_int(n_rows);
  file.read((char *)&n_cols, sizeof(n_cols));
  n_cols = this->reverse_int(n_cols);



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
        if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and
            rank < world_size - 1) {
          matrix[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE)temp;
        } else if (rank == world_size - 1 && i >= (rank)*chunk_size) {
          matrix[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE)temp;
        }
      }
    }
  }
  file.close();

  // Create Eigen::MatrixXf
  return matrix;
}
