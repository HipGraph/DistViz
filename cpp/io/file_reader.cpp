#include "file_reader.hpp"
#include <iostream>

Eigen::MatrixXf hipgraph::distviz::FileReader::load_data(string file_path) {
  // Read the file
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Error: Could not open the file." << std::endl;
    return Eigen::MatrixXf();
  }

  // Read the dimensions of the matrix (assuming it's a 2D matrix)
  int rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int));

  // Create Eigen::MatrixXf
  Eigen::MatrixXf matrix(rows, cols);

  // Read the data into the matrix
  file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(float));

  // Close the file
  file.close();

  return matrix;

}
