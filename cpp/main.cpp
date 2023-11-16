#include "lib/Mrpt.h"
#include <iostream>
#include "Eigen/Dense"
#include "io/file_reader.hpp"
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {

  string input_path = "";

  for (int p = 0; p < argc; p++)
  {
    if (strcmp(argv[p], "-input") == 0)
    {
      input_path = argv[p + 1];
    }
  }

  int k = 10;
  double target_recall = 0.9;
//  int n = 10000, d = 200, k = 10;
//  double target_recall = 0.9;
//  Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
//  Eigen::MatrixXf q = Eigen::VectorXf::Random(d);


  hipgraph::distviz::FileReader file_reader;
  Eigen::VectorXi indices(k),  indices_exact(k);

//  Eigen::VectorXf distances(k);

  std::cout << "calling data loading"<< std::endl;
  Eigen::MatrixXf X_trans = file_reader.load_data(input_path,60000,784,0,1);

  Eigen::MatrixXf X = X_trans.transpose();

  std::cout << "Number of Rows: " << X.rows() << std::endl;
  std::cout << "Number of Columns: " << X.cols() << std::endl;

  Eigen::MatrixXi neighbours(X.cols(),k);


//  Mrpt::exact_knn(X.row(0), X, k, indices_exact.data());
//  std::cout << indices_exact.transpose() << std::endl;

  auto start_index_buildling = high_resolution_clock::now();
  Mrpt mrpt(X);
  mrpt.grow_autotune(target_recall, k);


  for(int i=0;i<X.cols();i++){
    Eigen::VectorXi tempRow(k);
    mrpt.query(X.col(i), tempRow.data());
    neighbours.row(i) = tempRow;
  }

  auto stop_index_building = high_resolution_clock::now();

  auto duration_index_building = duration_cast<microseconds>(stop_index_building - start_index_buildling);

  std::cout << duration_index_building.count()/1000000 << std::endl;
//  std::cout << distances.transpose() << std::endl;

}
