#include "lib/Mrpt.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;

int main(int argc, char* argv[]) {

//  int n = 10000, d = 200, k = 10;
  double target_recall = 0.9;
//  Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
//  Eigen::MatrixXf q = Eigen::VectorXf::Random(d);

  Eigen::VectorXi indices(k), indices_exact(k);

  Mrpt::exact_knn(q, X, k, indices_exact.data());
  std::cout << indices_exact.transpose() << std::endl;

  Mrpt mrpt(X);
  mrpt.grow_autotune(target_recall, k);

  mrpt.query(q, indices.data());
  std::cout << indices.transpose() << std::endl;

}
