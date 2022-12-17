#include "optimizer.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "data.h"

int main() {
  // Read data
  // std::string file_name = "../../abalone_scale.txt";
  std::string file_name = "../../bodyfat_scale.txt";
  // std::string file_name = "../../housing_scale.txt";
  // std::string file_name = "../../bodyfat.txt";
  std::vector<std::vector<double>> X;
  std::vector<double> Y;

  read_lib_svm_data(file_name, X, Y);
  // print_data(X, Y);
  double regu_weight = 0;

  Options options;
  options.use_ceres = true;
  options.use_ceres = false;
  options.num_params = X[0].size() + 1;
  std::vector<double> params(options.num_params, 0);
  Optimizer optimizer(X, Y, options);

  if (options.use_ceres) {
    optimizer.SolveByCeres();
  } else {
    optimizer.SolveProblem();
  }

  optimizer.Fetch(params);
  optimizer.ShowResult();

  check_data(X, Y, params);
}