#include "optimizer.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "data.h"
void run(const std::vector<std::vector<double>> &X,
         const std::vector<double> &Y, Options &options,
         const std::string &log_name) {
  std::vector<double> params(options.num_params, 0);
  Optimizer optimizer(X, Y, options);

  if (options.use_ceres) {
    optimizer.SolveByCeres();
  } else {
    optimizer.SolveProblem();
  }

  optimizer.Fetch(params);
  optimizer.ShowResult();
  optimizer.SaveLog(log_name + std::to_string(options.method) + ".txt");

  check_data(X, Y, params);
}

int main() {
  // Read data
  // std::string file_name = "../data/abalone_scale.txt";
  std::string file_name = "../data/bodyfat_scale.txt";
  // std::string file_name = "../data/housing_scale.txt";
  // std::string file_name = "../data/bodyfat.txt";
  std::vector<std::vector<double>> X;
  std::vector<double> Y;

  read_lib_svm_data(file_name, X, Y);
  // print_data(X, Y);
  double regu_weight = 0;

  Options options;
  options.use_ceres = true;
  // options.use_ceres = false;
  options.num_params = X[0].size() + 1;
  options.method = OptimizationMethod::GRADIENT_DECENT;
  std::string log_name = file_name.replace(file_name.size() - 4, 4, ".log");
  run(X, Y, options, log_name);
  options.method = OptimizationMethod::CONJUGATE_GRADIENT;
  run(X, Y, options, log_name);
  options.method = OptimizationMethod::QUASI_NEWTON;
  run(X, Y, options, log_name);
}