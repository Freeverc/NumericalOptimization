#pragma once

#include "ceres/ceres.h"
#include "data_types.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct RegressionLoss {
  RegressionLoss(std::vector<double> x, double y, double alfa)
      : x_(x), y_(y), alfa_(alfa){};

  bool operator()(const double *const a, double *residual) const {
    residual[0] = x_[x_.size()] - y_;
    residual[1] = 0;
    for (int i = 0; i < x_.size(); ++i) {
      residual[0] += x_[i] * a[i];
      residual[1] += alfa_ * a[i];
      return true;
    }
  };

  const std::vector<double> x_;
  const double y_;
  const double alfa_;
};

struct RidgeRegressionLoss {
  RidgeRegressionLoss(std::vector<double> x, double y, double regu_weight)
      : x_(x), y_(y), regu_weight_(regu_weight) {}

  template <typename T>
  bool operator()(const T *const *params, T *residual) const {
    residual[x_.size()] = y_ - params[0][x_.size()]; // y-b
    for (int i = 0; i < x_.size(); ++i) {
      residual[i] = regu_weight_ * params[0][i];   //-wx
      residual[x_.size()] -= params[0][i] * x_[i]; // a*x
    }
    return true;
  }

private:
  // Observations for a sample.
  const std::vector<double> x_;
  const double y_;
  const double regu_weight_;
};

class Optimizer {
public:
  Optimizer(std::vector<std::vector<double>> x, std::vector<double> y,
            Options options)
      : X_(x), Y_(y), options_(options) {
    params_.resize(options_.num_params);
  };

  ~Optimizer(){};

  void Gradient(std::vector<double> &g, double &cost);

  void Init();

  void Update();

  void UpdateGradientDescent();

  void UpdateConjugateGradient();

  void UpdateQuasiNewton();

  void SolveProblem();

  void SolveByCeres();

  void Fetch(std::vector<double> &a);

  void ShowResult();

private:
  const std::vector<std::vector<double>> X_;
  const std::vector<double> Y_;
  std::vector<double> params_;
  std::vector<double> init_params_;
  Options options_;
};

void Optimizer::Init() {
  std::cout << "Initing... " << std::endl;
  for (int i = 0; i < options_.num_params; ++i) {
    params_[i] = rand() / double(RAND_MAX);
  }

  init_params_ = params_;
}

void Optimizer::Update() {
  if (options_.method == OptimizationMethod::GRADIENT_DECENT) {
    UpdateGradientDescent();
  } else if (options_.method == OptimizationMethod::CONJUGATE_GRADIENT) {
    UpdateConjugateGradient();
  } else if (options_.method == OptimizationMethod::QUASI_NEWTON) {
    UpdateQuasiNewton();
  }
}

void Optimizer::Gradient(std::vector<double> &g, double &cost) {
  for (int i = 0; i < X_.size(); ++i) {         // Data items
    double res = params_[X_[i].size()] - Y_[i]; // bi - yi
    for (int j = 0; j < X_[i].size(); ++j) {    // Params
      res += params_[j] * X_[i][j];             // wj * xij
    }

    cost += 0.5 * res * res;
    for (int j = 0; j < X_[i].size(); ++j) {
      g[j] += 2 * options_.regularization_weight * params_[j]; // -2*alfa*x
      g[j] += res * X_[i][j];                                  // wj * xij * xij
      cost += 0.5 * options_.regularization_weight * params_[j] * params_[j];
    }

    g[X_[i].size()] = res;
  }
}

void Optimizer::UpdateGradientDescent() {
  double cost = 0;
  std::vector<double> g(options_.num_params);
  Gradient(g, cost);

  for (int j = 0; j < params_.size(); ++j) {
    params_[j] += -options_.lr * g[j];
  }

  std::cout << cost << " ";
}

void Optimizer::UpdateConjugateGradient() {
  double cost = 0;
  std::vector<double> g(options_.num_params);
  Gradient(g, cost);

  for (int j = 0; j < options_.num_params; ++j) {
    params_[j] += -options_.lr * g[j];
  }

  std::cout << cost << " ";
}

void Optimizer::UpdateQuasiNewton() {
  double cost = 0;
  std::vector<double> g(options_.num_params);
  Gradient(g, cost);

  for (int j = 0; j < options_.num_params; ++j) {
    params_[j] += -options_.lr * g[j];
  }

  std::cout << cost << " ";
}

void Optimizer::Fetch(std::vector<double> &params) { params = params_; }

void Optimizer::SolveProblem() {
  Init();

  std::cout << "Iterating... " << std::endl;
  for (int i = 0; i < options_.iter; ++i) {
    std::cout << i << " ";
    Update();
    std::cout << options_.lr << std::endl;
    if (i % 50 == 0) {
      options_.lr *= options_.decay_rate;
    }
  }
}

void Optimizer::SolveByCeres() {
  Init();
  int n = options_.num_params;
  double **params_ptr;
  params_ptr = new double *[1];
  params_ptr[0] = new double[n];
  for (int i = 0; i < n; ++i) {
    params_ptr[0][i] = params_[i];
  }

  Problem problem;
  for (int i = 0; i < X_.size(); ++i) {
    DynamicAutoDiffCostFunction<RidgeRegressionLoss, 4> *cost_function =
        new DynamicAutoDiffCostFunction<RidgeRegressionLoss, 4>(
            new RidgeRegressionLoss(X_[i], Y_[i],
                                    options_.regularization_weight));
    cost_function->AddParameterBlock(n + 1);
    cost_function->SetNumResiduals(n + 1);
    problem.AddResidualBlock(cost_function, nullptr, params_ptr, 1);
  }

  Solver::Options ceres_options;
  ceres_options.minimizer_type = ceres::LINE_SEARCH;
  ceres_options.max_num_iterations = 1000;
  ceres_options.min_line_search_step_contraction = 0.6;
  ceres_options.line_search_direction_type = ceres::STEEPEST_DESCENT;
  // ceres_options.line_search_direction_type =
  //     ceres::NONLINEAR_CONJUGATE_GRADIENT;
  // ceres_options.line_search_direction_type = ceres::LBFGS;
  // ceres_options.line_search_direction_type = ceres::BFGS;
  ceres_options.minimizer_progress_to_stdout = true;
  // ceres_options.linear_solver_type = ceres::DENSE_QR;
  Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  for (int i = 0; i < n; ++i) {
    params_[i] = params_ptr[0][i];
  }

  delete[] params_ptr[0];
  delete[] params_ptr;
}

void Optimizer::ShowResult() {
  std::cout << "data item num : " << Y_.size() << std::endl;
  std::cout << "params num : " << options_.num_params << std::endl;
  if (params_.size() != options_.num_params ||
      init_params_.size() != options_.num_params) {
    std::cout << "Result error";
    return;
  }

  std::cout << "init params : ";
  for (int i = 0; i < init_params_.size(); ++i) {
    std::cout << init_params_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "params : ";
  for (int i = 0; i < params_.size(); ++i) {
    std::cout << params_[i] << " ";
  }
  std::cout << std::endl;
}
