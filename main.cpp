#include "ceres/ceres.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include "read_livsvm_data.h"

namespace Eigen {

typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> Matrix3x4f;
typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Matrix3x4d;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4x4f;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Matrix4x4d;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

} // namespace Eigen

struct RegressionLoss {
  RegressionLoss(std::vector<double> X, double Y, double alfa)
      : X_(X), Y_(Y), alfa_(alfa){};

  bool operator()(const double *const a, double *residual) const {
    for (int i = 0; i < X_.size(); ++i) {
      residual[0] = X_[i] * a[i] - Y_;
      residual[1] = alfa_ * a[i];
      return true;
    }
  };

  const std::vector<double> X_;
  const double Y_;
  const double alfa_;
};

struct RidgeRegressionLoss {
  RidgeRegressionLoss(std::vector<double> x, double y, double w)
      : x_(x), y_(y), w_(w) {}

  template <typename T> bool operator()(const T *const *m, T *residual) const {
    for (int i = 0; i < x_.size(); ++i) {
      // residual[2 * i] = (y_ - m[0][i] * x_[i]) * (y_ - m[0][i] * x_[i]);
      // residual[2 * i + 1] = w_ * m[0][i] * m[0][i];
      residual[2 * i] = y_ - m[0][i] * x_[i];
      residual[2 * i + 1] = w_ * m[0][i];
    }
    return true;
  }

private:
  // Observations for a sample.
  const std::vector<double> x_;
  const double y_;
  const double w_;
};

int main() {
  // Read data
  std::string file_name = "../../abalone_scale.txt";
  // std::string file_name = "../../bodyfat_scale.txt";
  // std::string file_name = "../../housing_scale.txt";
  std::vector<std::vector<double>> X;
  std::vector<double> Y;

  read_lib_svm_data(file_name, X, Y);
  // print_data(X, Y);
  double w = 10;

  int n = X[0].size();
  double **params;
  params = new double *[1];
  params[0] = new double[n];
  double *init_params = new double[n];
  for (int i = 0; i < n; ++i) {
    params[0][i] = rand() / double(RAND_MAX);
    init_params[i] = params[0][i];
  }

  Problem problem;
  for (int i = 0; i < X.size(); ++i) {
    DynamicAutoDiffCostFunction<RidgeRegressionLoss, 4> *cost_function =
        new DynamicAutoDiffCostFunction<RidgeRegressionLoss, 4>(
            new RidgeRegressionLoss(X[i], Y[i], w));
    cost_function->AddParameterBlock(n);
    cost_function->SetNumResiduals(2 * n);
    problem.AddResidualBlock(cost_function, nullptr, params, 1);
  }

  Solver::Options options;
  options.minimizer_type = ceres::LINE_SEARCH;
  // options.line_search_direction_type = ceres::STEEPEST_DESCENT;
  // options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
  options.line_search_direction_type = ceres::LBFGS;
  options.minimizer_progress_to_stdout = true;
  options.linear_solver_type = ceres::DENSE_QR;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  std::cout << "data item num : " << Y.size() << std::endl;
  std::cout << "params num : " << n << std::endl;
  std::cout << "init params : ";
  for (int i = 0; i < n; ++i) {
    std::cout << init_params[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "params : ";
  for (int i = 0; i < n; ++i) {
    std::cout << params[0][i] << " ";
  }
  std::cout << std::endl;

  delete params[0];
  delete params;
  delete init_params;
}