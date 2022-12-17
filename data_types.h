#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

enum OptimizationMethod {
  GRADIENT_DECENT,
  CONJUGATE_GRADIENT,
  QUASI_NEWTON,
};

struct Options {
  bool use_ceres = false;
  OptimizationMethod method = GRADIENT_DECENT;
  int iter = 2000;
  double lr = 0.002;
  int num_params = 15;
  const double regularization_weight = 1;
  double decay_rate = 0.8;
};

namespace Eigen {

typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> Matrix3x4f;
typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Matrix3x4d;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4x4f;
typedef Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Matrix4x4d;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

} // namespace Eigen