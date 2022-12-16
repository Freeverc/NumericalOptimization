#include "ceres/ceres.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
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

struct nonline_fun {
  // x=2,1
  template <typename T> bool operator()(const T *const x, T *residual) const {
    residual[1] = x[0] * x[0] + x[1] - T(5.0);
    residual[0] = x[0] + x[1] - T(3.0);
    residual[2] = T(4.0) * x[0] + x[1] * x[1] - T(9.0);
    return true;
  }
};

int main() {
  read_lib_svm_data("/home/freeverc/Projects/Other/NO/bodyfat_scale.txt");

  Problem problem;
  //待优化参数
  double y[2] = {0.0, 0.0};
  // 3:残差数量，2:()函数第一个参数的维度，这里是x的维度
  CostFunction *cost_function =
      new AutoDiffCostFunction<nonline_fun, 3, 2>(new nonline_fun);
  problem.AddResidualBlock(cost_function, NULL, y);
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  // std::chrono::system_clock::time_point t1 =
  // std::chrono::system_clock::now(); int t = 1000000;
  // std::chrono::duration<double> elapsed_seconds;
  // while (t--) {
  //   Eigen::Matrix3x4f proj_matrix = Eigen::Matrix3x4f::Random(3, 4);
  //   const Eigen::Vector4f X = Eigen::Vector4f::Random(4, 1);
  //   const Eigen::Vector3f proj = proj_matrix * X;
  // }
  // std::chrono::system_clock::time_point t2 =
  // std::chrono::system_clock::now(); elapsed_seconds = t2 - t1; double
  // duration = elapsed_seconds.count(); std::cout << "depth time : " <<
  // duration << std::endl; return 0;
}

// class LofFile{
//   public:
//     LofFile() {
//         f.open("log.txt");
//     }
//     void shared_print(std::string s, int num) {
//         // m_mutex.lock();
//         std::lock_guard<std::mutex> guard(m_mutex);
//         f << s << " " << num << std::endl;
//         // m_mutex.unlock();
//     }
//   private:
//     std::ofstream f;
//     std::mutex m_mutex;
// };
// void thread_func(LofFile & log)
// {
//     for(int i = 0;i<100;++i) {
//         log.shared_print("thread", i);
//     }
// }
// int main()
// {
//     LofFile log;
//     std::thread t1(thread_func, std::ref(log));
//     for(int i = 0;i<100;++i) {
//         log.shared_print("main", i);
//     }
//     t1.join();
//     return 0;
// }

// std::mutex mu;
// void shared_print(std::string s, int num)
// {
//     // mu.lock();
//     std::cout<<s<<"  "<<num<<std::endl;
//     // mu.unlock();
// }

// int main() {
//     std::cout<<"testing"<<std::endl;

//     std::thread t1(thread_func, "thread");
//     for(int i = 0;i<100;++i) {
//         shared_print("main", i);
//     }
//     t1.join();
//     shared_print("main end", -1);
//     return 0;
// }