#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void read_lib_svm_data(const std::string &file_name,
                       std::vector<std::vector<double>> &X,
                       std::vector<double> &Y) {
  std::cout << "Reading lib svm data : " << std::endl;
  std::ifstream edit_file_is;
  edit_file_is.open(file_name);

  if (!edit_file_is.is_open()) {
    std::cout << "Bad data : " << file_name << std::endl;
    return;
  }

  int i = 0;
  int j = 0;
  char c;
  double x, y;
  std::string s;
  while (edit_file_is.good() && std::getline(edit_file_is, s)) {
    if (s.empty()) {
      break;
    }

    std::stringstream ss(s);
    ss >> y;

    ++i;
    if (X.size() < i) {
      X.push_back(std::vector<double>(0));
    }

    if (Y.size() < i) {
      Y.push_back(y);
    }

    // std::cout << y << " : ";
    X[i - 1].resize(j, 0);
    while (ss >> j >> c >> x) {
      if (j <= 0)
        break;

      // std::cout << j << "=" << x << " ";
      if (X[i - 1].size() < j) {
        X[i - 1].push_back(0);
      }

      X[i - 1][j - 1] = x;
    }

    s.clear();
  }

  std::cout << "Finished reading data." << std::endl;
}

void print_data(std::vector<std::vector<double>> &X, std::vector<double> &Y) {
  for (int i = 0; i < X.size(); ++i) {
    std::cout << i << " " << Y[i] << " : ";
    for (int j = 0; j < X[i].size(); ++j) {
      std::cout << X[i][j] << " ";
    }
    std::cout << std::endl;
  }
}