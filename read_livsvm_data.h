#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void read_lib_svm_data(const std::string &file_name) {
  std::cout << "Reading lib svm data : " << std::endl;
  std::ifstream edit_file_is;
  edit_file_is.open(file_name);

  if (!edit_file_is.is_open()) {
    std::cout << "Bad data : " << file_name << std::endl;
    return;
  }

  std::string s;
  while (edit_file_is.good() && std::getline(edit_file_is, s)) {
    if (s.empty()) {
      break;
    }

    int i;
    char c;
    float x, y;
    std::stringstream ss(s);
    ss >> y;
    std::cout << y << "~ ";
    while (ss >> i >> c >> x) {
      std::cout << i << "-" << x << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Finished reading data." << std::endl;
}