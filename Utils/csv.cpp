#include "csv.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace Eigen;

std::vector<Image> csvToImage(std::string path, int numImages) {
  std::ifstream file(path);
  if (file.good()) {
    std::cout << "File is good!\n";
  }
  std::vector<Image> imgs;
  std::string line;

  // Skip the first line of the CSV file
  std::getline(file, line);
  int i = 0;

  while (std::getline(file, line) && i < numImages) {

    Image img;
    std::istringstream ss(line);
    int j = 0;

    std::string token;
    while (std::getline(ss, token, ',')) {
      if (j == 0) {
        img.label = std::stoi(token);
      } else {
        int row_idx = (j - 1) / 28;
        int col_idx = (j - 1) % 28;
        img.image_data(row_idx, col_idx) = std::stof(token) / 256.0;
      }
      j++;
    }
    imgs.push_back(img);
    i++;
  }
  file.close();
  return imgs;
}

void Image::printImage() {
  std::cout << image_data << "\n";
  std::cout << "Image Label: " << label << "\n";
}
