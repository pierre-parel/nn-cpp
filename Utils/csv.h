#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
using namespace Eigen;

class Image {

public:
  Image() : image_data(28, 28) {};
  MatrixXf image_data;
  int label;
  void printImage();
};

std::vector<Image> csvToImage(std::string path, int numImages);
