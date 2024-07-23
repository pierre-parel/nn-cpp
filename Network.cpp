// network.cpp
// ~~~~~~~~~~~
// A module to implement the stochastic gradient descent learning
// algorithm for a feedforward neural network. Gradients are calculated
// using backpropagation. Note that I have focused on making the code
// simple, easily readable, and easily modifiable. It is not optimized,
// and omits many desirable features.

#include "Utils/csv.h"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

using namespace Eigen;
using namespace std;

class Network {
public:
  Network(const vector<int> &sizes);
  VectorXd feedforward(VectorXd a);
  void SGD(vector<pair<VectorXd, VectorXd>> &training_data, int epochs,
           int mini_batch_size, double eta,
           vector<pair<VectorXd, int>> *test_data = nullptr);

private:
  int num_layers;
  vector<int> sizes;
  vector<VectorXd> biases;
  vector<MatrixXd> weights;

  void update_mini_batch(const vector<pair<VectorXd, VectorXd>> &mini_batch,
                         double eta);
  pair<vector<VectorXd>, vector<MatrixXd>> backprop(const VectorXd &x,
                                                    const VectorXd &y);
  int evaluate(const vector<pair<VectorXd, int>> &test_data);
  VectorXd cost_derivative(const VectorXd &output_activations,
                           const VectorXd &y);
  static double sigmoid(double z);
  static double sigmoid_prime(double z);
};

Network::Network(const vector<int> &sizes)
    : sizes(sizes), num_layers(sizes.size()) {
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> d(0, 1);

  for (int i = 1; i < sizes.size(); ++i) {
    biases.emplace_back(
        VectorXd::NullaryExpr(sizes[i], [&]() { return d(gen); }));
    weights.emplace_back(MatrixXd::NullaryExpr(sizes[i], sizes[i - 1],
                                               [&]() { return d(gen); }));
  }
}

VectorXd Network::feedforward(VectorXd a) {
  for (size_t i = 0; i < biases.size(); ++i) {
    a = (weights[i] * a + biases[i]).unaryExpr(&Network::sigmoid);
  }
  return a;
}

void Network::SGD(vector<pair<VectorXd, VectorXd>> &training_data, int epochs,
                  int mini_batch_size, double eta,
                  vector<pair<VectorXd, int>> *test_data) {
  int n = training_data.size();
  int n_test = test_data ? test_data->size() : 0;
  random_device rd;
  mt19937 gen(rd());

  for (int j = 0; j < epochs; ++j) {
    cout << "Shuffling data... \n";
    shuffle(training_data.begin(), training_data.end(), gen);
    for (int k = 0; k < n; k += mini_batch_size) {
      vector<pair<VectorXd, VectorXd>> mini_batch(
          training_data.begin() + k,
          training_data.begin() + min(k + mini_batch_size, n));
      update_mini_batch(mini_batch, eta);
    }
    if (test_data) {
      cout << "Epoch " << j << ": " << evaluate(*test_data) << " / " << n_test
           << endl;
    } else {
      cout << "Epoch " << j << " complete" << endl;
    }
  }
}

void Network::update_mini_batch(
    const vector<pair<VectorXd, VectorXd>> &mini_batch, double eta) {
  vector<VectorXd> nabla_b(biases.size());
  vector<MatrixXd> nabla_w(weights.size());

  for (auto &nb : nabla_b)
    nb.setZero();
  for (auto &nw : nabla_w)
    nw.setZero();

  for (const auto &[x, y] : mini_batch) {
    auto [delta_nabla_b, delta_nabla_w] = backprop(x, y);
    for (size_t i = 0; i < nabla_b.size(); ++i) {
      nabla_b[i] += delta_nabla_b[i];
      nabla_w[i] += delta_nabla_w[i];
    }
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= (eta / mini_batch.size()) * nabla_w[i];
    biases[i] -= (eta / mini_batch.size()) * nabla_b[i];
  }
}

pair<vector<VectorXd>, vector<MatrixXd>> Network::backprop(const VectorXd &x,
                                                           const VectorXd &y) {
  vector<VectorXd> nabla_b(biases.size());
  vector<MatrixXd> nabla_w(weights.size());

  for (auto &nb : nabla_b)
    nb.setZero();
  for (auto &nw : nabla_w)
    nw.setZero();

  vector<VectorXd> activations = {x};
  vector<VectorXd> zs;

  VectorXd activation = x;
  for (size_t i = 0; i < biases.size(); ++i) {
    VectorXd z = weights[i] * activation + biases[i];
    zs.push_back(z);
    activation = z.unaryExpr(&Network::sigmoid);
    activations.push_back(activation);
  }

  VectorXd delta =
      cost_derivative(activations.back(), y)
          .cwiseProduct(zs.back().unaryExpr(&Network::sigmoid_prime));
  nabla_b.back() = delta;
  nabla_w.back() = delta * activations[activations.size() - 2].transpose();

  for (int l = 2; l < num_layers; ++l) {
    VectorXd z = zs[zs.size() - l];
    VectorXd sp = z.unaryExpr(&Network::sigmoid_prime);
    delta =
        (weights[weights.size() - l + 1].transpose() * delta).cwiseProduct(sp);
    nabla_b[nabla_b.size() - l] = delta;
    nabla_w[nabla_w.size() - l] =
        delta * activations[activations.size() - l - 1].transpose();
  }

  return {nabla_b, nabla_w};
}

int Network::evaluate(const vector<pair<VectorXd, int>> &test_data) {
  int correct = 0;
  for (const auto &[x, y] : test_data) {
    VectorXd output = feedforward(x);
    if (output.maxCoeff() == output[y]) {
      ++correct;
    }
  }
  return correct;
}

VectorXd Network::cost_derivative(const VectorXd &output_activations,
                                  const VectorXd &y) {
  return output_activations - y;
}

double Network::sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

double Network::sigmoid_prime(double z) {
  return sigmoid(z) * (1 - sigmoid(z));
}

int main() {
  // Load training and test data
  vector<Image> training_images =
      csvToImage("./TrainingImages/mnist_train.csv", 500);
  vector<Image> test_images =
      csvToImage("./TrainingImages/mnist_test.csv", 100);

  // Convert Image data to training and test data format
  vector<pair<VectorXd, VectorXd>> training_data;
  vector<pair<VectorXd, int>> test_data;

  // Function to convert MatrixXf to VectorXd
  auto matrixToVector = [](const MatrixXf &mat) -> VectorXd {
    VectorXd vec(mat.size());
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        vec(i * mat.cols() + j) = mat(i, j);
      }
    }
    return vec;
  };

  int i = 0;
  for (const auto &img : training_images) {
    cout << "Processing training images: " << ++i << "\n";
    VectorXd x = matrixToVector(img.image_data);
    VectorXd y = VectorXd::Zero(10); // Assuming 10 classes for digits 0-9
    y(img.label) = 1.0;
    training_data.push_back(make_pair(x, y));
  }

  std::cout << "*************************Processed training_images";
  i = 0;
  for (const auto &img : test_images) {
    cout << "Processing test images: " << ++i << "\n";
    VectorXd x = matrixToVector(img.image_data);
    test_data.push_back(make_pair(x, img.label));
  }
  std::cout << "Processed test_images";

  // Define network sizes (e.g., 784 input neurons for 28x28 images, 30 hidden
  // neurons, 10 output neurons for digits 0-9)
  vector<int> sizes = {784, 10, 10};
  Network net(sizes);

  // Train the network using stochastic gradient descent
  net.SGD(training_data, 30, 10, 3.0, &test_data);

  return 0;
}
