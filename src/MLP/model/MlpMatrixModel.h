

#ifndef MLPMATRIXMODEL_H
#define MLPMATRIXMODEL_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ImageEmnist.h"
#include "metrics.h"
#include "simple_matrix.h"
using std::cout;
using std::endl;

namespace s21 {

class MlpMatrixModel {
 private:
  std::vector<int> sizes;
  int numberOfLayers;
  std::vector<s21::ImageEmnist> trainingData;
  std::vector<s21::ImageEmnist> testingData;

  std::vector<s21::SimpleMatrix<float>> biases;
  std::vector<s21::SimpleMatrix<float>> weights;

  std::vector<s21::SimpleMatrix<float>> deltaBiases;
  std::vector<s21::SimpleMatrix<float>> deltaWeights;

  s21::Metrics metrics;

  std::string trainingSource;
  std::string testingSource;

 public:
  void setTestingSource(const std::string s) { testingSource = s; }
  void setTrainingSource(const std::string s) { trainingSource = s; }
  void setSizes(std::vector<int> s);
  void setNum(int val) { numberOfLayers = val; }

  int getNumberOfLayers() { return numberOfLayers; }

  void initWeightsAndBiases();
  void saveConfig(const std::string dir);
  void loadConfig(const std::string &source);
  void initTrainingData();
  void initTestingData();
  void shuffleTrainingData();
  void shuffleTestingData();
  std::string trainModelForOneEpoch(bool evaluateModelAfterEachEpoch,
                                    float testingRatio, int epoch,
                                    int mini_batch_size,
                                    float learningConstant);

  void updateMiniBatch(int batch_beginning, int batch_ending, float constant);
  std::pair<std::vector<s21::SimpleMatrix<float>>,
            std::vector<s21::SimpleMatrix<float>>>
  backpropagate(const s21::ImageEmnist &image);
  s21::SimpleMatrix<float> feedforward(const s21::ImageEmnist &image);
  void evaluate(float ratio);
  s21::Metrics getMetrics();
  s21::SimpleMatrix<float> costDerivative(
      const s21::SimpleMatrix<float> &output_activation,
      const s21::SimpleMatrix<float> &output_desired);
  s21::SimpleMatrix<float> sigmoid(const s21::SimpleMatrix<float> &m);
  s21::SimpleMatrix<float> sigmoidPrime(const s21::SimpleMatrix<float> &m);
  char classifyImage(const std::vector<float> &image);
};

}  // namespace s21

#endif
