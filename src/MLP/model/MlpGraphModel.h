
#ifndef MLPGRAPHMODEL_H
#define MLPGRAPHMODEL_H

#include <algorithm>
#include <chrono>
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
using std::vector;

namespace s21 {

struct Neuron {
  std::vector<Neuron *> nextLayer;
  std::vector<Neuron *> prevLayer;

  std::vector<float> weights;
  float bias;

  std::vector<float> deltaWeights;
  float deltaBias;

  float activation;
  float errTerm;
};

class MlpGraphModel {
 private:
  std::vector<int> sizes;
  int numberOfLayers;

  std::vector<s21::ImageEmnist> trainingData;
  std::vector<s21::ImageEmnist> testingData;

  float learningRateConstant;

  vector<vector<Neuron>> neurons;

  std::string trainingSource;
  std::string testingSource;
  s21::Metrics metrics;

 public:
  void setTestingSource(const std::string s) { testingSource = s; }
  void setTrainingSource(const std::string s) { testingSource = s; }
  void setDefaultValuesToNeuronActivations();
  void setSizes(std::vector<int> s);
  void setNum(int val) { numberOfLayers = val; }
  int getNumberOfLayers() { return numberOfLayers; }
  void initSizes(std::initializer_list<int> args);
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
  void updateMiniBatch(int batch_beginning, int batch_ending,
                       float learningConstant);
  void updateWeightsAndBiases(float constant);
  void backpropagate(const s21::ImageEmnist &image);
  float sigmoid(float value) { return 1 / (1 + exp(value * -1)); }
  s21::SimpleMatrix<float> feedforward(const s21::ImageEmnist &image);
  void evaluate(float ratio);
  s21::Metrics getMetrics();
  char classifyImage(const std::vector<float> &image);
};

}  // namespace s21

#endif
