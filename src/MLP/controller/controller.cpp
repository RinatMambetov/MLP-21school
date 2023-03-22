#include "controller.h"

namespace s21 {
bool s21::Controller::setTrainingDataSource(const std::string &source) {
  graphModel->setTrainingSource(source);
  matrixModel->setTrainingSource(source);
  return true;
}

bool s21::Controller::setTestingDataSource(const std::string &source) {
  graphModel->setTestingSource(source);
  matrixModel->setTestingSource(source);
  return true;
}

void s21::Controller::setSizes(std::vector<int> s) {
  graphModel->setSizes(s);
  matrixModel->setSizes(s);
}
void s21::Controller::initModel(bool isMatrixModel) {
  if (isMatrixModel) {
    matrixModel->initTestingData();
    matrixModel->initTrainingData();
    matrixModel->initWeightsAndBiases();
  } else {
    graphModel->initTestingData();
    graphModel->initTrainingData();
    graphModel->initWeightsAndBiases();
  }
}
void s21::Controller::resetModels() {
  delete graphModel;
  delete matrixModel;
  graphModel = new s21::MlpGraphModel();
  matrixModel = new s21::MlpMatrixModel();
}
void s21::Controller::saveConfig(bool isMatrixModel, const std::string dir) {
  if (isMatrixModel) {
    matrixModel->saveConfig(dir);
  } else {
    graphModel->saveConfig(dir);
  }
}

std::string s21::Controller::trainModelForOneEpoch(
    bool isMatrixModel, bool evaluateModelAfterEachEpoch, float testingRatio,
    int epoch, int miniBatchSize, float learningConstant) {
  if (isMatrixModel) {
    return matrixModel->trainModelForOneEpoch(evaluateModelAfterEachEpoch,
                                              testingRatio, epoch,
                                              miniBatchSize, learningConstant);
  } else {
    return graphModel->trainModelForOneEpoch(evaluateModelAfterEachEpoch,
                                             testingRatio, epoch, miniBatchSize,
                                             learningConstant);
  }
  return nullptr;
}
s21::Metrics s21::Controller::getMetrics(bool isMatrixModel,
                                         float testingFraction) {
  if (isMatrixModel) {
    matrixModel->evaluate(testingFraction);
    return matrixModel->getMetrics();
  } else {
    graphModel->evaluate(testingFraction);
    return graphModel->getMetrics();
  }
}
void s21::Controller::loadConfig(const std::string path, bool isMatrixModel) {
  if (isMatrixModel) {
    matrixModel->loadConfig(path);
    matrixModel->initTestingData();
  } else {
    graphModel->loadConfig(path);
    graphModel->initTestingData();
  }
}

char s21::Controller::classifyImage(std::vector<float> pixels,
                                    bool isMatrixModel) {
  if (isMatrixModel) {
    return matrixModel->classifyImage(pixels);
  } else {
    return graphModel->classifyImage(pixels);
  }
}
}  // namespace s21