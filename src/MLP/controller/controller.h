#pragma once
#include <string>

#include "../model/MlpGraphModel.h"
#include "../model/MlpMatrixModel.h"

namespace s21 {

class Controller {
 private:
  s21::MlpGraphModel *graphModel;
  s21::MlpMatrixModel *matrixModel;

 public:
  Controller(s21::MlpGraphModel *g, s21::MlpMatrixModel *m)
      : graphModel(g), matrixModel(m) {}
  Controller() = delete;
  bool setTrainingDataSource(const std::string &source);
  bool setTestingDataSource(const std::string &source);

  void setSizes(std::vector<int> s);
  void initModel(bool isMatrixModel);
  void resetModels();
  void saveConfig(bool isMatrixModel, const std::string dir);
  std::string trainModelForOneEpoch(bool isMatrixModel,
                                    bool evaluateModelAfterEachEpoch,
                                    float testingRatio, int epoch,
                                    int miniBatchSize, float learningConstant);
  s21::Metrics getMetrics(bool isMatrixModel, float testingFraction);
  void loadConfig(const std::string path, bool isMatrixModel);
  char classifyImage(std::vector<float> pixels, bool isMatrixModel);
};

}  // namespace s21
