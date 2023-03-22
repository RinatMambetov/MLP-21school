#ifndef METRICS_H
#define METRICS_H
#define CLASSIFICATION_CATEGORIES_AMOUNT 26
#include <array>
#include <vector>

namespace s21 {

// This struct is designed to hold statistical information
// about performance of the model.
class Metrics {
 public:
  void clear();
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> getPrcision() {
    return precision;
  }
  float getPrecisionElem(int i) const { return precision[i]; }
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> getAccuracy() {
    return accuracy;
  }
  float getAccuracyElem(int i) const { return accuracy[i]; }
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> getRecall() {
    return recall;
  }
  float getRecallElem(int i) const { return recall[i]; }
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> getF1Measure() {
    return f1_measure;
  }
  float getF1MeasureElem(int i) const { return f1_measure[i]; }
  float getAveragePrecision() { return average_precision; }
  float getAverageAccuracy() { return average_accuracy; }
  float getAverageRecall() { return average_recall; }
  float getAverageF1Measure() { return average_f1_measure; }
  float getTotalGuessedAmount() { return total_guessed_amount; }
  float getTotalSampleSize() { return total_sample_size; }
  std::vector<int> getSizes() { return sizes; }
  int getNumberOfLayers() { return numberOfLayers; }

  void setTotalSampleSize(float val) { total_sample_size = val; }
  void setTotalGuessedAmount(float val) { total_guessed_amount = val; }
  void setPrecisionElem(int i, float val) { precision[i] = val; }
  void setAccuracyElem(int i, float val) { accuracy[i] = val; }
  void setRecallElem(int i, float val) { recall[i] = val; }
  void setF1MeasureElem(int i, float val) { f1_measure[i] = val; }
  void setAverageAccuracy(float val) { average_accuracy = val; }
  void setAveragePrecision(float val) { average_precision = val; }
  void setAverageRecall(float val) { average_recall = val; }
  void setAverageF1Measure(float val) { average_f1_measure = val; }
  void setSizes(std::vector<int> val) { sizes = val; }
  void setNumberOfLayers(int val) { numberOfLayers = val; }

 private:
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> precision;
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> accuracy;
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> recall;
  std::array<float, CLASSIFICATION_CATEGORIES_AMOUNT> f1_measure;
  float average_precision;
  float average_accuracy;
  float average_recall;
  float average_f1_measure;
  float total_guessed_amount;
  float total_sample_size;

  std::vector<int> sizes;
  int numberOfLayers;
};

}  // namespace s21

#endif  // METRICS_H
