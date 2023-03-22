#include "metrics.h"

namespace s21 {

void s21::Metrics::clear() {
  for (int i = 0; i < CLASSIFICATION_CATEGORIES_AMOUNT; ++i) {
    precision[i] = 0;
    accuracy[i] = 0;
    recall[i] = 0;
    f1_measure[i] = 0;
  }
  average_precision = 0;
  average_accuracy = 0;
  average_recall = 0;
  average_f1_measure = 0;
  total_guessed_amount = 0;
  total_sample_size = 0;
}

}  // namespace s21