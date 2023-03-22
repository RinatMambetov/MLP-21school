#include <gtest/gtest.h>

#include <exception>
#include <vector>

#include "../model/MlpGraphModel.h"
#include "../model/MlpMatrixModel.h"
#include "../model/simple_matrix.h"

TEST(MLP_TEST, MatrTest1) {
  s21::MlpMatrixModel matr;
  matr.setNum(5);
  matr.loadConfig("../resources/configuration.config");
  ASSERT_TRUE(matr.getNumberOfLayers() == 5);
}

TEST(MLP_TEST, MatrTest2) {
  s21::MlpMatrixModel matr;
  matr.setTrainingSource("../resource/emnist-letters-train.csv");
  matr.initTrainingData();
  matr.shuffleTrainingData();
}

TEST(MLP_TEST, MatrTest3) {
  s21::MlpMatrixModel matr;
  matr.setTestingSource("../resource/emnist-letters-test.csv");
  matr.initTestingData();
  matr.shuffleTestingData();
}

TEST(MLP_TEST, GraphTest1) {
  s21::MlpGraphModel matr;
  matr.setNum(5);
  matr.loadConfig("../resources/configuration.config");
  ASSERT_TRUE(matr.getNumberOfLayers() == 5);
}

TEST(MLP_TEST, GraphTest2) {
  s21::MlpGraphModel matr;
  matr.setTrainingSource("../resource/emnist-letters-train.csv");
  matr.initTrainingData();
  matr.shuffleTrainingData();
}

TEST(MLP_TEST, GraphTest3) {
  s21::MlpGraphModel matr;
  matr.setTestingSource("../resource/emnist-letters-test.csv");
  matr.initTestingData();
  matr.shuffleTestingData();
}

TEST(MLP_TEST, SMTest1) {
  s21::SimpleMatrix<float> ex(2, 2);
  ex(0, 0) = 1.;
  ex(0, 1) = 2.;
  ex(1, 0) = 3.;
  ex(1, 1) = 4.;
  s21::SimpleMatrix<float> copy = ex;
  for (int i = 0; i < ex.getRows(); ++i) {
    for (int j = 0; j < ex.getCols(); ++j) {
      GTEST_ASSERT_EQ(copy(i, j), ex(i, j));
    }
  }
}

TEST(MLP_TEST, SMTest2) {
  s21::SimpleMatrix<int> first(2, 3);
  first(0, 0) = 1;
  first(0, 1) = 2;
  first(0, 2) = 3;
  first(1, 0) = 4;
  first(1, 1) = 5;
  first(1, 2) = 6;
  s21::SimpleMatrix<int> second(3, 2);
  second(0, 0) = 1;
  second(0, 1) = 4;
  second(1, 0) = 2;
  second(1, 1) = 5;
  second(2, 0) = 3;
  second(2, 1) = 6;
  s21::SimpleMatrix ex = first.transpose();
  for (int i = 0; i < second.getRows(); i++) {
    for (int j = 0; j < second.getCols(); j++) {
      GTEST_ASSERT_EQ(ex(i, j), second(i, j));
    }
  }
}

TEST(MLP_TEST, SMTest3) {
  s21::SimpleMatrix<char> first(1, 1);
  first(0, 0) = 1;
  s21::SimpleMatrix<char> second(1, 1);
  second(0, 0) = 1;
  s21::SimpleMatrix<char> ex = first.transpose();
  for (int i = 0; i < second.getRows(); i++) {
    for (int j = 0; j < second.getCols(); j++) {
      GTEST_ASSERT_EQ(ex(i, j), second(i, j));
    }
  }
}

TEST(MLP_TEST, SMTest4) {
  s21::SimpleMatrix<short> first(1, 1);
  first(0, 0) = 15;
  s21::SimpleMatrix<short> result(1, 1);
  result(0, 0) = 15;
  GTEST_ASSERT_EQ(first(0, 0), result(0, 0));
}

TEST(MLP_TEST, SMTest5) {
  s21::SimpleMatrix<int> first(2, 2);
  first(0, 0) = 1;
  first(0, 1) = 2;
  first(1, 0) = 3;
  first(1, 1) = 4;
  s21::SimpleMatrix<int> result(2, 2);
  result(0, 0) = 1;
  result(0, 1) = 2;
  result(1, 0) = 3;
  result(1, 1) = 4;
  for (int i = 0; i < result.getRows(); i++) {
    for (int j = 0; j < result.getCols(); j++) {
      GTEST_ASSERT_EQ(first(i, j), result(i, j));
    }
  }
}

TEST(MLP_TEST, SMTest6) {
  s21::SimpleMatrix<float> first(2, 2);
  first(0, 0) = 1.;
  first(0, 1) = 2.;
  first(1, 0) = 3.;
  first(1, 1) = 4.;
  s21::SimpleMatrix<float> result(2, 2);
  result(0, 0) = 1.;
  result(0, 1) = 2.;
  result(1, 0) = 3.;
  result(1, 1) = 4.;
  float n = 1.0;
  first = first * n;
  for (int i = 0; i < result.getRows(); ++i) {
    for (int j = 0; j < result.getCols(); ++j) {
      GTEST_ASSERT_EQ(first(i, j), result(i, j));
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
