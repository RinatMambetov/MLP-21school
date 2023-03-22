#ifndef ARRAY_TPP
#define ARRAY_TPP

#include "simple_matrix.h"

namespace s21 {
template <typename T>
SimpleMatrix<T>::SimpleMatrix() : rows_(0), cols_(0) {}
template <typename T>
SimpleMatrix<T>::SimpleMatrix(int r, int c)
    : rows_(r), cols_(c), data(r * c, 0) {}

template <typename T>
T &SimpleMatrix<T>::operator()(int rows, int cols) {
  if (rows < rows_ && cols < cols_ && rows >= 0 && cols >= 0) {
    return data[rows * cols_ + cols];
  } else {
    throw std::invalid_argument("Indexes out of range");
  }
}

template <typename T>
T SimpleMatrix<T>::operator()(int rows, int cols) const {
  if (rows < rows_ && cols < cols_ && rows >= 0 && cols >= 0) {
    return data[rows * cols_ + cols];
  } else {
    throw std::invalid_argument("Indexes out of range");
  }
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::modifiedProduct(
    const SimpleMatrix &other) const {
  if (rows_ != other.rows_) {
    throw std::invalid_argument("Conflicting dimensions");
  }

  SimpleMatrix ret(other.cols_, 1);
  for (int i = 0, end = other.cols_; i < end; ++i) {
    for (int j = 0, e = rows_; j < e; ++j) {
      ret(i, 0) += data[j] * other.data[j * end + i];
    }
  }
  return ret;
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::modifiedProductRC(
    const SimpleMatrix &other) const {
  if (rows_ != other.cols_) {
    throw std::invalid_argument("Conflicting dimensions");
  }

  SimpleMatrix ret(other.rows_, 1);
  for (int i = 0, end = other.rows_; i < end; ++i) {
    for (int j = 0, e = other.cols_; j < e; ++j) {
      ret(i, 0) += data[j] * other(i, j);  //.data[i * e + j];
    }
  }
  return ret;
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::operator+(const SimpleMatrix &other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Conflicting dimensions");
  }

  SimpleMatrix ret(rows_, cols_);
  for (int i = 0, end = data.size(); i < end; ++i) {
    ret.data[i] = data[i] + other.data[i];
  }
  return ret;
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::operator*(const SimpleMatrix &other) const {
  if (cols_ != other.rows_) {
    throw std::invalid_argument("Conflicting dimensions");
  }
  SimpleMatrix<T> result(rows_, other.cols_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < other.cols_; j++) {
      result(i, j) = 0;
      for (int k = 0; k < cols_; k++) {
        result(i, j) += data[i * cols_ + k] * other(k, j);
      }
    }
  }
  return result;
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::operator*(T number) const {
  SimpleMatrix<T> result(rows_, cols_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result(i, j) = data[i * cols_ + j] * number;
    }
  }
  return result;
}

template <typename T>
SimpleMatrix<T> SimpleMatrix<T>::transpose() {
  SimpleMatrix<T> result(cols_, rows_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result(j, i) = data[i * cols_ + j];
    }
  }
  return result;
}

template <typename T>
void SimpleMatrix<T>::operator=(std::vector<T> dum) {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      data[i * cols_ + j] = dum[i * cols_ + j];
    }
  }
}

}  // namespace s21

#endif