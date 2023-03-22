#ifndef SIMPLE_MATRIX_H
#define SIMPLE_MATRIX_H

#include <stdexcept>
#include <vector>

namespace s21 {

template <typename T>
class SimpleMatrix {
 public:
  SimpleMatrix<T>();
  SimpleMatrix<T>(int r, int c);
  void operator=(std::vector<T> dum);
  T &operator()(int rows, int cols);
  T operator()(int rows, int cols) const;

  int getRows() const { return rows_; }
  int getCols() const { return cols_; }
  std::vector<T> getData() { return data; }
  int getDataSize() const { return data.size(); }
  auto getDataBegin() const { return data.begin(); }
  auto getDataEnd() const { return data.end(); }
  T getElem(int index) const { return data[index]; }
  void setElem(int index, T val) { data[index] = val; }

  SimpleMatrix<T>(std::vector<float> &&d, int r, int c)
      : data(d), rows_(r), cols_(c) {}

  SimpleMatrix modifiedProduct(const SimpleMatrix &other) const;
  SimpleMatrix modifiedProductRC(const SimpleMatrix &other) const;
  SimpleMatrix operator+(const SimpleMatrix &other) const;
  SimpleMatrix operator*(const SimpleMatrix &other) const;
  SimpleMatrix operator*(T number) const;
  SimpleMatrix transpose();

 private:
  std::vector<T> data;
  int rows_;
  int cols_;
};

}  // namespace s21

#include "simple_matrix.tpp"

#endif  // SIMPLE_MATRIX_H
