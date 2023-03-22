#ifndef IMAGEEMNIST_H
#define IMAGEEMNIST_H
#include <fstream>

#include "simple_matrix.h"

// This struct is designed to load and hold 1 image from Emnist Database (28 x
// 28 pixels) from the .csv file.
namespace s21 {

class ImageEmnist {
 public:
  ImageEmnist()
      : image(s21::SimpleMatrix<unsigned char>(28 * 28, 1)), value(-1) {}

  s21::SimpleMatrix<unsigned char> getImg() const { return image; }
  void setImg(int i, int j, int val) { image(i, j) = val; }
  int getValue() const { return value; }

  bool load(std::fstream &my_file);

 private:
  s21::SimpleMatrix<unsigned char> image;
  int value;
};

}  // namespace s21
#endif  // IMAGEEMNIST_H
