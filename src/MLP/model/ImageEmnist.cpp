#include "ImageEmnist.h"

namespace s21 {

bool s21::ImageEmnist::load(std::fstream &my_file) {
  char ch;
  int i = 0;
  int val = 0;
  bool prev = false;
  while (my_file >> ch && i < 28 * 28) {
    if (ch >= 48 && ch <= 57) {
      val = val * 10 + ch - '0';
      prev = true;
    } else if (prev) {
      image(i, 0) = val;
      val = 0;
      i = i + 1;
      prev = false;
    }
  }
  value = image(0, 0);
  if (my_file.peek() == std::ifstream::traits_type::eof()) {
    return false;
  } else {
    return true;
  }
}

}  // namespace s21
