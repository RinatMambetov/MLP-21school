#include "paint.h"

#include "model/MlpMatrixModel.h"
#include "ui_mainwindow.h"
Paint::Paint(QWidget *parent)
    : QWidget{parent}, is_locked_(false), is_write_(true) {
  pen_ = std::make_unique<QPen>(
      QPen(Qt::black, 40, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
}

void Paint::paintEvent(QPaintEvent *event) {  // override
  QPainter painter(this);
  painter.drawPixmap(0, 0, pixmap_);
  this->update();
}

void Paint::resizeEvent(QResizeEvent *event) {  // override
  auto newRect = pixmap_.rect().united(rect());
  if (!(newRect == pixmap_.rect())) {
    QPixmap newPixmap{newRect.size()};
    QPainter painter{&newPixmap};
    painter.fillRect(newPixmap.rect(), Qt::white);
    painter.drawPixmap(0, 0, pixmap_);
    pixmap_ = newPixmap;
  }
}

void Paint::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::RightButton && is_locked_ == false) {
    clear();
    is_write_ = false;
    lastPos_ = event->pos();
  } else if (event->button() == Qt::LeftButton) {
    is_write_ = true;
    lastPos_ = event->pos();
    draw(event->pos(), event->button());
  }
}

void Paint::mouseMoveEvent(QMouseEvent *event) {
  draw(event->pos(), event->button());
}

void Paint::draw(const QPoint &pos, Qt::MouseButton event) {  // override
  if (is_locked_ == false && is_write_) {
    QPainter painter{&pixmap_};
    painter.setPen(*pen_);
    painter.drawLine(lastPos_, pos);
    lastPos_ = pos;
    update();
  }
}

void Paint::clear() {
  pixmap_.fill(Qt::white);
  this->clearMask();
}

void Paint::setImg(QImage image) { pixmap_ = QPixmap::fromImage(image); }

// void Paint::setImage(s21::ImageEmnist img) {
//   for (int i = 0; i < 28 * 28; i++) {
//     // pixmap_.setPixel(i / 28, i % 28);
//   }
// }

void Paint::setLock(bool is_lock) { is_locked_ = is_lock; }

// void Paint::makeWhite() {
//   painter.drawPixmap(0, 0, pixmap_);
//}
QImage Paint::getCurrentImage() {
  int pad_width = 1;
  const int kPaintAreaSize = 512;
  const int kPixelSize = 28;
  QImage image(kPaintAreaSize, kPaintAreaSize, QImage::Format_Grayscale8);
  QImage padded_image(512 * pad_width, kPaintAreaSize * pad_width,
                      QImage::Format_Grayscale8);
  QPainter painter(&padded_image);
  this->render(&painter);
  image =
      padded_image.copy(pad_width, pad_width, kPaintAreaSize, kPaintAreaSize);
  image = image.scaled(QSize(kPixelSize, kPixelSize), Qt::IgnoreAspectRatio);
  return image;
}
