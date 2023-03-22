#ifndef PAINT_H
#define PAINT_H

// #include "../model/MlpMatrixModel.h"
#include <QMouseEvent>
#include <QPainter>
#include <QWidget>
#include <memory>
#include <vector>
class Paint : public QWidget {
  Q_OBJECT
 public:
  explicit Paint(QWidget *parent = nullptr);
  void paintEvent(QPaintEvent *event) override;

  void resizeEvent(QResizeEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  // void setImage(s21::ImageEmnist img);
  QImage getCurrentImage();
  void draw(const QPoint &pos, Qt::MouseButton event);
  void clear();
  void setImg(QImage image);
  void setLock(bool is_lock);

 private:
  static const int kDefaultPenWidth = 40;
  static const int kReportPenWidth = 40;

  QPixmap pixmap_;
  QPoint lastPos_;
  std::unique_ptr<QPen> pen_;

  bool is_locked_;
  bool is_write_;
};

#endif
