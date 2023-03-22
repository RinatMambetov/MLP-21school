#include <QApplication>

#include "controller/controller.h"
#include "mainwindow.h"
#include "model/MlpGraphModel.h"
#include "model/MlpMatrixModel.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  s21::MlpGraphModel *graphModel = new s21::MlpGraphModel();
  s21::MlpMatrixModel *matrixModel = new s21::MlpMatrixModel();

  s21::Controller *controller = new s21::Controller(graphModel, matrixModel);

  MainWindow w(controller, nullptr);
  w.setWindowTitle("MLP");
  w.show();
  return a.exec();
}
