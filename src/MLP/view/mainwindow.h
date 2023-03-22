#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <condition_variable>

#include "../controller/controller.h"
#include "../model/MlpMatrixModel.h"
#include "experimentinfo.h"
#include "paint.h"
#include "statuswindow.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(s21::Controller *c, QWidget *parent);
  ~MainWindow();
  void classify();
  std::vector<float> ReadPixels(const QImage &image);
 signals:
  void terminateTrainingProcess();

 private slots:
  void on_modelLoader_clicked();
  void on_classifyImage_clicked();
  void on_clear_clicked();
  void on_trainModel_clicked();
  void on_chooseTrainingDataset_clicked();
  void on_chooseTestingDataset_clicked();
  void updateStatusMessage(std::string s);
  void terminateProcess();
  void updateModeDisplayedlInfo();
  void on_load_img_clicked();
  void on_saveModel_clicked();
  void on_runExperiment_clicked();
  void hiddenLayers();
  void on_Research_clicked();

 private:
  Ui::MainWindow *ui;
  s21::Controller *controller;
  Statuswindow *statusWindow;  // statusWindow opens during training process and
                               // displays training progress
  ExperimentInfo *experimentInfo;
  bool isMatrixModel;
};

#endif  // MAINWINDOW_H
