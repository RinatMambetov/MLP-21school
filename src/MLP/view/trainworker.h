#ifndef TRAINWORKER_H
#define TRAINWORKER_H

#include <QDebug>
#include <QEventLoop>
#include <QMutexLocker>
#include <QObject>
#include <QRunnable>
#include <QScopedPointer>
#include <QThread>
#include <QTimer>

#include "../controller/controller.h"

class TrainWorker : public QObject, public QRunnable {
  Q_OBJECT
 public:
  explicit TrainWorker(s21::Controller *c, QObject *parent = nullptr);
  ~TrainWorker();
  void run();
  void setParameters(bool isMatrixM, bool evalModel, float tRatio, int eAmount,
                     int mBatchSize, float lConstant);
  void delay(int millisecondsWait);

  bool needEndNow();
 public slots:
  void requestEndNow();

 signals:
  void updateStatus(std::string message);
  void finished();

 private:
  std::mutex m;
  bool endNowRequested;
  s21::Controller *controller;
  bool isMatrixModel;
  float testingRatio;
  int epochesAmount;
  int miniBatchSize;
  float learningConstant;
  bool evaluateModelAfterEachEpoch;
};

#endif  // TRAINWORKER_H
