#include "mainwindow.h"

#include <QDebug>
#include <QFileDialog>
#include <QThread>
#include <QThreadPool>
#include <thread>
#include <vector>

#include "paint.h"
#include "ui_mainwindow.h"
#include "view/statuswindow.h"
#include "view/trainworker.h"

MainWindow::MainWindow(s21::Controller *c, QWidget *parent = nullptr)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  controller = c;
  ui->setupUi(this);
  ui->tab_2->setAutoFillBackground(true);
  statusWindow = new Statuswindow(this);
  experimentInfo = new ExperimentInfo();
  QObject::connect(statusWindow, &Statuswindow::terminateProcess, this,
                   &MainWindow::terminateProcess);
  QObject::connect(ui->hiddenLayersAmount, SIGNAL(valueChanged(int)), this,
                   SLOT(hiddenLayers()));
  ui->classifyImage->setEnabled(false);
  ui->runExperiment->setEnabled(false);
  if (ui->hiddenLayersAmount->value() == 2) {
    ui->label_9->hide();
    ui->layerSize_3->hide();
    ui->label_10->hide();
    ui->layerSize_4->hide();
    ui->label_11->hide();
    ui->layerSize_5->hide();
  }
}

MainWindow::~MainWindow() {
  delete ui;
  delete experimentInfo;
  delete statusWindow;
}
void MainWindow::on_modelLoader_clicked() {
  QString dir = QFileDialog::getOpenFileName(this, "Choose File");
  if (ui->matrixMode->isChecked()) {
    isMatrixModel = true;
    controller->loadConfig(dir.toStdString(), isMatrixModel);
    updateModeDisplayedlInfo();
  } else {
    isMatrixModel = false;
    controller->loadConfig(dir.toStdString(), isMatrixModel);
    updateModeDisplayedlInfo();
  }
  ui->runExperiment->setEnabled(true);
  ui->classifyImage->setEnabled(true);
}
void MainWindow::classify() {
  try {
    QImage image = ui->paint->getCurrentImage();
    std::vector<float> pixels = ReadPixels(image);
    char ans = controller->classifyImage(pixels, isMatrixModel);

    ui->answer->setText(QChar(ans));

  } catch (std::exception &ex) {
  }
}

std::vector<float> MainWindow::ReadPixels(const QImage &image) {
  std::vector<float> image_pixels;
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      image_pixels.push_back(
          image.pixelColor(i, j).black());  // / 255. (for 0 and 1)
    }
  }
  return image_pixels;
}

void MainWindow::on_classifyImage_clicked() { classify(); }

void MainWindow::on_clear_clicked() { ui->paint->clear(); }

void MainWindow::on_trainModel_clicked() {
  int layersAmount = ui->hiddenLayersAmount->value();
  std::vector<int> sizes;
  sizes.push_back(28 * 28);
  sizes.push_back(ui->layerSize_1->value());
  sizes.push_back(ui->layerSize_2->value());
  if (layersAmount >= 3) {
    sizes.push_back(ui->layerSize_3->value());
  }
  if (layersAmount >= 4) {
    sizes.push_back(ui->layerSize_4->value());
  }
  sizes.push_back(26);
  controller->setSizes(sizes);
  int epochesAmount = ui->epochAmount->value();
  int batchSize = ui->batchSize->value();
  float learningConstant = ui->learningConstant->value();
  bool evaluateModelAfterEachEpoch = ui->evaluateCheckBox->isChecked();
  QThread *thread = new QThread();
  TrainWorker *worker = new TrainWorker(this->controller);
  if (ui->matrixMode->isChecked()) {
    isMatrixModel = true;
    worker->setParameters(isMatrixModel, evaluateModelAfterEachEpoch, 1,
                          epochesAmount, batchSize, learningConstant);
  } else {
    isMatrixModel = false;
    worker->setParameters(isMatrixModel, evaluateModelAfterEachEpoch, 1,
                          epochesAmount, batchSize, learningConstant);
  }
  worker->moveToThread(thread);
  QObject::connect(thread, &QThread::started, worker, &TrainWorker::run);
  QObject::connect(thread, &QThread::started, statusWindow,
                   &Statuswindow::disableFinishButton);
  QObject::connect(worker, &TrainWorker::finished, worker,
                   &TrainWorker::deleteLater);
  QObject::connect(worker, &TrainWorker::finished, thread, &QThread::quit);
  QObject::connect(thread, &QThread::finished, thread, &QThread::deleteLater);
  QObject::connect(thread, &QThread::finished, statusWindow,
                   &Statuswindow::enableFinishButton);
  QObject::connect(worker, &TrainWorker::finished, this,
                   &MainWindow::updateModeDisplayedlInfo);
  QObject::connect(this, &MainWindow::terminateTrainingProcess, worker,
                   &TrainWorker::requestEndNow, Qt::DirectConnection);
  QObject::connect(worker, &TrainWorker::updateStatus, this,
                   &MainWindow::updateStatusMessage);
  thread->start();
  ui->classifyImage->setEnabled(true);
  ui->runExperiment->setEnabled(true);
  statusWindow->exec();
}
void MainWindow::terminateProcess() {
  emit terminateTrainingProcess();
  s21::MlpGraphModel *graphModel = new s21::MlpGraphModel();
  s21::MlpMatrixModel *matrixModel = new s21::MlpMatrixModel();
  controller = new s21::Controller(graphModel, matrixModel);
  controller->setTrainingDataSource(
      ui->trainingDatasetDirectory->text().toStdString());
  controller->setTestingDataSource(
      ui->testingDatasetDirectory->text().toStdString());
  ui->classifyImage->setEnabled(false);
  ui->runExperiment->setEnabled(false);
}

void MainWindow::updateModeDisplayedlInfo() {
  auto metrics = controller->getMetrics(isMatrixModel, 0.05);
  ui->layersCount->setText(QString::number(metrics.getNumberOfLayers()));
  QString layersSizes = "_";
  std::vector<int> arr = metrics.getSizes();
  for (auto i : arr) {
    layersSizes.append(QString::number(i) + "_");
  }
  ui->layersSizes->setText(layersSizes);
  ui->precision->setText(QString::number(metrics.getAveragePrecision()));
  ui->accuracy->setText(QString::number(metrics.getAverageAccuracy()));
  ui->f1Measure->setText(QString::number(metrics.getAverageF1Measure()));
  ui->recall->setText(QString::number(metrics.getAverageRecall()));
}

void MainWindow::on_chooseTrainingDataset_clicked() {
  QString dir = QFileDialog::getOpenFileName(this, "Choose File");
  ui->trainingDatasetDirectory->setText(dir);
  controller->setTrainingDataSource(dir.toStdString());
}

void MainWindow::on_chooseTestingDataset_clicked() {
  QString dir = QFileDialog::getOpenFileName(this, "Choose File");
  ui->testingDatasetDirectory->setText(dir);
  controller->setTestingDataSource(dir.toStdString());
}

void MainWindow::updateStatusMessage(std::string message) {
  statusWindow->appendText(message);
}

void MainWindow::on_saveModel_clicked() {
  QString filename = QFileDialog::getSaveFileName(this, "Save As");
  controller->saveConfig(isMatrixModel, filename.toStdString());
}

void MainWindow::on_runExperiment_clicked() {
  time_t now = time(0);
  auto metrics =
      controller->getMetrics(isMatrixModel, ui->testSampleFraction->value());
  std::ostringstream ss;
  ss << "Total guessed: " << metrics.getTotalGuessedAmount() << '\\'
     << metrics.getTotalSampleSize() << '\n';
  ss << "Accuracy: " << metrics.getAverageAccuracy()
     << " , precision: " << metrics.getAveragePrecision()
     << " , recall: " << metrics.getAverageRecall()
     << " , f1-measure: " << metrics.getAverageF1Measure() << '\n';
  ss << "Total time :" << time(0) - now << '\n';
  experimentInfo->clearText();
  experimentInfo->appendText(std::string(ss.str()));
  experimentInfo->exec();
}

void MainWindow::on_load_img_clicked() {
  QString fileName = QFileDialog::getOpenFileName(this, "Choose file");
  if (fileName.length() > 0) {
    QImage image;
    image.load(fileName);
    image = image.convertToFormat(QImage::Format_Grayscale8);
    image = image.scaled(QSize(512, 512), Qt::IgnoreAspectRatio);
    ui->paint->setImg(image);
  }
}

void MainWindow::hiddenLayers() {
  if (ui->hiddenLayersAmount->value() == 2) {
    ui->label_9->hide();
    ui->layerSize_3->hide();
    ui->label_10->hide();
    ui->layerSize_4->hide();
    ui->label_11->hide();
    ui->layerSize_5->hide();
  } else if (ui->hiddenLayersAmount->value() == 3) {
    ui->label_9->show();
    ui->layerSize_3->show();
    ui->label_10->hide();
    ui->layerSize_4->hide();
    ui->label_11->hide();
    ui->layerSize_5->hide();
  } else if (ui->hiddenLayersAmount->value() == 4) {
    ui->label_9->show();
    ui->layerSize_3->show();
    ui->label_10->show();
    ui->layerSize_4->show();
    ui->label_11->hide();
    ui->layerSize_5->hide();
  } else if (ui->hiddenLayersAmount->value() == 5) {
    ui->label_9->show();
    ui->layerSize_3->show();
    ui->label_10->show();
    ui->layerSize_4->show();
    ui->label_11->show();
    ui->layerSize_5->show();
  }
}

void MainWindow::on_Research_clicked() {
  time_t general_t = time(0);
  for (int i = 0; i < ui->numresearch->value(); i++) {
    time_t now = time(0);
    auto metrics =
        controller->getMetrics(isMatrixModel, ui->testSampleFraction->value());
    std::ostringstream ss;
    ss << "Total guessed: " << metrics.getTotalGuessedAmount() << '\\'
       << metrics.getTotalSampleSize() << '\n';
    ss << "Accuracy: " << metrics.getAverageAccuracy()
       << " , precision: " << metrics.getAveragePrecision()
       << " , recall: " << metrics.getAverageRecall()
       << " , f1-measure: " << metrics.getAverageF1Measure() << '\n';
    ss << "Total time :" << time(0) - now << '\n';
    std::cout << "General time :" << time(0) - general_t << "\n";
  }
}
