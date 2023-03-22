#include "statuswindow.h"

#include <string>

#include "ui_statuswindow.h"
Statuswindow::Statuswindow(QWidget *parent)
    : QDialog(parent), ui(new Ui::Statuswindow) {
  ui->setupUi(this);
  ui->finishButton->setEnabled(false);
  setWindowFlags(Qt::Window
	| Qt::WindowMinimizeButtonHint
	| Qt::WindowMaximizeButtonHint);
}
void Statuswindow::appendText(const std::string &text) {
  ui->trainStatusInfo->appendPlainText(QString::fromStdString(text));
}

Statuswindow::~Statuswindow() { delete ui; }

void Statuswindow::enableFinishButton() {
  ui->finishButton->setEnabled(true);
  ui->terminateProcess->setEnabled(false);
}
void Statuswindow::disableFinishButton() {}

void Statuswindow::clearText() {
  QMetaObject::invokeMethod(ui->trainStatusInfo, "clear");
}

void Statuswindow::on_terminateProcess_clicked() {
  emit terminateProcess();
  this->close();
  clearText();
}

void Statuswindow::on_finishButton_clicked() {
  this->close();
  clearText();
  ui->finishButton->setEnabled(false);
  ui->terminateProcess->setEnabled(true);
}
