#include "experimentinfo.h"

#include "ui_experimentinfo.h"

ExperimentInfo::ExperimentInfo(QWidget *parent)
    : QDialog(parent), ui(new Ui::ExperimentInfo) {
  ui->setupUi(this);
}

ExperimentInfo::~ExperimentInfo() { delete ui; }

void ExperimentInfo::appendText(const std::string &text) {
  ui->info->appendPlainText(QString::fromStdString(text));
}

void ExperimentInfo::clearText() {
  QMetaObject::invokeMethod(ui->info, "clear");
}

void ExperimentInfo::on_ok_clicked() { this->close(); }
