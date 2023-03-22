#ifndef EXPERIMENTINFO_H
#define EXPERIMENTINFO_H

#include <QDialog>

namespace Ui {
class ExperimentInfo;
}

class ExperimentInfo : public QDialog {
  Q_OBJECT

 public:
  explicit ExperimentInfo(QWidget *parent = nullptr);
  ~ExperimentInfo();

 public:
  void appendText(const std::string &text);

  void clearText();

 private slots:
  void on_ok_clicked();

 private:
  Ui::ExperimentInfo *ui;
};

#endif  // EXPERIMENTINFO_H
