#ifndef STATUSWINDOW_H
#define STATUSWINDOW_H

#include <QDialog>

namespace Ui {
class Statuswindow;
}

class Statuswindow : public QDialog {
  Q_OBJECT

 public:
  explicit Statuswindow(QWidget *parent = nullptr);
  void appendText(const std::string &text);
  ~Statuswindow();

  void clearText();
 public slots:
  void enableFinishButton();
  void disableFinishButton();

 signals:
  void terminateProcess();
 private slots:

  void on_terminateProcess_clicked();

  void on_finishButton_clicked();

 private:
  Ui::Statuswindow *ui;
};

#endif  // STATUSWINDOW_H
