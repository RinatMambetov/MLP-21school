#ifndef TRAINRUNNABLE_H
#define TRAINRUNNABLE_H
#include <QObject>

class TrainRunnable : public QObject {
  Q_OBJECT
  Q_PROPERTY(bool running READ running WRITE setRunning NOTIFY runningChanged)
  Q_PROPERTY(
      QString message READ message WRITE setMessage NOTIFY messageChanged)
 public:
  explicit TrainRunnable(QObject *parrent = 0);
  bool running() const;
  void setRunning(bool newRunning);

  const QString &message() const;
  void setMessage(const QString &newMessage);

 signals:
  void finished();
  void runningChanged();

  void messageChanged();

 public slots:
  void run();

 private:
  bool m_running;
  QString m_message;
};

#endif  // TRAINRUNNABLE_H
