#include "trainrunnable.h"

#include <QDebug>
TrainRunnable::TrainRunnable(QObject *parrent) : QObject(parrent) {}

void TrainRunnable::run() {
  int count = 0;
  while (m_running) {
    count++;
    qDebug() << m_message << ' ' << count;
  }
  emit finished();
}

bool TrainRunnable::running() const { return m_running; }

void TrainRunnable::setRunning(bool newRunning) {
  if (m_running == newRunning) return;
  m_running = newRunning;
  emit runningChanged();
}

const QString &TrainRunnable::message() const { return m_message; }

void TrainRunnable::setMessage(const QString &newMessage) {
  if (m_message == newMessage) return;
  m_message = newMessage;
  emit messageChanged();
}
