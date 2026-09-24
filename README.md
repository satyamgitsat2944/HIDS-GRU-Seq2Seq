# HIDS-GRU-Seq2Seq

## Host Intrusion Detection System using GRU Seq2Seq with Attention

A deep learning-based Host Intrusion Detection System (HIDS) that detects abnormal system behavior from system-call sequences.

The project uses a GRU-based Sequence-to-Sequence (Seq2Seq) model with Bahdanau Attention to learn sequential patterns in system calls. The predicted sequences are then combined with the observed sequences and passed to machine learning classifiers to identify normal and malicious behavior.

---

## 🚀 Features

- System-call sequence based intrusion detection
- GRU-based Seq2Seq Encoder-Decoder architecture
- Bahdanau Attention mechanism
- Sequence prediction for future system calls
- Multiple anomaly detection algorithms:
  - Random Forest
  - SVM with RBF kernel
  - Isolation Forest
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - False Positive Rate
  - True Positive Rate
  - Confusion Matrix
- Per-attack detection analysis
- Real-time monitoring and alert generation

---

## 🧠 System Architecture

The system follows a two-stage detection pipeline:

```text
             System Call Sequence
                     |
                     v
          +---------------------+
          |   GRU Seq2Seq       |
          |      Encoder        |
          +---------------------+
                     |
                     v
          +---------------------+
          | Bahdanau Attention  |
          +---------------------+
                     |
                     v
          +---------------------+
          |   GRU Decoder       |
          +---------------------+
                     |
                     v
          Predicted Syscalls
                     |
                     v
       Observed + Predicted Sequence
                     |
                     v
          +---------------------+
          | Feature Extraction  |
          +---------------------+
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Random      SVM      Isolation
       Forest                Forest
          |          |          |
          +----------+----------+
                     |
                     v
             Normal / Attack
