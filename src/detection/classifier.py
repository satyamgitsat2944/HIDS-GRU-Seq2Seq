"""
Anomaly Detection Classifier
Second stage of the HIDS pipeline (Paper Section 3.1, Fig. 2)

Pipeline:
    observed_seq  ──→  Seq2Seq  ──→  predicted_seq
                                           │
    observed_seq + predicted_seq = extended_seq
                                           │
                                    classifier
                                           │
                                  normal / intrusion
                                  
Paper Section 4.4 shows extended sequences improve
detection performance across all classifiers.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix)


def extract_features(token_sequence, vocab_size, window_size=20):
    """
    Extract statistical features from a syscall sequence.
    
    Features per window:
    - Frequency histogram (how often each syscall appears)
    - Normalized counts
    
    Then we pool across all windows using mean + std
    giving us a fixed-size feature vector regardless
    of sequence length.
    """
    if len(token_sequence) < window_size:
        # Pad short sequences
        token_sequence = token_sequence + [0] * (
            window_size - len(token_sequence))

    windows = []
    for i in range(len(token_sequence) - window_size + 1):
        window = token_sequence[i: i + window_size]
        # Frequency histogram normalized by window size
        hist = np.bincount(window, minlength=vocab_size).astype(float)
        hist /= window_size
        windows.append(hist)

    windows = np.array(windows)

    # Mean + std pooling across windows
    # Captures both average behavior and variability
    mean_feat = windows.mean(axis=0)
    std_feat = windows.std(axis=0)

    return np.concatenate([mean_feat, std_feat])


class HIDSClassifier:
    """
    Anomaly classifier for the HIDS second stage.
    
    Supports three classifiers from the paper (Section 4.3):
    - Random Forest  (best overall performance)
    - SVM with RBF   (good precision)
    - Isolation Forest (unsupervised, no attack data needed)
    """

    def __init__(self, clf_type='random_forest',
                 vocab_size=200, window_size=20):
        self.clf_type = clf_type
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = self._build()
        self.is_trained = False

    def _build(self):
        if self.clf_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1      # use all CPU cores
            )
        elif self.clf_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif self.clf_type == 'isolation_forest':
            return IsolationForest(
                contamination=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier: {self.clf_type}")

    def _featurize(self, traces):
        """Convert list of token sequences to feature matrix."""
        return np.array([
            extract_features(t, self.vocab_size, self.window_size)
            for t in traces
        ])

    def train(self, normal_traces, attack_traces=None):
        """
        Train classifier.
        
        normal_traces: encoded token lists — label 0
        attack_traces: encoded token lists — label 1
        
        Isolation Forest only needs normal traces (unsupervised).
        RF and SVM need both (supervised).
        """
        print(f"\n[Classifier] Training {self.clf_type}...")

        X_normal = self._featurize(normal_traces)
        X_normal = self.scaler.fit_transform(X_normal)

        if self.clf_type == 'isolation_forest':
            self.model.fit(X_normal)

        else:
            if not attack_traces:
                raise ValueError(
                    f"{self.clf_type} needs attack traces!")

            X_attack = self._featurize(attack_traces)
            X_attack = self.scaler.transform(X_attack)

            X = np.vstack([X_normal, X_attack])
            y = np.array(
                [0] * len(X_normal) + [1] * len(X_attack))

            self.model.fit(X, y)

        self.is_trained = True
        print(f"[Classifier] Training complete!")
        return self

    def predict(self, token_sequence):
        """
        Predict single sequence.
        Returns 0 (normal) or 1 (attack).
        """
        feat = extract_features(
            token_sequence, self.vocab_size, self.window_size)
        feat = self.scaler.transform(feat.reshape(1, -1))

        if self.clf_type == 'isolation_forest':
            pred = self.model.predict(feat)[0]
            return 1 if pred == -1 else 0
        else:
            return int(self.model.predict(feat)[0])

    def predict_proba(self, token_sequence):
        """
        Returns anomaly probability between 0 and 1.
        Higher = more likely to be an attack.
        """
        feat = extract_features(
            token_sequence, self.vocab_size, self.window_size)
        feat = self.scaler.transform(feat.reshape(1, -1))

        if self.clf_type == 'isolation_forest':
            score = -self.model.decision_function(feat)[0]
            return float(1 / (1 + np.exp(-score * 5)))
        else:
            proba = self.model.predict_proba(feat)[0]
            return float(proba[1])

    def evaluate(self, normal_test, attack_test):
        """
        Full evaluation with metrics from the paper:
        - AUC score
        - Accuracy, Precision, Recall, F1
        - False Positive Rate (FPR)
        - Confusion Matrix
        """
        X_normal = self.scaler.transform(self._featurize(normal_test))
        X_attack = self.scaler.transform(self._featurize(attack_test))

        X = np.vstack([X_normal, X_attack])
        y_true = np.array(
            [0] * len(X_normal) + [1] * len(X_attack))

        if self.clf_type == 'isolation_forest':
            raw = self.model.predict(X)
            y_pred = np.where(raw == -1, 1, 0)
            scores = -self.model.decision_function(X)
        else:
            y_pred = self.model.predict(X)
            scores = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        auc = roc_auc_score(y_true, scores)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Attack'],
            output_dict=True
        )

        results = {
            'auc': auc,
            'accuracy': report['accuracy'],
            'precision': report['Attack']['precision'],
            'recall': report['Attack']['recall'],
            'f1': report['Attack']['f1-score'],
            'fpr': cm[0][1] / max(cm[0].sum(), 1),
            'tpr': cm[1][1] / max(cm[1].sum(), 1),
            'confusion_matrix': cm
        }

        # Print results
        print(f"\n{'='*50}")
        print(f"  {self.clf_type.upper()} Results")
        print(f"{'='*50}")
        print(f"  AUC       : {auc:.4f}")
        print(f"  Accuracy  : {results['accuracy']:.4f}")
        print(f"  Precision : {results['precision']:.4f}")
        print(f"  Recall    : {results['recall']:.4f}")
        print(f"  F1 Score  : {results['f1']:.4f}")
        print(f"  FPR       : {results['fpr']:.4f}")
        print(f"  TPR       : {results['tpr']:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[Classifier] Saved to {path}")

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, 'rb') as f:
            obj.__dict__ = pickle.load(f)
        return obj