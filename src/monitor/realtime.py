"""
Real-Time HIDS Monitor
Watches incoming syscall stream and raises alerts.

Alert Levels:
    LOW      : score 0.50 - 0.60
    MEDIUM   : score 0.60 - 0.75
    HIGH     : score 0.75 - 0.85
    CRITICAL : score > 0.85
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from collections import deque


class AlertLevel:
    LOW      = 'LOW'
    MEDIUM   = 'MEDIUM'
    HIGH     = 'HIGH'
    CRITICAL = 'CRITICAL'


class Alert:
    def __init__(self, level, score, 
                 observed_syscalls, predicted_syscalls):
        self.timestamp         = datetime.now().isoformat()
        self.level             = level
        self.score             = score
        self.observed_syscalls = observed_syscalls
        self.predicted_syscalls = predicted_syscalls

    def to_dict(self):
        return {
            'timestamp'          : self.timestamp,
            'level'              : self.level,
            'anomaly_score'      : round(self.score, 4),
            'observed_syscalls'  : self.observed_syscalls,
            'predicted_syscalls' : self.predicted_syscalls,
        }

    def __str__(self):
        return (
            f"[{self.timestamp}] "
            f"[{self.level}] "
            f"Anomaly score: {self.score:.3f}\n"
            f"  Observed  : {self.observed_syscalls[-8:]}\n"
            f"  Predicted : {self.predicted_syscalls}"
        )


class HIDSMonitor:
    """
    Real-time sliding window monitor.
    
    How it works:
    1. Maintains a rolling buffer of recent syscall tokens
    2. Every time the buffer fills a window, it:
       a. Encodes the window using Seq2Seq encoder
       b. Predicts next syscalls using decoder
       c. Extends sequence: observed + predicted
       d. Classifies extended sequence
       e. Raises alert if score exceeds threshold
    """

    def __init__(self, seq2seq_model, classifier,
                 vocab, window_size=20,
                 threshold=0.60, use_prediction=True,
                 device='cpu'):

        self.seq2seq         = seq2seq_model
        self.classifier      = classifier
        self.vocab           = vocab
        self.window_size     = window_size
        self.threshold       = threshold
        self.use_prediction  = use_prediction
        self.device          = device

        # Rolling buffer of token IDs
        self.buffer = deque(maxlen=window_size * 3)

        # Alert history
        self.alerts = []

        # Stats
        self.stats = {
            'total_syscalls'   : 0,
            'windows_analyzed' : 0,
            'alerts_raised'    : 0,
            'start_time'       : None,
        }

    def _score_to_level(self, score):
        if score < 0.60:
            return AlertLevel.LOW
        elif score < 0.75:
            return AlertLevel.MEDIUM
        elif score < 0.85:
            return AlertLevel.HIGH
        else:
            return AlertLevel.CRITICAL

    def _get_prediction(self, token_window):
        """
        Use Seq2Seq model to predict next syscalls.
        Returns list of predicted token IDs.
        """
        if self.seq2seq is None or not self.use_prediction:
            return []

        try:
            self.seq2seq.eval()
            src = torch.tensor(
                [token_window], dtype=torch.long
            ).to(self.device)
            src_lens = torch.tensor(
                [len(token_window)], dtype=torch.long
            )

            predicted_steps = self.seq2seq.predict(
                src, src_lens,
                max_len=10,
                device=self.device
            )

            # Flatten predictions
            predicted = []
            for step in predicted_steps:
                token = int(step[0])
                if token == 2:  # EOS
                    break
                if token not in (0, 1):  # skip PAD, SOS
                    predicted.append(token)

            return predicted

        except Exception as e:
            return []

    def process_syscall(self, syscall_id):
        """
        Process one incoming syscall.
        Returns Alert if anomaly detected, else None.
        """
        self.stats['total_syscalls'] += 1

        # Encode syscall ID to token
        token = self.vocab.token2idx.get(
            str(syscall_id),
            self.vocab.token2idx.get('<UNK>', 3)
        )
        self.buffer.append(token)

        # Wait until buffer has enough data
        if len(self.buffer) < self.window_size:
            return None

        self.stats['windows_analyzed'] += 1

        # Get current window
        window = list(self.buffer)[-self.window_size:]

        # Predict next syscalls
        predicted = self._get_prediction(window)

        # Extended sequence: observed + predicted (paper Section 4.4)
        extended = window + predicted

        # Classify
        score = self.classifier.predict_proba(extended)

        if score >= self.threshold:
            level = self._score_to_level(score)

            # Decode tokens back to syscall names for readability
            observed_names  = self.vocab.decode(window[-8:])
            predicted_names = self.vocab.decode(predicted[:5])

            alert = Alert(
                level=level,
                score=score,
                observed_syscalls=observed_names,
                predicted_syscalls=predicted_names
            )

            self.alerts.append(alert)
            self.stats['alerts_raised'] += 1
            return alert

        return None

    def watch_trace(self, syscall_list, verbose=True):
        """
        Process a full list of syscall IDs.
        Used for testing on ADFA-LD trace files.
        """
        self.stats['start_time'] = time.time()

        if verbose:
            print(f"\n[Monitor] Watching trace "
                  f"({len(syscall_list)} syscalls)...")

        raised_alerts = []
        for syscall_id in syscall_list:
            alert = self.process_syscall(syscall_id)
            if alert:
                raised_alerts.append(alert)
                if verbose:
                    print(str(alert))

        elapsed = time.time() - self.stats['start_time']

        if verbose:
            print(f"\n[Monitor] Done in {elapsed:.2f}s")
            print(f"[Monitor] Syscalls    : "
                  f"{self.stats['total_syscalls']}")
            print(f"[Monitor] Windows     : "
                  f"{self.stats['windows_analyzed']}")
            print(f"[Monitor] Alerts      : "
                  f"{self.stats['alerts_raised']}")

        return raised_alerts

    def watch_file(self, filepath, verbose=True):
        """
        Monitor a trace file directly.
        File format: space separated syscall IDs on one line.
        e.g: '4 4 21 197 4 11 45 ...'
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            content = f.read().strip()

        syscalls = [int(x) for x in content.split()
                    if x.isdigit()]
        return self.watch_trace(syscalls, verbose=verbose)

    def save_report(self, path='logs/report.json'):
        """Save full alert report to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report = {
            'stats' : self.stats,
            'alerts': [a.to_dict() for a in self.alerts]
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[Monitor] Report saved to {path}")
        return report
