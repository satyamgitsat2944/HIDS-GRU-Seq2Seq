"""
HIDS Main Pipeline
==================
Full pipeline:
1. Load ADFA-LD dataset
2. Build vocabulary
3. Train Seq2Seq model (GRU + Attention)
4. Train anomaly classifiers
5. Evaluate with metrics
6. Run real-time monitor demo
"""

import os
import sys
import torch
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import (load_adfa_ld, SyscallVocab,
                               Seq2SeqDataset, collate_fn)
from src.model.seq2seq import Seq2SeqHIDS
from src.model.trainer import train_model
from src.detection.classifier import HIDSClassifier
from src.monitor.realtime import HIDSMonitor

# ─────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────
DATA_DIR   = 'ADFA-LD/ADFA-LD/ADFA-LD'   # path to dataset
MODEL_DIR  = 'models'
LOG_DIR    = 'logs'

# Model hyperparameters (paper Section 4.2)
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 3
DROPOUT     = 0.5
BATCH_SIZE  = 64
N_EPOCHS    = 30
LR          = 0.1
WINDOW_SIZE = 20
SEQ_LENGTHS = (10, 15, 20, 25)

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("\n" + "="*60)
    print("  HIDS — GRU Seq2Seq Intrusion Detection System")
    print("  Based on: Lv et al., Beijing Jiaotong University")
    print(f"  Device: {DEVICE}")
    print("="*60 + "\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ─────────────────────────────────────────
    #  STEP 1: Load Dataset
    # ─────────────────────────────────────────
    print("STEP 1: Loading ADFA-LD Dataset")
    print("-"*40)

    normal_train, normal_val, attack_dict = load_adfa_ld(DATA_DIR)

    # Flatten all attack traces
    all_attacks = []
    for traces in attack_dict.values():
        all_attacks.extend(traces)

    # ─────────────────────────────────────────
    #  STEP 2: Build Vocabulary
    # ─────────────────────────────────────────
    print("\nSTEP 2: Building Syscall Vocabulary")
    print("-"*40)

    vocab = SyscallVocab()
    vocab.build(normal_train + normal_val + all_attacks)

    # Save vocabulary
    vocab_path = os.path.join(MODEL_DIR, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"[Vocab] Saved to {vocab_path}")

    # ─────────────────────────────────────────
    #  STEP 3: Prepare Datasets
    # ─────────────────────────────────────────
    print("\nSTEP 3: Preparing Seq2Seq Training Data")
    print("-"*40)

    # Encode traces
    normal_train_enc = [vocab.encode(t) for t in normal_train]
    normal_val_enc   = [vocab.encode(t) for t in normal_val]
    attack_enc       = {k: [vocab.encode(t) for t in v]
                        for k, v in attack_dict.items()}
    all_attacks_enc  = [vocab.encode(t) for t in all_attacks]

    # Build Seq2Seq datasets
    train_dataset = Seq2SeqDataset(
        normal_train_enc + all_attacks_enc,
        vocab, seq_lengths=SEQ_LENGTHS
    )
    val_dataset = Seq2SeqDataset(
        normal_val_enc[:500],
        vocab, seq_lengths=SEQ_LENGTHS
    )

    # ─────────────────────────────────────────
    #  STEP 4: Train Seq2Seq Model
    # ─────────────────────────────────────────
    print("\nSTEP 4: Training GRU Seq2Seq Model")
    print("-"*40)

    model = Seq2SeqHIDS(
        vocab_size  = vocab.size,
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT
    ).to(DEVICE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    history = train_model(
        model       = model,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        n_epochs    = N_EPOCHS,
        batch_size  = BATCH_SIZE,
        lr          = LR,
        device      = DEVICE,
        save_dir    = MODEL_DIR
    )

    # Load best model
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, 'best_model.pt'),
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[Model] Best model loaded "
          f"(epoch {checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['val_loss']:.4f})")

    # ─────────────────────────────────────────
    #  STEP 5: Generate Extended Sequences
    # ─────────────────────────────────────────
    print("\nSTEP 5: Generating Extended Sequences")
    print("-"*40)

    def extend_sequence(token_seq, max_pred=10):
        """Extend a sequence with Seq2Seq predictions."""
        if len(token_seq) < WINDOW_SIZE:
            return token_seq
        model.eval()
        with torch.no_grad():
            window = token_seq[:WINDOW_SIZE]
            src = torch.tensor(
                [window], dtype=torch.long).to(DEVICE)
            src_lens = torch.tensor([len(window)])

            predicted_steps = model.predict(
                src, src_lens,
                max_len=max_pred,
                device=DEVICE
            )
            predicted = []
            for step in predicted_steps:
                token = int(step[0])
                if token == 2:
                    break
                if token not in (0, 1):
                    predicted.append(token)

        return token_seq + predicted

    print("[Extend] Extending normal training sequences...")
    normal_ext = [extend_sequence(t)
                  for t in normal_train_enc[:600]]

    print("[Extend] Extending attack sequences...")
    attack_ext = [extend_sequence(t)
                  for t in all_attacks_enc[:600]]

    # ─────────────────────────────────────────
    #  STEP 6: Train Classifiers
    # ─────────────────────────────────────────
    print("\nSTEP 6: Training Anomaly Classifiers")
    print("-"*40)

    classifiers = {}
    for clf_type in ['random_forest', 'svm', 'isolation_forest']:
        clf = HIDSClassifier(
            clf_type    = clf_type,
            vocab_size  = vocab.size,
            window_size = WINDOW_SIZE
        )
        if clf_type == 'isolation_forest':
            clf.train(normal_ext)
        else:
            clf.train(normal_ext, attack_ext)

        clf_path = os.path.join(
            MODEL_DIR, f'clf_{clf_type}.pkl')
        clf.save(clf_path)
        classifiers[clf_type] = clf

    # ─────────────────────────────────────────
    #  STEP 7: Evaluate
    # ─────────────────────────────────────────
    print("\nSTEP 7: Evaluating Classifiers")
    print("-"*40)

    # Test sets
    normal_test     = normal_val_enc[:200]
    all_attacks_test = all_attacks_enc[:200]

    # Extend test sequences
    normal_test_ext  = [extend_sequence(t) for t in normal_test]
    attack_test_ext  = [extend_sequence(t) for t in all_attacks_test]

    print("\n--- Results on ORIGINAL sequences ---")
    for clf_type, clf in classifiers.items():
        if clf_type != 'isolation_forest':
            clf.evaluate(normal_test, all_attacks_test)

    print("\n--- Results on EXTENDED sequences ---")
    print("(observed + Seq2Seq predicted — paper Section 4.4)")
    for clf_type, clf in classifiers.items():
        if clf_type != 'isolation_forest':
            clf.evaluate(normal_test_ext, attack_test_ext)

    # Per attack type breakdown
    print("\n--- Per-Attack Detection Rate ---")
    rf = classifiers['random_forest']
    for attack_name, traces in attack_enc.items():
        test = [extend_sequence(t) for t in
                [vocab.encode(t) for t in
                 attack_dict[attack_name][:30]]]
        detected = sum(1 for t in test if rf.predict(t) == 1)
        dr = detected / len(test) * 100
        print(f"  {attack_name:20s}: "
              f"{detected}/{len(test)} ({dr:.1f}%)")

    # ─────────────────────────────────────────
    #  STEP 8: Real-Time Monitor Demo
    # ─────────────────────────────────────────
    print("\nSTEP 8: Real-Time Monitor Demo")
    print("-"*40)

    monitor = HIDSMonitor(
        seq2seq_model = model,
        classifier    = classifiers['random_forest'],
        vocab         = vocab,
        window_size   = WINDOW_SIZE,
        threshold     = 0.60,
        use_prediction = True,
        device        = DEVICE
    )

    # Test on a real attack trace
    first_attack_name = list(attack_dict.keys())[0]
    first_attack_trace = attack_dict[first_attack_name][0]

    print(f"\n[Monitor] Testing on: {first_attack_name}")
    alerts = monitor.watch_trace(first_attack_trace)

    # Save report
    monitor.save_report(os.path.join(LOG_DIR, 'demo_report.json'))

    print("\n" + "="*60)
    print("  Pipeline Complete!")
    print(f"  Models saved to  : {MODEL_DIR}/")
    print(f"  Logs saved to    : {LOG_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
