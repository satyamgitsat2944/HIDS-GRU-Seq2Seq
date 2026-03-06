"""
ADFA-LD Dataset Loader
Loads system call traces from the ADFA-LD dataset.
"""

import os
import torch
from torch.utils.data import Dataset
from collections import Counter

# Special tokens
PAD = 0
SOS = 1
EOS = 2
UNK = 3


class SyscallVocab:
    """Builds a vocabulary from system call IDs."""

    def __init__(self):
        self.token2idx = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<UNK>': UNK}
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.size = 4

    def build(self, traces):
        counter = Counter()
        for trace in traces:
            counter.update(trace)
        for syscall in sorted(counter.keys()):
            key = str(syscall)
            if key not in self.token2idx:
                self.token2idx[key] = self.size
                self.idx2token[self.size] = key
                self.size += 1
        print(f"[Vocab] Built vocabulary with {self.size} tokens")
        return self

    def encode(self, trace):
        return [self.token2idx.get(str(s), UNK) for s in trace]

    def decode(self, indices):
        return [self.idx2token.get(i, '<UNK>') for i in indices]


def read_trace(filepath):
    """Read a single trace file — returns list of syscall IDs."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    if not content:
        return []
    return [int(x) for x in content.split() if x.isdigit()]


def load_adfa_ld(root_dir):
    """
    Load ADFA-LD dataset.
    Returns normal_train, normal_val, attack_dict
    """
    normal_train, normal_val, attack_dict = [], [], {}

    # Normal training traces
    train_dir = os.path.join(root_dir, 'Training_Data_Master')
    for fname in os.listdir(train_dir):
        t = read_trace(os.path.join(train_dir, fname))
        if t:
            normal_train.append(t)

    # Normal validation traces
    val_dir = os.path.join(root_dir, 'Validation_Data_Master')
    for fname in os.listdir(val_dir):
        t = read_trace(os.path.join(val_dir, fname))
        if t:
            normal_val.append(t)

    # Attack traces
    attack_dir = os.path.join(root_dir, 'Attack_Data_Master')
    for attack_name in os.listdir(attack_dir):
        attack_subdir = os.path.join(attack_dir, attack_name)
        if os.path.isdir(attack_subdir):
            attack_dict[attack_name] = []
            for fname in os.listdir(attack_subdir):
                t = read_trace(os.path.join(attack_subdir, fname))
                if t:
                    attack_dict[attack_name].append(t)

    print(f"[Data] Normal train : {len(normal_train)} traces")
    print(f"[Data] Normal val   : {len(normal_val)} traces")
    for k, v in attack_dict.items():
        print(f"[Data] {k:20s}: {len(v)} traces")

    return normal_train, normal_val, attack_dict


class Seq2SeqDataset(Dataset):
    """
    PyTorch Dataset for Seq2Seq training.
    Splits each trace into source (input) and target (output to predict).
    Paper uses dynamic length segmentation (Section 3.2).
    """

    def __init__(self, traces, vocab, seq_lengths=(10, 15, 20, 25)):
        self.vocab = vocab
        self.pairs = []

        for trace in traces:
            encoded = vocab.encode(trace)
            for L in seq_lengths:
                if len(encoded) < L + 2:
                    continue
                for start in range(0, len(encoded) - L, max(1, L // 2)):
                    window = encoded[start: start + L]
                    split = int(len(window) * 0.7)
                    if split < 2 or len(window) - split < 2:
                        continue
                    src = window[:split]
                    tgt = [SOS] + window[split:] + [EOS]
                    self.pairs.append((src, tgt))

        print(f"[Dataset] Built {len(self.pairs)} sequence pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """Pads sequences to same length for batching."""
    srcs, tgts = zip(*batch)

    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = [s + [PAD] * (max_src - len(s)) for s in srcs]
    tgt_padded = [t + [PAD] * (max_tgt - len(t)) for t in tgts]

    return (torch.tensor(src_padded, dtype=torch.long),
            torch.tensor(tgt_padded, dtype=torch.long),
            torch.tensor(src_lens, dtype=torch.long),
            torch.tensor(tgt_lens, dtype=torch.long))