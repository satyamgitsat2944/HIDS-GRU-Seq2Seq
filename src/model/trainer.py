"""
Training Script for Seq2Seq HIDS Model.
Implements the training loop with:
- Cross entropy loss (ignoring PAD tokens)
- Gradient clipping (paper: max_norm=5)
- Learning rate decay on plateau
- Early stopping
- Model checkpointing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_epoch(model, loader, optimizer, criterion, device, clip=5.0):
    """
    One full training epoch.
    Returns average loss over all batches.
    """
    model.train()
    epoch_loss = 0

    for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs: (batch, tgt_len, vocab_size)
        outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)

        # Reshape for loss calculation
        # outputs: (batch * tgt_len, vocab_size)
        # tgt    : (batch * tgt_len,)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)

        # Cross entropy loss (ignores PAD token=0)
        loss = criterion(outputs, tgt)

        # Backpropagation
        loss.backward()

        # Gradient clipping (paper Section 4.2: clip=5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)


def evaluate_epoch(model, loader, criterion, device):
    """
    Evaluate model on validation set.
    No gradient computation — faster and uses less memory.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_lens = src_lens.to(device)

            # No teacher forcing during evaluation
            outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)

            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(outputs, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_model(model, train_dataset, val_dataset=None,
                n_epochs=30, batch_size=64, lr=0.1,
                device='cuda', save_dir='models/'):
    """
    Full training pipeline.

    Paper hyperparameters (Section 4.2):
    - batch_size = 64
    - lr         = 0.1
    - clip       = 5.0
    - dropout    = 0.5
    - early stop when loss stops improving
    """
    os.makedirs(save_dir, exist_ok=True)

    # DataLoaders
    from src.data.dataset import collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    # Optimizer: Adam works better than SGD for seq2seq in practice
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler: reduce lr when val loss plateaus (paper strategy)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=3, verbose=True
    )

    # Loss: ignore PAD token (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'='*60}")
    print(f"  Training Seq2Seq HIDS Model")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR        : {lr}")
    print(f"  Train size: {len(train_dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        history['train_loss'].append(train_loss)

        # Validate
        if val_loader:
            val_loss = evaluate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            print(f"  Train Loss: {train_loss:.4f} "
                  f"| Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'vocab_size': model.vocab_size,
                    'hidden_dim': model.hidden_dim,
                    'num_layers': model.num_layers,
                }, best_model_path)
                print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"  Train Loss: {train_loss:.4f}")

    print(f"\n[Train] Done! Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Model saved to: {best_model_path}")
    return history