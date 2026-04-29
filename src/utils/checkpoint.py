"""
Save and load training checkpoints.

A checkpoint bundles everything needed to resume a training run:
    - model weights
    - optimizer state (e.g., Adam moments)
    - AMP grad scaler state (for mixed-precision training)
    - epoch number
    - best validation loss seen so far
    - patience counter (for early stopping)

Why all in one file: a single torch.save call gives atomic semantics —
either the entire state is on disk or nothing is. Splitting across files
risks resuming from an inconsistent state if the machine crashes mid-save.

Why optimizer and scaler are optional in load_checkpoint: evaluation and
inference scripts only need weights. Forcing them to instantiate an
unused optimizer + scaler would be wasteful and confusing.
"""

from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
) -> None:
    """
    Save full training state to a single .pt file.

    Args:
        path: Output path. Convention: '..._latest.pt' for the most recent
            epoch, '..._best.pt' for the best-val checkpoint.
        model: The model whose state_dict to save.
        optimizer: The optimizer whose state_dict to save.
        scaler: AMP grad scaler (or None for CPU/non-AMP training).
        epoch: Epoch number that this checkpoint represents (0-indexed).
        best_val_loss: Best validation loss seen so far across all epochs.
        patience_counter: Current early-stopping patience counter.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = 'cpu',
) -> dict:
    """
    Load a checkpoint from disk and apply state dicts in place.

    Args:
        path: Path to a .pt file written by save_checkpoint.
        model: Model to load weights into. Updated in place.
        optimizer: Optional. If given, its state is loaded too. Pass None
            for evaluation/inference where you only need weights.
        scaler: Optional. If given (and the checkpoint has scaler state),
            its state is loaded.
        device: Device to map tensors onto when loading. Default 'cpu' is
            safe — you can move the model to GPU after loading.

    Returns:
        Dict with metadata keys: 'epoch', 'best_val_loss', 'patience_counter'.
        Useful for resuming training from the exact stopped state.
    """
    # weights_only=False because we save Python floats and ints alongside
    # the tensors. Safe here because we trust our own checkpoints.
    state = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    if scaler is not None and state.get('scaler_state_dict') is not None:
        scaler.load_state_dict(state['scaler_state_dict'])

    return {
        'epoch': state['epoch'],
        'best_val_loss': state['best_val_loss'],
        'patience_counter': state['patience_counter'],
    }