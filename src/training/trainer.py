"""
3D-SwinSTB training loop.

Implements Pan et al. Section IV-F (training rules) plus standard
production-quality additions:
    - Mixed precision (AMP) on CUDA, full precision on CPU
    - Gradient clipping (max_norm=1.0) for stability with batch size 1
    - Early stopping (patience=4, min improvement 0.01% of best)
    - Latest + best checkpoint pattern for Colab session resumption
    - CSV training log for post-hoc analysis

What this does NOT do (intentionally):
    - LR scheduling (Pan et al. use constant lr=0.001)
    - TensorBoard / W&B integration (CSV is enough)
    - Multi-GPU
    - In-training visualisations (kept for separate evaluation phase)

The trainer is designed so that a single Colab session that times out
mid-training can resume cleanly: every epoch, both _latest.pt (always)
and _best.pt (when val improves) are written to disk, and the CSV log
is appended-to. Pass --resume to scripts/05_train_fm.py to continue.
"""

import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import SpectrogramSequenceDataset
from src.model.swinstb import SwinSTB
from src.utils.checkpoint import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch building blocks
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    use_amp: bool,
    grad_clip_max_norm: float = 1.0,
    smoke_test_max_batches: Optional[int] = None,
) -> float:
    """
    Run one training epoch. Returns mean train loss.

    Standard AMP loop:
        autocast → forward → loss → scaler.scale(loss).backward()
        → scaler.unscale_(optimizer) so we can clip on real-magnitude grads
        → clip_grad_norm_
        → scaler.step(optimizer) → scaler.update()

    On CPU (use_amp=False), autocast is a no-op and scaler is disabled.
    The same code path runs in both cases.
    """
    model.train()
    losses = []
    n_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        if smoke_test_max_batches is not None and batch_idx >= smoke_test_max_batches:
            break

        inputs = inputs.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu',
                                enabled=use_amp):
            preds = model(inputs)
            loss = F.mse_loss(preds, targets)  # mean reduction

        # Backward + clip + step. scaler is a no-op on CPU.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        # Light progress print every 50 batches (enough info, not too noisy)
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            running = sum(losses) / len(losses)
            print(f"  batch {batch_idx+1}/{n_batches}  "
                  f"loss={loss.item():.6f}  running_avg={running:.6f}",
                  flush=True)

    return float(np.mean(losses)) if losses else float('nan')


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
    smoke_test_max_batches: Optional[int] = None,
) -> float:
    """
    Run validation pass (no_grad, no scaler). Returns mean val loss.

    Uses autocast in val too: if we use mixed precision in training but
    full precision in val, the val numbers wouldn't reflect what the model
    "feels" during training.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if smoke_test_max_batches is not None and batch_idx >= smoke_test_max_batches:
                break

            inputs = inputs.to(device, non_blocking=use_amp)
            targets = targets.to(device, non_blocking=use_amp)

            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu',
                                    enabled=use_amp):
                preds = model(inputs)
                loss = F.mse_loss(preds, targets)

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, resume: bool = False, force_restart: bool = False,
          smoke_test: bool = False, seed: int = 42) -> dict:
    """
    Run end-to-end training as specified by `config`.

    Args:
        config: nested dict from configs/default.yaml.
        resume: if True, load checkpoint and continue. Fails if no checkpoint.
        force_restart: if True, ignore an existing _latest.pt and start fresh.
        smoke_test: if True, run 2 epochs × 10 batches each for pipeline
            verification. Real training runs the configured num_epochs.
        seed: RNG seed for torch + numpy reproducibility.

    Returns:
        Dict with keys: 'final_epoch', 'best_val_loss', 'best_epoch',
        'history' (list of per-epoch metric dicts).
    """
    # ─── RNG seed ────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ─── Device & AMP detection ──────────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    device = 'cuda' if use_amp else 'cpu'
    if not use_amp:
        print('=' * 70)
        print('WARNING: Running on CPU. This is intended for sanity-checking only.')
        print('Real training requires a CUDA GPU. One epoch on FM data would')
        print('take many hours on CPU. Use --smoke-test for a quick pipeline check.')
        print('=' * 70)
    else:
        # Modest A100 speedup once input shapes are stable
        torch.backends.cudnn.benchmark = True

    # ─── Paths from config ───────────────────────────────────────────────────
    paths = config['paths']
    cache_path = paths['processed_fm']
    checkpoint_dir = paths['checkpoint_dir']
    output_dir = paths['output_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    latest_path = os.path.join(checkpoint_dir, 'swinstb_fm_latest.pt')
    best_path = os.path.join(checkpoint_dir, 'swinstb_fm_best.pt')
    log_path = os.path.join(output_dir, 'training_log.csv')

    # ─── Datasets and dataloaders ────────────────────────────────────────────
    seq_cfg = config['data']['sequence']
    train_ds = SpectrogramSequenceDataset(
        cache_path=cache_path, split='train',
        input_length=seq_cfg['input_length'],
        target_length=seq_cfg['target_length'],
    )
    val_ds = SpectrogramSequenceDataset(
        cache_path=cache_path, split='val',
        input_length=seq_cfg['input_length'],
        target_length=seq_cfg['target_length'],
    )
    print(f"Train: {len(train_ds)} examples")
    print(f"Val:   {len(val_ds)} examples")

    train_cfg = config['training']
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=use_amp,  # only useful on CUDA
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        pin_memory=use_amp,
    )

    # ─── Model, optimizer, scaler ────────────────────────────────────────────
    model_cfg = config['model']
    model = SwinSTB(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        embed_dim=model_cfg['feature_size'],
        patch_size=tuple(model_cfg['patch_size']),
        window_size=tuple(model_cfg['window_size']),
        encoder_depths=tuple(model_cfg['depths'][:3]),
        encoder_heads=tuple(model_cfg['num_heads'][:3]),
        decoder_depths=(2, 4, 2),
        decoder_heads=(16, 8, 4),
        bottleneck_depth=2,
        mlp_ratio=model_cfg['mlp_ratio'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ─── Resume vs fresh-start ───────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    if resume:
        if not os.path.exists(latest_path):
            raise FileNotFoundError(
                f"--resume given but no checkpoint at {latest_path}"
            )
        meta = load_checkpoint(latest_path, model, optimizer, scaler, device=device)
        start_epoch = meta['epoch'] + 1
        best_val_loss = meta['best_val_loss']
        patience_counter = meta['patience_counter']
        print(f"Resumed from epoch {meta['epoch']} "
              f"(best val so far: {best_val_loss:.6f})")
    elif os.path.exists(latest_path) and not force_restart:
        raise FileExistsError(
            f"Found existing checkpoint at {latest_path}. "
            f"Pass --resume to continue or --force-restart to overwrite."
        )

    # ─── CSV log ─────────────────────────────────────────────────────────────
    log_columns = ['epoch', 'train_loss', 'val_loss', 'best_val_loss',
                   'patience', 'lr', 'gpu_mem_gb', 'elapsed_sec']
    # Fresh write only if not resuming
    write_header = not (resume and os.path.exists(log_path))
    csv_file = open(log_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(log_columns)
        csv_file.flush()

    # ─── Smoke test bounds ───────────────────────────────────────────────────
    smoke_max_batches = 10 if smoke_test else None
    n_epochs = 2 if smoke_test else train_cfg['num_epochs']
    if smoke_test:
        print('SMOKE TEST: 2 epochs × 10 batches each')

    # ─── Early-stopping config ───────────────────────────────────────────────
    es_cfg = train_cfg['early_stopping']
    patience = es_cfg['patience']
    min_improvement = es_cfg['min_improvement']

    # ─── Training loop ───────────────────────────────────────────────────────
    history = []
    print()
    for epoch in range(start_epoch, n_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        print(f"=== Epoch {epoch}/{n_epochs - 1} ===")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device=device, use_amp=use_amp,
            grad_clip_max_norm=1.0,
            smoke_test_max_batches=smoke_max_batches,
        )
        val_loss = validate(
            model, val_loader, device=device, use_amp=use_amp,
            smoke_test_max_batches=smoke_max_batches,
        )

        # Early stopping check
        improved = val_loss < best_val_loss * (1.0 - min_improvement)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - epoch_start
        gpu_mem = (torch.cuda.max_memory_allocated() / 1e9
                   if torch.cuda.is_available() else 0.0)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"best={best_val_loss:.6f}  patience={patience_counter}/{patience}  "
              f"elapsed={elapsed:.1f}s  gpu_mem={gpu_mem:.2f}GB"
              f"{'  ★ improved' if improved else ''}")

        # Always save latest
        save_checkpoint(latest_path, model, optimizer, scaler,
                        epoch=epoch, best_val_loss=best_val_loss,
                        patience_counter=patience_counter)
        # Save best on improvement
        if improved:
            save_checkpoint(best_path, model, optimizer, scaler,
                            epoch=epoch, best_val_loss=best_val_loss,
                            patience_counter=patience_counter)

        # CSV log
        csv_writer.writerow([
            epoch, f'{train_loss:.6f}', f'{val_loss:.6f}',
            f'{best_val_loss:.6f}', patience_counter,
            f'{current_lr:.6g}', f'{gpu_mem:.3f}', f'{elapsed:.1f}',
        ])
        csv_file.flush()

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'best_val_loss': best_val_loss, 'patience': patience_counter,
            'gpu_mem_gb': gpu_mem, 'elapsed_sec': elapsed,
        })

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement in {patience} epochs).")
            break

    csv_file.close()

    return {
        'final_epoch': history[-1]['epoch'] if history else start_epoch - 1,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'history': history,
    }