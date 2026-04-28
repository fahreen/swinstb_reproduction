# 3D-SwinSTB Reproduction Reference Card

**Paper:** Pan et al. (2025), "Spectrum Prediction With Deep 3D Pyramid Vision Transformer Learning," IEEE TWC Vol. 24 No. 1.
**Scope:** Reproduce 3D-SwinSTB only (the spectrogram predictor). 3D-SwinLinear is out of scope.
**Target platform:** Google Colab Pro with A100 (40 GB VRAM).
**Framework:** PyTorch + MONAI (SwinUNETR as base).

---

## 1. Architecture Specification

### Input and output

| | Shape (PyTorch convention) |
|---|---|
| Input | `(batch, 3, 20, 256, 256)` — (B, C, T, H, W) |
| Output | `(batch, 3, 20, 256, 256)` — same shape |

Note: PyTorch conventions put channels second. Pan et al.'s paper notation is `(T, H, W, C)`; convert when reading between them.

### Encoder (Sα3D)

| Stage | Operation | Output shape (without batch) |
|---|---|---|
| — | Input | 3 × 20 × 256 × 256 |
| 1a | 3D Patch Partition + Linear Embedding (one Conv3D) | 96 × 10 × 64 × 64 |
| 1b | Swin block × 2 (heads=4) → **S¹_en** | 96 × 10 × 64 × 64 |
| 2a | Patch Merging (spatial /2, channels ×2) | 192 × 10 × 32 × 32 |
| 2b | Swin block × 4 (heads=8) → **S²_en** | 192 × 10 × 32 × 32 |
| 3a | Patch Merging | 384 × 10 × 16 × 16 |
| 3b | Swin block × 2 (heads=16) → **S³_en** | 384 × 10 × 16 × 16 |

### Bottleneck

| Stage | Operation | Output shape |
|---|---|---|
| 4 | Swin block × 2 | 384 × 10 × 16 × 16 |

### Predictor (Dβ3D)

| Stage | Operation | Output shape |
|---|---|---|
| 5 | Concat(bottleneck, S³_en) → Swin block × 2 (heads=16) | 768 × 10 × 16 × 16 |
| 6 | Patch Expanding (spatial ×2, channels /2) | 384 × 10 × 32 × 32 |
| 7 | Concat with S²_en → Swin block × 4 (heads=8) | 576 × 10 × 32 × 32 |
| 8 | Patch Expanding | 288 × 10 × 64 × 64 |
| 9 | Concat with S¹_en → Swin block × 2 (heads=4) | 384 × 10 × 64 × 64 |
| 10 | 3D Projection Layer (transposed conv + channel reductions) | 3 × 20 × 256 × 256 |

### Hyperparameters (from Section VI-A)

| Parameter | Value |
|---|---|
| Patch size (Tp, Hp, Wp) | (2, 4, 4) |
| Window size (P, M, M) | (2, 7, 7) |
| Base channel dim C | 96 |
| Encoder block counts | {2, 4, 2} |
| Encoder head counts | {4, 8, 16} |
| Bottleneck blocks | 2 |
| Predictor block counts | {2, 4, 2} |
| Predictor head counts | {16, 8, 4} |
| MLP expansion ratio | **2** (not default 4 — override!) |

### 3D Projection Layer (final output step)

```python
nn.Sequential(
    # Main upsampler: (10, 64, 64) → (20, 256, 256), channels stay at 96
    nn.ConvTranspose3d(96, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4)),
    # Repeated channel reduction: 96 → 48 → 24 → 12 → 6 → 3
    nn.ConvTranspose3d(96, 48, kernel_size=1),
    nn.ConvTranspose3d(48, 24, kernel_size=1),
    nn.ConvTranspose3d(24, 12, kernel_size=1),
    nn.ConvTranspose3d(12, 6, kernel_size=1),
    nn.ConvTranspose3d(6, 3, kernel_size=1),
)
```

---

## 2. MONAI SwinUNETR Configuration

Starting point — use as-is with overrides (Option A, fastest path):

```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(20, 256, 256),          # (T, H, W)
    in_channels=3,
    out_channels=3,
    depths=(2, 4, 2, 2),              # encoder {2,4,2} + bottleneck 2
    num_heads=(4, 8, 16, 16),
    feature_size=96,                   # = C
    use_v2=True,
    spatial_dims=3,
)
```

### Known mismatches with Pan et al.

| Component | SwinUNETR default | Pan et al. | Action |
|---|---|---|---|
| Patch size | (2, 2, 2) | (2, 4, 4) | Override if exposed, else accept |
| Window size | (7, 7, 7) | (2, 7, 7) | Override if exposed, else accept |
| MLP ratio | 4 | 2 | Override |
| Decoder | Transposed conv blocks | Patch Expanding + Swin blocks | **Accept for Option A** |
| Output head | Single 1×1 conv | 5-step channel reduction | **Accept for Option A** |

**Verify all defaults by printing the instantiated model and checking against Pan et al.'s architecture before training.**

---

## 3. Dataset Preprocessing Pipeline

### Step 1 — Parse one FM file

```python
import numpy as np

def parse_fm_file(filepath):
    """Parse one .xls file. Returns complex I/Q array of length 32508."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Row 0 is header: -166 12 092023 118.7905 31.9378 12.10
    # First two values are I0, Q0
    header = lines[0].split()
    i0, q0 = int(header[0]), int(header[1])
    
    # Rows 1-32507 are I Q pairs
    iq_pairs = [(int(line.split()[0]), int(line.split()[1])) 
                for line in lines[1:]]
    
    # Combine: length 32508 complex array
    i_samples = np.array([i0] + [pair[0] for pair in iq_pairs])
    q_samples = np.array([q0] + [pair[1] for pair in iq_pairs])
    iq_complex = i_samples + 1j * q_samples
    
    return iq_complex  # shape (32508,), dtype complex128
```

### Step 2 — STFT and convert to RGB spectrogram

```python
import scipy.signal
import matplotlib.cm as cm

def iq_to_spectrogram(iq_complex, out_size=256):
    """I/Q samples → 256x256x3 RGB spectrogram via Jet colormap."""
    # STFT: Hann window, length 256, no overlap (hop = 256)
    # Produces (256 frequency bins, 126 time windows)
    f, t, Zxx = scipy.signal.stft(
        iq_complex,
        fs=31_250_000,      # effective sample rate (125 MHz / 4 decimation)
        window='hann',
        nperseg=256,
        noverlap=0,
        return_onesided=False,  # full spectrum (complex I/Q)
    )
    
    # Power spectrum
    spectrogram = np.abs(Zxx) ** 2  # shape (256, ~127)
    
    # Log compression (standard practice, paper silent)
    spectrogram = 10 * np.log10(spectrogram + 1e-12)
    
    # Normalize to [0, 1] for colormap
    s_min, s_max = spectrogram.min(), spectrogram.max()
    spectrogram = (spectrogram - s_min) / (s_max - s_min + 1e-12)
    
    # Resize to 256x256 (bilinear)
    from PIL import Image
    img = Image.fromarray((spectrogram * 255).astype(np.uint8))
    img = img.resize((out_size, out_size), Image.BILINEAR)
    spectrogram_resized = np.array(img) / 255.0
    
    # Apply Jet colormap → RGB
    rgb = cm.jet(spectrogram_resized)[:, :, :3]  # drop alpha
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    
    return rgb_uint8  # shape (256, 256, 3)
```

### Step 3 — Cache all files to disk (do this ONCE)

```python
import os
from tqdm import tqdm

def preprocess_all_files(input_dir, output_path):
    """Preprocess all files and save as single numpy array."""
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.xls')])
    
    all_spectrograms = np.zeros((len(files), 256, 256, 3), dtype=np.uint8)
    
    for i, filename in enumerate(tqdm(files)):
        filepath = os.path.join(input_dir, filename)
        iq = parse_fm_file(filepath)
        spec = iq_to_spectrogram(iq)
        all_spectrograms[i] = spec
    
    # Save to Drive
    np.save(output_path, all_spectrograms)
    print(f"Saved {len(files)} spectrograms to {output_path}")

# Usage:
# preprocess_all_files('/content/drive/MyDrive/fm_xls_files/',
#                      '/content/drive/MyDrive/fm_spectrograms.npy')
```

**Run this once.** Takes ~2 hours for 10,777 files. After this, training loads the pre-computed `.npy` file in seconds.

### Step 4 — PyTorch Dataset class

```python
import torch
from torch.utils.data import Dataset

class SpectrogramSequenceDataset(Dataset):
    def __init__(self, spectrograms_path, split='train', T=20, K=20):
        self.data = np.load(spectrograms_path, mmap_mode='r')  # (N, 256, 256, 3)
        self.T = T  # input frames
        self.K = K  # target frames
        self.window = T + K  # total span per example
        
        N = len(self.data)
        train_end = int(N * 4 / 6)     # 4:1:1 chronological split
        val_end = int(N * 5 / 6)
        
        if split == 'train':
            self.start, self.end = 0, train_end
        elif split == 'val':
            self.start, self.end = train_end, val_end
        elif split == 'test':
            self.start, self.end = val_end, N
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def __len__(self):
        return self.end - self.start - self.window + 1
    
    def __getitem__(self, idx):
        start = self.start + idx
        # Get T+K frames, first T are input, last K are target
        seq = self.data[start:start + self.window].copy()  # (T+K, 256, 256, 3)
        seq_float = seq.astype(np.float32) / 255.0         # normalize to [0,1]
        
        # Convert to (C, frames, H, W) PyTorch convention
        seq_tensor = torch.from_numpy(seq_float).permute(3, 0, 1, 2)  # (3, T+K, 256, 256)
        
        input_seq = seq_tensor[:, :self.T]   # (3, 20, 256, 256)
        target_seq = seq_tensor[:, self.T:]  # (3, 20, 256, 256)
        
        return input_seq, target_seq
```

---

## 4. Training Loop

### Setup

```python
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

device = 'cuda'
model = SwinUNETR(...).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

train_dataset = SpectrogramSequenceDataset(path, split='train')
val_dataset = SpectrogramSequenceDataset(path, split='val')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
```

### Training loop with AMP and checkpointing

```python
MAX_EPOCHS = 20
PATIENCE = 4
IMPROVEMENT_THRESHOLD = 0.0001  # 0.01%

best_val_loss = float('inf')
patience_counter = 0

# Resume from checkpoint if available
start_epoch = 0
checkpoint_path = '/content/drive/MyDrive/swinstb_checkpoint.pt'
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    patience_counter = ckpt['patience_counter']
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, MAX_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch}'):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with autocast():
            preds = model(inputs)
            loss = F.mse_loss(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast():
                preds = model(inputs)
                val_loss += F.mse_loss(preds, targets).item()
    
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f'Epoch {epoch}: train={avg_train:.4f} val={avg_val:.4f}')
    
    # Early stopping check
    if avg_val < best_val_loss * (1 - IMPROVEMENT_THRESHOLD):
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), 
                   '/content/drive/MyDrive/swinstb_best.pt')
    else:
        patience_counter += 1
    
    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }, checkpoint_path)
    
    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch}')
        break
```

---

## 5. Evaluation

### Per-frame metrics

```python
import torchmetrics
import lpips

psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_fn = lpips.LPIPS(net='alex').to(device)

def evaluate(model, test_loader, K=20):
    """Returns arrays of per-frame MSE, PSNR, SSIM, LPIPS averaged over test set."""
    model.eval()
    n_batches = 0
    mse_acc = np.zeros(K)
    psnr_acc = np.zeros(K)
    ssim_acc = np.zeros(K)
    lpips_acc = np.zeros(K)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)  # (1, 3, 20, 256, 256)
            
            # Per-frame metrics
            for k in range(K):
                p = preds[:, :, k]      # (1, 3, 256, 256)
                t = targets[:, :, k]    # (1, 3, 256, 256)
                
                mse_acc[k] += F.mse_loss(p, t, reduction='mean').item()
                psnr_acc[k] += psnr(p, t).item()
                ssim_acc[k] += ssim(p, t).item()
                lpips_acc[k] += lpips_fn(p * 2 - 1, t * 2 - 1).item()  # LPIPS expects [-1,1]
            
            n_batches += 1
    
    return {
        'mse': mse_acc / n_batches,
        'psnr': psnr_acc / n_batches,
        'ssim': ssim_acc / n_batches,
        'lpips': lpips_acc / n_batches,
    }
```

---

## 6. Transfer Learning (FM → LTE)

Same architecture, same hyperparameters, fine-tune on LTE data.

### LTE preprocessing differences

| | FM | LTE |
|---|---|---|
| Samples per file | 32,508 | 16,254 |
| Centre freq | 99 MHz | 700 MHz |
| Other STFT settings | — | Same |

Number of time windows (no overlap): `16254 // 256 = 63`. Resize 63×256 → 256×256 same as FM.

### TL procedure

```python
# Load FM-trained weights
model = SwinUNETR(...).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/swinstb_fm_best.pt'))

# Same optimizer, possibly same or slightly lower lr
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Same training loop, but with LTE dataset
lte_train_dataset = SpectrogramSequenceDataset(lte_path, split='train')
# ... continue training as before
```

Expected speedup per paper: 77% fewer training minutes (46.3 → 10.5 min).

---

## 7. Open Questions Tracked Throughout

| # | Question | Default choice | When to revisit |
|---|---|---|---|
| 1 | How to get 256×256 spectrogram from 32,508 I/Q samples? | Hann STFT, no overlap (126 time bins), resize to 256×256 | If MSE far off paper numbers |
| 2 | Log compression before colormap? | Yes (standard practice) | If spectrograms look weird |
| 3 | MONAI SwinUNETR patch/window override | Use defaults initially, check if exposed | Before final training run |
| 4 | Final output activation | None | If predictions drift outside [0,1] |
| 5 | MLP expansion ratio | 2 (override MONAI default of 4) | Verify in model instantiation |
| 6 | Channel reduction in projection layer (halving) | Halving: 96→48→24→12→6→3 | If SwinUNETR output head works fine, don't customise |

---

## 8. Expected Results (to aim for)

Ballpark numbers to hit on the FM test set (input-20-predict-20):

| Metric | Frame 1 | Frame 10 | Frame 20 |
|---|---|---|---|
| MSE | ~420 | ~450 | ~460 |
| PSNR | ~35.2 dB | ~35.0 dB | ~34.8 dB |
| SSIM | ~0.778 | ~0.776 | ~0.775 |
| LPIPS | ~0.120 | ~0.122 | ~0.123 |

If your results are within 10% of these numbers, reproduction is successful.

**Caveat:** your metrics depend heavily on normalization conventions (e.g., whether MSE is reported per-pixel-sum or per-pixel-mean, whether images are [0,255] or [0,1]). Pan et al. don't specify. Match the pattern of their frame-wise curves first, absolute values second.

---

## 9. Implementation Order with Finish Lines

### Stage 1 — Plumbing (3–5 days)

- [ ] Colab notebook with Drive mounted.
- [ ] Upload FM dataset to Drive.
- [ ] Write `parse_fm_file()` and test on 1 file.
- [ ] Write `iq_to_spectrogram()` and verify output is 256×256×3 RGB.
- [ ] Run preprocessing on full dataset (takes ~2 hours).
- [ ] Verify output `.npy` file shape is (10777, 256, 256, 3).
- [ ] Implement `SpectrogramSequenceDataset`, verify split sizes.
- [ ] Install MONAI, instantiate SwinUNETR.
- [ ] Run one forward pass with random input, verify output shape.

**Finish line:** `model(torch.randn(1, 3, 20, 256, 256).to('cuda')).shape == torch.Size([1, 3, 20, 256, 256])`.

### Stage 2 — Training loop (2–3 days)

- [ ] Write training loop with AMP.
- [ ] Run for 1 epoch on a tiny subset (50 examples).
- [ ] Verify loss decreases.
- [ ] Add validation loop.
- [ ] Add checkpointing to Drive.
- [ ] Add evaluation metrics.

**Finish line:** Loss drops monotonically on 50-example training, metrics report meaningful numbers.

### Stage 3 — Full training (1–2 days on A100)

- [ ] Full training run on all FM data.
- [ ] Monitor convergence.
- [ ] Save best model.
- [ ] Run full test set evaluation.
- [ ] Generate frame-wise metric curves.

**Finish line:** Frame-wise MSE/PSNR/SSIM/LPIPS curves that roughly match Pan et al.'s Figure 7.

### Stage 4 — Transfer learning (1–2 days)

- [ ] Preprocess LTE dataset.
- [ ] Load FM weights.
- [ ] Fine-tune on LTE.
- [ ] Evaluate LTE performance.

**Finish line:** LTE model reaches similar MSE to FM model with ~77% fewer training minutes.

### Stage 5 — Writeup (3–5 days)

- [ ] Document methodology.
- [ ] Present frame-wise result curves.
- [ ] Compare your numbers to Pan et al.
- [ ] Document which open questions you resolved and how.

---

## 10. Quick Reference: Key Commands

### Colab session start

```python
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
!pip install -q monai lpips torchmetrics einops
```

### Check GPU VRAM during training

```python
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Resume from checkpoint (boilerplate is in the training loop above)

---

## Notes

- **Treat Pan et al.'s STFT config (sampling freq 125 MHz, decimation 4, window 256, STFT number 32508) as ground truth.** Your preprocessing matches this exactly.
- **Don't apply DC correction, gap handling, or other analytics beyond what the paper does.** This is a reproduction, not an improved analysis.
- **Use MONAI SwinUNETR as-is for Option A.** The decoder mismatch is real but minor; faithful reproduction can come later if time allows.
- **Save checkpoints aggressively.** Colab sessions die. Losing an epoch of training is annoying; losing all of them is devastating.
- **When in doubt about a preprocessing choice, match the simplest sensible default.** Pan et al. didn't document every detail because they used defaults; match those defaults.
