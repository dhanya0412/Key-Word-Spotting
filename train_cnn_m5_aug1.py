#!/usr/bin/env python3
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import random
import argparse
import soundfile as sf
import librosa
import numpy as np
# ---- FIX TORCHAUDIO.LOAD (TorchCodec error on Colab) ----
import torchaudio
import soundfile as sf
import torch
import numpy as np

_orig_ta_load = torchaudio.load

def safe_ta_load(path, *args, **kwargs):
    try:
        # try normal loader
        return _orig_ta_load(path, *args, **kwargs)
    except Exception:
        # fallback to soundfile
        data, sr = sf.read(path, dtype='float32')

        # ensure shape [channels, time]
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T

        return torch.from_numpy(data), sr

torchaudio.load = safe_ta_load
# ---------------------------------------------------------


# -------------------------
# Config / arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/KWS_project", help="Drive project folder")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=10)   # paper-ish: 10
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--debug", action="store_true", help="Small debug run")
parser.add_argument("--device", type=str, default=None)
# Augmentation flags (mirror preprocess)
# Augmentation flags
parser.add_argument("--augment_train", action="store_true", help="apply noise augmentation to TRAIN")
parser.add_argument("--augment_valtest", action="store_true", help="apply noise augmentation to VAL and TEST")
parser.add_argument("--augment_prob", type=float, default=0.10)

args = parser.parse_args()
 # pass [] so Colab cell can run

BASE_DIR = args.base_dir
os.makedirs(BASE_DIR, exist_ok=True)

DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = args.batch_size
N_EPOCHS = 2 if args.debug else args.n_epochs
LR = args.lr
NEW_SR = 8000  # resample to 8 kHz (paper for M5)
CLIP_DURATION = 1.0  # seconds
CLIP_SAMPLES = int(NEW_SR * CLIP_DURATION)
NUM_WORKERS = 1 if DEVICE.startswith("cuda") else 0

print("Device:", DEVICE)
print("Base dir:", BASE_DIR)
print("Debug mode:", args.debug)
print("Epochs:", N_EPOCHS)
print("Augment train:", args.augment_train, "Augment valtest:", args.augment_valtest, "Augment prob:", args.augment_prob)

# -------------------------
# Subset dataset wrapper (same split files as repo)
# -------------------------
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = "/content"):
        super().__init__(root, download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

# -------------------------
# Helpers: label maps, collate, pad/truncate
# -------------------------
def get_labels(dataset):
    # build sorted label list from training set (stable across splits)
    labels = sorted(list(set(datapoint[2] for datapoint in dataset)))
    return labels

def pad_truncate_waveform(waveform, sr, target_sr=NEW_SR, target_len=CLIP_SAMPLES):
    # waveform: Tensor [channels, time]
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    length = waveform.shape[-1]
    if length > target_len:
        waveform = waveform[:, :target_len]
    elif length < target_len:
        pad = target_len - length
        waveform = F.pad(waveform, (0, pad))
    return waveform  # [1, target_len]

def label_to_index_fn(labels):
    mapping = {l: i for i, l in enumerate(labels)}
    def fn(word):
        return torch.tensor(mapping[word], dtype=torch.long)
    return fn

# -------------------------
# Noise loading & mixing utilities (match preprocess behavior)
# -------------------------
def safe_load_noise(path, target_sr=NEW_SR, dtype='float32'):
    """Load a noise wav (numpy float32 1D) robustly, resample to target_sr if needed."""
    try:
        data, sr = sf.read(path, dtype=dtype)
    except Exception:
        try:
            data, sr = librosa.load(path, sr=None)
            data = data.astype(np.float32)
        except Exception:
            return None, None
    if data is None:
        return None, None
    data = np.asarray(data)
    if data.size == 0:
        return None, None
    if data.ndim > 1:
        data = data[:,0]
    if sr != target_sr:
        try:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            return None, None
    data = np.asarray(data, dtype=np.float32).flatten()
    if data.size == 0:
        return None, None
    return data, sr

# prepare noise file list (same folder name as preprocess)
NOISE_DIR_CANDIDATES = ["/content/_background_noise_", os.path.join("/content", "_background_noise_")]
noise_files = []
for nd in NOISE_DIR_CANDIDATES:
    if os.path.isdir(nd):
        noise_files = [os.path.join(nd, f) for f in os.listdir(nd) if f.endswith(".wav")]
        break
print("Found noise files:", len(noise_files))

def mix_background_noise_tensor(waveform_tensor, noise_files, prob=0.1):
    """
    waveform_tensor: torch tensor shape [1, T] (float, in [-1,1] typically)
    returns mixed waveform tensor same shape
    """
    if not noise_files or random.random() >= prob:
        return waveform_tensor
    nf = random.choice(noise_files)
    n_wave, n_sr = safe_load_noise(nf, target_sr=NEW_SR)
    if n_wave is None:
        return waveform_tensor
    # ensure correct length: waveform_tensor is [1, T]
    T = waveform_tensor.shape[-1]
    if len(n_wave) > T:
        start = random.randint(0, len(n_wave) - T)
        n_seg = n_wave[start:start+T]
    else:
        n_seg = np.resize(n_wave, T)
    # convert noise to tensor and match device/dtype
    n_t = torch.from_numpy(n_seg).unsqueeze(0).to(waveform_tensor.device, dtype=waveform_tensor.dtype)
    # normalize both (matching preprocess)
    y_np = waveform_tensor.detach().cpu().numpy().astype(np.float32).flatten()
    y_np = y_np / (np.max(np.abs(y_np)) + 1e-9)
    y_t = torch.from_numpy(y_np).unsqueeze(0).to(waveform_tensor.device, dtype=waveform_tensor.dtype)
    mixed = 0.9 * y_t + 0.1 * n_t
    # clamp to valid audio range
    mixed = mixed.clamp(-1.0, 1.0)
    return mixed

def collate_fn(batch, labels, split):
    tensors = []
    targets = []
    for waveform, sr, label, *_ in batch:
        w = pad_truncate_waveform(waveform, sr)
        # apply augmentation here (on-the-fly)
        if split == "train" and args.augment_train:
            w = mix_background_noise_tensor(w, noise_files, prob=args.augment_prob)

        if split in ("val", "test") and args.augment_valtest:
            w = mix_background_noise_tensor(w, noise_files, prob=args.augment_prob)

        tensors.append(w)
        targets.append(label_to_index_fn(labels)(label))
    tensors = torch.stack(tensors)  # [B, 1, T]
    targets = torch.stack(targets)
    return tensors, targets

# -------------------------
# M5 model (matches repo)
# -------------------------
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        # x shape: [B, 1, T]
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])  # global average pool over time
        x = x.permute(0, 2, 1)            # [B, 1, channels]
        x = self.fc1(x)                   # [B, 1, n_output]
        return F.log_softmax(x, dim=2)    # match original repo (log-probs)

# -------------------------
# Prepare data loaders
# -------------------------
print("Preparing datasets (this may download first time)...")
train_set = SubsetSC("training", root="/content")
val_set = SubsetSC("validation", root="/content")
test_set = SubsetSC("testing", root="/content")

labels = get_labels(train_set)
print("Labels (n):", len(labels))
print(labels)

if args.debug:
    # tiny subset for fast debugging
    random.seed(42)
    train_set = torch.utils.data.Subset(train_set, list(range(min(2000, len(train_set)))))
    val_set = torch.utils.data.Subset(val_set, list(range(min(400, len(val_set)))))
    test_set = torch.utils.data.Subset(test_set, list(range(min(400, len(test_set)))))
    
collate_train = lambda batch: collate_fn(batch, labels, "train")
collate_val   = lambda batch: collate_fn(batch, labels, "val")
collate_test  = lambda batch: collate_fn(batch, labels, "test")


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_val,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))

# -------------------------
# Build model, optimizer, scheduler
# -------------------------
model = M5(n_input=1, n_output=len(labels)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# -------------------------
# Training / validation / test helpers
# -------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Train E{epoch}")
    for data, targets in pbar:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)                # [B,1,n_output]
        outputs = outputs.squeeze(1)         # [B,n_output]
        loss = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss/total, correct/total

def evaluate(loader, setname="Val"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Eval {setname}")
        for data, targets in pbar:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(data).squeeze(1)
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    print(f"{setname} Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# -------------------------
# Training loop
# -------------------------
best_val = 0.0
checkpoint_path = os.path.join(BASE_DIR, "cnn_m5_checkpoint.pth")

for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_acc = evaluate(val_loader, "Val")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "labels": labels,
            "val_acc": val_acc
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)
    scheduler.step()

# Load best checkpoint and test
ck = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(ck["model_state"])
test_acc = evaluate(test_loader, "Test")

# -------------------------
# Save final results
# -------------------------
results_csv = os.path.join(BASE_DIR, "results.csv")
if not os.path.exists(results_csv):
    pd.DataFrame(columns=["Model", "Accuracy"]).to_csv(results_csv, index=False)

df = pd.read_csv(results_csv)
df.loc[len(df)] = ["CNN-M5 (raw 8k) + bg-noise", test_acc]
df.to_csv(results_csv, index=False)
print("Saved results to", results_csv)
print("Model labels:", labels)