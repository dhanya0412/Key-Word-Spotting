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


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/KWS_project", help="Drive project folder")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=10)   # paper-ish: 10
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--debug", action="store_true", help="Small debug run")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args([])  # pass [] so Colab cell can run

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

import os

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

def collate_fn(batch, labels):
    tensors = []
    targets = []
    for waveform, sr, label, *_ in batch:
        w = pad_truncate_waveform(waveform, sr)
        tensors.append(w)
        targets.append(label_to_index_fn(labels)(label))
    tensors = torch.stack(tensors)  # [B, 1, T]
    targets = torch.stack(targets)
    return tensors, targets


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


print("Preparing datasets (this may download first time)...")
train_set = SubsetSC("training", root="/content")
val_set = SubsetSC("validation", root="/content")
test_set = SubsetSC("testing", root="/content")

labels = get_labels(train_set)
print("Labels (n):", len(labels))
print(labels)

if args.debug:
    random.seed(42)
    train_set = torch.utils.data.Subset(train_set, list(range(min(2000, len(train_set)))))
    val_set = torch.utils.data.Subset(val_set, list(range(min(400, len(val_set)))))
    test_set = torch.utils.data.Subset(test_set, list(range(min(400, len(test_set)))))

collate = lambda batch: collate_fn(batch, labels)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE.startswith("cuda")))


model = M5(n_input=1, n_output=len(labels)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


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


ck = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(ck["model_state"])
test_acc = evaluate(test_loader, "Test")


results_csv = os.path.join(BASE_DIR, "results.csv")
if not os.path.exists(results_csv):
    pd.DataFrame(columns=["Model", "Accuracy"]).to_csv(results_csv, index=False)

df = pd.read_csv(results_csv)
df.loc[len(df)] = ["CNN-M5 (raw 8k)", test_acc]
df.to_csv(results_csv, index=False)
print("Saved results to", results_csv)
print("Model labels:", labels)
