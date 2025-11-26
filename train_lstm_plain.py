
# /content/drive/MyDrive/KWS_project/train_lstm_plain.py

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.backends.cudnn.benchmark = True


# -------------------------------------------------------
# MFCC Dataset
# -------------------------------------------------------
class MFCCDataset(Dataset):
    def __init__(self, df, label_list):
        self.df = df.reset_index(drop=True)
        self.labels = label_list
        self.label2idx = {l: i for i, l in enumerate(label_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["path"])  # shape (36, T)
        feat = feat.T.astype(np.float32)  # (T, 36)
        label = self.label2idx[row["label"]]
        return torch.from_numpy(feat), label


def collate_fn(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs])

    T_max = lengths.max().item()
    F = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), T_max, F)

    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s

    return padded, lengths, labels


# -------------------------------------------------------
# ✨ Plain LSTM Model (same spirit as original repo)
# -------------------------------------------------------
class PlainLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]  # (batch, hidden)
        logits = self.classifier(last_hidden)
        return logits


# -------------------------------------------------------
# Training + Eval
# -------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for x, lengths, labels in tqdm(loader, desc="Training", leave=False):
        x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    total, correct, total_loss = 0, 0, 0

    with torch.no_grad():
        for x, lengths, labels in tqdm(loader, desc=desc, leave=False):
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, labels)

            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--outdir", default="/content/drive/MyDrive/KWS_project")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.meta)
    labels = sorted(df["label"].unique())
    print("Labels:", labels)

    df_train = df[df.split == "training"]
    df_val = df[df.split == "validation"]
    df_test = df[df.split == "testing"]
    print(len(df_train), len(df_val), len(df_test))

    train_ds = MFCCDataset(df_train, labels)
    val_ds = MFCCDataset(df_val, labels)
    test_ds = MFCCDataset(df_test, labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example = np.load(df_train.iloc[0]["path"])
    input_dim = example.shape[0]  # 36
    num_classes = len(labels)

    model = PlainLSTM(input_dim, args.hidden, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0

    for ep in range(1, args.epochs + 1):
        print(f"\nEPOCH {ep}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "Validation")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Testing")

        print(f"Train  acc: {tr_acc:.3f}")
        print(f"Val    acc: {val_acc:.3f}")
        print(f"Test   acc: {test_acc:.3f}")

        scheduler.step()

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, "lstm_plain_best.pth"))

    print("Done. Best validation accuracy:", best_val)


if __name__ == "__main__":
    main()
