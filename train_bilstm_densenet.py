"""
Train a BiLSTM -> DenseNet1D hybrid on MFCC features saved by your preprocess script.
Compatible with features: np.load(path) -> shape (36, T)  (mfcc + d1 + d2)
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.backends.cudnn.benchmark = True

# -------------------------
# Dataset + collate
# -------------------------
class MFCCDataset(Dataset):
    def __init__(self, df, label_list):
        self.df = df.reset_index(drop=True)
        self.labels = label_list
        self.label2idx = {l: i for i, l in enumerate(label_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["path"])   # (36, T)
        feat = feat.T.astype(np.float32)  # (T, 36)
        label = self.label2idx[row["label"]]
        return torch.from_numpy(feat), label

def collate_fn(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)

    T_max = lengths.max().item()
    F = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), T_max, F, dtype=torch.float32)

    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s

    return padded, lengths, labels

# -------------------------
# DenseNet-like 1D blocks
# -------------------------
class DenseLayer1D(nn.Module):
    def __init__(self, in_ch, growth, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.bn = nn.BatchNorm1d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_ch, growth, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(self.act(self.bn(x)))
        return out

class DenseBlock1D(nn.Module):
    def __init__(self, in_ch, n_layers, growth):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for i in range(n_layers):
            self.layers.append(DenseLayer1D(ch, growth))
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            inp = torch.cat(features, dim=1)
            out = layer(inp)
            features.append(out)
        return torch.cat(features, dim=1)  # (B, out_ch, T)

class Transition1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        # optional pooling could be added but we'll keep time dim same

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

# -------------------------
# Model: BiLSTM -> DenseNet1D -> classifier
# -------------------------
class BiLSTM_DenseNet(nn.Module):
    def __init__(self,
                 input_dim=36,
                 lstm_hidden=128,
                 lstm_layers=2,
                 bidirectional=True,
                 db_config=(3, 3),  # (n_blocks, layers_per_block)
                 growth=32,
                 transition_reduction=0.5,
                 num_classes=35,
                 dropout=0.2):
        """
        db_config: tuple -> (n_blocks, layers_per_block)
        """
        super().__init__()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout if lstm_layers>1 else 0.0)

        lstm_out_ch = lstm_hidden * num_directions  # features per time-step coming out of LSTM

        # We'll treat LSTM outputs as channels for 1D conv (B, C, T)
        in_ch = lstm_out_ch
        n_blocks, layers_per_block = db_config
        dense_blocks = []
        for b in range(n_blocks):
            db = DenseBlock1D(in_ch, layers_per_block, growth)
            dense_blocks.append(db)
            in_ch = db.out_ch
            # transition to reduce channels slightly
            tr_out = max( int(in_ch * transition_reduction), 16)
            dense_blocks.append(Transition1D(in_ch, tr_out))
            in_ch = tr_out
        self.dense_net = nn.Sequential(*dense_blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)  # pool over time -> (B, C, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x, lengths):
        """
        x: (B, T, F)  (padded)
        lengths: (B,)
        """
        # pack -> LSTM -> pad
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, C)
        # permute to conv format
        out = lstm_out.permute(0, 2, 1)  # (B, C, T)
        out = self.dense_net(out)        # (B, C2, T)
        out = self.gap(out)              # (B, C2, 1)
        logits = self.classifier(out)    # (B, num_classes)
        return logits

# -------------------------
# Training / eval helpers
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, lengths, labels in pbar:
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
        pbar.set_postfix(loss=total_loss/total, acc=100.*correct/total)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    total, correct, total_loss = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=total_loss/total, acc=100.*correct/total)
    return total_loss / total, correct / total

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="metadata.csv created by preprocess")
    parser.add_argument("--outdir", default="/content/drive/MyDrive/KWS_project")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--growth", type=int, default=32)
    parser.add_argument("--db_blocks", type=int, default=2)
    parser.add_argument("--db_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.meta)
    labels = sorted(df["label"].unique())
    print("Labels:", labels)
    df_train = df[df.split == "training"].reset_index(drop=True)
    df_val = df[df.split == "validation"].reset_index(drop=True)
    df_test = df[df.split == "testing"].reset_index(drop=True)
    print("Sizes (train/val/test):", len(df_train), len(df_val), len(df_test))

    train_ds = MFCCDataset(df_train, labels)
    val_ds = MFCCDataset(df_val, labels)
    test_ds = MFCCDataset(df_test, labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example = np.load(df_train.iloc[0]["path"])
    input_dim = example.shape[0]  # should be 36
    num_classes = len(labels)

    model = BiLSTM_DenseNet(input_dim=input_dim,
                            lstm_hidden=args.lstm_hidden,
                            lstm_layers=args.lstm_layers,
                            bidirectional=args.bidirectional,
                            db_config=(args.db_blocks, args.db_layers),
                            growth=args.growth,
                            num_classes=num_classes,
                            dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    ckpt_path = os.path.join(args.outdir, "bilstm_densenet_best(wo_noise).pth")

    for ep in range(1, args.epochs + 1):
        print(f"\nEPOCH {ep}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Validation")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Testing")

        print(f"Train acc: {tr_acc:.4f} | Val acc: {val_acc:.4f} | Test acc: {test_acc:.4f}")
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "labels": labels,
                "val_acc": val_acc
            }, ckpt_path)
            print("Saved best model to", ckpt_path)

    # -------------------------
# After training: evaluate best model on TEST and save CSV
# -------------------------
    print("\nLoading best validation checkpoint for final test evaluation...")
    best_ckpt = os.path.join(args.outdir, "bilstm_densenet_best(wo_noise).pth")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Final Test")
    print(f"Final Test Accuracy (best-val model): {test_acc:.4f}")

    # Save to CSV
    csv_path = os.path.join(args.outdir, "results.csv")

    # If CSV doesn't exist, create it with header
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Model", "Test_Accuracy"]).to_csv(csv_path, index=False)

    # Append new row
    df = pd.read_csv(csv_path)
    df.loc[len(df)] = ["BiLSTM-DenseNet (without noise)", test_acc]
    df.to_csv(csv_path, index=False)

    print("Saved results to", csv_path)



    print("Done. Best val acc:", best_val)

if __name__ == "__main__":
    main()
