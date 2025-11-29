import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True

# -------------------------
# Model Definition
# -------------------------
class AttRNNSpeechModel(nn.Module):
    def __init__(self, n_classes, input_dim=36):
        super().__init__()
        self.input_dim = int(input_dim)

        # conv stack: input will be (B, 1, F, T)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(1)

        # LSTMs: input_size = input_dim after conv-squeeze
        self.lstm1 = nn.LSTM(input_size=self.input_dim, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        # Attention and classifier
        self.query_dense = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"Expected input tensor shape (B, T, F). Got {x.shape}")

        # Permute to (B, F, T) and add channel -> (B, 1, F, T)
        m = x.permute(0, 2, 1).unsqueeze(1)   # (B,1,F,T)

        # conv stack
        c = F.relu(self.bn1(self.conv1(m)))   # (B,10,F,T)
        c = F.relu(self.bn2(self.conv2(c)))   # (B,1,F,T)
        c = c.squeeze(1)                      # (B, F, T)

        # Permute to (B, T, F) for LSTM input
        seq = c.permute(0, 2, 1).contiguous()  # (B, T, F)

        # LSTM layers
        out1, _ = self.lstm1(seq)   # (B, T, 128)
        out2, _ = self.lstm2(out1)  # (B, T, 128)

        # Attention: use last time-step as query source
        last = out2[:, -1, :]                # (B, 128)
        q = self.query_dense(last)           # (B, 128)

        # Dot-product attention scores: (B, T)
        scores = torch.bmm(out2, q.unsqueeze(-1)).squeeze(-1)
        att = F.softmax(scores, dim=1)       # (B, T)

        # Weighted sum -> context (B, 128)
        context = torch.bmm(att.unsqueeze(1), out2).squeeze(1)

        # Classifier
        x = F.relu(self.fc1(context))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, lengths, labels in pbar:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(x)
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
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    
    # Get input dimension from data
    example = np.load(df_train.iloc[0]["path"])
    input_dim = example.shape[0]  # should be 36
    num_classes = len(labels)

    model = AttRNNSpeechModel(n_classes=num_classes, input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    ckpt_path = os.path.join(args.outdir, "attn_rnn_best.pth")

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
    best_ckpt = os.path.join(args.outdir, "attn_rnn_best(wo_noise).pth")
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
    df_results = pd.read_csv(csv_path)
    df_results.loc[len(df_results)] = ["AttRNN", test_acc]
    df_results.to_csv(csv_path, index=False)

    print("Saved results to", csv_path)
    print("Done. Best val acc:", best_val)

if __name__ == "__main__":
    main()