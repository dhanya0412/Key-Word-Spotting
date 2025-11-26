"""
Generate confusion matrices for trained AttRNN and BiLSTM-DenseNet models.
Loads best .pth checkpoints and evaluates on test set.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# -------------------------
# AttRNN Model
# -------------------------
class AttRNNSpeechModel(nn.Module):
    def __init__(self, n_classes, input_dim=36):
        super().__init__()
        self.input_dim = int(input_dim)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(1)

        self.lstm1 = nn.LSTM(input_size=self.input_dim, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

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

        m = x.permute(0, 2, 1).unsqueeze(1)
        c = F.relu(self.bn1(self.conv1(m)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = c.squeeze(1)

        seq = c.permute(0, 2, 1).contiguous()
        out1, _ = self.lstm1(seq)
        out2, _ = self.lstm2(out1)

        last = out2[:, -1, :]
        q = self.query_dense(last)

        scores = torch.bmm(out2, q.unsqueeze(-1)).squeeze(-1)
        att = F.softmax(scores, dim=1)

        context = torch.bmm(att.unsqueeze(1), out2).squeeze(1)

        x = F.relu(self.fc1(context))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

# -------------------------
# BiLSTM-DenseNet Model
# -------------------------
class DenseLayer1D(nn.Module):
    def __init__(self, in_ch, growth, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.bn = nn.BatchNorm1d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_ch, growth, kernel_size, padding=padding, bias=False)

    def forward(self, x):
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
        return torch.cat(features, dim=1)

class Transition1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

class BiLSTM_DenseNet(nn.Module):
    def __init__(self,
                 input_dim=36,
                 lstm_hidden=128,
                 lstm_layers=2,
                 bidirectional=True,
                 db_config=(2, 3),
                 growth=24,
                 transition_reduction=0.5,
                 num_classes=36,
                 dropout=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional, 
                            dropout=dropout if lstm_layers>1 else 0.0)

        lstm_out_ch = lstm_hidden * num_directions
        in_ch = lstm_out_ch
        n_blocks, layers_per_block = db_config
        dense_blocks = []
        for b in range(n_blocks):
            db = DenseBlock1D(in_ch, layers_per_block, growth)
            dense_blocks.append(db)
            in_ch = db.out_ch
            tr_out = max(int(in_ch * transition_reduction), 16)
            dense_blocks.append(Transition1D(in_ch, tr_out))
            in_ch = tr_out
        self.dense_net = nn.Sequential(*dense_blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = lstm_out.permute(0, 2, 1)
        out = self.dense_net(out)
        out = self.gap(out)
        logits = self.classifier(out)
        return logits

# -------------------------
# Dataset
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
        feat = np.load(row["path"])
        feat = feat.T.astype(np.float32)
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
# Evaluation
# -------------------------
def get_predictions(model, loader, device, model_type="attrnn"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, lengths, labels in tqdm(loader, desc=f"Evaluating {model_type}"):
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            
            if model_type == "bilstm":
                logits = model(x, lengths)
            else:  # attrnn
                logits = model(x)
            
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with appropriate size
    plt.figure(figsize=(16, 14))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()
    
    # Also save normalized version
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    norm_path = save_path.replace('.png', '_normalized.png')
    plt.savefig(norm_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalized confusion matrix to {norm_path}")
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--outdir", required=True, help="directory with .pth checkpoints")
    parser.add_argument("--attrnn_ckpt", default="attn_rnn_best.pth", help="AttRNN checkpoint filename")
    parser.add_argument("--bilstm_ckpt", default="bilstm_densenet_best.pth", help="BiLSTM checkpoint filename")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    df = pd.read_csv(args.meta)
    df_test = df[df.split == "testing"].reset_index(drop=True)
    print(f"Test samples: {len(df_test)}")

    # Get labels from checkpoint
    attrnn_path = os.path.join(args.outdir, args.attrnn_ckpt)
    bilstm_path = os.path.join(args.outdir, args.bilstm_ckpt)

    if not os.path.exists(attrnn_path):
        print(f"ERROR: AttRNN checkpoint not found at {attrnn_path}")
        return
    if not os.path.exists(bilstm_path):
        print(f"ERROR: BiLSTM checkpoint not found at {bilstm_path}")
        return

    # Load labels from checkpoint
    attrnn_state = torch.load(attrnn_path, map_location=device)
    labels = attrnn_state["labels"]
    num_classes = len(labels)
    print(f"Number of classes: {num_classes}")
    print(f"Labels: {labels}")

    # Get input dimension
    example = np.load(df_test.iloc[0]["path"])
    input_dim = example.shape[0]

    # Create test dataset and loader
    test_ds = MFCCDataset(df_test, labels)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, 
                            collate_fn=collate_fn, pin_memory=True)

    # -------------------------
    # Evaluate AttRNN
    # -------------------------
    print("\n" + "="*50)
    print("Evaluating AttRNN")
    print("="*50)
    
    attrnn_model = AttRNNSpeechModel(n_classes=num_classes, input_dim=input_dim).to(device)
    attrnn_model.load_state_dict(attrnn_state["model_state"])
    
    attrnn_preds, attrnn_labels = get_predictions(attrnn_model, test_loader, device, "attrnn")
    
    # Calculate accuracy
    attrnn_acc = (attrnn_preds == attrnn_labels).mean()
    print(f"AttRNN Test Accuracy: {attrnn_acc:.4f}")
    
    # Save confusion matrix
    attrnn_cm_path = os.path.join(args.outdir, "confusion_matrix_attrnn.png")
    plot_confusion_matrix(attrnn_labels, attrnn_preds, labels, "AttRNN", attrnn_cm_path)
    

    # -------------------------
    # Evaluate BiLSTM-DenseNet
    # -------------------------
    print("\n" + "="*50)
    print("Evaluating BiLSTM-DenseNet")
    print("="*50)
    
    bilstm_state = torch.load(bilstm_path, map_location=device)
    
    # Reconstruct BiLSTM model with same config as training
    bilstm_model = BiLSTM_DenseNet(
        input_dim=input_dim,
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=True,
        db_config=(2, 3),
        growth=24,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    bilstm_model.load_state_dict(bilstm_state["model_state"])
    
    bilstm_preds, bilstm_labels = get_predictions(bilstm_model, test_loader, device, "bilstm")
    
    # Calculate accuracy
    bilstm_acc = (bilstm_preds == bilstm_labels).mean()
    print(f"BiLSTM-DenseNet Test Accuracy: {bilstm_acc:.4f}")
    
    # Save confusion matrix
    bilstm_cm_path = os.path.join(args.outdir, "confusion_matrix_bilstm.png")
    plot_confusion_matrix(bilstm_labels, bilstm_preds, labels, "BiLSTM-DenseNet", bilstm_cm_path)
    

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"AttRNN Test Accuracy: {attrnn_acc:.4f}")
    print(f"BiLSTM-DenseNet Test Accuracy: {bilstm_acc:.4f}")
    print(f"\nAll confusion matrices and reports saved to: {args.outdir}")

if __name__ == "__main__":
    main()