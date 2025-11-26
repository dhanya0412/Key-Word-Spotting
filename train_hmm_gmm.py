import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import pandas as pd


# Remove ALL Google Drive code - not needed on Lightning
BASE_DIR = "/teamspace/studios/this_studio"
import os
print("cwd:", os.getcwd())
print("process user:", os.getlogin() if hasattr(os, 'getlogin') else "unknown")
print("home:", os.path.expanduser("~"))
print("BASE_DIR env check (if set):", os.environ.get("BASE_DIR"))
# Create directories
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/processed", exist_ok=True)
os.makedirs(f"{BASE_DIR}/speech_commands_v0.02", exist_ok=True)

# Path Configuration
PROCESSED_ROOT = f"{BASE_DIR}/processed"
TRAIN_ROOT = PROCESSED_ROOT
VALID_ROOT = PROCESSED_ROOT
TEST_ROOT = PROCESSED_ROOT
OUT_MODEL_PATH = f"{BASE_DIR}/models/hmmgmm_model.pkl"

print("✅ Paths configured:")
print("  Processed root:", PROCESSED_ROOT)
print("  Model path:", OUT_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data loading helpers
def load_feature_file(path):
    """
    Load a .npy feature file created by your preprocess step.
    Expected original shape: (feat_dim, time) -> we return (time, feat_dim)
    """
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr.astype(np.float32)


def build_dataset(root_dir):
    """
    If root_dir contains class subfolders (yes/, no/, ...), this returns a list
    of (feat_array (T,D), label_str).
    """
    data = []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        for f in files:
            try:
                feat = load_feature_file(f)
            except Exception as e:
                print("Skipping", f, "load error:", e)
                continue
            data.append((feat, cls))
    return data


# Log gaussian and GMM class (diagonal covariances)
def log_gaussian(x, mean, cov_diag):
    """
    Return log N(x|mean, cov_diag). x: (...,D). mean and cov_diag must be broadcastable to x.
    Uses torch tensors.
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x)).to(mean.device)
    D = x.shape[-1]
    const = -0.5 * D * torch.log(torch.tensor(2 * np.pi, device=x.device))
    log_det = -0.5 * torch.sum(torch.log(cov_diag), dim=-1)
    diff = x - mean
    exp = -0.5 * torch.sum((diff * diff) / cov_diag, dim=-1)
    return const + log_det + exp


class GMM:
    def __init__(self, n_mix=2, n_iter=10, device='cpu'):
        self.n_mix = int(n_mix)
        self.n_iter = int(n_iter)
        self.device = torch.device(device)
        self.pi = None
        self.means = None
        self.vars = None

    def _normalize_Xlist(self, Xlist, expected_feat=36):
        """
        Convert list of numpy arrays (feat_dim, T) or (T, feat_dim) into list of (N_i, D) rows
        where each element is shape (N_i, D) (frames x features) and dtype float32.
        expected_feat = 36 by default (3 * n_mfcc)
        """
        seqs = []
        for x in Xlist:
            if x is None:
                continue
            a = np.asarray(x)
            if a.ndim != 2:
                continue
            if a.shape[0] == expected_feat and a.shape[1] != expected_feat:
                arr = a.T.astype(np.float32)
            elif a.shape[1] == expected_feat and a.shape[0] != expected_feat:
                arr = a.astype(np.float32)
            elif a.shape[0] == expected_feat and a.shape[1] == expected_feat:
                arr = a.T.astype(np.float32)
            else:
                if a.shape[0] < a.shape[1]:
                    arr = a.T.astype(np.float32)
                else:
                    arr = a.astype(np.float32)
            if arr.size > 0:
                seqs.append(arr)
        return seqs

    def init_from_data(self, Xlist):
        seqs = self._normalize_Xlist(Xlist)
        if len(seqs) == 0:
            raise ValueError("GMM.init_from_data: no usable data")
        X = np.concatenate(seqs, axis=0)
        X = torch.from_numpy(X).to(self.device)
        N, D = X.shape
        perm = torch.randperm(N, device=self.device)
        groups = torch.chunk(X[perm], self.n_mix)
        means = []
        vars = []
        pis = []
        for i, g in enumerate(groups):
            if g.numel() == 0:
                idx = torch.randint(0, N, (1,), device=self.device)
                sample = X[idx].squeeze(0)
                means.append(sample)
                vars.append(torch.var(X, dim=0) + 1e-6)
                pis.append(torch.tensor(1.0 / self.n_mix, device=self.device))
            else:
                means.append(g.mean(dim=0))
                v = ((g - g.mean(dim=0))**2).mean(dim=0) + 1e-6
                vars.append(v)
                pis.append(torch.tensor(g.shape[0] / float(N), device=self.device))
        self.means = torch.stack(means, dim=0)
        self.vars = torch.stack(vars, dim=0)
        self.pi = torch.stack(pis).to(self.device)
        self.pi = self.pi / self.pi.sum()
        return self

    def fit(self, Xlist):
        seqs = self._normalize_Xlist(Xlist)
        if len(seqs) == 0:
            raise ValueError("GMM.fit: no usable data")
        X = np.concatenate(seqs, axis=0)
        X = torch.from_numpy(X).to(self.device)
        N, D = X.shape
        if self.means is None or self.vars is None or self.pi is None:
            self.init_from_data(seqs)
        K = self.n_mix
        for it in range(self.n_iter):
            Xexp = X.unsqueeze(1)
            means_exp = self.means.unsqueeze(0)
            vars_exp = self.vars.unsqueeze(0)
            log_p = log_gaussian(Xexp, means_exp, vars_exp)
            log_pi = torch.log(self.pi.unsqueeze(0) + 1e-12)
            log_w = log_p + log_pi
            log_sum = torch.logsumexp(log_w, dim=1, keepdim=True)
            log_resp = log_w - log_sum
            resp = torch.exp(log_resp)
            Nk = resp.sum(dim=0) + 1e-8
            pi = Nk / Nk.sum()
            means = (resp.t() @ X) / Nk.unsqueeze(1)
            var_num = (resp.unsqueeze(2) * (Xexp - means.unsqueeze(0))**2).sum(dim=0)
            vars = var_num / Nk.unsqueeze(1) + 1e-6
            self.pi = pi.to(self.device)
            self.means = means.to(self.device)
            self.vars = vars.to(self.device)
        return self

    def score_samples(self, X):
        """
        X: numpy (T,D) or (D,T) or torch
        returns torch tensor shape (T,) with per-frame log-likelihoods
        """
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("GMM.score_samples: X must be 2D")
        expected_D = self.means.shape[1] if (self.means is not None) else 36
        if X_np.shape[0] == expected_D and X_np.shape[1] != expected_D:
            frames = X_np.T.astype(np.float32)
        else:
            frames = X_np.astype(np.float32)
        X_t = torch.from_numpy(frames).to(self.device)
        Xexp = X_t.unsqueeze(1)
        log_p = log_gaussian(Xexp, self.means.unsqueeze(0), self.vars.unsqueeze(0))
        log_pi = torch.log(self.pi.unsqueeze(0) + 1e-12)
        log_w = log_p + log_pi
        return torch.logsumexp(log_w, dim=1)


# HMM-GMM class with all methods INSIDE the class
class HMMGMM:
    def __init__(self, n_states=3, n_mix=2, n_iter=6, device='cpu', gmm_iter=8):
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.device = torch.device(device)
        self.gmm_iter = gmm_iter
        self.startprob = torch.zeros(n_states, device=self.device)
        self.startprob[0] = 1.0
        self.trans = torch.zeros(n_states, n_states, device=self.device)
        for i in range(n_states):
            if i == n_states - 1:
                self.trans[i, i] = 1.0
            else:
                self.trans[i, i] = 0.6
                self.trans[i, i+1] = 0.4
        self.gmms = [GMM(n_mix=n_mix, n_iter=gmm_iter, device=self.device) for _ in range(n_states)]

    def _compute_emission_loglik(self, X):
        """
        Robust: accept X as numpy or torch, shape (36, T) or (T, 36).
        Returns B (T, S) torch on self.device.
        """
        expected_feat = 3 * 12  # 36
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("_compute_emission_loglik: X must be 2D")
        a0, a1 = X_np.shape
        if a0 == expected_feat and a1 != expected_feat:
            frames = X_np.T.astype(np.float32)
        elif a1 == expected_feat and a0 != expected_feat:
            frames = X_np.astype(np.float32)
        elif a0 == expected_feat and a1 == expected_feat:
            frames = X_np.T.astype(np.float32)
        else:
            if a0 < a1:
                frames = X_np.T.astype(np.float32)
            else:
                frames = X_np.astype(np.float32)

        T = frames.shape[0]
        S = int(self.n_states)
        B_list = []
        for s in range(S):
            scores = self.gmms[s].score_samples(frames)
            if not torch.is_tensor(scores):
                scores = torch.from_numpy(np.asarray(scores)).to(self.device)
            else:
                scores = scores.to(self.device)
            B_list.append(scores.unsqueeze(1))
        B = torch.cat(B_list, dim=1)
        return B

    def fit(self, Xseqs):
        S = self.n_states
        device = self.device

        expected_feat = 3 * 12
        seqs = []
        for x in Xseqs:
            if hasattr(x, "detach") and torch.is_tensor(x):
                try:
                    a = x.detach().cpu().numpy()
                except Exception:
                    continue
            else:
                a = np.asarray(x)

            if a.ndim != 2:
                continue

            if a.shape[0] == expected_feat and a.shape[1] != expected_feat:
                seqs.append(a.T.astype(np.float32))
            elif a.shape[1] == expected_feat and a.shape[0] != expected_feat:
                seqs.append(a.astype(np.float32))
            elif a.shape[0] == expected_feat and a.shape[1] == expected_feat:
                seqs.append(a.T.astype(np.float32))
            else:
                continue

        if len(seqs) == 0:
            raise ValueError("HMMGMM.fit: no valid sequences after normalization")

        all_frames = np.concatenate(seqs, axis=0)
        idx_splits = np.array_split(np.arange(all_frames.shape[0]), S)
        splits = [
            (all_frames[idxs] if len(idxs) > 0 else np.zeros((0, all_frames.shape[1]), dtype=np.float32))
            for idxs in idx_splits
        ]

        for s in range(S):
            self.gmms[s].init_from_data([splits[s]])
            self.gmms[s].fit([splits[s]])

        for it in range(self.n_iter):
            trans_num = torch.zeros_like(self.trans, device=device)
            trans_den = torch.zeros(self.n_states, device=device)
            state_frame_lists = [[] for _ in range(S)]

            for X in seqs:
                B = self._compute_emission_loglik(X)
                T = B.shape[0]
                logA = torch.log(self.trans + 1e-12)

                la = torch.empty((T, S), device=device)
                la[0] = torch.log(self.startprob + 1e-12) + B[0]
                for t in range(1, T):
                    la[t] = torch.logsumexp(la[t-1].unsqueeze(1) + logA, dim=0) + B[t]

                lb = torch.empty((T, S), device=device)
                lb[-1] = 0.0
                for t in range(T-2, -1, -1):
                    lb[t] = torch.logsumexp(logA + (B[t+1] + lb[t+1]).unsqueeze(0), dim=1)

                log_gamma = la + lb
                log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
                gamma = torch.exp(log_gamma)

                for t in range(T-1):
                    log_xi_t = la[t].unsqueeze(1) + logA + (B[t+1] + lb[t+1]).unsqueeze(0)
                    log_xi_t = log_xi_t - torch.logsumexp(log_xi_t, dim=(0,1))
                    xi_t = torch.exp(log_xi_t)
                    trans_num += xi_t
                    trans_den += xi_t.sum(dim=1)

                X_np = X
                for s in range(S):
                    weights = gamma[:, s].cpu().numpy()
                    rep_counts = (weights * 10).astype(int)
                    if rep_counts.sum() == 0:
                        state_frame_lists[s].append(X_np)
                    else:
                        rows = []
                        for i, rc in enumerate(rep_counts):
                            if rc <= 0:
                                continue
                            rows.append(np.repeat(X_np[i:i+1, :], rc, axis=0))
                        if rows:
                            state_frame_lists[s].append(np.concatenate(rows, axis=0))
                        else:
                            state_frame_lists[s].append(X_np)

            trans_den = trans_den + 1e-8
            new_trans = trans_num / trans_den.unsqueeze(1)
            for i in range(S):
                for j in range(S):
                    if not (j == i or j == i+1):
                        new_trans[i, j] = 0.0
            new_trans = new_trans / (new_trans.sum(dim=1, keepdim=True) + 1e-12)
            self.trans = new_trans

            for s in range(S):
                if len(state_frame_lists[s]) == 0:
                    continue
                merged = np.concatenate(state_frame_lists[s], axis=0)
                self.gmms[s].fit([merged])

        return self

    def score_sequence(self, X):
        B = self._compute_emission_loglik(X)
        T, S = B.shape
        logA = torch.log(self.trans + 1e-12)
        la = torch.empty((T, S), device=self.device)
        la[0] = torch.log(self.startprob + 1e-12) + B[0]
        for t in range(1, T):
            la[t] = torch.logsumexp(la[t-1].unsqueeze(1) + logA, dim=0) + B[t]
        loglik = torch.logsumexp(la[-1], dim=0)
        return loglik.item()


print("✅ HMMGMM class defined with all methods")


# Training and prediction wrapper functions
def train_GMMHMM(dataset, n_states=3, n_mix=2, n_iter=5, device='cpu'):
    device = torch.device(device)

    by_label = {}
    print("Grouping sequences by label...")
    for x, y in tqdm(dataset, desc="Collecting"):
        by_label.setdefault(y, []).append(x)

    models = {}
    print("\n=== Training models ===")

    for label in tqdm(sorted(by_label.keys()), desc="Labels"):
        seqs = by_label[label]
        print(f"\nTraining HMM-GMM for label '{label}' with {len(seqs)} sequences")

        model = HMMGMM(
            n_states=n_states,
            n_mix=n_mix,
            n_iter=n_iter,
            device=device,
            gmm_iter=8
        )

        model.fit(seqs)
        models[label] = model

    print("\nAll models trained.")
    return models


def predict_GMMHMM(dataset, models, type='test'):
    correct = 0
    total = 0
    per_label_counts = {}
    per_label_correct = {}

    print(f"\n=== Predicting ({type}) ===")
    for X, true_label in tqdm(dataset, desc=f"Predict {type}"):
        total += 1

        best_label = None
        best_score = -1e18

        for lab, model in models.items():
            try:
                score = model.score_sequence(X)
            except Exception:
                score = -1e12

            if score > best_score:
                best_score = score
                best_label = lab

        if best_label == true_label:
            correct += 1
            per_label_correct[true_label] = per_label_correct.get(true_label, 0) + 1

        per_label_counts[true_label] = per_label_counts.get(true_label, 0) + 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n[{type}] Accuracy: {acc*100:.2f}% ({correct}/{total})")

    for lab in sorted(per_label_counts.keys()):
        c = per_label_correct.get(lab, 0)
        t = per_label_counts[lab]
        print(f"  {lab}: {c}/{t} = {c/t*100:.2f}%")

    return acc


def build_dataset_from_meta(meta_csv, split_name):
    df = pd.read_csv(meta_csv)
    df = df[df['split'] == split_name].reset_index(drop=True)
    dataset = []
    for _, row in df.iterrows():
        try:
            X = np.load(row['path'])
        except Exception as e:
            continue
        y = row['label']
        dataset.append((X, y))
    return dataset


# Main execution
if __name__ == "__main__":
    meta_csv = f"{BASE_DIR}/processed/metadata.csv"

    print("\n📂 Loading datasets...")
    train_dataset = build_dataset_from_meta(meta_csv, 'training')
    valid_dataset = build_dataset_from_meta(meta_csv, 'validation')
    test_dataset = build_dataset_from_meta(meta_csv, 'testing')

    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # --- LOAD SAVED MODEL ---
    print("\n📦 Loading saved model...")
    with open(OUT_MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    print("✅ Model loaded.")

    # --- EVALUATION ONLY ---
    '''
    print("\n📊 Evaluating on training set...")
    predict_GMMHMM(train_dataset, models, type='train')

    print("\n📊 Evaluating on validation set...")
    predict_GMMHMM(valid_dataset, models, type='validation')
    ''' 
    print("\n📊 Evaluating on test set...")
    predict_GMMHMM(test_dataset, models, type='test')
