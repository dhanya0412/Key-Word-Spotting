#!/usr/bin/env python3
"""
Preprocess Speech Commands (v0.02) into MFCC + d1 + d2 .npy features,
preserve original wav basename, add speaker column, and assign split
(train/validation/testing) using Google's split lists.
"""
import os
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

def find_raw_base(candidates=["/content/speech_commands_v0.02","/content"]):
    for p in candidates:
        if os.path.isdir(p) and any(x in os.listdir(p) for x in ["yes","no","up","down"]):
            return p
    raise RuntimeError("Raw dataset folder not found. Please extract v0.02 into /content or /content/speech_commands_v0.02")

def load_split_lists(raw_base):
    val_list = []
    test_list = []
    vfile = os.path.join(raw_base, "validation_list.txt")
    tfile = os.path.join(raw_base, "testing_list.txt")
    if os.path.exists(vfile):
        with open(vfile) as f: val_list = [l.strip() for l in f.readlines() if l.strip()]
    if os.path.exists(tfile):
        with open(tfile) as f: test_list = [l.strip() for l in f.readlines() if l.strip()]
    return set(val_list), set(test_list)

def get_speaker_from_wav(wav_name):
    # wav_name like "9ab4c1d2_nohash_0.wav" -> speaker "9ab4c1d2"
    return os.path.basename(wav_name).split('_')[0]

def extract_features(wav_path, sr=16000, n_mfcc=12, top_db=20):
    y, _ = librosa.load(wav_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if y.size == 0:
        return None
    # normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < 3:
        return None
    width = min(5, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1]-1)
    if width < 3:
        width = 3
    d1 = librosa.feature.delta(mfcc, width=width)
    d2 = librosa.feature.delta(mfcc, order=2, width=width)
    feat = np.vstack([mfcc, d1, d2]).astype(np.float32)
    return feat

def main(args):
    raw_base = args.raw if args.raw else find_raw_base()
    out_base = args.out
    os.makedirs(out_base, exist_ok=True)

    # load split lists
    val_list, test_list = load_split_lists(raw_base)

    # background noise for silence augmentation
    noise_dir = os.path.join(raw_base, "_background_noise_")
    noise_files = []
    if os.path.isdir(noise_dir):
        noise_files = [os.path.join(noise_dir,f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
    print(f"Found {len(noise_files)} background noise files.")

    classes = sorted([d for d in os.listdir(raw_base)
                      if os.path.isdir(os.path.join(raw_base,d)) and not d.startswith("_")])
    print("Classes found:", classes)

    meta = []
    # Optionally limit classes? user wants all 35, so we keep all
    for cls in tqdm(classes, desc="Processing classes"):
        in_dir = os.path.join(raw_base, cls)
        out_dir = os.path.join(out_base, cls)
        os.makedirs(out_dir, exist_ok=True)

        wavs = sorted([f for f in os.listdir(in_dir) if f.endswith(".wav")])
        if args.debug:
            wavs = wavs[:args.debug_samples_per_class]

        for w in wavs:
            wav_relpath = os.path.join(cls, w)  # relative path used in split lists
            full_wav = os.path.join(in_dir, w)
            feat = extract_features(full_wav, sr=args.sr, n_mfcc=args.n_mfcc, top_db=args.top_db)
            if feat is None:
                continue

            # optional noise augmentation (small fraction)
            if args.augment and noise_files and random.random() < args.augment_prob:
                nf = random.choice(noise_files)
                n, _ = librosa.load(nf, sr=args.sr)
                if len(n) > len(y := librosa.load(full_wav, sr=args.sr)[0]):
                    start = random.randint(0, len(n)-len(y))
                    n_seg = n[start:start+len(y)]
                else:
                    n_seg = np.resize(n, len(y))
                y = y / (np.max(np.abs(y))+1e-9)
                y = 0.9*y + 0.1*n_seg
                # recompute mfcc from augmented y
                mfcc = librosa.feature.mfcc(y=y, sr=args.sr, n_mfcc=args.n_mfcc)
                if mfcc.shape[1] < 3:
                    continue
                width = min(5, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1]-1)
                if width < 3:
                    width = 3
                d1 = librosa.feature.delta(mfcc, width=width)
                d2 = librosa.feature.delta(mfcc, order=2, width=width)
                feat = np.vstack([mfcc, d1, d2]).astype(np.float32)

            out_name = os.path.splitext(w)[0] + ".npy"   # preserve original basename
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, feat)
            speaker = get_speaker_from_wav(w)
            # determine split
            if wav_relpath in test_list:
                split = "testing"
            elif wav_relpath in val_list:
                split = "validation"
            else:
                split = "training"
            meta.append((out_path, cls, w, speaker, split))

    # create silence class from background noise (same as earlier)
    silence_out = os.path.join(out_base, "silence")
    os.makedirs(silence_out, exist_ok=True)
    for nf in noise_files:
        n, _ = librosa.load(nf, sr=args.sr)
        # create several 1-second segments
        for i in range(5):
            if len(n) <= args.sr: continue
            start = random.randint(0, len(n)-args.sr)
            seg = n[start:start+args.sr]
            mfcc = librosa.feature.mfcc(y=seg, sr=args.sr, n_mfcc=args.n_mfcc)
            if mfcc.shape[1] < 3: continue
            d1 = librosa.feature.delta(mfcc, width=3)
            d2 = librosa.feature.delta(mfcc, order=2, width=3)
            feat = np.vstack([mfcc, d1, d2]).astype(np.float32)
            out_path = os.path.join(silence_out, f"{os.path.basename(nf).replace('.wav','')}_{i}.npy")
            np.save(out_path, feat)
            meta.append((out_path, "silence", os.path.basename(out_path), "noise", "training"))

    df = pd.DataFrame(meta, columns=["path","label","orig_wav","speaker","split"])
    out_meta = os.path.join(out_base, "metadata.csv")
    df.to_csv(out_meta, index=False)
    print("Saved metadata:", out_meta)
    print("Total samples:", len(df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=None, help="raw speech_commands_v0.02 path (auto-detect if empty)")
    parser.add_argument("--out", default="/content/drive/MyDrive/KWS_project/processed", help="output processed folder")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=12)
    parser.add_argument("--top_db", type=int, default=20)
    parser.add_argument("--debug", action="store_true", help="debug (small run)")
    parser.add_argument("--debug_samples_per_class", type=int, default=50)
    parser.add_argument("--augment", action="store_true", help="apply light noise augmentation")
    parser.add_argument("--augment_prob", type=float, default=0.10)
    args = parser.parse_args()
    main(args)
