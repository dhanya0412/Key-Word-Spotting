
import os
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def get_speaker_from_wav(wav_name):
    return wav_name.split("_")[0]


def load_background_noises(noise_dir, sr=16000):
    if not os.path.isdir(noise_dir):
        print("Warning: no noise directory found.")
        return []

    noises = []
    files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]
    print("Loading noise files...", files)

    for nf in files:
        try:
            n, _ = librosa.load(os.path.join(noise_dir, nf), sr=sr)
            noises.append(n)
        except:
            print("Skipping unreadable noise:", nf)

    print("Loaded", len(noises), "noise samples.")
    return noises


def add_background_noise_linear(clean, noises, mix_ratio=0.1, rng=None):
    if len(noises) == 0:
        return clean

    if rng is None:
        rng = random.Random()

    noise = rng.choice(noises)

    # tile noise if too short
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean)/len(noise))))

    # crop noise random segment
    start = rng.randint(0, len(noise) - len(clean))
    noise = noise[start:start+len(clean)]

    # normalize both
    c = clean / (np.max(np.abs(clean)) + 1e-9)
    n = noise / (np.max(np.abs(noise)) + 1e-9)

    y = (1-mix_ratio)*c + mix_ratio*n
    y = y / (np.max(np.abs(y)) + 1e-9)

    return y


def extract_features(y, sr, n_mfcc, n_fft, hop_length, top_db):
    if y is None or len(y) == 0:
        return None

    y, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y) == 0:
        return None

    # pre-emphasis
    y = np.append(y[0], y[1:] - 0.97*y[:-1])
    y = y / (np.max(np.abs(y))+1e-9)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    if mfcc.shape[1] < 3:
        return None

    width = mfcc.shape[1]
    if width % 2 == 0:
        width -= 1
    width = min(5, width)
    if width < 3:
        width = 3

    d1 = librosa.feature.delta(mfcc, width=width)
    d2 = librosa.feature.delta(mfcc, width=width, order=2)

    return np.vstack([mfcc, d1, d2]).astype(np.float32)


def split_speakers_global(per_class_files, seed=1234, train_frac=0.8, val_frac=0.1):
    rng = random.Random(seed)

    speaker_map = {}
    for cls, files in per_class_files.items():
        for f in files:
            sp = f.split("_")[0]
            speaker_map.setdefault(sp, []).append((cls, f))

    speakers = list(speaker_map.keys())
    rng.shuffle(speakers)

    N = len(speakers)
    n_train = int(round(N * train_frac))
    n_val = int(round(N * val_frac))
    if n_train + n_val > N:
        n_val = N - n_train

    train_s = set(speakers[:n_train])
    val_s   = set(speakers[n_train:n_train+n_val])
    test_s  = set(speakers[n_train+n_val:])

    train, val, test = [], [], []

    for sp, items in speaker_map.items():
        if sp in train_s:
            train.extend(items)
        elif sp in val_s:
            val.extend(items)
        else:
            test.extend(items)

    return train, val, test


def process_split(items, split_name, raw_base, out_base, sr, n_mfcc, n_fft, hop_length,
                  top_db, noises, aug_prob, mix_ratio, rng, meta):

    orig_count = 0
    aug_count = 0

    for cls, wavname in tqdm(items, desc=f"Processing {split_name}"):

        in_path = os.path.join(raw_base, cls, wavname)

        try:
            y, _ = librosa.load(in_path, sr=sr)
        except:
            print("Skipping unreadable:", in_path)
            continue

        # original features
        feat = extract_features(y, sr, n_mfcc, n_fft, hop_length, top_db)
        if feat is None:
            continue

        out_dir = os.path.join(out_base, cls)
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(wavname)[0]
        out_path = os.path.join(out_dir, base + ".npy")
        np.save(out_path, feat)

        speaker = get_speaker_from_wav(wavname)
        meta.append((out_path, cls, wavname, speaker, split_name, False))
        orig_count += 1

        # ------------- AUGMENTATION FOR ALL SPLITS ----------------
        if noises and rng.random() < aug_prob:
            y_aug = add_background_noise_linear(y, noises, mix_ratio=mix_ratio, rng=rng)

            feat_aug = extract_features(y_aug, sr, n_mfcc, n_fft, hop_length, top_db)
            if feat_aug is not None:
                aug_path = os.path.join(out_dir, base + "_aug.npy")
                np.save(aug_path, feat_aug)
                meta.append((aug_path, cls, wavname, speaker, split_name, True))
                aug_count += 1

    print(f"{split_name}: {orig_count} original, {aug_count} augmented")


def main(args):
    raw_base = args.raw
    if raw_base is None or not os.path.isdir(raw_base):
        raise RuntimeError("Invalid --raw dataset path")

    out_base = args.out
    os.makedirs(out_base, exist_ok=True)

    noise_dir = os.path.join(raw_base, "_background_noise_")
    noises = load_background_noises(noise_dir, sr=args.sr)

    # Collect class files
    classes = sorted([
        d for d in os.listdir(raw_base)
        if os.path.isdir(os.path.join(raw_base, d)) and not d.startswith("_")
    ])

    per_class_files = {}
    for cls in classes:
        wavs = sorted([f for f in os.listdir(os.path.join(raw_base, cls)) if f.endswith(".wav")])
        if args.debug:
            wavs = wavs[:args.debug_samples_per_class]
        per_class_files[cls] = wavs

    # global speaker-disjoint split
    train_list, val_list, test_list = split_speakers_global(
        per_class_files, seed=args.seed
    )

    print("Split sizes:", len(train_list), len(val_list), len(test_list))

    meta = []
    rng = random.Random(args.seed + 999)

    # process all splits
    process_split(train_list, "training", raw_base, out_base,
                  args.sr, args.n_mfcc, args.n_fft, args.hop_length,
                  args.top_db, noises, args.aug_prob, args.mix_ratio, rng, meta)

    process_split(val_list, "validation", raw_base, out_base,
                  args.sr, args.n_mfcc, args.n_fft, args.hop_length,
                  args.top_db, noises, args.aug_prob, args.mix_ratio, rng, meta)

    process_split(test_list, "testing", raw_base, out_base,
                  args.sr, args.n_mfcc, args.n_fft, args.hop_length,
                  args.top_db, noises, args.aug_prob, args.mix_ratio, rng, meta)

    df = pd.DataFrame(meta, columns=["path","label","orig_wav","speaker","split","augmented"])
    df.to_csv(os.path.join(out_base, "metadata.csv"), index=False)

    print("Saved metadata.csv")
    print(df["split"].value_counts())
    print("Augmentation count:", df["augmented"].sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default="/teamspace/studios/this_studio/processed")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=12)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--top_db", type=int, default=20)
    parser.add_argument("--aug_prob", type=float, default=0.10)
    parser.add_argument("--mix_ratio", type=float, default=0.10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_samples_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)