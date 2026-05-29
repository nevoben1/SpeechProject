"""
Acoustic Feature Extraction
Reads a results CSV (iemocap_full or cremad_full), extracts 4 acoustic features
per WAV file, and writes an enriched CSV with new columns appended.

Features extracted (voiced frames only):
    mean_f0, std_f0  — pitch via Praat/parselmouth
    energy           — RMS energy via librosa
    speech_rate      — voiced frames / total frames ratio

Usage:
    python extract_acoustic_features.py --input cremad_full.csv --output cremad_features.csv
    python extract_acoustic_features.py --input iemocap_full.csv --output iemocap_features.csv --workers 8
"""

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

FEATURE_COLS = ["mean_f0", "std_f0", "energy", "speech_rate"]


def extract_features(file_path: str) -> dict:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # --- Pitch via Praat ---
        snd = parselmouth.Sound(str(path))
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]  # exclude unvoiced frames

        if len(voiced) >= 2:
            mean_f0 = float(np.mean(voiced))
            std_f0  = float(np.std(voiced))
        else:
            mean_f0 = std_f0 = np.nan

        # Speech rate: ratio of voiced frames to total frames
        total_frames = len(f0_values)
        speech_rate = len(voiced) / total_frames if total_frames > 0 else np.nan

        # --- Energy via librosa ---
        y, sr = librosa.load(str(path), sr=None, mono=True)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        return {
            "mean_f0":     mean_f0,
            "std_f0":      std_f0,
            "energy":      energy,
            "speech_rate": speech_rate,
        }

    except Exception as e:
        return {col: np.nan for col in FEATURE_COLS}


def process_row(row):
    feats = extract_features(row["file_path"])
    return feats


def main():
    parser = argparse.ArgumentParser(description="Extract acoustic features from audio CSVs")
    parser.add_argument("--input",   required=True,  help="Input CSV path (iemocap_full.csv or cremad_full.csv)")
    parser.add_argument("--output",  required=True,  help="Output CSV path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"{len(df)} rows loaded.")

    rows = df.to_dict("records")

    print(f"Extracting features with {args.workers} workers...")
    results = Parallel(n_jobs=args.workers, backend="loky")(
        delayed(process_row)(row) for row in rows
    )

    feat_df = pd.DataFrame(results, columns=FEATURE_COLS)
    out_df  = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    out_df.to_csv(args.output, index=False)
    print(f"Done. Saved to {args.output}")

    # Quick summary
    n_failed = feat_df["mean_f0"].isna().sum()
    print(f"Failed/missing: {n_failed}/{len(df)} files")


if __name__ == "__main__":
    main()
