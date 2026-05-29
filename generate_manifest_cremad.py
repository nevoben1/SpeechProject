"""
CREMA-D Manifest Generator
Parses CREMA-D AudioWAV filenames to produce a manifest CSV
ready for step1_inference.py. Ground truth is taken from the filename-encoded
emotion (standard in literature).

Usage:
    python generate_manifest_cremad.py \
        --root "C:/path/to/CREMA-D" \
        --output cremad_manifest.csv

Expected CREMA-D structure:
    CREMA-D/
        AudioWAV/                        ← .wav audio files

Output columns:
    file_path, ground_truth, speaker_id, dataset
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# ── Label mapping to 4-class palette ──────────────────────────────────────────
# CREMA-D raw labels → shared 4-class set
# Full set: ANG, DIS, FEA, HAP, NEU, SAD
LABEL_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "DIS": None,   # disgust — drop
    "FEA": None,   # fear    — drop
}

# ── Filename parser ────────────────────────────────────────────────────────────
# CREMA-D filenames encode metadata:
# 1001_DFA_ANG_XX.wav
#  ^^^^ actor ID (1001–1091)
#       ^^^ sentence ID
#           ^^^ emotion
#               ^^ emotion level (LO, MD, HI, XX)

def parse_filename(fname: str) -> dict | None:
    """Extract actor_id and emotion from a CREMA-D filename."""
    stem = Path(fname).stem          # e.g. 1001_DFA_ANG_XX
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return {
        "actor_id": parts[0],        # e.g. 1001
        "emotion":  parts[2].upper() # e.g. ANG
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(root: Path, output: Path, keep_unmapped: bool):
    audio_dir     = root / "AudioWAV"
    summary_path  = root / "processedResults" / "summaryTable.csv"

    # Validate paths
    if not audio_dir.exists():
        print(f"ERROR: AudioWAV folder not found at {audio_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No .wav files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_files)} .wav files in {audio_dir}")

    rows = []
    skipped_label  = 0
    skipped_no_wav = 0

    for wav_path in wav_files:
        # ── Get label from filename ──
        parsed = parse_filename(wav_path.name)
        if not parsed:
            skipped_no_wav += 1
            continue
        raw_label = parsed["emotion"]

        # ── Map to 4-class ──
        mapped = LABEL_MAP.get(raw_label)
        if mapped is None:
            skipped_label += 1
            if keep_unmapped:
                mapped = raw_label.lower()
            else:
                continue

        # ── Speaker ID from filename ──
        speaker_id = parsed["actor_id"]

        rows.append({
            "file_path":    str(wav_path.resolve()),
            "ground_truth": mapped,
            "speaker_id":   speaker_id,
            "dataset":      "cremad",
        })

    # Write CSV
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "ground_truth", "speaker_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──
    print("\n" + "═" * 55)
    print(f"Total .wav files found  : {len(wav_files)}")
    print(f"Skipped (rare label)    : {skipped_label}  (disgust/fear)")
    print(f"Skipped (parse error)   : {skipped_no_wav}")
    print(f"Written to manifest     : {len(rows)} rows")
    print(f"Output file             : {output}")
    print("═" * 55)

    dist = Counter(r["ground_truth"] for r in rows)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"  {label:<12} {count:>5}  ({pct:.1f}%)")

    speaker_count = len(set(r["speaker_id"] for r in rows))
    print(f"\nUnique speakers: {speaker_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CREMA-D manifest CSV")
    parser.add_argument("--root",   required=True, help="Path to CREMA-D root folder")
    parser.add_argument("--output", default="cremad_manifest.csv", help="Output CSV path")
    parser.add_argument("--keep-unmapped", action="store_true",
                        help="Keep disgust/fear instead of dropping them")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    run(root, Path(args.output), args.keep_unmapped)
