"""
IEMOCAP Manifest Generator
Parses EmoEvaluation annotation files and produces a manifest CSV
ready for step1_inference.py

Usage:
    python generate_manifest_iemocap.py \
        --root "C:/Users/nevo/.cache/kagglehub/datasets/dejolilandry/iemocapfullrelease/versions/1/IEMOCAP_full_release" \
        --output iemocap_manifest.csv

Output columns:
    file_path, ground_truth, speaker_id, dataset
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# ── Label mapping to 4-class palette ──────────────────────────────────────────
# IEMOCAP raw labels → shared 4-class set
LABEL_MAP = {
    "ang": "angry",
    "sad": "sad",
    "hap": "happy",
    "exc": "happy",      # excitement → happiness (standard in literature)
    "neu": "neutral",
    "fru": "neutral",    # frustration → neutral (common mapping)
    "fea": None,         # fear — <1% of samples, typically dropped
    "sur": None,         # surprise — <1% of samples, typically dropped
    "dis": None,         # disgust — <1% of samples, typically dropped
    "oth": None,         # other — dropped
    "xxx": None,         # uncertain — dropped
}

# ── Parser ─────────────────────────────────────────────────────────────────────

# EmoEvaluation line format:
# [6.2901 - 8.2357]	Ses01F_impro01_F000	ang	[2, 2, 2]
LINE_RE = re.compile(
    r"\[[\d.]+ - [\d.]+\]\s+"   # time range
    r"(\w+)\s+"                  # utterance ID  (group 1)
    r"(\w+)\s+"                  # emotion label (group 2)
    r"\[[\d., ]+\]"              # annotator votes (int or float)
)

def parse_emo_evaluation(txt_path: Path) -> list[dict]:
    """Parse one EmoEvaluation .txt file, return list of utterance dicts."""
    utterances = []
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("C-"):  # skip category lines
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            utt_id    = m.group(1)   # e.g. Ses01F_impro01_F000
            raw_label = m.group(2).lower()
            utterances.append({"utt_id": utt_id, "raw_label": raw_label})
    return utterances


def find_wav(wav_root: Path, utt_id: str) -> Path | None:
    """
    Locate the .wav file for an utterance ID.
    IEMOCAP stores them at:
      Session{N}/sentences/wav/{dialog_name}/{utt_id}.wav
    e.g. Ses01F_impro01_F000 → Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
    """
    # dialog name = everything up to the last underscore + gender/index token
    # Ses01F_impro01_F000 → dialog = Ses01F_impro01
    parts = utt_id.rsplit("_", 1)
    if len(parts) < 2:
        return None
    dialog_name = parts[0]

    # Session number from utterance ID: Ses01 → 1
    session_match = re.match(r"Ses0?(\d)", utt_id)
    if not session_match:
        return None
    session_num = session_match.group(1)

    wav_path = wav_root / f"Session{session_num}" / "sentences" / "wav" / dialog_name / f"{utt_id}.wav"
    return wav_path if wav_path.exists() else None


# ── Main ───────────────────────────────────────────────────────────────────────

def run(root: Path, output: Path, keep_unmapped: bool):
    sessions = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Session")])
    if not sessions:
        print(f"ERROR: No Session* folders found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sessions)} sessions: {[s.name for s in sessions]}")

    rows = []
    skipped_label  = 0
    skipped_no_wav = 0
    total_parsed   = 0

    for session in sessions:
        emo_eval_dir = session / "dialog" / "EmoEvaluation"
        if not emo_eval_dir.exists():
            print(f"  WARNING: EmoEvaluation not found in {session.name}, skipping")
            continue

        txt_files = sorted(emo_eval_dir.glob("*.txt"))
        print(f"  {session.name}: {len(txt_files)} annotation files")

        for txt_file in txt_files:
            utterances = parse_emo_evaluation(txt_file)
            total_parsed += len(utterances)

            for utt in utterances:
                utt_id    = utt["utt_id"]
                raw_label = utt["raw_label"]

                # Map label
                mapped = LABEL_MAP.get(raw_label)
                if mapped is None:
                    skipped_label += 1
                    if keep_unmapped:
                        mapped = raw_label  # keep as-is
                    else:
                        continue

                # Find wav
                wav_path = find_wav(root, utt_id)
                if wav_path is None:
                    skipped_no_wav += 1
                    continue

                # Speaker ID: first part before underscore position 2
                # Ses01F_impro01_F000 → speaker = Ses01F
                speaker_id = utt_id.split("_")[0]

                rows.append({
                    "file_path":    str(wav_path),
                    "ground_truth": mapped,
                    "speaker_id":   speaker_id,
                    "dataset":      "iemocap",
                })

    # Write CSV
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "ground_truth", "speaker_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print("\n" + "═" * 55)
    print(f"Total utterances parsed : {total_parsed}")
    print(f"Skipped (rare label)    : {skipped_label}  (fear/surprise/disgust/other)")
    print(f"Skipped (wav not found) : {skipped_no_wav}")
    print(f"Written to manifest     : {len(rows)} rows")
    print(f"Output file             : {output}")
    print("═" * 55)

    # Label distribution
    from collections import Counter
    dist = Counter(r["ground_truth"] for r in rows)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"  {label:<12} {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IEMOCAP manifest CSV")
    parser.add_argument("--root",   required=True, help="Path to IEMOCAP_full_release folder")
    parser.add_argument("--output", default="iemocap_manifest.csv", help="Output CSV path")
    parser.add_argument("--keep-unmapped", action="store_true",
                        help="Keep rare labels (fear/surprise/disgust) instead of dropping them")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    run(root, Path(args.output), args.keep_unmapped)
