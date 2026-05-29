"""
Step 1 — Baseline Inference Client
Sends audio samples to the RunPod server and collects predictions
from Emotion2Vec into a CSV for analysis.

Usage:
    python step1_inference.py \
        --server http://<your-runpod-host>:<port> \
        --dataset iemocap \
        --manifest manifest.csv \
        --output results_iemocap.csv

Manifest CSV format (one row per utterance):
    file_path,ground_truth,speaker_id,dataset
    /path/to/audio.wav,angry,Ses01F,iemocap

The script handles:
- Sending each .wav to the /analyze endpoint
- Retrying on transient failures
- Writing results incrementally (safe to resume if interrupted)
- Printing a live summary as it runs
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

# ── Label normalization ────────────────────────────────────────────────────────
LABEL_MAP_4CLASS = {
    # anger family
    "angry":      "angry",
    "anger":      "angry",
    "ANGRY":      "angry",
    "ang":        "angry",
    "ANG":        "angry",
    # sadness family
    "sad":        "sad",
    "sadness":    "sad",
    "SAD":        "sad",
    # happiness family
    "happy":      "happy",
    "happiness":  "happy",
    "HAPPY":      "happy",
    "joy":        "happy",
    "excited":    "happy",
    "excitement": "happy",
    "hap":        "happy",
    "HAP":        "happy",
    # neutral family
    "neutral":    "neutral",
    "NEUTRAL":    "neutral",
    "neu":        "neutral",
    "NEU":        "neutral",
}

def normalize_label(raw: str) -> str:
    return LABEL_MAP_4CLASS.get(raw, raw.lower())


# ── Server communication ───────────────────────────────────────────────────────

def send_audio(server_url: str, audio_path: str, retries: int = 3, timeout: int = 30) -> dict | None:
    """POST an audio file to /analyze, return parsed JSON or None on failure."""
    url = f"{server_url.rstrip('/')}/analyze"
    for attempt in range(1, retries + 1):
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"file": (Path(audio_path).name, f, "audio/wav")},
                    timeout=timeout,
                )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"  [timeout] attempt {attempt}/{retries} — {audio_path}")
        except requests.exceptions.RequestException as e:
            print(f"  [error]   attempt {attempt}/{retries} — {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


# ── CSV helpers ────────────────────────────────────────────────────────────────

CONF_LABELS = ["ANGRY", "DISGUSTED", "FEARFUL", "HAPPY", "NEUTRAL", "SAD", "SURPRISED", "UNKNOWN"]

OUTPUT_FIELDS = [
    "file_path",
    "dataset",
    "speaker_id",
    "ground_truth_raw",
    "ground_truth_4class",
    # Emotion2Vec
    "e2v_raw",
    "e2v_4class",
    "e2v_correct",
    "e2v_conf_top",
    # Full confidence scores per emotion
    *[f"conf_{lbl.lower()}" for lbl in CONF_LABELS],
    # 768-dim embedding as JSON string
    "embedding",
]


def load_existing_results(output_path: str) -> set:
    """Return set of file_paths already written (for resuming)."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["file_path"])
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    with open(args.manifest, newline="") as f:
        samples = list(csv.DictReader(f))

    print(f"Loaded {len(samples)} samples from {args.manifest}")
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limiting to first {args.limit} samples (test mode)")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}\n")

    done = load_existing_results(args.output)
    if done:
        print(f"Resuming — {len(done)} samples already done, skipping.\n")

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    out_file = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    total = len(samples)
    processed = len(done)
    errors = 0
    e2v_correct_count = 0

    try:
        for i, sample in enumerate(samples, 1):
            fp = sample["file_path"]

            if fp in done:
                continue

            gt_raw     = sample.get("ground_truth", "unknown")
            speaker_id = sample.get("speaker_id", "")
            dataset    = sample.get("dataset", args.dataset)
            gt_4class  = normalize_label(gt_raw)

            result = send_audio(args.server, fp, retries=args.retries)

            if result is None:
                errors += 1
                print(f"[{i}/{total}] FAILED — {fp}")
                row = {
                    "file_path": fp, "dataset": dataset, "speaker_id": speaker_id,
                    "ground_truth_raw": gt_raw, "ground_truth_4class": gt_4class,
                    "e2v_raw": "ERROR", "e2v_4class": "ERROR", "e2v_correct": "",
                    "e2v_conf_top": "", "embedding": "",
                }
                for lbl in CONF_LABELS:
                    row[f"conf_{lbl.lower()}"] = ""
                writer.writerow(row)
                out_file.flush()
                continue

            # ── Parse response ──
            e2v_raw = result["emotion"]

            # emotion2vec sometimes returns "生气/ANGRY" format — extract English part
            if "/" in e2v_raw:
                e2v_raw = e2v_raw.split("/")[-1].strip()

            e2v_4class  = normalize_label(e2v_raw)
            e2v_correct = (e2v_4class == gt_4class)
            conf        = result.get("confidence", {})
            conf_top    = conf.get(e2v_raw, conf.get(e2v_raw.lower(), ""))
            embedding   = result.get("embedding", None)
            emb_str     = json.dumps(embedding) if embedding is not None else ""

            processed += 1
            if e2v_correct:
                e2v_correct_count += 1

            row = {
                "file_path":           fp,
                "dataset":             dataset,
                "speaker_id":          speaker_id,
                "ground_truth_raw":    gt_raw,
                "ground_truth_4class": gt_4class,
                "e2v_raw":             e2v_raw,
                "e2v_4class":          e2v_4class,
                "e2v_correct":         e2v_correct,
                "e2v_conf_top":        conf_top,
                "embedding":           emb_str,
            }
            for lbl in CONF_LABELS:
                row[f"conf_{lbl.lower()}"] = conf.get(lbl, "")
            writer.writerow(row)
            out_file.flush()

            if processed % 10 == 0:
                pct_done = processed / total * 100
                pct_e2v  = e2v_correct_count / processed * 100
                print(
                    f"[{processed}/{total} | {pct_done:.1f}%] "
                    f"E2V acc: {pct_e2v:.1f}%  "
                    f"Errors: {errors}"
                )

    finally:
        out_file.close()

    print("\n" + "═" * 60)
    print(f"DONE  —  {processed} samples processed, {errors} errors")
    if processed > 0:
        print(f"Emotion2Vec acc : {e2v_correct_count/processed*100:.1f}%")
    print(f"Results written to: {args.output}")
    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 — Baseline inference collector")
    parser.add_argument("--server",   required=True,  help="RunPod server URL, e.g. http://1.2.3.4:8080")
    parser.add_argument("--dataset",  required=True,  choices=["iemocap", "meld", "cremad"], help="Dataset name tag")
    parser.add_argument("--manifest", required=True,  help="CSV with file_path, ground_truth, speaker_id columns")
    parser.add_argument("--output",   required=True,  help="Output CSV path")
    parser.add_argument("--retries",  default=3, type=int, help="Retry attempts per file (default: 3)")
    parser.add_argument("--limit",    default=None, type=int, help="Only process this many samples (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    run(args)
