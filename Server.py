from fastapi import FastAPI, UploadFile, File
from funasr import AutoModel
import tempfile, os, json
import numpy as np

app = FastAPI()

# --- emotion2vec model ---
emotion2vec_model = AutoModel(
    model="iic/emotion2vec_plus_large",
    device="cuda"
)

EMOTION2VEC_LABEL_MAP = {
    "happy":     "HAPPY",
    "sad":       "SAD",
    "angry":     "ANGRY",
    "fear":      "FEARFUL",
    "fearful":   "FEARFUL",
    "disgust":   "DISGUSTED",
    "disgusted": "DISGUSTED",
    "surprised": "SURPRISED",
    "surprise":  "SURPRISED",
    "neutral":   "NEUTRAL",
    "unknown":   "UNKNOWN",
}

def run_emotion2vec(audio_path: str) -> dict:
    result = emotion2vec_model.generate(
        input=audio_path,
        output_dir=None,
        granularity="utterance",
        extract_embedding=True,
    )
    scores = result[0]["scores"]
    labels = result[0]["labels"]

    def clean_label(lbl: str) -> str:
        if "/" in lbl:
            lbl = lbl.split("/")[-1].strip()
        return EMOTION2VEC_LABEL_MAP.get(lbl.lower(), lbl.upper())

    best_idx = scores.index(max(scores))
    emotion = clean_label(labels[best_idx])
    confidence = {
        clean_label(lbl): round(score, 4)
        for lbl, score in zip(labels, scores)
    }
    feats = result[0].get("feats", None)
    if feats is not None:
        embedding = np.array(feats).flatten().tolist()
    else:
        embedding = None
    return {
        "emotion":    emotion,
        "confidence": confidence,
        "embedding":  embedding,
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = run_emotion2vec(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result