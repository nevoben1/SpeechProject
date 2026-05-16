from fastapi import FastAPI, UploadFile, File
from funasr import AutoModel
import tempfile, os

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
        extract_embedding=False,
    )
    scores = result[0]["scores"]
    labels = result[0]["labels"]
    best_idx = scores.index(max(scores))
    raw_label = labels[best_idx].lower()
    emotion = EMOTION2VEC_LABEL_MAP.get(raw_label, raw_label.upper())
    confidence = {
        EMOTION2VEC_LABEL_MAP.get(lbl.lower(), lbl.upper()): round(score, 4)
        for lbl, score in zip(labels, scores)
    }
    return {
        "emotion": emotion,
        "confidence": confidence,
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