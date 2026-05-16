# TODO

## Infrastructure
- [ ] Redeploy Server.py to RunPod
- [ ] Rerun inference on IEMOCAP and CREMA-D (`step1_inference.py`)

## 01 — Acoustic Feature Extraction
- [ ] Rerun acoustic feature extraction on both datasets (`extract_acoustic_features.py`)
- [ ] Compare feature distributions (pitch, energy, speech rate) between correctly and incorrectly classified utterances per emotion class

## 02 — Embedding Analysis
- [ ] Extract Emotion2Vec encoder embeddings for all samples (via updated server + inference script)
- [ ] Apply UMAP / t-SNE projection on embeddings
- [ ] Visualize where failure samples fall relative to model decision boundaries

## 03 — Confidence Analysis
- [ ] Examine softmax score distributions on misclassified samples
- [ ] Determine if failures show low confidence (ambiguous signal) or high confidence wrong predictions (systematic model bias)

## 04 — Speaker Variability Analysis
- [ ] Expand IEMOCAP coverage to all 5 sessions (currently only Sessions 1–2)
- [ ] Compute per-speaker accuracy across all 10 speakers
- [ ] Determine if neutral failures are speaker-specific or universal
