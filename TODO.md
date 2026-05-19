# TODO

## Infrastructure
- [x] Redeploy Server.py to RunPod
- [~] Rerun inference on IEMOCAP and CREMA-D (`step1_inference.py`) — CREMA-D processing, IEMOCAP pending

## EDA Phase 1 — Error & Confidence Profiling
- [ ] Per-class error profiling: confusion matrix per dataset (IEMOCAP + CREMA-D separately)
- [ ] Identify dominant confusion pairs (e.g. neutral→sad)
- [ ] Confidence distribution: softmax score histograms correct vs wrong
- [ ] Flag high-confidence wrong predictions (systematic bias) vs low-confidence wrong (ambiguous)

## EDA Phase 2 — Speaker Analysis (IEMOCAP)
- [ ] Expand IEMOCAP coverage to all 5 sessions (currently only Sessions 1–2)
- [ ] Compute per-speaker accuracy across all 10 speakers
- [ ] Determine if neutral failures are speaker-specific or universal

## EDA Phase 3 — Acoustic Feature Breakdown
- [ ] Rerun acoustic feature extraction on both datasets (`extract_acoustic_features.py`)
- [ ] Box plots pitch/energy/speech rate per emotion: correct vs misclassified
- [ ] Correlation matrix: which acoustic features correlate with misclassification

## EDA Phase 4 — Embedding Visualization
- [ ] Extract Emotion2Vec encoder embeddings for all samples (via updated server + inference script)
- [ ] Apply UMAP / t-SNE projection on embeddings
- [ ] Color by: true label, predicted label, confidence, correct/wrong
- [ ] Visualize where failure samples fall relative to model decision boundaries
