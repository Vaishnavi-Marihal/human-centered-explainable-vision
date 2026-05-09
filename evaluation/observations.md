# Evaluation Observations

## Protocol
25 images evaluated across 5 categories of increasing visual complexity.
Each image processed through ResNet50 classification, entropy-based 
uncertainty quantification, and Grad-CAM saliency mapping.

Logged metrics per image:
- Top-1 and Top-2 predictions with probabilities
- Normalised Shannon entropy
- Confidence band (high / cautious / ambiguous)
- Effective classes
- Grad-CAM focus pattern (focused / moderate / diffuse)
- Top-2 probability overlap

---

## Category Summary

| Category | N | Avg Entropy | Avg Top-1 Prob | Diffuse Heatmaps |
|---|---|---|---|---|
| Clear | 3 | 0.0939 | 0.7905 | 1/3 |
| Moderate | 5 | 0.2324 | 0.5693 | 2/5 |
| Cluttered | 5 | 0.3658 | 0.5217 | 5/5 |
| Similar | 5 | 0.0652 | 0.8281 | 4/5 |
| OOD | 4 | 0.4207 | 0.3414 | 1/4 |

---

## Key Observations

**O1 — Entropy tracks visual complexity monotonically**
Normalised entropy increased consistently from clear (0.0939) through 
moderate (0.2324) to cluttered (0.3658) and OOD (0.4207) categories. 
This validates entropy as a meaningful proxy for image-level 
uncertainty, not merely model noise.

**O2 — Cluttered images universally produced diffuse Grad-CAM activation**
All 5 cluttered images produced diffuse heatmaps (5/5). No other 
category showed this uniformity. When scenes contain multiple objects 
or complex backgrounds, the model cannot localise its attention to a 
specific region — activation spreads broadly across the image. This 
directly supports the project's central hypothesis: visual complexity 
degrades both confidence and spatial interpretability simultaneously.

**O3 — Confidence and attention focus are partially independent**
Several high-confidence predictions produced diffuse Grad-CAM maps:
- market street (69.8% confidence, diffuse, incorrect label)
- husky face (56.0% confidence, high band, diffuse)
- cheetah (100% confidence, high band, diffuse)
This suggests that entropy-based confidence and Grad-CAM spatial 
focus capture different properties of model behaviour. A system 
reporting only confidence scores would miss this distinction entirely.

**O4 — Inter-class similar images showed unexpectedly low entropy**
The similar category produced the lowest average entropy (0.0652), 
lower even than clear images (0.0939). The model committed decisively 
to one class even for visually confusable pairs (husky/wolf, 
leopard/cheetah). Low entropy does not guarantee correct 
fine-grained discrimination — it reflects model commitment, 
not necessarily model correctness.

**O5 — OOD images correctly triggered high uncertainty**
Out-of-distribution images (surreal art, costumes, robots, abstract 
paintings) produced the highest average entropy (0.4207) and lowest 
average top-1 probability (0.3414). The uncertainty module correctly 
flagged these with cautious or ambiguous confidence bands, 
demonstrating appropriate epistemic humility on unfamiliar inputs.

**O6 — One clear-category image was misclassified with moderate confidence**
red_apple_white_background.jpg was classified as "strawberry" 
(37.5%, high band). This represents a failure case where the 
confidence band does not signal the error — the model is moderately 
confident in an incorrect prediction. This highlights the limitation 
that confidence bands reflect distributional certainty, not 
ground-truth correctness.

---

## Implications for Explanation Design

These observations suggest that a human-centered explainable vision 
system should surface at minimum three distinct signals to a user:

1. Confidence score (top-1 probability)
2. Uncertainty estimate (entropy-based band)
3. Attention pattern (Grad-CAM focus description)

No single signal is sufficient. A system reporting only confidence 
would miss the attention diffusion pattern observed in O2 and O3. 
A system reporting only the Grad-CAM map would miss the entropy 
signal that correctly flagged OOD inputs in O5.