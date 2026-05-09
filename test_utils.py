from classifier import load_image, classify
from gradcam import GradCAM
from utils import (
    overlay_heatmap_on_pil,
    get_heatmap_focus_description,
    get_top2_overlap,
    format_prob_bar
)
import cv2

image_path = "test.jpg"
pil_image, tensor = load_image(image_path)
labels, probs, indices, full_probs = classify(tensor)

gcam = GradCAM()
heatmap = gcam.compute(tensor, indices[0])

# Test overlay
overlaid_rgb = overlay_heatmap_on_pil(heatmap, pil_image)
print(f"Overlaid shape     : {overlaid_rgb.shape}")
print(f"Overlaid dtype     : {overlaid_rgb.dtype}")

# Test focus description
focus = get_heatmap_focus_description(heatmap)
print(f"Heatmap focus      : {focus}")

# Test top2 overlap
overlap = get_top2_overlap(probs)
print(f"Top-2 overlap      : {overlap}")

# Test prob bar
bar = format_prob_bar(labels, probs)
print(f"Prob bar           : {bar}")