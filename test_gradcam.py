import cv2
from classifier import load_image, classify
from gradcam import GradCAM

image_path = "test.jpg"
pil_image, tensor = load_image(image_path)
labels, probs, indices, full_probs = classify(tensor)

print(f"Explaining prediction: {labels[0]}")

# Compute Grad-CAM for top prediction
gcam = GradCAM()
heatmap = gcam.compute(tensor, indices[0])
overlaid = gcam.overlay_heatmap(heatmap, pil_image)
peak = gcam.get_peak_region(heatmap)

print(f"Heatmap shape : {heatmap.shape}")
print(f"Heatmap range : {heatmap.min():.3f} to {heatmap.max():.3f}")
print(f"Peak region   : {peak}")

# Save output image to verify visually
cv2.imwrite("gradcam_output.jpg", overlaid)
print("Saved gradcam_output.jpg — open it to verify heatmap")