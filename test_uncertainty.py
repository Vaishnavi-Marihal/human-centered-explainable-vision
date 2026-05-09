from classifier import load_image, classify
from uncertainty import get_uncertainty_summary

image_path = "test.jpg"
pil_image, tensor = load_image(image_path)
labels, probs, indices, full_probs = classify(tensor)

summary = get_uncertainty_summary(full_probs)

print(f"Top prediction    : {labels[0]} ({probs[0]*100:.2f}%)")
print(f"Normalised entropy: {summary['entropy']}")
print(f"Confidence band   : {summary['band']}")
print(f"Message           : {summary['message']}")
print(f"Effective classes : {summary['effective_classes']}")