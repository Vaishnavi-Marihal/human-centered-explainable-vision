from classifier import load_image, classify


image_path = "test.jpg"
pil_image, tensor = load_image(image_path)
labels, probs, indices, full_probs = classify(tensor)

print("Top 5 Predictions:")
for i in range(5):
    print(f"  {i+1}. {labels[i]:<30} {probs[i]*100:.2f}%")
print(f"\nFull prob tensor shape: {full_probs.shape}")