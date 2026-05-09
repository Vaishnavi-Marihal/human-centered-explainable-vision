import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import torch
from pathlib import Path
from classifier import load_image, classify
from gradcam import GradCAM
from uncertainty import get_uncertainty_summary
from utils import get_heatmap_focus_description, get_top2_overlap

# ------------------------------------------------------------------
# Category mapping by filename order
# Images are processed alphabetically so naming matters
# We assign categories manually based on intended grouping
# ------------------------------------------------------------------

CATEGORIES = {
    "clear": "Category 1 — Clear/Unambiguous",
    "moderate": "Category 2 — Moderate Complexity",
    "cluttered": "Category 3 — Visually Cluttered",
    "similar": "Category 4 — Inter-class Similar",
    "ood": "Category 5 — Out of Distribution",
}

def get_category(filename: str) -> str:
    """
    Assign category based on filename keywords.
    Falls back to 'unknown' if no keyword matches.
    """
    name = filename.lower()

    # Category 1 — clear single objects
    if any(k in name for k in [
        "banana", "apple", "guitar", "schoolbus", "school_bus",
        "golden_retriever", "white_background", "single"
    ]):
        return "clear"

    # Category 2 — moderate complexity
    elif any(k in name for k in [
        "labrador", "park", "cat_window", "elephant", "sports_car",
        "toucan", "window", "savanna", "moderate"
    ]):
        return "moderate"

    # Category 3 — cluttered
    elif any(k in name for k in [
        "crowd", "market", "flock", "traffic", "forest",
        "busy", "many", "cluttered", "street"
    ]):
        return "cluttered"

    # Category 4 — inter-class similar
    elif any(k in name for k in [
        "husky", "wolf", "leopard", "cheetah", "persian",
        "similar", "versus", "compare"
    ]):
        return "similar"

    # Category 5 — out of distribution
    elif any(k in name for k in [
        "surreal", "costume", "robot", "abstract", "dali",
        "unusual", "ood", "painting", "melting"
    ]):
        return "ood"

    else:
        return "unknown"


def evaluate_image(
    image_path: str,
    gcam: GradCAM
) -> dict:
    """
    Run full evaluation pipeline on a single image.
    Returns dict of all logged metrics.
    """
    filename = Path(image_path).name

    try:
        pil_image, tensor = load_image(image_path)
        labels, probs, indices, full_probs = classify(tensor)
        summary = get_uncertainty_summary(full_probs)

        # Grad-CAM for top prediction
        heatmap = gcam.compute(tensor, indices[0])
        focus = get_heatmap_focus_description(heatmap)
        overlap = get_top2_overlap(probs)

        # Top-2 probability gap
        top2_gap = round(probs[0] - probs[1], 4) if len(probs) > 1 else 1.0

        category = get_category(filename)

        return {
            "filename": filename,
            "category": category,
            "top1_label": labels[0],
            "top1_prob": round(probs[0], 4),
            "top2_label": labels[1] if len(labels) > 1 else "N/A",
            "top2_prob": round(probs[1], 4) if len(probs) > 1 else 0.0,
            "top2_gap": top2_gap,
            "entropy": summary["entropy"],
            "confidence_band": summary["band"],
            "effective_classes": summary["effective_classes"],
            "gradcam_focus": focus,
            "top2_overlap": overlap,
            "status": "ok",
        }

    except Exception as e:
        return {
            "filename": filename,
            "category": "error",
            "top1_label": "ERROR",
            "top1_prob": 0.0,
            "top2_label": "ERROR",
            "top2_prob": 0.0,
            "top2_gap": 0.0,
            "entropy": 0.0,
            "confidence_band": "error",
            "effective_classes": 0.0,
            "gradcam_focus": "error",
            "top2_overlap": False,
            "status": f"error: {str(e)}",
        }


def run_evaluation():
    """
    Run evaluation on all images in test_images folder.
    Saves results to evaluation_log.csv and prints summary.
    """
    test_dir = Path(__file__).parent / "test_images"
    output_csv = Path(__file__).parent / "evaluation_log.csv"

    # Collect all image files
    extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in test_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        print("No images found in evaluation/test_images/")
        return

    print(f"Found {len(image_files)} images. Running evaluation...\n")

    gcam = GradCAM()
    results = []

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i:02d}/{len(image_files)}] {image_path.name}")
        result = evaluate_image(str(image_path), gcam)
        results.append(result)

        print(
            f"         → {result['top1_label']:<30} "
            f"prob={result['top1_prob']:.3f}  "
            f"entropy={result['entropy']}  "
            f"band={result['confidence_band']:<10} "
            f"focus={result['gradcam_focus']}"
        )

    # Save CSV
    fieldnames = [
        "filename", "category", "top1_label", "top1_prob",
        "top2_label", "top2_prob", "top2_gap", "entropy",
        "confidence_band", "effective_classes",
        "gradcam_focus", "top2_overlap", "status"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {output_csv}")

    # Print summary by category
    print("\n" + "="*60)
    print("SUMMARY BY CATEGORY")
    print("="*60)

    categories_seen = {}
    for r in results:
        cat = r["category"]
        if cat not in categories_seen:
            categories_seen[cat] = []
        categories_seen[cat].append(r)

    for cat, items in categories_seen.items():
        avg_entropy = sum(i["entropy"] for i in items) / len(items)
        avg_prob = sum(i["top1_prob"] for i in items) / len(items)
        focus_counts = {}
        for i in items:
            f = i["gradcam_focus"]
            focus_counts[f] = focus_counts.get(f, 0) + 1
        overlap_count = sum(1 for i in items if i["top2_overlap"])

        print(f"\n{cat.upper()} ({len(items)} images)")
        print(f"  Avg entropy    : {avg_entropy:.4f}")
        print(f"  Avg top1 prob  : {avg_prob:.4f}")
        print(f"  Focus counts   : {focus_counts}")
        print(f"  Top2 overlaps  : {overlap_count}/{len(items)}")

    print("\n" + "="*60)
    print("Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()