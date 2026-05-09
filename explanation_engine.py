from gradcam import GradCAM
from uncertainty import get_uncertainty_summary
from utils import (
    overlay_heatmap_on_pil,
    get_heatmap_focus_description,
    get_top2_overlap,
    format_prob_bar
)
import numpy as np


class ExplanationEngine:
    """
    Maps structured query intents to visual and textual explanations.

    Each intent handler returns a dict with:
        text         : string explanation
        visual       : numpy RGB array or None
        visual_label : string describing what the visual shows
        extras       : dict of any additional data for UI rendering
    """

    def __init__(self):
        self.gcam = GradCAM()

    def explain(
        self,
        intent: str,
        pil_image,
        tensor,
        labels: list,
        probs: list,
        indices: list,
        full_probs,
    ) -> dict:
        """
        Route intent to correct explanation handler.

        Args:
            intent     : string from query_interpreter
            pil_image  : original PIL image
            tensor     : preprocessed torch tensor
            labels     : top-5 label strings
            probs      : top-5 probability floats
            indices    : top-5 class indices
            full_probs : full 1000-class softmax tensor

        Returns:
            dict with keys: text, visual, visual_label, extras
        """
        handlers = {
            "why_prediction":        self._why_prediction,
            "confidence":            self._confidence,
            "important_region":      self._important_region,
            "alternative_prediction": self._alternative_prediction,
            "comparison":            self._comparison,
            "unknown":               self._unknown,
        }

        handler = handlers.get(intent, self._unknown)
        return handler(
            pil_image, tensor, labels, probs, indices, full_probs
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _why_prediction(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """
        Explain why the model made its top prediction.
        Shows Grad-CAM heatmap with textual explanation.
        """
        heatmap = self.gcam.compute(tensor, indices[0])
        visual = overlay_heatmap_on_pil(heatmap, pil_image)
        focus = get_heatmap_focus_description(heatmap)
        peak = self.gcam.get_peak_region(heatmap)

        focus_text = {
            "focused": (
                "The model's attention was concentrated on a specific "
                "region, suggesting clear distinguishing features were present."
            ),
            "moderate": (
                "The model's attention was moderately distributed, "
                "drawing on several regions of the image."
            ),
            "diffuse": (
                "The model's attention was broadly distributed across "
                "the image, suggesting the prediction draws on global "
                "rather than local features."
            ),
        }

        text = (
            f"The model classified this image as '{labels[0]}' "
            f"with {probs[0]*100:.1f}% confidence.\n\n"
            f"The highlighted regions show which parts of the image "
            f"contributed most strongly to this prediction. "
            f"Peak activation was detected in the {peak} of the image.\n\n"
            f"{focus_text[focus]}"
        )

        return {
            "text": text,
            "visual": visual,
            "visual_label": f"Grad-CAM: Why '{labels[0]}'?",
            "extras": {"focus": focus, "peak_region": peak},
        }

    def _confidence(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """
        Explain model confidence using entropy-based uncertainty.
        """
        summary = get_uncertainty_summary(full_probs)
        bar_data = format_prob_bar(labels, probs)

        band_display = {
            "high": "HIGH CONFIDENCE",
            "cautious": "MODERATE CONFIDENCE",
            "ambiguous": "LOW CONFIDENCE — HIGH AMBIGUITY",
        }

        text = (
            f"Confidence band: {band_display[summary['band']]}\n\n"
            f"Top prediction: '{labels[0]}' at "
            f"{probs[0]*100:.1f}%\n\n"
            f"Normalised entropy: {summary['entropy']} / 1.0\n"
            f"(0 = perfectly certain, 1 = maximally uncertain)\n\n"
            f"Effective classes considered: "
            f"{summary['effective_classes']}\n"
            f"(how many classes carry meaningful probability mass)\n\n"
            f"{summary['message']}"
        )

        return {
            "text": text,
            "visual": None,
            "visual_label": "Top-5 Prediction Probabilities",
            "extras": {
                "bar_data": bar_data,
                "entropy": summary["entropy"],
                "band": summary["band"],
            },
        }

    def _important_region(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """
        Identify and describe the most important spatial region.
        """
        heatmap = self.gcam.compute(tensor, indices[0])
        visual = overlay_heatmap_on_pil(heatmap, pil_image)
        peak = self.gcam.get_peak_region(heatmap)
        focus = get_heatmap_focus_description(heatmap)

        focus_description = {
            "focused": (
                "Attention is tightly concentrated — "
                "a specific local feature drove this prediction."
            ),
            "moderate": (
                "Attention is moderately spread across "
                "several regions of the image."
            ),
            "diffuse": (
                "Attention is broadly distributed — "
                "the prediction relies on global image patterns "
                "rather than a single localised feature."
            ),
        }

        text = (
            f"The most important region for predicting '{labels[0]}' "
            f"is the {peak} of the image.\n\n"
            f"Activation pattern: {focus.upper()}\n"
            f"{focus_description[focus]}\n\n"
            f"Red and yellow areas in the heatmap indicate high "
            f"importance. Blue and purple areas contributed minimally "
            f"to this prediction."
        )

        return {
            "text": text,
            "visual": visual,
            "visual_label": f"Activation Map: Important Regions for '{labels[0]}'",
            "extras": {"peak_region": peak, "focus": focus},
        }

    def _alternative_prediction(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """
        Show top-3 alternative predictions the model considered.
        """
        overlap = get_top2_overlap(probs)
        bar_data = format_prob_bar(labels[:3], probs[:3])

        alt_text = "\n".join([
    f"  {i}. {labels[i]:<30} {probs[i]*100:.2f}%"
    for i in range(1, min(4, len(labels)))
])

        overlap_note = (
            "\nNote: The gap between the top two predictions is small "
            "(less than 15%). The model shows meaningful uncertainty "
            "between these classes."
            if overlap else
            "\nThe top prediction is substantially more probable than "
            "the alternatives, indicating the model's primary choice "
            "is relatively stable."
        )

        text = (
            f"Beyond '{labels[0]}', the model also considered:\n\n"
            f"{alt_text}\n"
            f"{overlap_note}"
        )

        return {
            "text": text,
            "visual": None,
            "visual_label": "Top-3 Alternative Predictions",
            "extras": {
                "bar_data": bar_data,
                "overlap": overlap,
            },
        }

    def _comparison(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """
        Compare Grad-CAM activations for top-1 vs top-2 predictions.
        Shows side-by-side heatmaps to highlight distinguishing regions.
        """
        heatmap1 = self.gcam.compute(tensor, indices[0])
        heatmap2 = self.gcam.compute(tensor, indices[1])

        visual1 = overlay_heatmap_on_pil(heatmap1, pil_image)
        visual2 = overlay_heatmap_on_pil(heatmap2, pil_image)

        peak1 = self.gcam.get_peak_region(heatmap1)
        peak2 = self.gcam.get_peak_region(heatmap2)
        focus1 = get_heatmap_focus_description(heatmap1)
        focus2 = get_heatmap_focus_description(heatmap2)

        same_region = peak1 == peak2

        if same_region:
            region_note = (
                f"Both predictions activate the same region ({peak1}), "
                f"suggesting these classes share visual features in "
                f"this image — which explains the model's difficulty "
                f"distinguishing between them."
            )
        else:
            region_note = (
                f"The predictions attend to different regions: "
                f"'{labels[0]}' focused on the {peak1}, while "
                f"'{labels[1]}' focused on the {peak2}. "
                f"These differing attention patterns reveal what "
                f"visual features distinguish the two classes."
            )

        text = (
            f"Comparing '{labels[0]}' ({probs[0]*100:.1f}%) "
            f"vs '{labels[1]}' ({probs[1]*100:.1f}%):\n\n"
            f"'{labels[0]}' attention: {peak1} — {focus1}\n"
            f"'{labels[1]}' attention: {peak2} — {focus2}\n\n"
            f"{region_note}"
        )

        return {
            "text": text,
            "visual": visual1,
            "visual2": visual2,
            "visual_label": f"Grad-CAM: '{labels[0]}'",
            "visual2_label": f"Grad-CAM: '{labels[1]}'",
            "extras": {
                "peak1": peak1,
                "peak2": peak2,
                "same_region": same_region,
            },
        }

    def _unknown(
        self, pil_image, tensor, labels, probs, indices, full_probs
    ) -> dict:
        """Fallback for unrecognised queries."""
        text = (
            "Query not recognised.\n\n"
            "You can ask about:\n"
            "  • Why the model made this prediction\n"
            "  • How confident the model is\n"
            "  • Which part of the image mattered most\n"
            "  • What other predictions were considered\n"
            "  • How the top two predictions compare"
        )
        return {
            "text": text,
            "visual": None,
            "visual_label": None,
            "extras": {},
        }