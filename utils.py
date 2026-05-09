import numpy as np
import cv2
from PIL import Image


def pil_to_numpy_rgb(pil_image, size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image to RGB numpy array at given size."""
    return np.array(pil_image.resize(size))


def numpy_bgr_to_rgb(bgr_array: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR array to RGB for Streamlit display."""
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)


def overlay_heatmap_on_pil(
    heatmap: np.ndarray,
    pil_image,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on PIL image.
    Returns RGB numpy array suitable for Streamlit st.image().

    Args:
        heatmap   : (224, 224) numpy array in [0, 1]
        pil_image : original PIL image
        alpha     : heatmap blend strength

    Returns:
        RGB numpy array (224, 224, 3)
    """
    img_np = pil_to_numpy_rgb(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlaid_bgr = cv2.addWeighted(
        img_bgr, 1 - alpha,
        colored, alpha,
        0
    )

    # Return RGB for Streamlit
    return cv2.cvtColor(overlaid_bgr, cv2.COLOR_BGR2RGB)


def get_heatmap_focus_description(heatmap: np.ndarray) -> str:
    """
    Describe how focused or diffuse the heatmap activation is.
    Used in evaluation logging.

    Computes what fraction of pixels carry above-threshold activation.
    Low fraction  = focused (model attention concentrated)
    High fraction = diffuse (model attention spread broadly)

    Returns: 'focused', 'moderate', or 'diffuse'
    """
    threshold = 0.5
    active_fraction = (heatmap > threshold).mean()

    if active_fraction < 0.15:
        return "focused"
    elif active_fraction < 0.35:
        return "moderate"
    else:
        return "diffuse"


def get_top2_overlap(probs: list) -> bool:
    """
    Check if top-2 predictions are close in probability.
    Signals potential inter-class confusion.

    Returns True if gap between top-1 and top-2 is less than 15%.
    """
    if len(probs) < 2:
        return False
    return (probs[0] - probs[1]) < 0.15


def format_prob_bar(labels: list, probs: list) -> dict:
    """
    Format top-5 predictions for Streamlit bar chart.
    Returns dict of {label: probability_percent}
    """
    return {
        label: round(prob * 100, 2)
        for label, prob in zip(labels, probs)
    }