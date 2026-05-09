import torch
import math


def compute_entropy(full_probs: torch.Tensor) -> float:
    """
    Compute Shannon entropy of the softmax probability distribution.
    
    H(p) = -sum(p_i * log(p_i))
    
    Higher entropy = more uncertainty across classes.
    Lower entropy = model is concentrated on fewer classes.
    
    Args:
        full_probs: softmax probability tensor of shape (1000,)
    Returns:
        normalised_entropy: float between 0.0 and 1.0
    """
    # Clamp to avoid log(0)
    probs = torch.clamp(full_probs, min=1e-9)
    
    # Raw Shannon entropy
    raw_entropy = -torch.sum(probs * torch.log(probs)).item()
    
    # Maximum possible entropy for 1000 classes = log(1000)
    max_entropy = math.log(1000)
    
    # Normalise to [0, 1]
    normalised = raw_entropy / max_entropy
    
    return round(normalised, 4)


def get_confidence_band(normalised_entropy: float) -> tuple:
    """
    Map normalised entropy to a confidence band.
    
    Three bands:
        high       -> model is concentrated, prediction reliable
        cautious   -> moderate spread, interpret carefully  
        ambiguous  -> high spread, model genuinely uncertain
    
    Returns:
        band  : string label
        message : human-readable explanation
    """
    if normalised_entropy < 0.30:
        band = "high"
        message = (
            "The model is highly confident in this prediction. "
            "Probability mass is concentrated on very few classes."
        )
    elif normalised_entropy < 0.55:
        band = "cautious"
        message = (
            "This prediction should be interpreted cautiously. "
            "The model shows moderate uncertainty across multiple classes."
        )
    else:
        band = "ambiguous"
        message = (
            "High visual ambiguity detected. "
            "The model is uncertain between several categories — "
            "the prediction may not be reliable."
        )
    
    return band, message


def get_uncertainty_summary(full_probs: torch.Tensor) -> dict:
    """
    Full uncertainty analysis in one call.
    
    Returns dict with:
        entropy         : normalised entropy float
        band            : confidence band string
        message         : human-readable band message
        top1_prob       : top prediction probability
        effective_classes: approximate number of classes 
                          carrying meaningful probability
    """
    entropy = compute_entropy(full_probs)
    band, message = get_confidence_band(entropy)
    
    top1_prob = full_probs.max().item()
    
    # Effective number of classes = exp(raw_entropy)
    # Tells us how many classes the model is "spreading" over
    raw_entropy = -torch.sum(
        torch.clamp(full_probs, min=1e-9) * 
        torch.log(torch.clamp(full_probs, min=1e-9))
    ).item()
    effective_classes = round(math.exp(raw_entropy), 1)
    
    return {
        "entropy": entropy,
        "band": band,
        "message": message,
        "top1_prob": round(top1_prob, 4),
        "effective_classes": effective_classes
    }