import re


# Intent definitions with keyword patterns
# Each intent has primary keywords and optional phrase patterns
# Order matters — more specific patterns checked first

INTENT_PATTERNS = {
    "comparison": [
        r"instead of",
        r"rather than",
        r"versus",
        r"\bvs\b",
        r"compare",
        r"difference between",
        r"why not",
        r"over other",
    ],
    "alternative_prediction": [
        r"what else",
        r"other option",
        r"alternative",
        r"second choice",
        r"other possibilit",
        r"also consider",
        r"\belse\b",
        r"other than",
        r"besides",
    ],
    "important_region": [
        r"which part",
        r"what part",
        r"where.*model",
        r"which area",
        r"what area",
        r"which region",
        r"focus",
        r"look.*at",
        r"attend",
        r"location",
        r"\bwhere\b",
        r"spatial",
    ],
    "confidence": [
        r"how confident",
        r"how certain",
        r"how sure",
        r"confidence",
        r"certain",
        r"probability",
        r"how likely",
        r"trust",
        r"reliable",
        r"sure about",
        r"accuracy",
        r"uncertain",
        r"doubt",
    ],
    "why_prediction": [
        r"\bwhy\b",
        r"how did",
        r"what made",
        r"reason",
        r"basis",
        r"explain",
        r"how come",
        r"what caused",
        r"justify",
        r"evidence",
    ],
}

# Fallback response when no intent matches
FALLBACK_INTENT = "unknown"


def interpret_query(query: str) -> tuple:
    """
    Map a natural language query to one of 5 structured intents.

    Uses rule-based regex pattern matching.
    Comparison checked first as it requires two-class framing
    and would otherwise partially match why_prediction.

    Args:
        query: raw user input string

    Returns:
        intent : string — one of the 5 intent keys or 'unknown'
        matched_pattern : string — which pattern triggered (for transparency)
    """
    query_lower = query.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent, pattern

    return FALLBACK_INTENT, None


def get_intent_description(intent: str) -> str:
    """
    Return human-readable description of what each intent does.
    Used in UI to show the user what type of explanation they requested.
    """
    descriptions = {
        "why_prediction": (
            "Explaining why the model made this prediction"
        ),
        "confidence": (
            "Analysing model confidence and uncertainty"
        ),
        "important_region": (
            "Identifying which image regions drove the prediction"
        ),
        "alternative_prediction": (
            "Showing alternative predictions the model considered"
        ),
        "comparison": (
            "Comparing top two predictions and their distinguishing features"
        ),
        "unknown": (
            "Query not recognised — try asking about why, confidence, "
            "regions, alternatives, or comparisons"
        ),
    }
    return descriptions.get(intent, "Unknown intent")