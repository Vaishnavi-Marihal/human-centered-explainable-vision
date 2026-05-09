from classifier import load_image, classify
from query_interpreter import interpret_query
from explanation_engine import ExplanationEngine

image_path = "test.jpg"
pil_image, tensor = load_image(image_path)
labels, probs, indices, full_probs = classify(tensor)
engine = ExplanationEngine()

test_queries = [
    "Why did you classify this?",
    "How confident are you?",
    "Which part of the image mattered most?",
    "What else could this be?",
    "Why this instead of the second option?",
]

for query in test_queries:
    intent, _ = interpret_query(query)
    result = engine.explain(
        intent, pil_image, tensor,
        labels, probs, indices, full_probs
    )
    print(f"Query  : {query}")
    print(f"Intent : {intent}")
    print(f"Text   :\n{result['text']}")
    print(f"Visual : {'yes' if result['visual'] is not None else 'none'}")
    print("-" * 60)