from query_interpreter import interpret_query, get_intent_description

test_queries = [
    "Why did you classify this as a dog?",
    "How confident are you?",
    "Which part of the image did you focus on?",
    "What else could this be?",
    "Why golden retriever instead of a labrador?",
    "What is the weather like today?",
    "Can you explain your reasoning?",
    "Where did the model look?",
    "How certain is this prediction?",
    "Compare the top two predictions",
]

print(f"{'Query':<45} {'Intent':<25} {'Pattern'}")
print("-" * 90)

for query in test_queries:
    intent, pattern = interpret_query(query)
    print(f"{query:<45} {intent:<25} {pattern}")

print()
print("Intent descriptions:")
for intent in [
    "why_prediction",
    "confidence", 
    "important_region",
    "alternative_prediction",
    "comparison",
    "unknown"
]:
    print(f"  {intent}: {get_intent_description(intent)}")