import csv
import json
from collections import Counter

def review_feedback_and_suggest():
    """Reads the CSV, finds negative feedback, and suggests new patterns."""
    
    # 1. Load current intents
    with open("data/intents.json", "r") as f:
        intents_data = json.load(f)
    
    # 2. Read feedback
    try:
        with open("feedback_log.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print("No feedback log found yet.")
        return
    
    # 3. Filter for negative feedback with a corrected intent
    corrections = []
    for row in rows:
        if row["feedback"] == "negative" and row["corrected_intent"]:
            corrections.append({
                "user_input": row["user_input"],
                "corrected_intent": row["corrected_intent"]
            })
    
    if not corrections:
        print("No corrections to process. Great job! 🎉")
        return
    
    # 4. Group corrections by intent
    intent_suggestions = {}
    for item in corrections:
        intent = item["corrected_intent"]
        if intent not in intent_suggestions:
            intent_suggestions[intent] = []
        intent_suggestions[intent].append(item["user_input"])
    
    # 5. Print suggestions for manual addition
    print("📝 Suggested new patterns to add to your intents.json:")
    print("-" * 50)
    for intent, patterns in intent_suggestions.items():
        # Remove duplicates
        unique_patterns = list(set(patterns))
        print(f"\nIntent: '{intent}'")
        print(f"Add these patterns: {unique_patterns}")
        
        # Check if this intent exists in the current JSON
        existing = next((item for item in intents_data["intents"] if item["tag"] == intent), None)
        if existing:
            print(f"Current patterns: {existing['patterns']}")
            print(f"🟢 Merge these new patterns in.")
        else:
            print(f"🟡 This is a NEW intent! You'll need to create a new entry.")
    
    print("\n" + "-" * 50)
    print("✅ Copy these suggestions into your intents.json, then retrain!")

if __name__ == "__main__":
    review_feedback_and_suggest()