import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load the data from your JSON file
with open("data/intents.json", "r") as file:
    data = json.load(file)

# 2. Automatically parse patterns and tags out of the JSON structure
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)      # The question/phrase
        labels.append(intent["tag"]) # The intent label

# Convert to a Pandas DataFrame instantly
df = pd.DataFrame({"text": texts, "intent": labels})
print(f"📊 Loaded {len(df)} training phrases across {len(data['intents'])} different intents.")

# 3. Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df["text"])
y_train = df["intent"]

# 4. Train the Machine Learning Classifier
model = LogisticRegression(C=10.0, max_iter=1000) # Added scale parameters for larger datasets
model.fit(X_train, y_train)

# 5. Package both components together
model_artifact = {
    "vectorizer": vectorizer,
    "model": model
}

# 6. Overwrite the old model file with the newly expanded one
with open("models/chatbot_model.pkl", "wb") as file:
    pickle.dump(model_artifact, file)

print("🎉 New massive brain successfully trained and saved into 'models/chatbot_model.pkl'!")
