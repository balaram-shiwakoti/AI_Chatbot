import time

import streamlit as st
import pickle
import random
import json
import csv
from datetime import datetime
import os   # To check if the CSV exists

# Set up page styling
st.set_page_config(page_title="AI Assistant Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Intelligent AI Assistant")
st.write("Ask me anything! I am powered by a Scikit-Learn NLP model.")
st.markdown("---")


def log_feedback(user_input, predicted_intent, confidence, feedback_type, corrected_intent=""):
    """
    Appends feedback to a CSV file for later analysis and retraining.
    feedback_type: 'positive' or 'negative'
    """
    file_exists = os.path.isfile("feedback_log.csv")
    
    with open("feedback_log.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "user_input", "predicted_intent", "confidence", "feedback", "corrected_intent"])
        
        # Write the feedback row
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_input,
            predicted_intent,
            f"{confidence:.3f}",
            feedback_type,
            corrected_intent
        ])
# 1. Load the trained ML components from the pickle file safely
@st.cache_resource # Keeps the model loaded in memory so the app stays fast
def load_chatbot_brain():
    with open("models/chatbot_model.pkl", "rb") as file:
        artifact = pickle.load(file)
    return artifact["vectorizer"], artifact["model"]

vectorizer, model = load_chatbot_brain()

# 2. Dynamically load responses directly from our data file
with open("data/intents.json", "r") as file:
    intents_data = json.load(file)

# Build a mapping dictionary: {"greeting": ["Hello!", "Hi!"], "goodbye": [...]}
responses_map = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

# 3. Maintain conversational history using Streamlit's built-in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages if the page reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle Live User Input
# -- Inside your user_input handling block --
if user_input := st.chat_input("Type your message here..."):
    
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # --- NLP Prediction ---
    input_vector = vectorizer.transform([user_input])
    predicted_intent = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence_score = max(probabilities)

    # --- Generate Bot Reply ---
    if confidence_score < 0.45:
        bot_reply = "I'm sorry, I'm not entirely sure I understood that. Could you rephrase your question?"
    else:
        bot_reply = random.choice(responses_map.get(predicted_intent, ["I'm processing that standard intent!"]))

    # --- Store the CURRENT prediction in session state for the feedback buttons ---
    st.session_state.last_user_input = user_input
    st.session_state.last_predicted_intent = predicted_intent
    st.session_state.last_confidence = confidence_score

    # Display Bot Message
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        
        # --- 🆕 FEEDBACK BUTTONS (Appear right under the bot reply) ---
        # Use a unique key based on the timestamp or message length to avoid button conflicts
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍", key=f"pos_{len(st.session_state.messages)}"):
                log_feedback(
                    st.session_state.last_user_input,
                    st.session_state.last_predicted_intent,
                    st.session_state.last_confidence,
                    "positive"
                )
                st.success("Thanks for the positive feedback! ✅", icon="🙌")
        
        with col2:
            if st.button("👎", key=f"neg_{len(st.session_state.messages)}"):
                # Store the fact that we need to ask for correction
                st.session_state.show_correction = True
        
        # --- If user clicked thumbs down, show a correction input ---
        if st.session_state.get("show_correction", False):
            corrected_intent = st.text_input("What should I have said? (e.g., 'joke', 'greeting')", 
                                             key=f"corr_{len(st.session_state.messages)}")
            if st.button("Submit Correction", key=f"sub_{len(st.session_state.messages)}"):
                if corrected_intent:
                    log_feedback(
                        st.session_state.last_user_input,
                        st.session_state.last_predicted_intent,
                        st.session_state.last_confidence,
                        "negative",
                        corrected_intent
                    )
                    st.success("Thanks! I'll learn from this. 📚")
                    st.session_state.show_correction = False  # Hide the input box
                    st.rerun()  # Refresh to clear the input box
                else:
                    st.warning("Please enter what I should have said.")

    # Append to session history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    
        
    # 5. Out-of-Bounds handling (Fallback if confidence falls below 15%)
    if confidence_score < 0.45:
        bot_reply = "I'm sorry, I'm not entirely sure I understood that. Could you rephrase your question?"
    elif predicted_intent == "time":
        from datetime import datetime
        now = datetime.now().strftime("%I:%M %p")
        bot_reply = f"⏰ The current time is {now}."
    else:
        # Match the intent to one of our mapped phrases randomly
        bot_reply = random.choice(responses_map.get(predicted_intent, ["I'm processing that standard intent!"]))

    # Render bot response on screen with a slight delay fallback
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ *Typing...*")
        time.sleep(0.8)  # Simulate thinking
        placeholder.markdown(bot_reply)  # Replace with actual reply
        
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
