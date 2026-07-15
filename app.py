import streamlit as st
import pickle
import random
import json

# Set up page styling
st.set_page_config(page_title="AI Assistant Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Intelligent AI Assistant")
st.write("Ask me anything! I am powered by a Scikit-Learn NLP model.")
st.markdown("---")

# 1. Load the trained ML components from the pickle file safely
@st.cache_resource # Keeps the model loaded in memory so the app stays fast
def load_chatbot_brain():
    with open("models/chatbot_model.pkl", "rb") as file:
        artifact = pickle.load(file)
    return artifact["vectorizer"], artifact["model"]

vectorizer, model = load_chatbot_brain()

# 2. Hardcoded fallback responses mapped to the intents we trained
# (In a larger app, you would load these directly from your intents.json data file)
responses_map = {
    "greeting": ["Hello! How can I help you today?", "Hi there!", "Hey! What's up?"],
    "goodbye": ["Goodbye! Have a great day.", "See you later!", "Bye! Take care."],
    "bot_name": ["I am your friendly AI Assistant.", "You can call me Chatbot v1."],
    "thanks": ["You're very welcome!", "Happy to help!", "Anytime!"]
}

# 3. Maintain conversational history using Streamlit's built-in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages if the page reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle Live User Input
if user_input := st.chat_input("Type your message here..."):
    
    # Render user message on screen immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # -- NLP Engine Pipeline processing --
    # Transform user input text using our fitted vectorizer vocabulary
    input_vector = vectorizer.transform([user_input])
    
    # Predict intent tag and compute its raw confidence probability
    predicted_intent = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence_score = max(probabilities)

    # 5. Out-of-Bounds handling (Fallback if confidence falls below 45%)
    if confidence_score < 0.45:
        bot_reply = "I'm sorry, I'm not entirely sure I understood that. Could you rephrase your question?"
    else:
        # Match the intent to one of our mapped phrases randomly
        bot_reply = random.choice(responses_map.get(predicted_intent, ["I'm processing that standard intent!"]))

    # Render bot response on screen with a slight delay fallback
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
