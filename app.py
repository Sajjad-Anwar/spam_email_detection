# app.py

import streamlit as st
import pandas as pd
import pickle

# Set the title and description of the app
st.title("Spam vs Ham Email Detection App")
st.write("""
### Detect whether an email is Spam or Ham.
Enter the email text below, and the trained model will predict whether it is spam or not.
""")

# Input field for the email text
email_text = st.text_area("Enter the Email Text", "")

# Load the trained model and TF-IDF vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, tfidf = load_model()

# Predict when the "Detect" button is clicked
if st.button("Detect"):
    if not email_text.strip():
        st.warning("Please enter the email text.")
    else:
        # Transform the input text using the TF-IDF vectorizer
        text_features = tfidf.transform([email_text])
        
        # Make the prediction
        prediction = model.predict(text_features)[0]

        # Display the result
        if prediction == 1:
            st.error("⚠️ This is a Spam email!")
        else:
            st.success("✅ This is a Ham (not spam) email.")

# Footer message
st.write("Built with ❤️ using Streamlit")
