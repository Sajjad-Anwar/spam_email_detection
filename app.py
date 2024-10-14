import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open('spam_detection_model.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

# Streamlit app title
st.title("Spam Email Detection")

# Input from the user
input_email = st.text_area("Enter the email text:")

if st.button("Predict"):
    if input_email.strip() != "":
        # Transform the input text using the vectorizer
        input_data = vectorizer.transform([input_email])

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.error("This email is SPAM!")
        else:
            st.success("This email is NOT SPAM!")
    else:
        st.warning("Please enter some email text.")

