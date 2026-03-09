import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup page config
st.set_page_config(page_title="Spam Detector", page_icon="📧")

# Download NLTK resources
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

load_nltk()

ps = PorterStemmer()

# Text preprocessing for ml models
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Load trained model and vectorizer
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Machine Learning model (Naive Bayes) to classify messages as Spam or Not Spam."
)

# Main UI
st.title("📧 Email / SMS Spam Classifier")
st.markdown("---")

input_sms = st.text_area(
    "Enter the message below:",
    placeholder="Type your message here...",
    height=150
)

# Classification button
if st.button('Classify Message'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        with st.spinner("Analyzing message..."):

            # 1. Preprocess text
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]

            # 4. Probability
            probability = model.predict_proba(vector_input)[0]
            confidence = max(probability) * 100
            spam_prob = probability[1] * 100
            ham_prob = probability[0] * 100

            # 5. Display result
            st.subheader("Result")

            if result == 1:
                st.error("🚨 Spam ")
            else:
                st.success("✅ Not Spam (Ham)")

            st.markdown("---")

            st.write(f"Confidence Level: **{confidence:.2f}%**")
            st.write(f"Spam Probability: **{spam_prob:.2f}%**")
            st.write(f"Ham Probability: **{ham_prob:.2f}%**")

# Footer
st.markdown("---")
st.caption("Machine Learning Project - SMS Spam Detection")