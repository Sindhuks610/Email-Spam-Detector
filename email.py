
import streamlit as st
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("email_spam_dataset.csv")


def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text.strip()


data['cleaned'] = data['message'].apply(clean_text)


data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['cleaned'])
y = data['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


st.title("📧 Email Spam Detector.")
st.write("Enter an email message to check if it's spam or not:")

user_input = st.text_area("Type your email message here:")

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        confidence = model.predict_proba(vectorized_input)[0][prediction] * 100

        if prediction == 1:
            st.error(f"⚠️ Spam detected! (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ Not spam. (Confidence: {confidence:.2f}%)")
