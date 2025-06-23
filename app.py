#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📰 Fake News Classifier")

text_input = st.text_area("Enter a news headline or article text:")

if st.button("Classify"):
    if text_input:
        cleaned_input = [' '.join(text_input.lower().split())]
        vectorized_input = vectorizer.transform(cleaned_input)
        prediction = model.predict(vectorized_input)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader(label)
    else:
        st.warning("Please enter some text.")


# In[ ]:





# In[ ]:




