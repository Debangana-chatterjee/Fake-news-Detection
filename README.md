# Fake News Detection

This repository contains a machine learning notebook for detecting fake news using natural language processing (NLP) techniques. The model is trained to classify news articles as either **real** or **fake** based on their content.

---
##  Overview

The goal of this project is to build a binary classification model that can effectively detect misleading or false news articles. The model leverages text processing, vectorization (TF-IDF), and a machine learning classifier to make predictions.


---
##  Features
- Clean and preprocess raw text input using NLP techniques.

- Predict whether the news is real or fake using a pre-trained model.

- Simple and interactive user interface.


---
## Dataset
The dataset used in this project consists of labeled news articles with title, text, and label columns. Typically:

- label = 0: Fake news

- label = 1: Real news

📥 [Download the Dataset](https://drive.google.com/drive/folders/1uDiDqvD7jKLsvoetN64ppb3e7QqpUxYc?usp=drive_link)

---
##  Files
`app.py`: Main Streamlit app that loads the model and provides the UI.

`MNB`: Serialized trained model (via joblib).

`FakeNewsDetection.ipynb`: Google colab notebook containing the data processing, model training, and evaluation.

---
##  How it Works
1. User enters a news snippet in the text box.

2. The input is preprocessed:

   - Lowercased

   - Non-alphabetic characters removed

   - Stopwords removed

   - Lemmatized 

3. The cleaned text is passed to a Multinomial Naive Bayes model.

4. Model predicts and displays whether the news is Real or Fake.
