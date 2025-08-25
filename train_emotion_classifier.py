# File: /sanchi_nlp/emotion_engine/train_emotion_classifier.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from emotion_engine.load_and_preprocess import load_goemotions_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_emotion_classifier():
    print("🔍 Loading dataset...")
    df = load_goemotions_dataset()

    # Features and target
    X = df['text']
    y = df['emotions']

    print("🔢 Binarizing emotions...")
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    print("✂️ Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

    print("📐 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🤖 Training model...")
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train_vec, y_train)

    print("💾 Saving model and vectorizer...")
    joblib.dump(model, "model/emotion_model.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    joblib.dump(mlb, "model/emotion_binarizer.pkl")

    print("✅ Model training complete! Model saved to /model/")

if __name__ == "__main__":
    train_emotion_classifier()
