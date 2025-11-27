# src/model_utils.py

import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def create_training_data():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 1, 0, 1, 0, 1])  # Even = 0, Odd = 1
    return X, y

def train_model():
    X, y = create_training_data()
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
