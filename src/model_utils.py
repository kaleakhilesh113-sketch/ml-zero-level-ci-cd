import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

MODEL_PATH = "model.pkl"

def prepare_data():
    X = np.arange(0, 100).reshape(-1, 1)
    y = np.array([1 if i % 2 == 0 else 0 for i in X])
    return X, y

def train_model():
    X, y = prepare_data()
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model
