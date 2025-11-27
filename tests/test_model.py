# tests/test_model.py

import os
from src.model_utils import train_model, save_model

def test_model_training():
    model = train_model()
    assert model is not None

def test_model_save():
    model = train_model()
    save_model(model, "model.pkl")
    assert os.path.exists("model.pkl")

def test_model_predict():
    model = train_model()
    pred = model.predict([[2]])
    assert pred is not None
