import os
from src.model_utils import train_model, MODEL_PATH

def test_training():
    model = train_model()
    assert model is not None

def test_model_file_saved():
    train_model()
    assert os.path.exists(MODEL_PATH)

def test_predict_method():
    model = train_model()
    pred = model.predict([[4]])
    assert pred is not None
