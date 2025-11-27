# src/train.py

from model_utils import train_model, save_model

def main():
    model = train_model()
    save_model(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    main()
