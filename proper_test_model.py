import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path):
    img = cv2.imread(img_path)  
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded.")
    img = cv2.resize(img, (128, 128))  
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img


def classify_image(model, img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    category = np.argmax(pred)
    categories = ["BROCOLI", "ONION", "PAPAYA"]
    return categories[category]


def load_regression_model(category):
    if category == "BROCOLI":
        return load_model("broccoli_final_regression_model.keras")
    elif category == "ONION":
        return load_model("onion_final_regression_model.keras")
    elif category == "PAPAYA":
        return load_model("papaya_final_regression_model.keras")
    else:
        raise ValueError("Unknown category. Choose from 'BROCOLI', 'ONION', or 'PAPAYA'.")


def predict_freshness_and_lifespan(model, img_path, regression_model, category):
    img = preprocess_image(img_path)
    pred = regression_model.predict(img)
    freshness = pred[0][0]  
    lifespan = pred[0][1]  

    if category == "BROCOLI":
        freshness *= 3
        lifespan *= 2.5
    elif category == "ONION":
        freshness *= 4.5
        lifespan *= 4
    elif category == "PAPAYA":
        freshness *= 3
        lifespan *= 3

    return freshness, lifespan


def main(img_path):
    classification_model = load_model("final_classification_model.keras")

    category = classify_image(classification_model, img_path)
    print(f"Predicted category: {category}")

    regression_model = load_regression_model(category)

    freshness, lifespan = predict_freshness_and_lifespan(regression_model, img_path, regression_model, category)

    print(f"Freshness: {freshness} days")
    print(f"Expected lifespan: {lifespan} days")


if __name__ == "__main__":
    img_path = input("Enter the full path of the image: ").strip()

    main(img_path)
