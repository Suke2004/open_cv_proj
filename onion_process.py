import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

base_folder = r"C:\Users\royma\OneDrive\Desktop\FLIPCART GRID\PROJECTSTART" # give the path to csv and images
csv_file = os.path.join(base_folder, "onion.csv")

data = pd.read_csv(csv_file)

images = []
freshness = []
lifespan = []

for _, row in data.iterrows():
    if row["Category"] == "ONION":  
        img_path = os.path.join(base_folder, row["Category"], row["Image Name"])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0  
            images.append(img)
            freshness.append(row["Freshness"])
            lifespan.append(row["Expected Lifespan"])

images = np.array(images, dtype="float32")
freshness = np.array(freshness)
lifespan = np.array(lifespan)

X_train, X_test, y_train_fresh, y_test_fresh = train_test_split(images, freshness, test_size=0.2, random_state=42)
_, _, y_train_life, y_test_life = train_test_split(images, lifespan, test_size=0.2, random_state=42)

np.savez("onion_preprocessed_data.npz",
         X_train=X_train, X_test=X_test,
         y_train_fresh=y_train_fresh, y_test_fresh=y_test_fresh,
         y_train_life=y_train_life, y_test_life=y_test_life)

print("Onion data preprocessing completed. Saved to 'onion_preprocessed_data.npz'.")
