import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Current folder:", os.getcwd())
print("Dataset exists:", os.path.exists("dataset"))
print("Dataset files:", os.listdir("dataset"))

DATASET_PATH = "dataset"

X = []
y = []

print("Loading dataset...")

for file in os.listdir(DATASET_PATH):
    if file.endswith(".npy"):
        label = file.split("_")[0]
        path = os.path.join(DATASET_PATH, file)

        try:
            data = np.load(path)

            # проверяем форму
            if data.shape != (30, 126):
                print("Skipping invalid shape:", file, data.shape)
                continue

            data = data.flatten()

            X.append(data)
            y.append(label)

        except Exception as e:
            print("Skipping broken file:", file)

X = np.array(X)
y = np.array(y)

print("Valid samples:", len(X))

model = RandomForestClassifier(n_estimators=300)

print("Training model...")
model.fit(X, y)

joblib.dump(model, "gesture_model.pkl")

print("Model saved: gesture_model.pkl")
