import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

data = np.load("preprocessed_data.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train_cat = data["y_train_cat"]
y_test_cat = data["y_test_cat"]

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  #
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

checkpoint = ModelCheckpoint("best_classification_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint]
)

model.save("final_classification_model.keras")
print("Classification training completed.")
