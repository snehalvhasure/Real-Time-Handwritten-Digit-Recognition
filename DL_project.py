import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

# ==========================
# Load MNIST Dataset
# ==========================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ==========================
# Scaling
# ==========================
X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

# ==========================
# Model Building
# ==========================
seq_model = Sequential()

seq_model.add(Input(shape=(28, 28)))
seq_model.add(Flatten())

seq_model.add(Dense(128, activation='relu'))
seq_model.add(Dense(256, activation='relu'))
seq_model.add(Dense(512, activation='relu'))
seq_model.add(Dense(256, activation='relu'))
seq_model.add(Dense(128, activation='relu'))

seq_model.add(Dense(10, activation='softmax'))

seq_model.summary()

# ==========================
# Compile
# ==========================
seq_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================
seq_model.fit(
    X_train_scale,
    y_train,
    epochs=10
)

# ==========================
# Evaluate
# ==========================
seq_model.evaluate(
    X_test_scale,
    y_test
)

# WEBCAM / CCTV DIGIT RECOGNITION

print("\nStarting Webcam Digit Recognition...")
print("Press 'q' to quit.\n")

cap = cv2.VideoCapture(0)

# For CCTV stream use:
# cap = cv2.VideoCapture("YOUR_CCTV_STREAM_URL")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize image to MNIST size
    digit_img = cv2.resize(gray, (28, 28))

    # Invert colors
    # MNIST digits are white on black background
    digit_img = 255 - digit_img

    # Normalize
    digit_img = digit_img / 255.0

    # Reshape for prediction
    digit_img = digit_img.reshape(1, 28, 28)

    # Predict
    prediction = seq_model.predict(
        digit_img,
        verbose=0
    )

    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Print prediction in terminal
    print(
        f"Predicted Digit: {digit} | Confidence: {confidence:.4f}"
    )

    # Display prediction on frame
    cv2.putText(
        frame,
        f"Digit: {digit} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "MNIST Digit Recognition - Webcam",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================================
# RELEASE RESOURCES
# =====================================================

cap.release()
cv2.destroyAllWindows()

print("\nProgram Finished Successfully.")