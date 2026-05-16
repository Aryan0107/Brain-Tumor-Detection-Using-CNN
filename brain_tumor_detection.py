import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset path
DATASET_PATH = "dataset"

# Image size
IMG_SIZE = 150

images = []
labels = []

# Loading images from dataset folders
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(category)
        except:
            pass

# Converting to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalizing images
images = images / 255.0

# Encoding labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

# Building CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

# Compiling model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluating model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

# Prediction on one sample image
index = 0
sample_image = x_test[index]

prediction = model.predict(sample_image.reshape(1, IMG_SIZE, IMG_SIZE, 3))
predicted_class = np.argmax(prediction)

print("Predicted Class:", label_encoder.inverse_transform([predicted_class])[0])
print("Actual Class:", label_encoder.inverse_transform([y_test[index]])[0])

# Displaying sample image
plt.imshow(sample_image)
plt.title(
    "Predicted: " + label_encoder.inverse_transform([predicted_class])[0] +
    " | Actual: " + label_encoder.inverse_transform([y_test[index]])[0]
)
plt.axis("off")
plt.show()

# Saving model
model.save("brain_tumor_cnn_model.h5")
