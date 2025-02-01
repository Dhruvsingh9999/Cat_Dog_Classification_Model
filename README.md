# Cat_Dog_Classification_Model / Image Classification Model

This repository contains an image classification model built using TensorFlow and Keras. The model is trained to classify images into different categories using a Convolutional Neural Network (CNN).

## Features
- Loads an image dataset from a directory
- Normalizes images for better training
- Implements a CNN model for classification
- Trains and evaluates the model
- Visualizes training accuracy and loss

## Installation
To run this project, install the required dependencies:
```bash
pip install tensorflow matplotlib
```

## Dataset
The dataset should be structured as follows:
```
/dataset_directory/
    /train/
        /class_1/
        /class_2/
    /test/
        /class_1/
        /class_2/
```
Replace `class_1` and `class_2` with your actual class names.

## Steps to Run
### 1. Load Dataset
```python
train_ds = keras.utils.image_dataset_from_directory(
    'path_to_train_directory', labels='inferred', label_mode='int',
    batch_size=32, image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    'path_to_test_directory', labels='inferred', label_mode='int',
    batch_size=32, image_size=(256, 256)
)
```
### 2. Normalize Images
```python
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

### 3. Build CNN Model
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 4. Compile Model
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 5. Train Model
```python
history = model.fit(train_ds, validation_data=validation_ds, epochs=10)
```

### 6. Evaluate Model
```python
model.evaluate(validation_ds)
```

### 7. Visualize Training Results
```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Conclusion
This repository provides an easy-to-use deep learning model for image classification. Modify the dataset path and class names to suit your needs. Happy coding!
