
# Importing
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import cv2 as cv
import numpy as np

"""
Creates a convolutional neural network to predict which of three classes (entered by user) an image belongs to.
"""
predictCounter = 1

class Model2:
    # Used for each new image captured for prediction
    
    """
    Creates the model and trains it
    """
    def __init__(self, classNames, dataDir):
        batchSize = 32
        self.width = 150
        self.height = 150

        # creates the training data set
        train = tf.keras.utils.image_dataset_from_directory(dataDir, label_mode='categorical', image_size=(150,150), batch_size=batchSize)
        self.classNames = train.class_names

        # checking the shape of the images and labels
        for imgBatch, labelsBatch in train:
            print(imgBatch.shape)
            print(labelsBatch.shape)
            break
        
        # Improving quality of data
        AUTOTUNE = tf.data.AUTOTUNE
        train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        num_classes = len(classNames)

        # Building the model
        self.model = Sequential([
        layers.Rescaling(1./255, input_shape=(self.height, self.width, 3)),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])

        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        self.model.summary()

        epochs=20
        self.model.fit(train, epochs=epochs)

        # Confirmation
        print("Model successfully trained")

    """
    Predicts which class a new image belongs to
    """
    def makePrediction(self, image):
        global predictCounter

        # Setting up the image path and creating the path in the predictions folder

        imgPath = f'predictions/frame{predictCounter}.jpg'
        cv.imwrite(imgPath, image)

        # Preprocessing the image
        image = cv.resize(image, (150, 150))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        print(f"Image: {image}")

        print(f"Image Shape: {image.shape}")

        print(f"Min Pixel Value: {np.min(image)}, Max Pixel Value: {np.max(image)}")

        predictCounter += 1
        
        # Loading the image with keras
        newImg = tf.keras.utils.load_img(imgPath, target_size=(self.height, self.width))

        # Preparing the image
        img_array = tf.keras.utils.img_to_array(newImg)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # Making the prediction
        predictions = self.model.predict(img_array)

        # Determines probability of the image belonging to each class
        score = tf.nn.softmax(predictions[0])

        # Class with highest probability is selected
        print(f'Predictions = {predictions}\nScore = {score}')

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.classNames[np.argmax(score)], 100 * np.max(score))
        )
