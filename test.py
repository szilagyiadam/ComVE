# This is a neuron network for testing purposes

# Tensorflow library
import tensorflow as tf

# Other helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Load the whole dataset from keras
fashion_mnist = tf.keras.datasets.fashion_mnist

# Initialize the training and testing datasets
(train_images, train_labels), (test_images, test_labes) = fashion_mnist.load_data()

# 60000 images, each 28x28 pixels
print(train_images.shape)

# Each pixel have a value between 0-255 representing the greyscale value to the corresponding pixel
print(train_images[3,16,16])

# Each label is between 0-9 representing different types of clothing
print(train_labels[:10])

# We create an array for the clothing classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bad', 'Ankle boot']

# Here is what an image looks like from the dataset
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.show()

# Preprocessing the data (converting each pixel value to be between [0, 1]) to help the neuron network in the
# classification process so it doesn't have to deal with big numbers
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer
    tf.keras.layers.Dense(128, activation='relu'),  # hidden layers
    tf.keras.layers.Dense(10, activation='softmax') # output layer
])
# The input layer consist of 784 neurons (28x28). the Flatten method converts the input data (which is a 28x28 array)
# into a one dimension array of 784 elements

# The hidden layer has 128 neurons. Dense layers means that each neuron in the layer has a connection to all neurons in
# the previous layer. We will use the rectify linear unit activation function in this layer, which will plot every
# neurons value between [0, +∞]

# The output layer have 10 neurons, each for every clothing class. The softmax activation function is used on this layer
# to calculate a probability distribution for each class. Because of this, all neruo valu in this layer will be between
# [0, 1], and they all add up to 1

# Defining the optimizer, the loss function and the metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# The optimizer does the gradient descent algorithm, which finds the local minimum point, where the model needs to move
# for better results. The loss function defines how good the actual neuron network is at the moment. Finally, we are
# interested in the accuracy of the prediction, so that's why we chose this metric

# Train the model (epochs means how many iterations of training we want)
model.fit(train_images, train_labels, epochs=3)

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labes, verbose=1)

# Print the model's accuracy
print('Model\'s accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[5])])
plt.figure()
plt.imshow(test_images[5])
plt.colorbar()
plt.show()

