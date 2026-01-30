import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def evaluate_mdl_acc(model, x_data, y_data):
    y_likelihoods = model.predict(x_data, verbose=0)
    y_predictions = np.argmax(y_likelihoods, axis=1)
    res = y_data - y_predictions
    return sum(x == 0 for x in res) / y_data.shape[0]

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input images from 0-255 to 0-1 range 
# (improves training stability and convergence, and is standard practice in DL;
# Model seems to get stuck without normalization)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the image samples from 28x28 to 1x784
x_train = x_train.reshape(x_train.shape[0], -1) # 60000x784 training data matrix
x_test = x_test.reshape(x_test.shape[0], -1) # 10000x784 test data matrix

# One-hot encode the class ids 
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

# Build & compile the model
model = Sequential()
# Using Input object instructed at https://keras.io/guides/sequential_model/:
model.add(Input(shape=(784,))) # not an actual layer, just predefines the input shape for weights
# see note 1
model.add(Dense(64, activation='sigmoid')) # input layer with X neurons <------------- CHANGE FOR TESTING
model.add(Dense(10, activation='sigmoid')) # output layer with 10 neurons (one per class)
learning_rate = 0.5  # <-------------------------------------------------------------- CHANGE FOR TESTING
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='mean_squared_error')

# Train the model with training data
epochs = 30 # <----------------------------------------------------------------------- CHANGE FOR TESTING
history = model.fit(x_train, y_train_one_hot, epochs=epochs, verbose=1)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Training Loss (MSE) over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Evaluate the model accuracy on training & test data; see note 2
acc_tr = evaluate_mdl_acc(model, x_train, y_train)
print(f"Classification accuracy with training data is {acc_tr * 100:.2f} %")
acc_te = evaluate_mdl_acc(model, x_test, y_test)
print(f"Classification accuracy with test data is {acc_te * 100:.2f} %")

plt.show()


# Note 1: *******************************************************************************
# Other activation methods in the input and output layers give better results; 
# e.g. with epochs = 20; learning_rate = 0.5 and
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'));
# ---> Classification accuracy with training data is 98.50 %;
#      Classification accuracy with test data is 97.16 %
# Changing the loss function to something else, sa. categorical_crossentropy might also help
# https://keras.io/api/losses/

# Note 2: *******************************************************************************
# Can also be done by including the 'accuracy' metric in model.compile() @ln43, e.g.
# 'model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])' and later calling
# model.evaluate(), e.g. 'model.evaluate(model.evaluate(x_train, y_train_one_hot, verbose=0))'
# which would return the training data loss and accuracy directly after compiling
