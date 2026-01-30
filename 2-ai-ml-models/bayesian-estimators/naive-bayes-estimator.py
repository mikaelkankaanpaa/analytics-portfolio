import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import sys


def load_mnist_data(name):
    if name == 'fashion':
        print("Using the MNIST Fashion dataset")
        return fashion_mnist.load_data()
    elif name == 'original':
        print("Using the original MNIST dataset")
        return mnist.load_data()
    else:
        sys.exit("Invalid dataset name. Use 'mnist' or 'fashion'.")


def get_mean_and_var(x_train, y_train, num_of_classes=10):
    means = []
    variances = []
    for c in range(num_of_classes):
        samples = x_train[y_train == c]
        # Mean and variance for each sample (across axis 0, ie. rows of the data matrix) 
        means.append(np.mean(samples, axis=0))
        variances.append(np.var(samples, axis=0))

    return np.array(means), np.array(variances)


def log_likelihood(x, mean, var):
    return -0.5 * np.sum(((x - mean) ** 2) / var + np.log(var) + np.log(2 * np.pi), axis=1)


def classify(x_test, means, variances):
    num_of_classes = means.shape[0]
    log_likelihoods = np.zeros((x_test.shape[0], num_of_classes))
    
    for c in range(num_of_classes):
        # log likelihood for each class
        log_likelihoods[:, c] = log_likelihood(x_test, means[c], variances[c])
    
    # Classify each sample by the class with the highest log likelihood
    return np.argmax(log_likelihoods, axis=1)


def class_acc(pred,gt):
    res = gt-pred
    acc = sum(x == 0 for x in res) / gt.shape[0]
    return acc


def main():
    # Load the correct MNIST dataset & convert the image samples from 28x28 to 1x784
    (x_train, y_train), (x_test, y_test) = load_mnist_data(sys.argv[-1])
    x_train = x_train.reshape(x_train.shape[0], -1) # 60000x784 training data matrix
    x_test = x_test.reshape(x_test.shape[0], -1) # 10000x784 test data matrix

    # Add zero-mean Gaussian noise to the training data
    noise_std = 25.0 # <-------------------------------------------------------- CHANGE FOR TESTING
    print(f"Classifying with noise_std of {noise_std}")
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=noise_std, size=x_train.shape)

    # Compute mean and variance for each class using noisy data
    means, variances = get_mean_and_var(x_train_noisy, y_train)

    # Classify test samples
    y_pred = classify(x_test, means, variances)

    # Calculate & print the classification accuracy
    acc = class_acc(y_pred, y_test)
    print(f"Classification accuracy is {acc * 100:.2f} %")

if __name__ == "__main__":
    main()