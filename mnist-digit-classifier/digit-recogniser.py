import pandas as pd
import numpy as np

data = pd.read_csv('mnist-digit-classifier/Data/train.csv')
inputs = data.drop('label', axis=1).values.T   # (784, 42000)
labels = data['label'].values # (42000,)
# Gives 2D array of the correct digit that was written e.g. [1, 6, 8, 2...] 
# *First element means Image 1 (the pixels in column 1) was a 1 etc

Y = np.zeros((10, 42000))
# 10 rows (one per digit 0-9), 42000 columns (one per image)
# Each column will be all zeros except a single 1 at the row of the correct digit

Y[labels , np.arange(42000)] = 1
# labels[i] gives the correct digit for image i
# np.arange(42000) = [0, 1, 2, ... 41999] - column indices for each image
# Together they set Y[correct_digit, image_index] = 1 for every image at once
# e.g. labels = [1, 6, 8, 2, ...] sets Y[1,0]=1, Y[6,1]=1, Y[8,2]=1, Y[2,3]=1

# Kaggle test data has no labels so we split our training data instead
# This gives us labelled data for both training and evaluating accuracy

N_TRAIN_SAMPLES = 37000
# 37000 training
N_TEST_SAMPLES = 5000
# 5000 testing

train_inputs = inputs[ :, :N_TRAIN_SAMPLES] # (784, 37000)
train_labels = labels[:N_TRAIN_SAMPLES] # (37000, ) 
test_inputs  = inputs[ :, N_TRAIN_SAMPLES: ] # (784, 5000)
test_labels  = labels[N_TRAIN_SAMPLES:] # (5000, )


# Input layer fixed at 784 neurons (28x28 pixels), output fixed at 10 neurons (digits 0-9)
# Number and size of hidden layers can vary
def initialise_hidden_layers(*hidden_layers):
    layers = [784] + list(hidden_layers) + [10]
    #e.g. [784, 512, 600, 512, 10]
    n_layers = len(layers)
    params = {}
    for i in range(1, n_layers):
        weight_scale = np.sqrt(2/layers[i-1])
        params[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * weight_scale
        params[f'B{i}'] = np.zeros((layers[i], 1))
    return params, (n_layers - 1)
    #The input layer doesn't technically cout as a layer, we subtract 1 to remove counting input layer


def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    return (z > 0).astype(float)

def calculate_loss(A, Y, m):
    return -np.sum(Y * np.log(A)) / m
    # L=−1/m x ​∑Y⊙log(A)
    # ⊙ means element-wise multiplication

def softmax(z):
    # Softmax unlike sigmoid, turns each of the ouputs into probabilites which all sum to 1
    # It exp() all the outputs first as some could have been negative, and so we would've got negative exponenetial
    # Axis=0 means "collapse along rows"

    pass



def forward(A0, params, n_layers):
    cached = {'A0' : A0}
    for i in range(1, n_layers+1):
        z = cached[f'Z{i}'] = params[f'W{i}'] @ cached[f'A{i-1}'] + params[f'B{i}']
        cached[f'A{i}'] = softmax(z) if (i == n_layers) else relu(z)