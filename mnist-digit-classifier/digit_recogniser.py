import pandas as pd
import numpy as np

data = pd.read_csv('mnist-digit-classifier/Data/train.csv')
inputs = data.drop('label', axis=1).values.T / 255.0   # (784, 42000)
# Normalise pixel values from 0-255 to 0-1
# Large pixel values cause huge weighted sums which overflow softmax, producing NaN
# Dividing by 255 keeps values small enough for stable computation

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
test_inputs  = inputs[ :, N_TRAIN_SAMPLES: ] # (784, 5000)
Y_train = Y[:, :N_TRAIN_SAMPLES]             # (10, 37000)
Y_test  = Y[:, N_TRAIN_SAMPLES:]             # (10, 5000)


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

def loss_deriv(A,Y):
    return A-Y
    #This is dz for the last layer

def softmax(z):
    z = np.exp(z) 
    z /= np.sum(z, axis=0, keepdims=True)
    # Perform element-wise division, so element in row is divided by the total, creating probabilites
    return z

    # Softmax unlike sigmoid, turns each of the ouputs into probabilites which all sum to 1
    # It exp() all the outputs first as some could have been negative, and so we would've got negative exponenetial
    # Axis=0 means "collapse along rows"


def forward(A0, params, n_layers):
    cached = {'A0' : A0}
    for i in range(1, n_layers+1):
        z = cached[f'Z{i}'] = params[f'W{i}'] @ cached[f'A{i-1}'] + params[f'B{i}']
        cached[f'A{i}'] = softmax(z) if (i == n_layers) else relu(z)
    return cached

def back_prop(params, cached, n_layers, m, Y):
    grads = {}
    prev_dz = None
    for i in range(n_layers, 0, -1):
        if i == n_layers:
            A = cached[f'A{i}']
            dz = loss_deriv(A, Y)
            # Finding dz for last layer is different than finding dz for the layers after 
            # Softmax + cross entropy shortcut: dL/dZ_n = A_n - Y
            # Combines dL/dA_n and dA_n/dZ_n into one clean expression
        else:  
            dz = params[f'W{i+1}'].T @ prev_dz * relu_deriv(cached[f'Z{i}'])
            # Finding dz for the layers after the last layer is the same
            # Z_{i+1} = W_{i+1} @ A_i + B_{i+1}
            # A_i = relu(Z_i)
            # Chain: dL/dZ_i = dL/dZ_{i+1} @ dZ_{i+1}/dA_i * dA_i/dZ_i
            # dZ_{i+1}/dA_i = W_{i+1}.T
            # dA_i/dZ_i = relu'(Z_i)  [1 if Z>0, 0 if Z<=0]

        dw = (dz @ cached[f'A{i-1}'].T) / m
        # Z_i = W_i @ A_{i-1} + B_i
        # Chain: dL/dW_i = dL/dZ_i @ dZ_i/dW_i
        # dZ_i/dW_i = A_{i-1}.T
        # divide by m to average gradient across batch

        db = np.sum(dz, axis=1, keepdims=True) / m
        # Z_i = W_i @ A_{i-1} + B_i
        # Chain: dL/dB_i = dL/dZ_i * dZ_i/dB_i
        # dZ_i/dB_i = 1
        # db originally not a num_neurons x 1 like the biases, so we sum axis 1 which is columns, and keep dimensions so it remains a 2d array
        # sum across batch (axis=1) to collapse to (neurons, 1), divide by m to average

        prev_dz = dz
        # store dL/dZ_i to use next iteration as dL/dZ_{i+1}

        grads[f'DW{i}'] = dw
        grads[f'DB{i}'] = db

    return grads


def update_params(params, grads, lr):
    for key in params: #W1, W2, B1, B2 
        params[key] -= lr * grads[f'D{key}'] 
        # Works because the endings for W1 and DW1


def epoch_stats(A0, params, num_layers, Y):
    cached = forward(A0, params, num_layers)
    loss = calculate_loss(cached[f'A{num_layers}'], Y, 37000)
    outputs = cached[f'A{num_layers}'] # (10, 37000)
    prediction = np.argmax(outputs, axis=0) # Returns index of highest neuron
    expected = np.argmax(Y, axis=0)
    return loss, np.mean(prediction == expected) * 100
    

def train_model(A0, Y, params, batch_size, n_layers, n_samples, epochs, lr=0.01):
    number_of_batches = n_samples//batch_size
    for epoch_num in range(1, epochs+1):
        # Gonna ensure all inputs run in batches of size batch_size
        # Column numbers increase in size batch_size for equal spaced batches
        for col in range(0, n_samples, batch_size): # There gonna be n_samples/batch_size iterations
            batch_inputs = A0[:, col: col+batch_size]
            e_output = Y[:, col: col+batch_size]

            cached = forward(batch_inputs, params, n_layers)
            grads = back_prop(params, cached, n_layers, batch_size, e_output)

            update_params(params, grads, lr)

        # Epoch stats here: Gonna do a full epoch pass at the end and print the stats of that
        epoch_loss, epoch_accuracy = epoch_stats(A0, params, n_layers, Y)
        print(f"Epoch {epoch_num}   |   Loss: {epoch_loss}  |   Accuracy: {epoch_accuracy}%")

    return params


def test_model(test_inputs, params, num_layers, Y_test):
    cached = forward(test_inputs, params, num_layers)
    outputs = cached[f'A{num_layers}']
    predictions = np.argmax(outputs, axis=0)
    expected = np.argmax(Y_test, axis=0)
    accuracy = np.mean(predictions == expected) * 100
    print(f"Test Accuracy: {accuracy}%")

if __name__ == "__main__":
    N_OF_EPOCHS = 15
    BATCH_SIZE = 64
    # 64 doesn't divide evenly into 37000 but numpy slicing handles the smaller last batch automatically

    initial_params, num_layers = initialise_hidden_layers(128, 64)
    trained_params = train_model(train_inputs, Y_train, initial_params, BATCH_SIZE, num_layers, N_TRAIN_SAMPLES, N_OF_EPOCHS, lr=0.1)


    # saves my params (weights and biases)
    np.save('params.npy', trained_params)


    # Final evaluation on 5000 samples the network has never seen to check the network generalises beyond training samples
    test_model(test_inputs, trained_params, num_layers, Y_test)