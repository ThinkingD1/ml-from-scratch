import numpy as np
np.set_printoptions(precision=3, suppress=True) # Only a printing thing


# Input layer and Output layer are fixed at 2 neurons, number of hidden layers can vary
def initialise_hidden_layers(*hidden_layers):
    layers = [2] + list(hidden_layers) + [2]
    #e.g. [2, 3, 4, 2]
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
    return np.sum(np.square(A-Y)) / m

def loss_deriv(A, Y):
    return 2 * (A-Y)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deriv(A):
    return A * (1-A)

# n represents the number of layers

def forward(params, A0, n):
    cached = {'A0' : A0}
    for i in range(1, n+1):
        z = cached[f'Z{i}'] = ( params[f'W{i}'] @ cached[f'A{i-1}'] ) + params[f'B{i}']
        cached[f'A{i}'] = relu(z) if (i < n) else sigmoid(z)
        # Last layer is uses sigmoid as its the outputs, hidden layers use ReLU
    return cached


def backprop(params, cached, n, m, Y):
    grads = {}
    prev_dz = None
    for i in range(n, 0, -1):
        if i == n:
            A = cached[f'A{i}']
            dz = loss_deriv(A, Y) * sigmoid_deriv(A)
            # Finding dz for last layer is different than finding dz for the layers after 
            # L = (A_n - Y)^2
            # A_n = sigmoid(Z_n)
            # Chain: dL/dZ_n = dL/dA_n * dA_n/dZ_n
            # dL/dA_n = 2(A_n - Y) [Loss derivative]
            # dA_n/dZ_n = A_n(1-A_n)  [sigmoid derivative]
        else:  
            dz = params[f'W{i+1}'].T @ prev_dz * relu_deriv(cached[f'Z{i}'])
            # Finding dz for the layers after the last layer is the same
            # Z_{i+1} = W_{i+1} @ A_i + B_{i+1}
            # A_i = relu(Z_i)
            # Chain: dL/dZ_i = dL/dZ_{i+1} * dZ_{i+1}/dA_i * dA_i/dZ_i
            # dZ_{i+1}/dA_i = W_{i+1}.T
            # dA_i/dZ_i = relu'(Z_i)  [1 if Z>0, 0 if Z<=0]

        dw = (dz @ cached[f'A{i-1}'].T) / m
        # Z_i = W_i @ A_{i-1} + B_i
        # Chain: dL/dW_i = dL/dZ_i * dZ_i/dW_i
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



def batch_stats(batch_num, A_last, Y, batch_size, num_of_batches):
    batch_loss = calculate_loss(A_last, Y, batch_size)

    # For each column (sample), find which row (neuron) has the highest activation
    # index 0 = predicts XOR=1, index 1 = predicts XOR=0
    #returns 1D Array of the indexes of the row which was largest e.g. [0, 1, 1, 0]
    predictions = np.argmax(A_last, axis=0)
    
    # For each column (sample), find which row (neuron) should have been highest using the expected outcomes
    # Also returns 1D array e.g. [0, 1, 0] *the first element means in the first col, row one had greater value etc
    labels = np.argmax(Y, axis=0)
    
    # predictions == labels gives [True, False, True ...]
    # True = 1, False = 0, so mean gives proportion correct
    # multiply by 100 to convert to percentage
    accuracy = np.mean(predictions == labels) * 100


    print(f"Batch {batch_num}/{num_of_batches}  |   Loss: {batch_loss}  |   Accuracy: {accuracy}%")

    return accuracy




def epoch_stats(epoch_num, total_epochs, accuracy_sum, num_of_batches):
    print(f"\n\nEpoch {epoch_num}/{total_epochs}  |   Accuracy: {accuracy_sum/num_of_batches}%")
    print("\n------------------------------------------------------------------")




def train_model(A0, Y, params, batch_size, n_layers, n_samples, epochs, lr=0.01):
    number_of_batches = n_samples//batch_size
    for epoch_num in range(1, epochs+1):
        accuracy_sum = 0
        # Gonna ensure all inputs run in batches of size batch_size
        # Column numbers increase in size batch_size for equal spaced batches
        for col in range(0, n_samples, batch_size): # There gonna be n_samples/batch_size iterations
            batch_inputs = A0[:, col: col+batch_size]
            e_output = Y[:, col: col+batch_size]

            cached = forward(params, batch_inputs, n_layers)
            grads = backprop(params, cached, n_layers, batch_size, e_output)

            batch_num = (col//batch_size) + 1
            accuracy_sum += batch_stats(batch_num, cached[f'A{n_layers}'], e_output, batch_size, number_of_batches)

            update_params(params, grads, lr)

        epoch_stats(epoch_num, epochs, accuracy_sum, number_of_batches)
    return params



def predict(x1, x2, params):
    n_layers = len(params) // 2
    inputs = np.array([
        [x1],
        [x2]
    ])
    cached = forward(params, inputs, n_layers)
    output = cached[f'A{n_layers}']
    #Returns a 2x1, with the outputs
    prediction = np.argmax(output, axis = 0)[0]
    #Top Row is XOR=1, bottom row is XOR=0
    #Return 0 means top row largest, return 1 means bottom row largest

    print("XOR = 1" if prediction == 0 else "XOR = 0")


N_OF_EPOCHS = 150
BATCH_SIZE = 3
N_OF_SAMPLES = 12

A0 = np.array([
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
])

Y = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],   # is XOR=1?
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]    # is XOR=0?
])  

initial_params, num_layers = initialise_hidden_layers(8)
trained_params = train_model(A0, Y, initial_params, BATCH_SIZE, num_layers, N_OF_SAMPLES, N_OF_EPOCHS, 0.01)


predict(1, 1, trained_params)


