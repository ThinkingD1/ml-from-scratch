import numpy as np
np.set_printoptions(precision=3, suppress=True) #Only a printing thing

"""
- 3 layers: 2 input neurons, 3 hidden neurons, 2 output neuron
- Given two bits and then outputs the xor value
- Data set 12 samples
- Batches of 5
- 3 Epochs

- Z = W_i @ A_i-1 + b_i
- Loss
"""

num_epochs = 5000
batch_size = 3
sample_size = 12
num_layers = 2 # One hidden layer and one output layer, inputs dont count as a layer
weight_scale = 0.01

A0 = np.array([
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
])

Y = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],   # is XOR=1?
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]    # is XOR=0?
])   


W1 = np.random.randn(3,2) * weight_scale
W2 = np.random.randn(2,3) * weight_scale
b1 = np.zeros((3,1))
b2 = np.zeros((2,1))


params = {
    'W1' : W1,
    'W2' : W2,
    'B1' : b1,
    'B2' : b2
}


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
# since we forward in batches of , we slice A0 (inputs for only first 3 columbs)

def forward(params, A0, n):
    cached = {'A0' : A0}
    for i in range(1, n+1):
        z = cached[f'Z{i}'] = ( params[f'W{i}'] @ cached[f'A{i-1}'] ) + params[f'B{i}']
        cached[f'A{i}'] = relu(z) if (i < n) else sigmoid(z)
    return cached


def backprop(params, cached, n, m, Y):
    grads = {}
    prev_dz = None
    for i in range(n, 0, -1):
        if i == n:
            A = cached[f'A{i}']
            dz = loss_deriv(A, Y) * sigmoid_deriv(A)
        else:  
            dz = params[f'W{i+1}'].T @ prev_dz * relu_deriv(cached[f'Z{i}']) # This was an error i originally only element wise multiplied everything, fixed now

        dw = (dz @ cached[f'A{i-1}'].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m # db originally not a nx1 like the biases, so we sum axis 1 which is columns, and keep dimnesions so it remains a 2d array
        prev_dz = dz

        grads[f'DW{i}'] = dw
        grads[f'DB{i}'] = db

    return grads

    # dz2/d2 always = A1, transpose A1
    # dz2/da1 = w2
    # dl/dw2 = dl/a2 x da2/dz2 x dz2/w2
    # Then dl/dz2 stored 
    # dl/b2 =  dl/dz2

    # dl/dw1 = dz2/da1 (T) @ dl/dz2 x da1/dz1 x dz1/dw1
    #dz2/da1 = W2

    # For biases our db size for first layer is a 3 x m so a 3 x 3 as m = batch size
    # But we have a bias matrix of 3x1 so we must find the avergae gradient by summing all columns in db and dividing by m



def update_params(params, grads, lr):
    for key in params: #W1, W2, B1, B2 
        params[key] -= lr * grads[f'D{key}'] # Works because the endings for W1 and DW1



def batch_stats(batch_num, A_last, Y, batch_size, num_of_batches):
    batch_loss = calculate_loss(A_last, Y, batch_size)

    #Compares the highest in a row, showing the prediction. If row 1 higher it predicts XOR = 1 vice versa
    predictions = np.argmax(A_last, axis=0)
    #Contains the expected outputs
    labels = np.argmax(Y, axis=0)
    #Compares each prediction with corresponding outputs [True, False, True ..] and adds them up, true means they were the same so you add one, false = 0
    accuracy = np.mean(predictions == labels) * 100 
    # Mean: total correct/number of predictions (batch size)


    print(f"Batch {batch_num}/{num_of_batches}  |   Loss: {batch_loss}  |   Accuracy: {accuracy}%")

    return accuracy


def epoch_stats(epoch_num, total_epochs, accuracy_sum, num_of_batches):
    print(f"\n\nEpoch {epoch_num}/{total_epochs}  |   Accuracy: {accuracy_sum/num_of_batches}%")
    print("------------------------------------------------------------------")


def train_model(A0, Y, params, batch_size, n_layers, n_samples, epochs, lr=0.01):
    n_outputs = Y.shape[0]
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



print("\n")
train_model(A0, Y, params, batch_size, num_layers, sample_size, num_epochs, 0.01)
