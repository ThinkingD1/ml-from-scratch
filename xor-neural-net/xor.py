import numpy as np
np.set_printoptions(precision=3, suppress=True) #Only a printing thing

"""
- 3 layers, 2 input neurons, 3 hidden neurons, 1 output neuron
- Given two bits and then outputs the xor value
- Data set 20 samples
- Batches of 4
- 5 iterations
- 3 Epochs

- Z = W_i @ A_i-1 + b_i
- Loss
"""

dz = None
batch_size = 3
sample_size = 12
num_layers = 2 # One hidden layer and one output layer, inputs dont count as a layer

A0 = np.array([
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
])

Y = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],   # is XOR=1?
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]    # is XOR=0?
])   


W1 = np.random.randn(3,2) * 0.5
W2 = np.random.randn(2,3) * 0.5
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

def loss_deriv(A, Y , m):
    return 2/m * (A-Y)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deriv(A):
    return A * (1-A)

#n represents the number of layers
# since we forward in batches of , we slice A0 (inputs for only first 3 columbs)

def forward(params, A0, n):
    cached = {'A0' : A0}
    for i in range(1, n+1):
        z = cached[f'Z{i}'] = ( params[f'W{i}'] @ cached[f'A{i-1}'] ) + params[f'B{i}']
        cached[f'A{i}'] = relu(z) if (i < n) else sigmoid(z)
    return cached

def backprop(cached):
    # dl/dw2 = dl/a2 x da2/dz2 x dz2/w2

    # dl/db2 = dl/a2 x da2/dz2 x dz2/b2
    # dz2/db2 = 1 so
    # dl/b2 =  dl/dz2

    # Then dl/dz2 stored 
    # dl/dw1 = dl/dz2 x dz2/da1 x da1/dz1 x dz1/dw1
    pass
    

"""

cached_vals = forward(params, A0[:, 0:3], num_layers)

for key,value in cached_vals.items():
    print(f"{key}:\n{value}\n")

"""

