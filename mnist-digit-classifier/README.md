# MNIST Digit Classifier

A handwritten digit classifier built entirely from scratch in Python using NumPy - no ML frameworks.

Implements forward propagation, backpropagation via the chain rule, batch gradient descent, He initialisation, ReLU and Softmax activations, all as matrix operations.

It's trained on 42,000 MNIST images from Kaggle, achieving 97%+ accuracy on unseen digits.

## Run the drawing app (no training needed)

```bash
pip install numpy pillow pandas
python draw_digit.py
```

Draw any digit on the canvas and the network predicts it live with a confidence score. Trained weights are included in `params.npy`.

## Retrain the model

1. Download `train.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
2. Place it in a `Data/` folder inside the project directory
3. Run:

```bash
python digit_recognizer.py
```

This trains for 15 epochs and saves new weights to `params.npy`.

## Architecture

- Input: 784 neurons (28x28 flattened)
- Hidden layers: 128 → 64 neurons (ReLU)
- Output: 10 neurons (Softmax)