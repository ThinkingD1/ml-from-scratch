import pandas as pd

data = pd.read_csv('mnist-digit-classifier/Data/train.csv')
inputs = data.drop('label', axis=1).values.T   # (784, 42000)
labels = data['label'].values # (42000,)
# Gives 2D array of the correct digit that was written e.g. [1, 6, 8, 2...] 
# *First element means Image 1 (the pixels in column 1) was a 1 etc

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


