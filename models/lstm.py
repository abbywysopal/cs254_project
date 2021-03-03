import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
from torch.utils import data
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

#https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

import torch
from torch.autograd import Variable
 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x.double())
        return out

# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = torch.nn.Linear(n_inputs, 1)
        self.activation = torch.nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.layer.weight)
 
    # forward propagate input
    def forward(self, X):
        out = self.layer(X)
        out = self.activation(X)
        return out

#Partitioning the dataset
class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]
        return X, y

def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        
        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            input_data = []
            output = []
            for item in sequence:
                if(item != 'cycles taken'):
                    input_data.append(float(sequence[item]))
                else:
                    output.append(float(sequence[item]))

            targets.append(output)
            inputs.append(input_data)
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    # print(inputs_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set


# tests/data/test/json
sequence = []
test_filenames = glob.glob('../tests/data/test/json/*')
for filename in test_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)

train_filenames = glob.glob('../tests/data/train/json/*')
for filename in train_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)

valid_filenames = glob.glob('../tests/data/valid/json/*')
for filename in valid_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)

training_set, validation_set, test_set = create_datasets(sequence, Dataset)

train_dl = DataLoader(training_set, batch_size=len(training_set), shuffle=True)
print(train_dl)
test_dl = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

model = MLP(32).float()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # enumerate epochs
for epoch in range(len(training_set)):
    # enumerate mini batches

    inputs, targets = train_dl.dataset.__getitem__(epoch)
    inp = Variable(torch.from_numpy(np.asarray(inputs)), requires_grad=True)
    targ = Variable(torch.from_numpy(np.asarray(targets)), requires_grad=True)
    inp = inp.float()

    # clear the gradients
    optimizer.zero_grad()
    # compute the model output
    yhat = model(inp)
    # calculate loss
    loss = criterion(yhat, targ)
    # credit assignment
    loss.backward()
    # update model weights
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

predictions, actuals = [], []

for epoch in range(len(test_set)):
    inputs, targets = test_dl.dataset.__getitem__(epoch)
    inp = Variable(torch.from_numpy(np.asarray(inputs)), requires_grad=False)
    target = Variable(torch.from_numpy(np.asarray(targets)), requires_grad=False)
    inp = inp.float()

    # evaluate the model on the test set
    yhat = model(inp)

    yhat = yhat.detach().numpy()
    actual = target.detach().numpy()
    actual = actual.reshape((len(actual), 1))
    # round to class values
    yhat = yhat.round()
    # store
    # predictions[epoch] = yhat
    # actuals[epoch] = actual
    # predictions, actuals = vstack(predictions), vstack(actuals)
    print(yhat)
    print(actual)
    # acc = accuracy_score(actuals, predictions)
    # print(acc)



# targets = []
# testing = []
# with torch.no_grad():
#     for epoch in range(len(test_set)):
#         inputs, target = test_dl.dataset.__getitem__(epoch)
#         inp = Variable(torch.from_numpy(np.asarray(inputs)))
#         inp = inp.float()
#         print(target)

#         predicted = model(inp).data.numpy()

#         print(predicted)

#     plt.clf()
#     plt.plot(epoch, target, 'go', label='True data', alpha=0.5)
#     plt.plot(epoch, predicted, '--', label='Predictions', alpha=0.5)
#     plt.legend(loc='best')
#     plt.show()