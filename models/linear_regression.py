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
from torch.autograd import Variable
import numpy as np
import glob
import json
import torch

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
 
#Partitioning the dataset
class Dataset(Dataset):
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

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
 
# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(len(training_set)):
        inputs, targets = train_dl.dataset.__getitem__(epoch)
        inp = Variable(torch.from_numpy(np.asarray(inputs)), requires_grad=True)
        # targ = Variable(torch.from_numpy(np.asarray(targets)), requires_grad=True)
        targ = torch.from_numpy(np.asarray(targets))
        targ = targ.float()
        inp = inp.float()

        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inp)
        print(yhat)
        # calculate loss
        loss = criterion(yhat, targ)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
 
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals, acc = list(), list(), list()
    for epoch in range(len(test_set)):
        inputs, targets = test_dl.dataset.__getitem__(epoch)
        inp = Variable(torch.from_numpy(np.asarray(inputs)), requires_grad=False)
        # targ = Variable(torch.from_numpy(np.asarray(targets)), requires_grad=False)
        targ = torch.from_numpy(np.asarray(targets))
        inp = inp.float()
        targ = targ.float()
        
        # evaluate the model on the test set
        yhat = model(inp)
        print(yhat)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targ.numpy()
        actual = actual.reshape((len(actual), 1))

        # round to class values
        yhat = yhat.round()
        print(yhat)
        # store
        predictions.append(yhat)
        actuals.append(actual)
        acc.append(accuracy_score(actual, yhat))
        print(accuracy_score(actual, yhat))

    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    return sum(acc)/len(acc)
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data
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
                    # input_data.append(sequence[item])
                    # string_seq = "0x" + str(sequence[item])
                    # print("before", string_seq)
                    # string_seq = string_seq[:string_seq.find('\n') - 1]
                    # inst = int(string_seq, 16)
                    # print(inst)
                    input_data.append(float(sequence[item]))
                else:
                    string_num = sequence[item]
                    string_num = string_num[2:]
                    output.append(float(int(string_num,2)))

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
test_dl = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

# define the network
model = MLP(32).float()

# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

# make a single prediction (expect class=1)
MAX_VAL = 4294967295
row = [0, 16712242956, 8, 32776, 49919253, 89339795, 14397587, 6827397134, 43932821, 2825758977, 37833108, 8471865102, 177646736, 3980513294, 44479251, 14540181, 458437259, 14552855055, 4887832206, 136978065, 1437703, 462857, 990440846, 22488722, 385030, 2396987536, 24582, 75831185, 585735, 2728594, 116, 4294967295]
row = np.asarray(row)/MAX_VAL

# print(len(row))
yhat = predict(row, model)

print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))

'''
input is all binary and output is 8bits of binary then convert to dec

'''