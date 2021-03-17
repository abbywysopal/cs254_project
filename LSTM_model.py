import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.activations import *

import json
import glob
import numpy as np

def create_dataset(path):
    '''
        creating dataset from data generated and fed through cpusim
    '''
    dataset = []
    test_filenames = glob.glob(path + '*')
    for filename in test_filenames:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            dataset.append(data)

    labels = []
    nmaps = []
    targets = []

    for item in dataset: 
        label = item['instr_cycle']
        nmaps.append(item['nmap'])
        targets.append(int(item['total_cycles']))
        labels.append(label)

    return nmaps, targets, labels

def pre_processing(instructions, targets, labels, max_length = 4):
    trunc_type='post'
    padding_type='post'
    TRAINING_SIZE = int((len(targets)) * .8)

    training_instructions = nmaps[0:TRAINING_SIZE]
    testing_instructions = nmaps[TRAINING_SIZE:]
    training_labels = labels[0:TRAINING_SIZE]
    testing_labels = labels[TRAINING_SIZE:]
    training_targets = targets[0:TRAINING_SIZE]
    testing_targets = targets[TRAINING_SIZE:]

    training_sequences = training_instructions
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)
    testing_sequences = testing_instructions
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type)

    training_padded = np.array(training_padded)
    testing_padded = np.array(testing_padded)
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)
    training_targets = np.array(training_targets)
    testing_targets = np.array(testing_targets)

    return training_padded, training_targets, training_labels, testing_padded, testing_targets, testing_labels

def pre_processing_testdata(instructions, targets, labels, max_length = 4):
    trunc_type='post'
    padding_type='post'

    testing_instructions = nmaps
    testing_labels = labels
    testing_targets = targets

    testing_sequences = testing_instructions
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type)

    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)
    testing_targets = np.array(testing_targets)

    return testing_padded, testing_targets, testing_labels

def create_and_train_model(training_padded, training_targets, training_labels, testing_padded, testing_targets, testing_labels, epochs=10, max_length=4):
    vocab_size = 10000
    embedding_dim = 16
    TRAINING_SIZE = len(training_targets)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='relu'),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='softsign'),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    correct = 0

    '''
        continue to train until atleast 5% accuracte
    '''
    while(correct < len(testing_labels)/30):
        history = model.fit(training_padded, training_labels, 
        epochs=epochs, validation_data=(testing_padded, testing_labels), verbose=2)
        epochs += 1
        pred = model.predict(testing_padded)

        correct = 0
        for i in range(len(pred)):
            print("pred:", pred[i])
            print("sum:", sum(pred[i]))
            print("target:", testing_targets[i])
            if(round(sum(pred[i])) == testing_targets[i]):
                correct += 1

        print("num correct:", correct)
        print("out of:", len(pred))

    return model

max_length = 4

nmaps, targets, labels = create_dataset('./cpusim/tests/data/train/json/')

training_padded, training_targets, training_labels, testing_padded, testing_targets, testing_labels = pre_processing(instructions=nmaps, targets=targets, labels=labels, max_length=max_length)

model = create_and_train_model(training_padded=training_padded, training_targets=training_targets, 
    training_labels=training_labels, testing_padded=testing_padded, testing_targets=testing_targets, 
    testing_labels=testing_labels, epochs=40, max_length=4)

pred = model.predict(testing_padded)
correct = 0
for j in range(len(pred)):
    if(round(sum(pred[j])) == testing_targets[j]):
        correct += 1
print("num correct:", correct)
print("out of:", len(pred))

for i in range(1,32):
    nmaps, targets, labels = create_dataset(f'./cpusim/tests/data/test{i}/json/')
    testing_padded, testing_targets, testing_labels = pre_processing_testdata(nmaps, targets, labels, max_length)
    pred = model.predict(testing_padded)
    correct = 0
    for j in range(len(pred)):
        if(round(sum(pred[j])) == testing_targets[j]):
            correct += 1

    print("FOR ", i , "instruction size")
    print("num correct:", correct)
    print("out of:", len(pred))
