import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow_core.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.activations import *

import joblib
import json
import glob
import numpy as np

# instructions = ["XOR R0,R0,R0",
#                 "R R1,R1,R1",
#                 "ADDI R1,R1,0",
#                 "SUBI R2,R0,0",
#                 "BEQZ R2,L0",
#                 "BNEZ R2,L3",
# ]

def create_dataset(path):
    dataset = []
    test_filenames = glob.glob(path + '*')
    for filename in test_filenames:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            dataset.append(data)

    # instructions = []
    # xmls = []
    nmaps = []
    targets = []

    for item in dataset: 
        # instructions.append(item['instr'])
        # xmls.append(item['xml'])
        nmaps.append(item['nmap'])
        # targets.append(int(item['total_cycles']))

        # could train with cycle time per instruction
        targets.append(np.asarray(item['instr_cycle']))

    return nmaps, targets

def pre_processing(instructions, targets, max_length = 4):
    trunc_type='post'
    padding_type='post'
    TRAINING_SIZE = int((len(instructions)) * .8)

    training_instructions = instructions[0:TRAINING_SIZE]
    testing_instructions = instructions[TRAINING_SIZE:]

    training_targets = targets[0:TRAINING_SIZE]
    testing_targets = targets[TRAINING_SIZE:]
    training_targets = np.array(training_targets)
    testing_targets = np.array(testing_targets)

    training_padded = pad_sequences(training_instructions, maxlen=max_length, padding=padding_type)
    training_padded = np.array(training_padded)
    testing_padded = pad_sequences(testing_instructions, maxlen=max_length, padding=padding_type)
    testing_padded = np.array(testing_padded)

    print("train targets", training_targets)
    print("test targets:", testing_targets)

    return training_padded, training_targets, testing_padded, testing_targets


def create_and_train_model(training_padded, training_targets, testing_padded, testing_targets, epochs=10, max_length=4):
    vocab_size = 10000
    embedding_dim = 16

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=True), #input_shape=(1,1,4)
        tf.keras.layers.Dense(32, activation='softsign'),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='selu')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(training_padded, training_targets, 
    epochs=epochs, validation_data=(testing_padded, testing_targets), verbose=2, shuffle=True)

    return model

max_length = 4
nmaps, targets = create_dataset('./cpusim/tests/data/test/json/')
training_padded, training_targets, testing_padded, testing_targets = pre_processing(nmaps, targets, max_length)

trained_model = create_and_train_model(training_padded=training_padded, training_targets=training_targets, 
        testing_padded=testing_padded, testing_targets=testing_targets, epochs=10, max_length=max_length)

# ten_correct = 0 
# epochs = 23
# square_correct = 0
# min_correct = len(training_targets)/10
# scalar = 0
# while (square_correct < min_correct):
#     trained_model = create_and_train_model(training_padded=training_padded, training_targets=training_targets, 
#         testing_padded=testing_padded, testing_targets=testing_targets, epochs=epochs, max_length=max_length)

#     epochs += 1
#     pred = trained_model.predict(training_padded)
#     ten_correct = 0
#     square_correct = 0
#     for i in range(len(pred)):
#         print("pred:", pred[i])
#         print("target:", training_targets[i])
#         # print(training_targets[i]/pred[i])
#         scalar += training_targets[i]/pred[i]

#         if(round(sum(pred[i]) * 10) == training_targets[i]):
#             ten_correct += 1

#         if(round(sum(pred[i] * pred[i])) == training_targets[i]):
#             square_correct += 1

#     print("ten correct:", ten_correct)
#     print("square correct:", square_correct)
#     print("out of:", len(pred))
#     print("scalar:", scalar/len(pred))
#     scalar = 0

# # save the model to disk
# trained_model.save('trained_model.h5')