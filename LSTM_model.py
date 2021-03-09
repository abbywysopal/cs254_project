import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow_core.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.activations import *

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

vocab_size = 10000
embedding_dim = 16
max_length = 4
trunc_type='post'
padding_type='post'

dataset = []
test_filenames = glob.glob('./cpusim/tests/data/test/json/*')
for filename in test_filenames:
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        dataset.append(data)

instructions = []
labels = []
xmls = []
nmaps = []
targets = []

for item in dataset: 
    label = item['instr_cycle']
    instructions.append(item['instr'])
    xmls.append(item['xml'])
    nmaps.append(item['nmap'])
    targets.append(int(item['total_cycles']))
    labels.append(label)

TRAINING_SIZE = int((len(dataset)) * .8)

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

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=True), #input_shape=(1,1,4)
    tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='relu'),
    tf.keras.layers.LSTM(32),
    # tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
    tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='softsign'),
    tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
    # tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='softsign'),
    # tf.keras.layers.Dense(1, kernel_initializer='lecun_normal', activation='selu')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 10
correct = 0

while(correct < TRAINING_SIZE/10):

    history = model.fit(training_padded, training_labels, 
    epochs=epochs, validation_data=(testing_padded, testing_labels), verbose=2)
    epochs += 1
    pred = model.predict(training_padded)

    sum_pred_correct = 0
    correct = 0
    for i in range(len(pred)):
        # print("sum pred:", sum(pred[i]))
        # print("target:", training_targets[i])
        if(round(sum(pred[i])) == training_targets[i]):
            correct += 1

    print("num correct:", correct)
    print("out of:", len(pred))

# pred = model.predict(training_padded)
# correct = 0
# for i in range(len(pred)):
#     # print(sum(pred[i]) * 10)
#     # print(testing_targets[i])
#     # print("pred:", pred[i])
#     # print("pred round:", round(sum(pred[i]) * 10))
#     # print("target:", testing_targets[i])
#     if(round(sum(pred[i]) * 10) == testing_targets[i]):
#         correct += 1

# print("num correct:", correct)
# print("out of:", len(pred))