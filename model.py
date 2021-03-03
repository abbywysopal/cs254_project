import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence

import json
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
TRAINING_SIZE = 2

with open('tests/boolexpr/boolexpr.ass_json.txt', 'r') as f:
    data = json.load(f)

instructions = []
labels = []

# print(data)
for item in data:

    instructions.append(item['instr'])
    labels.append(item['cycles'])

# instructions = np.array(instructions)
# labels = np.array(labels)


training_instructions = instructions[0:TRAINING_SIZE]
testing_instructions = instructions[TRAINING_SIZE:]
training_labels = labels[0:TRAINING_SIZE]
testing_labels = labels[TRAINING_SIZE:]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_instructions)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_instructions)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)
testing_sequences = tokenizer.texts_to_sequences(testing_instructions)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type)

print(training_padded.shape)
# print(training_sequences.shape)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# training_padded = np.reshape(training_padded, (training_padded.shape[0], 1, training_padded.shape[1]))
# testing_padded = np.reshape(testing_padded, (testing_padded.shape[0], 1, testing_padded.shape[1]))


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(64), #input_shape=(1,1,4)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.build()
# model.summary()

epochs = 5

# print(training_sequences)

history = model.fit(training_padded, training_labels, 
epochs=epochs, validation_data=(testing_padded, testing_labels), verbose=2)
    
# print(history)