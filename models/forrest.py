import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import glob
import json

sequence = []
duration = []
instructions = ""
train_data = []
test_data = []
x_test = []
y_test = []
x_train = []
y_train = []

test_filenames = glob.glob('../tests/data/test/json/*')
for filename in test_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]
        test_data.append(data)
        x_test.append(data["inst"])
        y_test.append(data["duration"])

train_filenames = glob.glob('../tests/data/train/json/*')
for filename in train_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]
        train_data.append(data)
        x_train.append(data["inst"])
        y_train.append(data["duration"])

valid_filenames = glob.glob('../tests/data/valid/json/*')
for filename in valid_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]

# split data into two
# train_data, test_data = train_test_split(n, test_size=0.33)

# x_train = train_data.drop("inst", axis=1)
# y_train = train_data['duration']

# x_test = test_data.drop(["inst"], axis=1)
# y_test = test_data['duration']
# print(x_test)

# create and fit the forest
forest = RandomForestRegressor(n_estimators=100)
forest.fit(x_train, y_train)

# predict
y_pred = forest.predict(x_test)

# print the score
forest_score = round(forest.score(x_test, y_test) * 100, 2)
print(forest_score)

# plot predicted v measured
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()