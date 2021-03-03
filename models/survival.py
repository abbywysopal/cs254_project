import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import glob
import json

sequence = []
duration = []
instructions = ""
test_filenames = glob.glob('../tests/data/test/json/*')
for filename in test_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]

train_filenames = glob.glob('../tests/data/train/json/*')
for filename in train_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]

valid_filenames = glob.glob('../tests/data/valid/json/*')
for filename in valid_filenames:
    with open(filename) as json_file:
        data = json.load(json_file)
        sequence.append(data)
        duration.append(data["duration"])
        # instructions.append(data["inst"])
        instructions += data["inst"]

kmf = KaplanMeierFitter()
kmf.fit(duration)

print(kmf.event_table)
kmf.plot()
plt.show()