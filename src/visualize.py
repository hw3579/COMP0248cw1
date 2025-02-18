import matplotlib.pyplot as plt
import os

import json

with open('results/evaluation.json', 'r') as f:
    data = json.load(f)
ax, pl = plt.subplots(1, 2)
pl[0].plot(data['loss'])
pl[0].set_title('Loss')
pl[1].plot(data['accuracy'])
pl[1].set_title('Accuracy')

plt.show()