import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
import pandas as pd
import numpy as np
sns.set()

train_set = load_data(kind="train").dataset
test_set = load_data(kind="test").dataset
val_set = load_data(kind="val").dataset

labels = {v: k for k, v in train_set.class_to_idx.items()}
get_targets = lambda dataset: pd.Series(dataset.targets).map(lambda x: labels[x]).value_counts().sort_index()
train_targets = get_targets(train_set)
test_targets = get_targets(test_set)
val_targets = get_targets(val_set)

fig, ax = plt.subplots()
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, train_targets, width, label="Train")
ax.bar(x + width/2, test_targets, width, label="Test")
# No point plotting val because it isn't even visible
# plt.bar(x + width, val_targets, width, label="Val")
ax.set_xticks(x)
ax.set_xticklabels(labels.values())

ax.legend()
plt.show()