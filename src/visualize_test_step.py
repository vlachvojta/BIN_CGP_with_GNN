import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='training/GraphRegressorBasicBlocksUsed_2/test_step_log_1200.tsv')
args = parser.parse_args()

# read file
data = pd.read_csv(args.file, sep='\t')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

def plot_pred_vs_label(ax, x, y, title):
    ax.scatter(x, y, alpha=0.5)
    ax.set_xlabel('label')
    ax.set_ylabel('pred')
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(title)
    ax.grid(True)

plot_pred_vs_label(ax[0], data['fitness'], data['fitness_pred'], 'Fitness')
plot_pred_vs_label(ax[1], data['blocks_used'], data['blocks_used_pred'], 'Blocks Used')

plt.tight_layout()
path = os.path.dirname(args.file)
file_name = os.path.basename(args.file)

plt.savefig(os.path.join(path, f'{file_name}.png'))
plt.clf()
