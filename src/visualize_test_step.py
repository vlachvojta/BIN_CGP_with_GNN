import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='training/GraphRegressorBasicBlocksUsed_2/test_step_log_1200.tsv')
args = parser.parse_args()


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


def visualize_test_step(file):
    print(f'Processing file: {file}')
    # read file
    data = pd.read_csv(file, sep='\t')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))


    plot_pred_vs_label(ax[0], data['fitness'], data['fitness_pred'], 'Fitness')
    plot_pred_vs_label(ax[1], data['blocks_used'], data['blocks_used_pred'], 'Blocks Used')

    plt.tight_layout()
    path = os.path.dirname(file)
    file_name = os.path.basename(file)

    plt.savefig(os.path.join(path, f'{file_name}.png'))
    plt.clf()

if __name__ == '__main__':
    args = parser.parse_args()
    visualize_test_step(args.file)
