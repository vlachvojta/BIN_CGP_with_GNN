import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default='training-new/GraphRegressorEmbedNew')
    return parser.parse_args()

def main(args):
    for file in os.listdir(args.folder):
        if file.endswith('.tsv'):
            visualize_test_step_file(os.path.join(args.folder, file))

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


def visualize_test_step_file(file):
    path = os.path.dirname(file)
    file_name = os.path.basename(file)
    out_file = os.path.join(path, f'{file_name}.png')
    if os.path.exists(out_file):
        return

    print(f'Processing file: {file}')
    # read file
    data = pd.read_csv(file, sep='\t')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    plot_pred_vs_label(ax[0], data['fitness'], data['fitness_pred'], 'Fitness')
    plot_pred_vs_label(ax[1], data['blocks_used'], data['blocks_used_pred'], 'Blocks Used')

    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()

if __name__ == '__main__':
    main(parse_args())
