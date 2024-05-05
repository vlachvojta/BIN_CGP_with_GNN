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

plt.scatter(list(data.blocks_used), list(data.blocks_used_pred), alpha=0.5)
plt.xlabel('label')
plt.ylabel('pred')
plt.title('Test Step')
plt.grid(True)
plt.tight_layout()
path = os.path.dirname(args.file)

plt.savefig(os.path.join(path, 'test_step.png'))
plt.clf()
