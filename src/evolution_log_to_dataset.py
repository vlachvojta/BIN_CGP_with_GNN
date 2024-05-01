# Description: This script is used to convert the evolution log files to a dataset that can be used for training the model.
# OUTPUT: The output dataset is json config file and CSV file with following columns is a CSV file that contains the following columns:
# The evolution log files contain all chromosomes generated by mutating, each chromosome line starts by its fitness value and blocks used.

# Example run: `python src/evolution_log_to_dataset.py -i cgp_lab/cgp/multi3/evolution.log -o datasets/multi3`

# Example of evolution log:
# Dataset: json_config: {"maxfitness": 384, "param_in": 6, "param_out": 6, "param_m": 7, "param_n": 7, "l_back": 2}
# LogName: log  l-back: 2  popsize:6
# ----------------------------------------------------------------
# Run: 0           Wed May  1 15:33:23 2024
# ----------------------------------------------------------------
# Dataset: New generation: 3
# Dataset: Chromosome: 1,257,18,"{6,6, 7,7, 2,2,18}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,22,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 1,247,14,"{6,6, 7,7, 2,2,14}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]1,5,1)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(54,43,15,6,16,52)"
# Dataset: Chromosome: 1,255,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]9,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 1,255,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,1)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 1,257,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 1,255,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,2)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,9,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,27,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: New generation: 4
# Dataset: Chromosome: 2,253,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,2)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]35,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 2,265,17,"{6,6, 7,7, 2,2,17}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,21,3)([38]27,33,1)([39]21,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 2,251,13,"{6,6, 7,7, 2,2,13}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,2)([39]0,0,1)([40]29,4,2)([41]5,32,2)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,29,15,6,16,52)"
# Dataset: Chromosome: 2,257,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]47,38,2)([50]35,46,2)([51]41,3,3)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 2,257,16,"{6,6, 7,7, 2,2,16}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]6,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]37,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"
# Dataset: Chromosome: 2,259,17,"{6,6, 7,7, 2,2,17}([6]3,2,1)([7]4,1,0)([8]0,3,3)([9]5,2,1)([10]5,5,0)([11]5,3,0)([12]0,0,1)([13]1,11,1)([14]6,1,3)([15]11,11,3)([16]7,9,3)([17]6,0,1)([18]5,5,2)([19]6,6,2)([20]8,2,3)([21]0,1,2)([22]13,1,3)([23]2,2,2)([24]7,3,0)([25]14,4,0)([26]2,7,0)([27]5,4,1)([28]21,20,3)([29]16,19,3)([30]3,19,1)([31]22,5,2)([32]24,25,3)([33]23,0,0)([34]20,33,3)([35]3,27,3)([36]33,30,2)([37]25,0,3)([38]27,33,1)([39]0,0,1)([40]29,4,2)([41]5,32,3)([42]36,0,1)([43]40,30,1)([44]32,36,0)([45]28,31,1)([46]27,2,1)([47]5,34,2)([48]42,46,1)([49]34,38,2)([50]35,46,2)([51]41,3,1)([52]4,46,1)([53]41,37,2)([54]2,1,2)(37,43,15,6,16,52)"

import argparse
import os
import json
import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Convert evolution log files to dataset')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory path')
    return parser.parse_args()

def main():
    args = parse_args()

    DatasetCreator(args.input, args.output_dir)()

class DatasetCreator:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = utils.ensure_folder_created_with_overwrite_prompt(output_dir)

        if not os.path.exists or not os.path.isfile(input_file):
            raise FileNotFoundError(f'Input file not found: {input_file}')

        self.dataset_file_path = os.path.join(self.output_dir, 'dataset.csv')
        self.dataset_config_file_path = os.path.join(self.output_dir, 'dataset_config.json')
        self.stats_file_path = os.path.join(self.output_dir, 'stats.txt')

    def __call__(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()

        print(f'Parsing {len(lines)} lines from {self.input_file}...')

        chromosome_lines = []

        for line in lines:
            if line.startswith('Dataset: json_config:'):
                start = len('Dataset: json_config: ')
                print(f'Parsing json config: {line}, {line[start:]}')
                json_config = json.loads(line[start:])
                continue

            if line.startswith('Dataset: Chromosome:'):
                start = len('Dataset: Chromosome: ')
                chromosome_lines.append(line[start:])

        chromosomes = list(csv.reader(chromosome_lines))
        print(f'Parsed {len(chromosomes)} chromosome lines from csv')

        # save csv to a file with headers 'generation_id', 'fitness', 'blocks_used', 'chromosome'
        with open(self.dataset_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['generation_id', 'fitness', 'blocks_used', 'chromosome'])
            for row in chromosomes:
                writer.writerow(row)

        # save dataset config
        with open(self.dataset_config_file_path, 'w') as f:
            json.dump(json_config, f)
        
        print(f'Dataset saved to {self.dataset_file_path}')

        self.make_stats(chromosomes)

    def make_stats(self, chromosomes):
        # get stats, save them to self.stats_file_path
        data = pd.DataFrame(chromosomes, columns=['generation_id', 'fitness', 'blocks_used', 'chromosome'])
        data['fitness'] = data['fitness'].astype(int)
        data['blocks_used'] = data['blocks_used'].astype(int)
        data['generation_id'] = data['generation_id'].astype(int)

        stats = data.groupby('generation_id').agg({'fitness': ['min', 'max', 'mean', 'std'], 'blocks_used': ['min', 'max', 'mean', 'std']})
        stats.to_csv(self.stats_file_path)
        print(f'Stats saved to {self.stats_file_path}')

        # Make plot made from from 2 subplot histograms of fitness and blocks_used
        fig, axs = plt.subplots(2)
        axs[0].hist(data['fitness'], bins=20)
        axs[0].set_title('Fitness histogram')
        axs[0].set_xlabel('Fitness')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(data['blocks_used'], bins=20)
        axs[1].set_title('Blocks used histogram')
        axs[1].set_xlabel('Blocks used')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fitness_blocks_used_histogram.png'))
        plt.clf()


if __name__ == '__main__':
    main()
