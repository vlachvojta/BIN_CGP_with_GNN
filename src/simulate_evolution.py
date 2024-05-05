import os
import argparse
import sys
import json
import csv
import time

import numpy as np

import utils
from chr_to_digraph import Chromosome

def parse_args():
    print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", default="datasets/multi3_3000",
                        help="Source path (default: datasets/multi3_3000")
    parser.add_argument('-o', '--output_dir', type=str, default='testing/without_bayes_models',
                        help='Where to store testing stats.')
    parser.add_argument('-m', '--mode', default='normal', choices=['normale', 'bayes'],
                        help='How to simulate stuff.')
    parser.add_argument('-c', '--circuit_function', default='multiplicator3', choices=['multiplicator3'],
                        help='What function the circuir do.')

    # parser.add_argument('-s', '--save_step', type=int, default=200,
    #                     help='How often to save checkpoint')
    # parser.add_argument('-t', '--test_step', type=int, default=50,
    #                     help='How often to test model with validation set')
    # parser.add_argument('-m', '--model_config', type=str, default=None,
    #                     help='Path to model config file. If not provided, will try to find output_train_dir/model_config.json')
    # parser.add_argument('-k', '--task', choices=list(task_choices.keys()), default='regression',
    #                     help=f'Which task should the net learn. Choices: {list(task_choices.keys())}')
    # parser.add_argument('--model_output', nargs='*', default=['fitness', 'blocks_used'],
    #                     help=f'Which outputs should the model predict. Default: ["fitness", "blocks_used"]')

    # parser.add_argument('-b', '--batch_size', default=20, type=int)
    # parser.add_argument('-e', '--max_steps', default=1_200, type=int)

    args = parser.parse_args()
    print(f'args: {args}')
    return args

def main():
    args = parse_args()

    # init dataset + model?
    test_dataset = TestDataset(args.dataset_path, part='test')
    simulation_data = prepare_data_for_simulation(test_dataset.inputs, args.circuit_function)

    simulation_runs = 0
    wrong_decisions = 0
    skipped_generations = 0

    start_time = time.time()

    # go through all data and simulate all possibilities to get fitness
    for generation_id in range(len(test_dataset)):
        chromosomes = parse_generation(test_dataset[generation_id])
        if chromosomes is None: skipped_generations += 1; continue

        # simulate inputs and check outputs
        for sim_input, sim_output in simulation_data:
            [chr.simulate(sim_input, sim_output) for chr in chromosomes]

        best_candidate_id = find_best_candidate(chromosomes)
        # Check correctly categorized the best fitness candidate, print simulation_runs

        if best_candidate_id != 0:
            wrong_decisions += 1
        simulation_runs += sum([chr.simulations for chr in chromosomes])
    time.sleep(2)

    print(f'Skipped generations due to identical fitness of more candidates: {skipped_generations}.')
    print(f'Simulation_runs: {simulation_runs}')
    print(f'Wrong decisions: {wrong_decisions}')
    print(f'Whole think took {utils.timeit(start_time)}')


def prepare_data_for_simulation(input_bits: int, circuit_function: str):
    inputs = []
    outputs = []

    if circuit_function == 'multiplicator3':
        for in1 in range(2**3):
            for in2 in range(2**3):
                out = in1 * in2
                outputs.append(out)
                inputs.append((in1 & 0b111) << 3 | (in2 & 0b111))

    inputs = np.array(inputs).reshape(-1, 1)
    outputs = np.array(outputs).reshape(-1, 1)
    both = np.concatenate([inputs, outputs], axis=1)
    np.random.shuffle(both)

    return both


def parse_generation(generation):
    candidates = sorted(generation, key=lambda x : x[0], reverse=True) # sort by fitness
    if candidates[0][0] == candidates[1][0]:  # two candidates have the same best fitness
        return None

    return [ChromosomeWrapper(chr) for (fit, blocks, chr) in candidates]

def find_best_candidate(chromosomes):
    chromosomes_with_id = [(i, chr.succ) for i, chr in enumerate(chromosomes)]
    sorted_chroms = sorted(chromosomes_with_id, key=lambda x : x[1], reverse=True)
    return sorted_chroms[0][0]


class ChromosomeWrapper:
    def __init__(self, chromosome: Chromosome | str):
        self.chromosome = chromosome if isinstance(chromosome, Chromosome) else Chromosome.from_str(chromosome)
        self.succ = 0  # succesfull simulation according to GT
        self.simulations = 0

    def simulate(self, input_, gt):
        inputs = self.unsqueeze(input_, self.chromosome.n_inputs)
        gt = self.unsqueeze(gt, self.chromosome.n_outputs)

        out = self.chromosome.simulate_input(inputs)

        self.succ += gt == out
        self.simulations += 1

    @staticmethod
    def unsqueeze(input_, size):
        return [((input_ >> i) & 0b1) for i in reversed(range(size))]


class TestDataset:
    def __init__(self, dataset_path, part='test'): #, outputs=['fitness', 'blocks_used'], input_features=4):
        self.dataset_path = dataset_path
        self.part = part

        self.config = self.load_config()
        self.data = self.load_data()
        self.inputs, self.outputs = self.get_circuit_in_out()

    def load_config(self):
        with open(os.path.join(self.dataset_path, 'dataset_config.json'), 'r') as f:
            return json.load(f)

    def load_data(self):
        with open(os.path.join(self.dataset_path, f'{self.part}.csv'), 'r') as f:
            data = list(csv.reader(f))
        data = data[1:]  # ignore headers

        chromosomes_len = len(data)

        data_by_generation = {}

        for dato in data:
            generation_id, *rest = dato

            if not generation_id in data_by_generation.keys():
                data_by_generation[generation_id] = [rest]
            else:
                data_by_generation[generation_id].append(rest)

        print(f'Loaded {chromosomes_len} in {len(data_by_generation)} generations.')

        return data_by_generation

    def get_circuit_in_out(self):
        fitness, blocks_used, chromosome = self.data[list(self.data.keys())[0]][0] # generation 0, 1st chromosome
        chromosome = Chromosome.from_str(chromosome)
        return chromosome.n_inputs, chromosome.n_outputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # fitness, blocks_used, chromosome = self.data[idx]
        # graph = chr_to_digraph(chromosome)

        # # limit input features to self.input_features
        # graph.x = graph.x[:, :min(graph.x.shape[-1], self.input_features)].float()

        # labels = []
        # for output in self.outputs:
        #     labels.append(float(eval(output)))

        # return graph, torch.tensor(labels, dtype=torch.float32)
        return self.data[list(self.data.keys())[idx]]


if __name__ == '__main__':
    main()
