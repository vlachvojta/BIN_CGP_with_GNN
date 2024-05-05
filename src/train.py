import os
import argparse
import sys
import json
import csv
import time
import re

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as torch_dataset
import matplotlib.pyplot as plt
import torch_geometric
import pandas as pd

import utils
from net_definitions import *
from chr_to_digraph import chr_to_digraph


class TaskRegression():
    def __init__(self, output_train_dir, model_config_path: str, criterion = torch.nn.MSELoss(),
                 optimizer = None, device = 'cpu', outputs = ['fitness', 'blocks_used']):
        self.output_train_dir = utils.ensure_folder_created(output_train_dir)
        self.model_config, self.model_config_path = self.load_model_config(model_config_path)
        self.criterion = criterion
        self.device = device
        self.outputs = outputs

        self.checkpoint_key = 'trainedsteps'
        self.model, self.trained_steps = self.load_model(self.model_config, device=device)
        if self.model is None:
            self.model = self.init_model(self.model_config)

        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters())

        self.train_losses_file = os.path.join(self.output_train_dir, 'train_losses.npy')
        self.val_losses_file = os.path.join(self.output_train_dir, 'val_losses.npy')
        self.train_losses, self.val_losses = self.load_loss_stats()

        self.save_model_config()

    def save_model_config(self):
        with open(self.model_config_path, 'w') as f:
            json.dump(self.model.config, f)

    def load_model_config(self, model_config_path):
        if model_config_path is None:
            model_config_path = os.path.join(self.output_train_dir, 'model_config.json')

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f'Model config not found at {model_config_path}')
            # # inform user that model config does not exist and ask if they want to load from different path
            # print(f'Model config not found at {model_config_path}.')
            # # ask if user wants to load from different path, create default model config, or cancel
            # while True:
            #     choice = input(f'Options: [L]oad from different path, [C]reate default model config, [A]bort: ').lower()
            #     if choice == 'l':
            #         model_config_path = input('Enter model config path: ')
            #         if os.path.exists(model_config_path):
            #             if os.path.isdir(model_config_path):
            #                 model_config_path = os.path.join(model_config_path, 'model_config.json')
            #             break
            #         else:
            #             print(f'Error: Model config not found at {model_config_path}')
            #     elif choice == 'c':
            #         model_config_path = os.path.join(self.output_train_dir, 'model_config.json')
            #         with open(model_config_path, 'w') as f:
            #             json.dump(self.model.config, f)
            #         break
            #     elif choice == 'a':
            #         raise FileNotFoundError(f'Model config not found at {model_config_path}')
            #     else:
            #         print('Invalid choice. Please enter L, C, or A.')

        with open(model_config_path, 'r') as f:
            return json.load(f), model_config_path        

    def init_model(self, model_config):
        # if not utils.class_exists(model_config['model_class']):
        #     raise ValueError(f'Class {model_config["model_class"]} does not exist.')

        try: 
            model = eval(model_config['model_class']).from_config(model_config)
        except Exception as e:
            raise ValueError(f'Model class {model_config["model_class"]} does not exist or does not have from_config method.')
        model.to(self.device)
        return model

    def save_model(self):
        model_path = os.path.join(self.output_train_dir, f'{self.model.__class__.__name__}_{self.trained_steps}{self.checkpoint_key}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load_model(self, model_config, device = 'cpu'):
        path, model_name = utils.find_last_model(self.output_train_dir, self.checkpoint_key)
        if not path or not model_name:
            return None, 0

        trained_steps = 0
        match_obj = re.match(rf'\S+_(\d+){self.checkpoint_key}', model_name)
        if match_obj:
            trained_steps = int(match_obj.groups(1)[0])

        print(f'Loading model from {os.path.join(path, model_name)}')
        # print(f'Hidden dim: {hidden_dim}, epochs: {epochs}, n_layers: {n_layers}, bidirectional: {bidirectional}, bias: {bias}')

        model = self.init_model(model_config)
        model.load_state_dict(torch.load(os.path.join(path, model_name), map_location=device))
        model.eval()

        return model, trained_steps

    def test_step(self, dataloader) -> float:
        self.model.eval()
        losses = []
        log = torch.zeros([1, len(self.outputs * 2)])

        with torch.no_grad():
            for i, (graph, labels) in enumerate(dataloader):
                graph = graph.to(self.device)
                labels = labels.to(self.device)

                out = self.model(graph)
                loss = self.criterion(out, labels)
                losses.append(loss.item())

                if i < 5:
                    log_step = torch.concat((labels, out), dim=1)
                    log = torch.concat((log, log_step), dim=0)
                    # reorder log_step to match test_step_log headers
                    # log_step = log_step[:, [0, 1, 3, 4]]
                    # save labels and predictions to test_step_log

        order = [i+b*2 for i in range(2) for b in range(len(self.outputs))]  # re-order columns to match test_step_log headers
        log = log[1:, order]
        df = pd.DataFrame(log.numpy(), columns=[out + sign for out in self.outputs for sign in ['', '_pred']])
        df.to_csv(os.path.join(self.output_train_dir, f'test_step_log_{self.trained_steps}.tsv'), sep='\t', index=False)

        mean_val_loss = sum(losses) / len(losses) if losses else 0
        self.val_losses.append(mean_val_loss)
        return mean_val_loss

    def plot_stats(self):
        # produce 2 figure chart of train_losses and val_losses
        fig, ax = plt.subplots(2, 2)
        for i in range(2):
            ax[0, i].plot(self.train_losses)
            ax[0, i].set_title('Train losses')
            ax[0, i].set_xlabel('Trained steps')
            ax[1, i].plot(self.val_losses)
            ax[1, i].set_title('Val losses')
            ax[1, i].set_xlabel('Test steps')

        ax[0, 0].set_yscale('log')
        ax[1, 0].set_yscale('log')
        plt.tight_layout()
        print(f'Saving losses to {os.path.join(self.output_train_dir, "losses.png")}')
        plt.savefig(os.path.join(self.output_train_dir, 'losses.png'))
        plt.close()

        # save train_losses and val_losses to .npy or something for future reference
        np.save(self.train_losses_file, self.train_losses)
        np.save(self.val_losses_file, self.val_losses)

    def load_loss_stats(self):
        train_losses = list(np.load(self.train_losses_file)) if os.path.exists(self.train_losses_file) else []
        val_losses = list(np.load(self.val_losses_file)) if os.path.exists(self.val_losses_file) else []
        return train_losses, val_losses


class CustomDataset(torch_dataset):
    def __init__(self, dataset_path, part='train', outputs=['fitness', 'blocks_used'], input_features=4):
        self.dataset_path = dataset_path
        self.part = part
        self.outputs = outputs
        self.input_features = input_features

        self.config = self.load_config()
        self.data = self.load_data()

    def load_config(self):
        with open(os.path.join(self.dataset_path, 'dataset_config.json'), 'r') as f:
            return json.load(f)

    def load_data(self):
        with open(os.path.join(self.dataset_path, f'{self.part}.csv'), 'r') as f:
            data = list(csv.reader(f))
        return data[1:]  # ignore headers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generation_id, fitness, blocks_used, chromosome = self.data[idx]
        graph = chr_to_digraph(chromosome)

        # limit input features to self.input_features
        graph.x = graph.x[:, :min(graph.x.shape[-1], self.input_features)].float()

        labels = []
        for output in self.outputs:
            labels.append(float(eval(output)))

        return graph, torch.tensor(labels, dtype=torch.float32)

task_choices = {
    'regression': TaskRegression,
}

def parse_args():
    print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", default="datasets/multi3_3000",
                        help="Source path (default: datasets/multi3_3000")
    parser.add_argument('-o', '--output_train_dir', type=str, default='training/Regression_basic',
                        help='Where to store training progress.')
    parser.add_argument('-s', '--save_step', type=int, default=200,
                        help='How often to save checkpoint')
    parser.add_argument('-t', '--test_step', type=int, default=50,
                        help='How often to test model with validation set')
    parser.add_argument('-m', '--model_config', type=str, default=None,
                        help='Path to model config file. If not provided, will try to find output_train_dir/model_config.json')
    parser.add_argument('-k', '--task', choices=list(task_choices.keys()), default='regression',
                        help=f'Which task should the net learn. Choices: {list(task_choices.keys())}')
    parser.add_argument('--model_output', nargs='*', default=['fitness', 'blocks_used'],
                        help=f'Which outputs should the model predict. Default: ["fitness", "blocks_used"]')

    parser.add_argument('-b', '--batch_size', default=20, type=int)
    parser.add_argument('-e', '--max_steps', default=1_200, type=int)

    args = parser.parse_args()
    print(f'args: {args}')
    return args

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # prepare task
    task = TaskRegression(args.output_train_dir, args.model_config, device=device, outputs=args.model_output)
    print(f'Task loaded')
    print(f'model: \n{task.model}')

    # load datasets
    train_dataset = CustomDataset(args.dataset_path, part='train', outputs=args.model_output, input_features=task.model.config['in_features'])
    val_dataset = CustomDataset(args.dataset_path, part='val', outputs=args.model_output, input_features=task.model.config['in_features'])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # train
    training = True
    test_step_time = time.time()
    start_time = time.time()
    print(f'starting training')

    while training:
        epoch_time = time.time()

        for _, (graph, labels) in enumerate(train_dataloader):
            if task.trained_steps >= args.max_steps + 1:
                training = False; break

            if task.trained_steps % args.test_step == 0:
                mean_val_loss = task.test_step(val_dataloader)
                print('\nTest step:')
                print(f'trained_step: {task.trained_steps}, '
                      f'mean train loss: {sum(task.train_losses[-args.test_step:]) / args.test_step:.2f}, '
                      f'mean val loss: {mean_val_loss:.2f}, time of test_step: {utils.timeit(test_step_time)}, '
                      f'time from start: {utils.timeit(start_time)}')
                task.plot_stats()
                test_step_time = time.time()

            if task.trained_steps % args.save_step == 0:
                model_path = task.save_model()
                print('\nModel saved to', model_path)

            graph = graph.to(device)
            labels = labels.to(device)

            task.model.train()
            task.optimizer.zero_grad()
            out = task.model(graph)
            loss = task.criterion(out, labels)
            loss.backward()
            task.optimizer.step()
            task.train_losses.append(loss.item())
            task.trained_steps += 1


if __name__ == '__main__':
    main()
