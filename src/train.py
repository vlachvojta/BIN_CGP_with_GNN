import os
import argparse
import sys
import json
import csv
import time

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset as torch_dataset
import matplotlib.pyplot as plt
import torch_geometric

import utils
from net_definitions import GraphRegressorBasic
from chr_to_digraph import chr_to_digraph

net_choices_classes = {
    'GraphRegressorBasic': GraphRegressorBasic,
}




class TaskRegression():
    def __init__(self, output_train_dir, model_class, criterion = torch.nn.MSELoss(), 
                 optimizer = None, device = 'cpu', outputs = ['fitness', 'blocks_used']):
        self.output_train_dir = utils.ensure_folder_created(output_train_dir)
        self.model_class = model_class
        self.criterion = criterion
        self.device = device
        self.outputs = outputs

        self.checkpoint_key = 'trainedsteps'
        self.model, self.trained_steps = self.load_model(model_class, self.output_train_dir, self.device)
        if self.model is None:
            self.model = self.create_model(model_class)

        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters())

        self.train_losses_file = os.path.join(self.output_train_dir, 'train_losses.npy')
        self.val_losses_file = os.path.join(self.output_train_dir, 'val_losses.npy')
        self.train_losses, self.val_losses = self.load_loss_stats()

    def create_model(self, cls):
        model = cls(out_features=len(self.outputs))
        model.to(self.device)
        return model

    def save_model(self):
        model_path = os.path.join(self.output_train_dir, f'{self.model_class.__name__}_{self.trained_steps}{self.checkpoint_key}.pth')
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def load_model(self, model_class, path:str = 'models', device = 'cpu'):
        path, model_name = utils.find_last_model(path, self.checkpoint_key)
        if not path or not model_name:
            return None, 0
        
        trained_steps = 0
        match_obj = re.match(rf'\S+_(\d+){self.checkpoint_key}', model_name)
        if match_obj:
            trained_steps = int(match_obj.groups(1)[0])

        print(f'Loading model from {os.path.join(path, model_name)}')
        # print(f'Hidden dim: {hidden_dim}, epochs: {epochs}, n_layers: {n_layers}, bidirectional: {bidirectional}, bias: {bias}')

        model = model_class()
        model.load_state_dict(torch.load(os.path.join(path, model_name), map_location=device))
        model.eval()

        print(f'Model loaded from {os.path.join(path, model_name)}. {trained_steps} trained steps.')
        return model, trained_steps

    def test_step(self, dataloader) -> float:
        self.model.eval()
        losses = []

        with torch.no_grad():
            for i, (graph, labels) in enumerate(dataloader):
                graph = graph.to(self.device)
                labels = labels.to(self.device)
                # print(f'{graph=}, {labels.shape=}')

                out = self.model(graph)
                # print(f'{out.shape=}, {labels.shape=}')
                loss = self.criterion(out, labels)
                losses.append(loss.item())

        return sum(losses) / len(losses) if losses else 0

    def plot_stats(self):
        # print(f'\nTest step. Train losses: [{train_losses_str}]') # , Val IoUs: {val_losses}')

        # produce 2 figure chart of train_losses and val_losses
        fig, ax = plt.subplots(2)
        ax[0].plot(self.train_losses)
        ax[0].set_title('Train losses')
        ax[1].plot(self.val_losses)
        ax[1].set_title('Val losses')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_train_dir, 'losses.png'))
        plt.clf()

        # save train_losses and val_losses to .npy or something for future reference
        np.save(self.train_losses_file, self.train_losses)
        np.save(self.val_losses_file, self.val_losses)

    def load_loss_stats(self):
        train_losses = list(np.load(self.train_losses_file)) if os.path.exists(self.train_losses_file) else []
        val_losses = list(np.load(self.val_losses_file)) if os.path.exists(self.val_losses_file) else []
        return train_losses, val_losses


class CustomDataset(torch_dataset):
    def __init__(self, dataset_path, part='train'):
        self.dataset_path = dataset_path
        self.part = part

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
        # print(f'Dataset getting item: {generation_id=}, {fitness=}, {blocks_used=}, {chromosome=}')
        graph = chr_to_digraph(chromosome)

        return graph, torch.tensor([float(fitness), float(blocks_used)], dtype=torch.float32)


def parse_args():
    print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", default="datasets/multi3_3000",
                        help="Source path (default: datasets/multi3_3000")
    parser.add_argument('-o', '--output_train_dir', type=str, default='training/Regression_basic',
                        help='Where to store training progress.')
    parser.add_argument('-s', '--save_step', type=int, default=50,
                        help='How often to save checkpoint')
    parser.add_argument('-t', '--test_step', type=int, default=10,
                        help='How often to test model with validation set')
    parser.add_argument('-m', '--model_class', choices=list(net_choices_classes.keys()), default='GraphRegressorBasic',
                        help='Which model to use for training')

    parser.add_argument('-b', '--batch_size', default=20, type=int)
    parser.add_argument('-e', '--max_steps', default=1_000, type=int)
    # parser.add_argument('--lr', default=0.1, type=float)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    print(f'args: {args}')
    return args

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # load datasets
    train_dataset = CustomDataset(args.dataset_path, part='train')
    val_dataset = CustomDataset(args.dataset_path, part='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # prepare task
    task = TaskRegression(args.output_train_dir, net_choices_classes[args.model_class], device=device)
    print(f'Task loaded')
    print(f'model: \n{task.model}')

    # train
    training = True
    test_step_time = time.time()
    start_time = time.time()
    print(f'starting training')

    while training:
        epoch_time = time.time()

        for _, (graph, labels) in enumerate(train_dataloader):
            if task.trained_steps >= args.max_steps:
                training = False; break

            if task.trained_steps % args.test_step == 0:
                print('\nTest step:')
                print(f'trained_step: {task.trained_steps}, mean loss: {sum(task.train_losses[-args.test_step:]) / args.test_step:.2f}, '
                      f'time of test_step: {utils.timeit(test_step_time)}, '
                      f'time from start: {utils.timeit(start_time)}')
                val_iou = task.test_step(val_dataloader)
                task.val_losses.append(val_iou)
                task.plot_stats()
                test_step_time = time.time()

            if task.trained_steps % args.save_step == 0:
                print('\nSave step:')
                task.save_model()
                print(f'Model saved at {task.output_train_dir}')

            graph = graph.to(device)
            labels = labels.to(device)
            # print(f'{graph.shape=}, {labels.shape=}')

            task.model.train()
            task.optimizer.zero_grad()
            out = task.model(graph)
            # print(f'{out.shape=}, {labels.shape=}')
            loss = task.criterion(out, labels)
            loss.backward()
            task.optimizer.step()
            task.train_losses.append(loss.item())
            task.trained_steps += 1


# def train():
#     num_epochs = 100

#     for epoch in range(1, num_epochs):
#         model.train()
#         out = model(data)
#         loss = criterion(out, data.y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.inference_mode():
#             out = model(data)
#             loss = criterion(out, data.y)

#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# def train(net, train_data: Dataset, val_data: Dataset, charset: Charset,
#           learn_rate, device='cpu', batch_size=32, epochs=5, save_step=5, view_step=1,
#           hidden_dim=8, GRU_layers=1, bidirectional=False, bias=False,
#           training_path:str = 'models'):

#     model, epochs_trained = None, 0
#     if training_path:
#         if not os.path.isdir(training_path):
#             os.makedirs(training_path)
#         else:
#             model, epochs_trained = helpers.load_model(net, training_path)
#             epochs += epochs_trained

#     if not model:
#         model = net(hidden_dim=hidden_dim, device=device, batch_size=batch_size, n_layers=GRU_layers, bidirectional=bidirectional, bias=bias)

#     input_example, _ = next(train_data.batch_iterator(5))
#     writer.add_graph(model, input_example)
#     writer.flush()
#     writer.close()

#     model.to(device)
#     print(f'Using device: {device}')
#     print(f'Using model:\n{model}')

#     # Defining loss function and optimizer
#     criterion = charset.task.criterion()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#     model.train()
#     print("Starting Training")
#     print('')
#     epoch_times = []
#     trn_losses = []
#     trn_losses_lev = []
#     val_losses_lev = []
#     h = None  # model.init_hidden(batch_size).to(device)

#     for epoch in range(epochs_trained, epochs + 1):
#         export_path = helpers.get_save_path(training_path, hidden_dim, epoch, batch_size, n_layers=GRU_layers, bidirectional=bidirectional, bias=bias)
#         epoch_outputs = []
#         epoch_labels = []
#         start_time = time.time()
#         model.train()

#         for i, (x, labels) in enumerate(train_data.batch_iterator(batch_size), start=1):
#             model.zero_grad()
#             out, _ = model(x.to(device), h)
#             loss = criterion(out, labels.to(device))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             outputs = [charset.tensor_to_word(o) for o in out]
#             epoch_outputs += outputs
#             labels = [charset.tensor_to_word(l) for l in labels]
#             epoch_labels += labels
#         trn_losses_lev.append(helpers.levenstein_loss(epoch_outputs, epoch_labels))
#         trn_losses.append(loss.item())

#         if epoch % view_step == 0:
#             model.eval()
#             val_loss_lev, val_in_words, val_out_words, val_labels_words = test_val(model, val_data, device, batch_size, charset)
#             val_losses_lev.append(val_loss_lev)

#             print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.5f}, {trn_losses_lev[-1]:.5f} %, val losses: {val_losses_lev[-1]:.3f} %")
#             print(f"Average epoch time in this view_step: {np.mean(epoch_times[-view_step:]):.2f} seconds")
#             print('Example:')
#             print(f'\tin:  {val_in_words[:100]}')
#             print(f'\tout: {val_out_words[:100]}')
#             print(f'\tlab: {val_labels_words[:100]}')
#             print('')

#         if epoch % save_step == 0:
#             helpers.plot_losses(trn_losses, trn_losses_lev, val_losses_lev, epoch, view_step=view_step, path=export_path)
#             helpers.save_model(model, path=export_path)
#             helpers.save_out_and_labels(val_out_words, val_labels_words, path=export_path)
#         current_time = time.time()
#         print(f'epoch time: {current_time-start_time:.2f} seconds')
#         epoch_times.append(current_time-start_time)

#     print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({np.mean(epoch_times):.2f} seconds per epoch)")

#     return model


if __name__ == '__main__':
    main()
