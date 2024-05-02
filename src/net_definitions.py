import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GraphRegressorBasic(torch.nn.Module):
    def __init__(self, in_features: int = 2, conv_features: int = 16, conv_features_2: int = 32, out_features: int = 2):
        super(GraphRegressorBasic, self).__init__()
        self.conv1 = GCNConv(in_features, conv_features)
        self.conv2 = GCNConv(conv_features, conv_features_2)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(conv_features_2, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Aggregate node features into a single vector
        x = self.pool(x, data.batch)  # TODO data MUST be batched (or add if-else)

        # Process the graph representation
        x = self.fc(x)

        return x
