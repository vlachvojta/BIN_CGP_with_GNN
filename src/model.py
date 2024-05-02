import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# Chromosome data
chromosome_data = [
    (3, 1, 3), (2, 3, 3), (2, 0, 3), (4, 6, 2), (4, 4, 3), (5, 6, 2),
    (9, 8, 0), (9, 0, 3), (7, 1, 2), (11, 7, 5), (7, 11, 1), (11, 12, 3), (14, 13, 15)
]

# Function to convert chromosome to graph
def chromosome_to_graph(chromosome_data):
    G = nx.DiGraph()

    for element in chromosome_data[:-1]:
        input1, input2, function_id = element
        # Add nodes for inputs
        G.add_node(input1, type='input')
        G.add_node(input2, type='input')
        # Add node for the function
        G.add_node(function_id, type='function', function_id=function_id)
        # Add edges from inputs to function
        G.add_edge(input1, function_id, label=function_id)
        G.add_edge(input2, function_id, label=function_id)
        # Add edges from function to outputs
        for output_id in element[2:]:
            G.add_edge(function_id, output_id, label=output_id)
    
    return G

# Convert chromosome to graph
graph = chromosome_to_graph(chromosome_data)


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

# Example usage
def graph_to_pyg_data(G):
    # Convert networkx DiGraph to PyTorch Geometric Data
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor([G.nodes[node]['feature'] for node in G.nodes], dtype=torch.float)
    y = torch.tensor([G.nodes[node]['label'] for node in G.nodes], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Example graph conversion and model usage
G = nx.DiGraph()
# Add nodes and edges to G
# For example:
G.add_node(1, feature=torch.tensor([0.0]))
G.add_edge(1, 2, label=torch.tensor([0.0]))

# Convert to PyTorch Geometric Data
data = graph_to_pyg_data(G)

# Initialize the model
model = GraphNeuralNetwork(num_node_features=1, num_classes=1)

# Forward pass
output = model(data)
print(output)
