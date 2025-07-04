# src/chem_augur/models/gnn/gcn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels, num_layers=2):
        super(SimpleGCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))

        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x