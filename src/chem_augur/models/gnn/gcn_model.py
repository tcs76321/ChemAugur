# models/gnn_architectures/simple_gcn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels) # Single GCN layer
        self.lin = torch.nn.Linear(hidden_channels, output_channels) # Linear output layer

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GCN Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 2. Graph-level readout (mean pooling)
        x = global_mean_pool(x, batch)

        # 3. Linear layer for prediction
        x = self.lin(x)
        return x