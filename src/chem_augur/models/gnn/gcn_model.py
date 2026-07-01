# src/chem_augur/models/gnn/gcn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear


class SimpleGCN(torch.nn.Module):
    """Minimal Graph Convolutional Network for molecular property prediction.

    Architecture:
        GCNConv -> ReLU -> Dropout  (× num_layers)
        global_mean_pool
        Linear -> output
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        output_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_param = dropout

        # First conv ingests raw node features
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))

        # Intermediate convs
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # Final readout
        self.lin = Linear(hidden_channels, output_channels)

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_param, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x