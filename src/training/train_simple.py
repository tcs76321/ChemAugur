# training/train_simple.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from features.molecular_graph import smiles_to_pyg_graph_simple # Import simplified graph conversion
from models.gnn_architectures.simple_gcn_model import SimpleGCN # Import simple GCN model

# --- Synthetic Data (Tiny Dataset) ---
smiles_list = ["CCO", "CCC", "C=C", "O=C=O"] # Ethanol, Propane, Ethene, Carbon Dioxide
property_values = [1.0, 2.0, 1.5, 0.5] # Example property values (replace with something meaningful if you have a target property in mind)

# Convert SMILES to PyG graphs and create labels
data_list = []
for smiles, value in zip(smiles_list, property_values):
    graph_data = smiles_to_pyg_graph_simple(smiles)
    if graph_data is not None:
        graph_data.y = torch.tensor([value], dtype=torch.float)
        data_list.append(graph_data)

# --- Data Loader ---
batch_size = 2 # Tiny batch size for this example
data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# --- Model, Loss, Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(num_node_features=1, hidden_channels=32, output_channels=1).to(device) # Input node features is 1 (atomic number)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop (Simplified) ---
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")

# --- Save Trained Model ---
torch.save(model.state_dict(), "models/simple_gcn_model_minimal.pth")
print("Minimal SimpleGCN model saved to models/simple_gcn_model_minimal.pth")
