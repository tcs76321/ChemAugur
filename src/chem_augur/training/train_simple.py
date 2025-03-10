# training/train_simple.py
from datetime import datetime, timezone, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from chem_augur.features.molecular_graph import smiles_to_pyg_graph_simple
from chem_augur.models.gnn.gcn_model import SimpleGCN

smiles_list = ["CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC"]
property_values = [184.2, 231.2, 272.1, 309.2, 341.8, 371.6, 398.8, 424.1]

data_list = []
for smiles, value in zip(smiles_list, property_values):
    graph_data = smiles_to_pyg_graph_simple(smiles)
    if graph_data is not None:
        graph_data.y = torch.tensor([[value]], dtype=torch.float)
        data_list.append(graph_data)

data_loader = DataLoader(data_list, batch_size=8, shuffle=True)

# --- Model, Loss, Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(num_node_features=1, hidden_channels=64, output_channels=1, num_layers=4).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Simplified Training Loop ---
target_loss = 1.0
target_hour = 10  # 10 AM
target_timezone = timezone(timedelta(hours=-6)) # CST
epoch_count = 0

while True:
    model.train()
    total_loss = 0
    epoch_count += 1

    for batch in data_loader:
        batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(data_loader)
    current_time_cst = datetime.now(target_timezone)
    current_time_str = current_time_cst.strftime("%I:%M:%S %p %Z")

    print(f"Epoch {epoch_count}, Time (CST): {current_time_str}, Loss: {epoch_loss:.4f}")

    if epoch_loss < target_loss or current_time_cst.hour >= target_hour:
        if epoch_loss < target_loss:
            print(f"Reached target loss of {target_loss:.4f}.")
        else:
            print(f"Reached target time of {target_hour}:00 AM CST.")
        break

print("Training loop finished.")

torch.save(model.state_dict(), "src/chem_augur/models/gnn/simple_gcn_model_minimal.pth")
print("Minimal SimpleGCN model saved.")