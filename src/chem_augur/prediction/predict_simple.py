# prediction/predict_simple.py
import torch
from chem_augur.models.gnn.gcn_model import SimpleGCN
from chem_augur.features.molecular_graph import smiles_to_pyg_graph_simple
from torch_geometric.data import Data

def predict_property_simple(smiles, model_path="src/chem_augur/models/gnn/simple_gcn_model_minimal.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Convert SMILES to Graph
    graph_data = smiles_to_pyg_graph_simple(smiles)
    if graph_data is None:
        return "Invalid SMILES string"
    graph_data.to(device)

    # 2. Load Trained Model
    model = SimpleGCN(num_node_features=1, hidden_channels=32, output_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Make Prediction
    with torch.no_grad():
        prediction = model(Data(x=graph_data.x.unsqueeze(0), edge_index=graph_data.edge_index, batch=torch.tensor([0])))
    return prediction.item()

if __name__ == '__main__':
    smiles_input = "CCCCCCCCCC"
    predicted_value = predict_property_simple(smiles_input)
    print(f"Predicted property for SMILES '{smiles_input}': {predicted_value}")