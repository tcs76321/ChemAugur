# prediction/predict_simple.py
import torch
from chem_augur.models.gnn.gcn_model import SimpleGCN # Import simple GCN model
from chem_augur.features.molecular_graph import smiles_to_pyg_graph_simple # Import simplified graph conversion

def predict_property_simple(smiles, model_path="models/simple_gcn_model_minimal.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Convert SMILES to Graph
    graph_data = smiles_to_pyg_graph_simple(smiles)
    if graph_data is None:
        return "Invalid SMILES string"
    graph_data.to(device)

    # 2. Load Trained Model
    model = SimpleGCN(num_node_features=1, hidden_channels=32, output_channels=1) # Match model definition
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Make Prediction
    with torch.no_grad():
        prediction = model(graph_data.unsqueeze(0))
    return prediction.item()

if __name__ == '__main__':
    smiles_input = "C=C" # Example SMILES
    predicted_value = predict_property_simple(smiles_input)
    print(f"Predicted property for SMILES '{smiles_input}': {predicted_value}")