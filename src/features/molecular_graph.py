# features/molecular_graph.py
from rdkit import Chem
from torch_geometric.data import Data
import torch

def smiles_to_pyg_graph_simple(smiles):
    """Converts a SMILES string to a very basic PyTorch Geometric Data graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: Just atom type (atomic number)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features = [atom.GetAtomicNum()] # Just atomic number
        atom_features_list.append(atom_features)
    x = torch.tensor(atom_features_list, dtype=torch.float)

    # Edge index (connectivity)
    edge_index_list = []
    for bond in mol.GetBonds():
        start_atom_index = bond.GetBeginAtomIdx()
        end_atom_index = bond.GetEndAtomIdx()
        edge_index_list.append((start_atom_index, end_atom_index))
        edge_index_list.append((end_atom_index, start_atom_index)) # Bidirectional
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # No edge features for now - we can add them later

    data = Data(x=x, edge_index=edge_index) # No edge_attr for now
    return data
