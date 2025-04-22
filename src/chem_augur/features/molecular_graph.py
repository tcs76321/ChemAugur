# features/molecular_graph.py
from rdkit import Chem
from torch_geometric.data import Data
import torch
from torch_geometric.utils import coalesce


def sdf_mol_to_graph(mol, target_property):
    """Convert an RDKit molecule (from SDF) to a PyG Data object."""
    try:
        Chem.SanitizeMol(mol)
    except:
        print("Invalid molecule structure, skipping.")
        return None

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_indices.append((start, end))
        edge_indices.append((end, start))  # Ensure bidirectional edges

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_index = coalesce(edge_index)

    try:
        y_value = mol.GetProp(target_property)
    except KeyError:
        print(f"Warning: Missing property '{target_property}' for this molecule.")
        return None
    y = torch.tensor([[float(y_value)]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


def smiles_to_pyg_graph_simple(smiles):
    """Converts a SMILES string to a PyG Data graph with atomic features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index_list = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index_list.append((start, end))
        edge_index_list.append((end, start))

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_index = coalesce(edge_index)

    return Data(x=x, edge_index=edge_index)