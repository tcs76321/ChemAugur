# features/molecular_graph.py
"""
Molecular graph feature extraction.

Converts RDKit molecules (from SDF or SMILES) into PyTorch Geometric Data objects
with rich atom and bond feature representations suitable for GCN training.
"""
from typing import Optional, List

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


# ---------------------------------------------------------------------------
# Atom feature encoding
# ---------------------------------------------------------------------------

def _one_hot(value: int, num_classes: int) -> List[float]:
    """Return a one-hot encoded list."""
    return [1.0 if i == value else 0.0 for i in range(num_classes)]


HYBRIDIZATION_MAP = {
    Chem.HybridizationType.S: 0,
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 4,
    Chem.HybridizationType.SP3D2: 5,
}
NUM_HYBRIDIZATION_CLASSES = 6

ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Na",
    "Mg", "Al", "K", "Ca", "Fe", "Zn", "Other"
]
NUM_ATOM_CLASSES = len(ATOM_SYMBOLS)  # 19

# ---------------------------------------------------------------------------
# One-hot dimension sizes for categorical atom features
# ---------------------------------------------------------------------------
NUM_DEGREE_CLASSES = 6  # degree 0-5 (clamped)
CHARGE_CLASSES = 5       # formal charge -2..+2
CHIRAL_CLASSES = 3       # R-NONE, R-CLIW, R-CW (0..2)
NUM_H_COUNT_CLASSES = 6  # total H count 0-5 (clamped)
NUM_VALENCE_CLASSES = 6  # total valence 0-5 (clamped)


def get_atom_feature(atom: Chem.Atom) -> List[float]:
    """
    Extract a rich feature vector for a single atom.

    Features (in order):
      1. Atomic number (raw, scaled)                          – 1 dim
      2. Degree (valency)                                     – 6 dim  (0-5 + clamp)
      3. Formal charge (clamped -2..2)                        – 5 dim
      4. Hybridization                                       – 6 dim
      5. Aromatic flag                                        – 1 dim
      6. Chiral tag                                           – 2 dim
      7. Number of H atoms attached                           – 6 dim  (0-5 + clamp)
      8. Total valence                                        – 6 dim  (0-5 + clamp)
      9. In-ring flag                                         – 1 dim

    Total: ATOM_FEATURE_DIM dimensions
    """
    # 1. Atomic number (normalized by 100 for scale)
    atomic_num = atom.GetAtomicNum() / 100.0

    # 2. Degree
    degree = min(atom.GetDegree(), 5)
    degree_oh = _one_hot(degree, NUM_DEGREE_CLASSES)

    # 3. Formal charge (clamped to -2 .. 2 → indices 0..4)
    formal_charge = max(-2, min(2, atom.GetFormalCharge()))
    charge_idx = formal_charge + 2  # shift to 0-based
    charge_oh = _one_hot(charge_idx, CHARGE_CLASSES)

    # 4. Hybridization
    hybrid_idx = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5)
    hybrid_oh = _one_hot(hybrid_idx, NUM_HYBRIDIZATION_CLASSES)

    # 5. Aromatic
    aromatic = float(atom.GetIsAromatic())

    # 6. Chiral tag (0=R-NONE, 1=R-CLIW, 2=R-CW)
    chiral_tag = min(int(atom.GetChiralTag()), 2)
    chiral_oh = _one_hot(chiral_tag, CHIRAL_CLASSES)

    # 7. Total H count
    total_h = min(atom.GetTotalNumHs(), 5)
    h_oh = _one_hot(total_h, NUM_H_COUNT_CLASSES)

    # 8. Total valence
    total_valence = min(atom.GetTotalValence(), 5)
    valence_oh = _one_hot(total_valence, NUM_VALENCE_CLASSES)

    # 9. In ring
    in_ring = float(atom.IsInRing())

    return [atomic_num] + degree_oh + charge_oh + hybrid_oh + [aromatic] + chiral_oh + h_oh + valence_oh + [in_ring]


ATOM_FEATURE_DIM = 1 + 6 + 5 + 6 + 1 + 3 + 6 + 6 + 1  # = 34

# ---------------------------------------------------------------------------
# Bond (edge) feature encoding
# ---------------------------------------------------------------------------

BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
NUM_BOND_TYPE_CLASSES = 5  # 4 types + unknown


def get_bond_feature(bond: Chem.Bond) -> List[float]:
    """
    Extract a feature vector for a single bond.

    Features:
      1. Bond type one-hot              – 5 dim
      2. Conjugated flag                – 1 dim
      3. In-ring flag                   – 1 dim

    Total: 7 dimensions
    """
    bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 4)
    bond_type_oh = _one_hot(bond_type, NUM_BOND_TYPE_CLASSES)
    conjugated = float(bond.GetIsConjugated())
    in_ring = float(bond.IsInRing())
    return bond_type_oh + [conjugated, in_ring]


BOND_FEATURE_DIM = 5 + 1 + 1  # = 7

# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def mol_to_graph(mol: Chem.Mol, target_property: Optional[str] = None) -> Optional[Data]:
    """
    Convert an RDKit molecule to a PyG Data object with rich features.

    Args:
        mol: RDKit molecule (must be sanitized).
        target_property: If provided, extract this property as the label ``y``.
            Return None if the property is not present on the molecule.

    Returns:
        PyG Data object with:
          - x:        [num_atoms, ATOM_FEATURE_DIM] node features
          - edge_index: [2, num_edges]
          - edge_attr:  [num_edges, BOND_FEATURE_DIM] edge features
          - y:        [1] regression target (only if target_property given)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # --- Node features ---
    atom_features = [get_atom_feature(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # --- Edge features ---
    edge_indices: List[tuple] = []
    edge_features: List[List[float]] = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bfeat = get_bond_feature(bond)
        # Bidirectional edges
        edge_indices.append((start, end))
        edge_features.append(bfeat)
        edge_indices.append((end, start))
        edge_features.append(bfeat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    # Coalesce edge_index and apply the same remapping to edge_attr.
    # coalesce returns (coalesced_edge_index, None) or (coalesced_edge_index, edge_attr)
    # depending on PyG version. Handle both.
    coalesced = coalesce(edge_index, edge_attr)
    if isinstance(coalesced, tuple):
        edge_index, edge_attr = coalesced
    else:
        edge_index = coalesced

    # --- Label ---
    y = None
    if target_property is not None:
        try:
            y_value = float(mol.GetProp(target_property))
        except (KeyError, ValueError):
            return None  # molecule doesn't have this property
        y = torch.tensor([[y_value]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (minimal feature set for legacy code)
# ---------------------------------------------------------------------------

def sdf_mol_to_graph(mol: Chem.Mol, target_property: str) -> Optional[Data]:
    """
    Legacy wrapper: convert SDF molecule to graph.

    Now uses rich atom features.  Callers that expect ``num_node_features=1``
    will need to update to ``num_node_features=ATOM_FEATURE_DIM``.
    """
    return mol_to_graph(mol, target_property=target_property)


def smiles_to_pyg_graph_simple(smiles: str, target_value: Optional[float] = None) -> Optional[Data]:
    """
    Legacy wrapper: convert SMILES to graph with optional label.

    Now uses rich atom features.
    """
    mol = Chem.MolFromSmiles(smiles)
    data = mol_to_graph(mol, target_property=None)
    if data is not None and target_value is not None:
        data.y = torch.tensor([[target_value]], dtype=torch.float)
    return data