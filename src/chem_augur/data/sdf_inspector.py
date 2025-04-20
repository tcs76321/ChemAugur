# sdf_inspector.py
import os
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict


def list_sdf_files():
    """List SDF files in the nested data directory"""
    data_dir = os.path.abspath("src/chem_augur/data")  # Explicit path from project root
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        return []

    return [
        f for f in os.listdir(data_dir)
        if f.endswith(('.sdf', '.sdf.gz')) and os.path.isfile(os.path.join(data_dir, f))
    ]


def display_molecule_info(mol):
    """Show detailed molecule information with key descriptors"""
    if mol is None:
        print("Invalid molecule entry")
        return

    print("\nMolecule Properties:")
    print("-------------------")
    # Sort properties alphabetically
    for prop in sorted(mol.GetPropNames()):
        print(f"{prop}: {mol.GetProp(prop)}")

    print("\nMolecular Descriptors:")
    print("---------------------")
    print(f"Molecular Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"Molecular Weight: {Descriptors.MolWt(mol):.2f}")
    print(f"LogP: {Descriptors.MolLogP(mol):.2f}")
    print(f"TPSA: {rdMolDescriptors.CalcTPSA(mol):.2f}")
    print(f"Number of Rings: {rdMolDescriptors.CalcNumRings(mol)}")
    print(f"Number of Atoms: {mol.GetNumAtoms()}")
    print(f"Number of Bonds: {mol.GetNumBonds()}")


def analyze_sdf_file(mols):
    """Analyze SDF contents with advanced statistics"""
    total_mols = len(mols)
    valid_mols = sum(1 for mol in mols if mol is not None)

    if valid_mols == 0:
        print("No valid molecules found in the file.")
        return

    property_stats = defaultdict(int)
    required_props = set()
    invalid_count = total_mols - valid_mols

    for mol in mols:
        if mol is not None:
            props = mol.GetPropNames()
            for prop in props:
                property_stats[prop] += 1
        else:
            invalid_count += 1  # Already accounted

    # Calculate required properties (present in all valid mols)
    if valid_mols > 0:
        required_props = [prop for prop, count in property_stats.items() if count == valid_mols]

    print("\nFile Analysis Summary:")
    print(f"Total entries: {total_mols}")
    print(f"Valid molecules: {valid_mols} ({valid_mols / total_mols * 100:.1f}%)")
    print(f"Invalid entries: {invalid_count} ({invalid_count / total_mols * 100:.1f}%)")

    print("\nProperty Statistics (Presence in Valid Molecules):")
    sorted_props = sorted(property_stats.items(), key=lambda x: (-x[1], x[0]))
    for prop, count in sorted_props:
        print(f"- {prop:20s}: {count:5d} ({count / valid_mols * 100:5.1f}%)")

    print("\nConsistently Present Properties:")
    if required_props:
        for prop in sorted(required_props):
            print(f"- {prop}")
    else:
        print("No properties present in all valid molecules")


def main():
    # List available SDF files
    sdf_files = list_sdf_files()
    if not sdf_files:
        print("No SDF files found in current directory")
        return

    print("Available SDF files:")
    for idx, filename in enumerate(sdf_files):
        print(f"{idx + 1}: {filename}")

    # Get user selection with validation
    while True:
        try:
            selection = int(input("\nEnter file number to analyze: ")) - 1
            if 0 <= selection < len(sdf_files):
                break
            print(f"Please enter a number between 1 and {len(sdf_files)}")
        except ValueError:
            print("Invalid input. Please enter a number")

    selected_file = sdf_files[selection]
    print(f"\nAnalyzing file: {selected_file}")

    full_path = os.path.abspath(os.path.join("src/chem_augur/data", selected_file))

    # Read molecules using the absolute path
    suppl = SDMolSupplier(full_path)
    mols = list(suppl)

    # Show first valid molecule's details
    first_valid = next((mol for mol in mols if mol is not None), None)
    if first_valid:
        display_molecule_info(first_valid)
    else:
        print("No valid molecules to display")

    # Perform full analysis
    analyze_sdf_file(mols)


if __name__ == "__main__":
    print("SDF File Inspector")
    print("----------------------------------------------------")
    main()
    print("----------------------------------------------------")
