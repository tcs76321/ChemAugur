# sdf_inspector.py
import os
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import rdMolDescriptors  # Explicit import [[1]][[5]]


def list_sdf_files():
    """List all SDF files in current directory"""
    return [f for f in os.listdir('.')
            if f.endswith('.sdf') and os.path.isfile(f)]


def display_molecule_info(mol):
    """Show detailed information about an RDKit molecule"""
    if mol is None:
        print("Invalid molecule entry")
        return

    print("\nMolecule Properties:")
    print("-------------------")
    for prop in mol.GetPropNames():
        value = mol.GetProp(prop)
        print(f"{prop}: {value}")

    # Corrected reference to rdMolDescriptors [[1]][[5]]
    print("\nMolecular Formula:", rdMolDescriptors.CalcMolFormula(mol))
    print("Number of Atoms:", mol.GetNumAtoms())
    print("Number of Bonds:", mol.GetNumBonds())


def analyze_sdf_file(file_path):
    """Analyze SDF file contents and training readiness"""
    suppl = SDMolSupplier(file_path)
    total_mols = 0
    valid_mols = 0
    property_stats = {}

    for mol in suppl:
        total_mols += 1
        if mol is not None:
            valid_mols += 1
            props = mol.GetPropNames()
            for prop in props:
                property_stats[prop] = property_stats.get(prop, 0) + 1

    print("\nFile Analysis:")
    print(f"Total entries: {total_mols}")
    print(f"Valid molecules: {valid_mols} ({valid_mols / total_mols * 100:.1f}%)")

    print("\nProperty Statistics:")
    for prop, count in property_stats.items():
        print(f"- {prop}: {count} entries ({count / valid_mols * 100:.1f}%)")


def main():
    # List available SDF files
    sdf_files = list_sdf_files()
    if not sdf_files:
        print("No SDF files found in current directory")
        return

    # Display file selection
    print("Available SDF files:")
    for idx, filename in enumerate(sdf_files):
        print(f"{idx + 1}: {filename}")

    # Get user selection
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

    # Show first valid molecule's information
    suppl = SDMolSupplier(selected_file)
    for mol in suppl:
        if mol is not None:
            display_molecule_info(mol)
            break  # Show first valid molecule

    # Perform full file analysis
    analyze_sdf_file(selected_file)


if __name__ == "__main__":
    main()