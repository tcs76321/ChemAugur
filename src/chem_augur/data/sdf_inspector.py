# sdf_inspector.py
import os
from collections import defaultdict
from datetime import datetime
from rdkit.Chem import SDMolSupplier
import pandas as pd


def list_sdf_files():
    """List SDF files in the data directory"""
    data_dir = os.path.abspath("src/chem_augur/data")
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        return []

    return sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith(('.sdf', '.sdf.gz')) and os.path.isfile(os.path.join(data_dir, f))
    ])


def display_files(file_list):
    """Display available SDF files"""
    print("\nAvailable SDF files:")
    for idx, filename in enumerate(file_list, 1):
        print(f"{idx}: {filename}")


def get_user_selection(file_list):
    """Get validated user selection"""
    while True:
        try:
            selection = int(input("\nEnter file number to analyze: ")) - 1
            if 0 <= selection < len(file_list):
                return file_list[selection]
            print(f"Please enter a number between 1 and {len(file_list)}")
        except ValueError:
            print("Invalid input. Please enter a number")


def read_sdf_file(file_path):
    """Read molecules from SDF file"""
    return list(SDMolSupplier(file_path))


def analyze_sdf_file(mols):
    """Generate analysis report from molecules using pandas"""
    report = []
    total = len(mols)
    valid = sum(1 for mol in mols if mol is not None)
    invalid = total - valid

    if valid == 0:
        return "\nNo valid molecules found in the file."

    # Collect property counts using pandas DataFrame
    props = defaultdict(int)
    for mol in mols:
        if mol:
            for prop in mol.GetPropNames():
                props[prop] += 1

    # Convert to DataFrame and calculate statistics
    props_df = pd.DataFrame(props.items(), columns=['Property', 'Count'])
    props_df['Percentage'] = (props_df['Count'] / valid * 100).round(1)

    # Build report content
    report.append(f"\n{'File Analysis Summary':-^60}")
    report.append(f"Total entries: {total}")
    report.append(f"Valid molecules: {valid} ({valid / total * 100:.1f}%)")
    report.append(f"Invalid entries: {invalid} ({invalid / total * 100:.1f}%)")

    report.append(f"\n{'Property Statistics':-^60}")
    report.append(f"{'Property':<30} | {'Count':<6} | {'Percentage':<10}")

    # Use pandas for sorting and iteration
    for _, row in props_df.sort_values(
            by=['Count', 'Property'], ascending=[False, True]
    ).iterrows():
        report.append(f"{row['Property']:<30} | {row['Count']:>6} | {row['Percentage']:>6.1f}%")

    return "\n".join(report)


def save_report(report, sdf_filename):
    """Save report to text file with timestamp in data directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(sdf_filename)[0]
    filename = f"analysis_{base_name}_{timestamp}.txt"

    # Define the target directory
    data_dir = os.path.abspath("src/chem_augur/data")
    file_path = os.path.join(data_dir, filename)  # Combine directory and filename [[6]]

    with open(file_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {os.path.abspath(file_path)}")


def main():
    sdf_files = list_sdf_files()
    if not sdf_files:
        print("No SDF files found in data directory")
        return

    display_files(sdf_files)
    selected_file = get_user_selection(sdf_files)

    print(f"\nAnalyzing: {selected_file}")
    full_path = os.path.join(os.path.abspath("src/chem_augur/data"), selected_file)

    try:
        mols = read_sdf_file(full_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    report = analyze_sdf_file(mols)
    print(report)
    save_report(report, selected_file)


if __name__ == "__main__":
    print("SDF File Inspector - START")
    print("----------------------------------------------------")
    main()
    print("----------------------------------------------------")
    print("SDF File Inspector - END")
