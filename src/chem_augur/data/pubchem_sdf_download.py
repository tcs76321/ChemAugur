# pubchem_sdf_download.py
# Downloads PubChem Compound SDF files with interactive control and MD5 verification

import os
import hashlib
from ftplib import FTP

# Configuration
FTP_SERVER = "ftp.ncbi.nlm.nih.gov"
FTP_DIR = "/pubchem/Compound/CURRENT-Full/SDF/"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_md5(file_path):
    """Compute MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def list_files(ftp):
    """List and filter SDF files from FTP"""
    files = ftp.nlst()
    return [f for f in files
            if f.endswith(".sdf.gz")
            and not f.endswith(".md5")]


def show_file_status(available, existing):
    """Display file status information"""
    print("\nAvailable files to download:")
    for f in available[:10]:
        print(f" - {f}")
    if len(available) > 10:
        print(f" ... and {len(available) - 10} more files")

    print("\nExisting files:")
    for f in existing[:10]:
        print(f" - {f}")
    if len(existing) > 10:
        print(f" ... and {len(existing) - 10} more files")


def get_download_count(max_count):
    """Prompt user for download count with validation"""
    while True:
        response = input("\nEnter number of files to download (or 'all'): ").strip().lower()
        if response == 'all':
            return max_count
        try:
            num = int(response)
            if 1 <= num <= max_count:
                return num
            print(f"Please enter a number between 1 and {max_count}")
        except ValueError:
            print("Invalid input. Please enter a number or 'all'")


def download_sdf_files():
    try:
        with FTP(FTP_SERVER) as ftp:
            ftp.login()
            ftp.cwd(FTP_DIR)

            # Get file lists
            all_files = list_files(ftp)
            existing_files = [f for f in all_files if os.path.exists(os.path.join(CURRENT_DIR, f))]
            available_files = [f for f in all_files if f not in existing_files]

            # Show status
            show_file_status(available_files, existing_files)
            if not available_files:
                print("All files already downloaded!")
                return

            # Get user input
            num_to_download = get_download_count(len(available_files))
            if num_to_download == 0:
                return

            # Download loop
            for idx in range(num_to_download):
                filename = available_files[idx]
                sdf_path = os.path.join(CURRENT_DIR, filename)
                md5_path = f"{sdf_path}.md5"

                print(f"\nProcessing file {idx + 1}/{num_to_download}: {filename}")

                # Download SDF
                print("  Downloading SDF...")
                with open(sdf_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)

                # Download MD5
                print("  Downloading MD5...")
                with open(md5_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}.md5", f.write)

                # Verify checksum
                print("  Verifying checksum...")
                with open(md5_path, "r") as f:
                    expected_md5 = f.read().split()[0].strip()

                actual_md5 = compute_md5(sdf_path)

                if actual_md5 == expected_md5:
                    print(f"  ✅ Checksum verified ({actual_md5})")
                    os.remove(md5_path)
                else:
                    print(f"  ❌ Checksum FAILED! Expected: {expected_md5}, Got: {actual_md5}")
                    os.remove(sdf_path)
                    os.remove(md5_path)
                    raise ValueError("Checksum verification failed")

                # Continue prompt
                if idx < num_to_download - 1:
                    response = input("\nDownload next file? (Y/n): ").strip().lower()
                    if response == 'n':
                        print("Stopping download.")
                        break

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    print("PubChem SDF Downloader")
    print("======================")
    download_sdf_files()