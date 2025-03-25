# pubchem_sdf_download.py
# Downloads PubChem Compound SDF files with MD5 verification

import os
import hashlib
from ftplib import FTP

# Configuration
FTP_SERVER = "ftp.ncbi.nlm.nih.gov"
FTP_DIR = "/pubchem/Compound/CURRENT-Full/SDF/"

def compute_md5(file_path):
    """Compute MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_sdf_files():
    try:
        # Connect to FTP
        ftp = FTP(FTP_SERVER)
        ftp.login()
        ftp.cwd(FTP_DIR)

        # List files
        files = ftp.nlst()
        sdf_files = [f for f in files
                     if f.endswith(".sdf.gz")
                     and not f.endswith(".md5")]

        for filename in sdf_files:
            sdf_path = os.path.join(DATA_DIR, filename)
            md5_path = f"{sdf_path}.md5"

            # Skip existing verified files
            if os.path.exists(sdf_path):
                print(f"Skipping existing file: {filename}")
                continue

            print(f"\nProcessing: {filename}")

            # Download SDF file
            print("  Downloading SDF...")
            with open(sdf_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)

            # Download MD5 file
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
                os.remove(md5_path)  # Clean up MD5 file
            else:
                print(f"  ❌ Checksum FAILED! Expected: {expected_md5}, Got: {actual_md5}")
                os.remove(sdf_path)
                os.remove(md5_path)
                raise ValueError("Checksum verification failed")

        ftp.quit()
        print(f"\nAll files downloaded to: {os.path.abspath(DATA_DIR)}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        ftp.quit()
        raise


if __name__ == "__main__":
    download_sdf_files()