"""
Data downloader for UCI Diabetes 130-US Hospitals (1999-2008) dataset.
Source: UCI Machine Learning Repository
Direct archive: https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

def download_and_extract(target_dir: str = "data/raw") -> str:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = target_path / "diabetic_data.csv"
    if csv_file.exists():
        print(f"Dataset already exists at: {csv_file} ({csv_file.stat().st_size / (1024*1024):.2f} MB)")
        return str(csv_file)
        
    zip_path = target_path / "dataset.zip"
    print(f"Downloading UCI Diabetes dataset from: {DATA_URL} ...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(DATA_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
        data = resp.read()
        out_file.write(data)
        print(f"Downloaded {len(data) / (1024*1024):.2f} MB to {zip_path}")
        
    print("Extracting zip archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_path)
        
    # Check if diabetic_data.csv is extracted directly or inside a nested folder
    extracted_files = list(target_path.glob("**/diabetic_data.csv"))
    if not extracted_files:
        # Check if there is an inner zip file (sometimes UCI has dataset.zip containing another zip)
        for inner_zip in target_path.glob("*.zip"):
            if inner_zip != zip_path:
                print(f"Extracting nested zip: {inner_zip}")
                with zipfile.ZipFile(inner_zip, "r") as iz:
                    iz.extractall(target_path)
                    
    extracted_files = list(target_path.glob("**/diabetic_data.csv"))
    if extracted_files:
        main_csv = extracted_files[0]
        if main_csv != csv_file:
            import shutil
            shutil.move(str(main_csv), str(csv_file))
        print(f"Dataset ready at: {csv_file}")
        return str(csv_file)
    else:
        raise FileNotFoundError(f"diabetic_data.csv not found in extracted files in {target_path}")

if __name__ == "__main__":
    download_and_extract()
