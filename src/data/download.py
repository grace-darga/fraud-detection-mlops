import os
import subprocess
from pathlib import Path

def download_dataset():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Téléchargement du dataset creditcardfraud...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "mlg-ulb/creditcardfraud",
        "-p", str(output_dir),
        "--unzip"
    ], check=True)
    print(f"✅ Dataset téléchargé dans {output_dir}")

if __name__ == "__main__":
    download_dataset()