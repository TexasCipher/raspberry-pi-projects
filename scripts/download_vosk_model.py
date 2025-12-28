#!/usr/bin/env python3
"""
Download and extract a VOSK model (small English by default) into `models/`.

Usage:
  python scripts/download_vosk_model.py

This script downloads the model zip and extracts it to models/<model-name>.
"""
import os
import sys
import argparse
import shutil
from urllib.request import urlopen
from urllib.error import URLError
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIRNAME = "vosk-model-small-en-us-0.15"


def download(url, out_path):
    print(f"Downloading {url} -> {out_path}")
    try:
        with urlopen(url) as resp, open(out_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    except URLError as e:
        print("Download failed:", e)
        sys.exit(1)


def extract(zip_path, target_dir):
    print(f"Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=MODEL_URL)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_name = os.path.join(args.out_dir, os.path.basename(args.url))
    if not os.path.exists(zip_name):
        download(args.url, zip_name)
    else:
        print("Zip already downloaded:", zip_name)

    # extract
    extract(zip_name, args.out_dir)

    final_path = os.path.join(args.out_dir, MODEL_DIRNAME)
    if os.path.exists(final_path):
        print("Model available at:", final_path)
    else:
        print("Extraction finished but model folder not found. Check the zip contents.")


if __name__ == "__main__":
    main()
