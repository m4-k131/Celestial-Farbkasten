import os
import argparse
import requests

from tqdm import tqdm

import pandas as pd


BASE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
DEFAULT_TIMEOUT = 300


def download_with_tqdm(url, local_filename):
    """Downloads a file using tqdm for a progress bar."""
    headers = {
        "Authorization": f"token 9bc4249132e7467e851c6d24f86bc96b"
    }
    try:
        with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT, headers=headers) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 1024 # 1 KB
            with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as download_bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    download_bar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def main(csv, download_jpgs=True, download_fits=True, outdir = None, must_contain=None):
    if outdir is None:
        outdir=csv.split("/")[:-4]
    os.makedirs(outdir, exist_ok=True)
    print(f"Reading observation data from '{csv}'...")
    df = pd.read_csv(csv)
    print(f"Starting download of {len(df) * 2} files...")
    for index, row in df.iterrows():
        uris_to_download = []
        if download_jpgs:
            uris_to_download.append(row['jpegURL'])
        if download_fits:
            uris_to_download.append(row['dataURL'])

        for uri in uris_to_download:
            if pd.notna(uri):
                if must_contain is not None:
                    if not uri.contains(str(must_contain)):
                        continue
                download_url = f"{BASE_URL}{uri}"
                filename = uri.split('/')[-1]
                local_filepath = os.path.join(outdir, filename)
                if not os.path.isfile(local_filepath):     
                    download_with_tqdm(download_url, local_filepath)              
                else:
                    print(f"{local_filepath} already exists")

    print(f"\n✅ Download complete. Files are in the '{outdir}' folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=False)
    parser.add_argument("--ignore_jpgs", action="store_true")
    parser.add_argument("--ignore_fits", action="store_true")
    parser.add_argument("--must_contain", required=False)
    args = parser.parse_args()
    main(args.csv, not args.ignore_jpgs, not args.ignore_fits, args.outdir)
