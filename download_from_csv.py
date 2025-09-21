import os

import pandas as pd

from download_utils import download_with_tqdm

csv_filename = "OMC2-SE.csv"
output_dir = csv_filename[:-4]

os.makedirs(output_dir, exist_ok=True)

print(f"Reading observation data from '{csv_filename}'...")
df = pd.read_csv(csv_filename)
#df = df[df['obs_id'].str.contains("o039")]

base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

print(f"Starting download of {len(df) * 2} files...")

for index, row in df.iterrows():
    uris_to_download = [row['dataURL'], row['jpegURL']]
    jpg_uris_to_download = [row['jpegURL']]
    for uri in jpg_uris_to_download:
        if pd.notna(uri):
            download_url = f"{base_url}{uri}"
            filename = uri.split('/')[-1]
            local_filepath = os.path.join(output_dir, filename)
            if not os.path.isfile(local_filepath):     
                download_with_tqdm(download_url, local_filepath)
                """
                       
                print(f"  -> Downloading {filename}...")
                try:
                    with requests.get(download_url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_filepath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"     ERROR downloading {filename}: {e}")
                """
            
            else:
                print(f"{local_filepath} already exists")

print(f"\n✅ Download complete. Files are in the '{output_dir}' folder.")
