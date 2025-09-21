import time
import sys

import requests

from tqdm import tqdm

class Constants:
    DEFAULT_TIMEOUT = 300

def format_size(byte_size):
    """Converts bytes to a human-readable format (KB, MB, GB)."""
    if byte_size is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while byte_size > power and n < len(power_labels) -1 :
        byte_size /= power
        n += 1
    return f"{byte_size:.2f} {power_labels[n]}B"

def format_time(seconds):
    """Converts seconds into a HH:MM:SS format."""
    if seconds is None:
        return "N/A"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def download_file_verbose(url, local_filename):
    """Downloads a file from a URL with verbose progress output."""
    try:
        with requests.get(url, stream=True, timeout=Constants.DEFAULT_TIMEOUT) as r:
            r.raise_for_status()
            total_size = r.headers.get('content-length')
            if total_size is not None:
                total_size = int(total_size)
            downloaded_size = 0
            start_time = time.time()
            chunk_size = 8192 # 8 KB
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    elapsed_time = time.time() - start_time
                    speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                    remaining_size = total_size - downloaded_size if total_size is not None else None
                    eta = remaining_size / speed if speed > 0 and remaining_size is not None else None
                    percent_done = (downloaded_size / total_size * 100) if total_size is not None else 0
                    #Output construction
                    progress_bar_len = 30
                    filled_len = int(progress_bar_len * downloaded_size / total_size) if total_size is not None else 0
                    download_bar = '█' * filled_len + '-' * (progress_bar_len - filled_len)
                    status = (
                        f"\r[{download_bar}] {percent_done:.1f}% | "
                        f"{format_size(downloaded_size)}/{format_size(total_size)} | "
                        f"Speed: {format_size(speed)}/s | "
                        f"ETA: {format_time(eta)}"
                    )
                    sys.stdout.write(status)
                    sys.stdout.flush()
        print("\nDownload completed successfully.")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")
def download_with_tqdm(url, local_filename):
    """Downloads a file using tqdm for a progress bar."""
    try:
        with requests.get(url, stream=True, timeout=Constants.DEFAULT_TIMEOUT) as r:
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
