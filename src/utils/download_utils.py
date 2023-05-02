import os
from urllib.request import urlopen

def download_tif_file(url: str, out_dir: str):
    """
    Download a tif file from a url
    """
    if os.path.exists(out_dir):
        print(f"File {os.path.basename(url)} already exists, skipping download")
    else:
        print(f"Downloading {url} to {out_dir}")
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        try:
            response = urlopen(url)
            with open(out_dir,'wb') as f:
                f.write(response.read())
        except Exception as e:
            print(f"Error downloading {url}: {e}")