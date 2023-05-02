import argparse
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
from urllib.request import urlopen

import pandas as pd
from numpy import nan

from utils.download_utils import download_tif_file

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_dir", type=str, help='Path to save tif files to', default='data')
parser.add_argument("--metadata", type=str, help='Path to metadata.csv', default='data/metadata.csv')
args = parser.parse_args()

def main():
    """Download the tif files from the links in the metadata.csv file"""
    metadata = pd.read_csv(args.metadata)
    
    assets = ['visual', 'ms_analytic', 'pan_analytic', 'data_mask', 'building_footprints', 'cloud_mask']
    
    links = []
    filepaths = []
    for asset in assets:
        for b, l in zip(metadata['base_url'], metadata[asset]):
            if l not in [None, nan, '']:
                links.append(os.path.dirname(b) + '/' + os.path.basename(l))
                filepaths.append(args.out_dir + '/' + asset + '/' + '-'.join(os.path.dirname(b).split('/')[-2:]) + os.path.basename(l))
                
    pool = Pool(cpu_count())
    results = pool.starmap(download_tif_file, zip(links, filepaths))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
