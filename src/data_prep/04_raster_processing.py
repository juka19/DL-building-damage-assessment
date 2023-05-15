import argparse
import ast
import glob
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
from affine import Affine
from PIL import Image
from pyproj import CRS
from rasterio.merge import merge
from rasterio.windows import Window, get_data_window, shape, transform
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split
from sympy import im
from tqdm import tqdm
from utils.data_prep import (ImageExistsExc, get_aoi_from_zipfile,
                             get_filename_from_url, write_filenames,
                             write_geotiff)

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=str, default='data/metadata_gdf.geojson')
parser.add_argument('--aoi_zipfile', type=str, default='copernicus/EMSR648_AOI04_GRA_MONIT01_r1_VECTORS_v1_vector.zip')
parser.add_argument('--footprint_file', type=str, default='data/footprints/AOI04.geojson')
parser.add_argument('--extract_dir', type=str, default='data/AOI04')
parser.add_argument('--img_folder', type=str, default='data/tifs/visual/')
parser.add_argument('--out_dir', type=str, default='data/AOI04/subset2')
parser.add_argument('--data_tifs', type=str, default='data/data_tifs/')
parser.add_argument('--root_dir', type=str, default='/media/julian/Extreme Pro/')
parser.add_argument('--data_mask_dir', type=str, default='data/tifs/data_mask/')
args = parser.parse_args()

# Im gona be burtally honst here, processing the raster files is pretty tricky, memory intensive, and you need to know a lot about the data you are working with.
# I would not reccomend running this script blindly from the command line, but rather to use it as a reference for how to process the data.
# The script is not very well documented, and I appologize for that, but I will try to explain what is going on here:
# The main purpose of the scripts is to read in the raster files overlay them with eachother, finding the intersection between the pre and post even iamges, 
# the building footpritns and the area of interst (AOI) in which th damage information is mapped
# The script will unify all of those to the same CRS, overlay them, crop where neccessary, and then process them one by one to create a final dataset that is composed 
# 512x512 images, and the corresponding segmentation masks, that encodes the damage levels. Finally, the script also create a train/test split of the data, and saves the
# filenames to a csv file, so that the data can be loaded from a dataloader in pytorch.
# The raster files are very large files, so you might want to free up some memory (may take up to 14gb of RAM) before running this script.

metadata = args.metadata
aoi_zipfile = args.aoi_zipfile
data_mask_dir = args.data_mask_dir
data_tif_dir = args.data_tifs
root_dir = args.root_dir
out_dir = args.out_dir
img_folder = args.img_folder
extract_dir = args.extract_dir
footprint_file = args.footprint_file


def check_crs(file):
    # Check the CRS of the file
    with rio.open(file) as src:
        src_crs = src.crs
        src_bounds = box(*src.bounds)
        return src_bounds


def binary_mask(polys, window_transform, r_shape):
    # Create a binary mask from the polygons within the window. (Not really binary, you get what I mean, one value for each damage grade)
    value_mapp = {'no_label': 1, 'No visible damage': 1, 'Possibly damaged': 2, 'Damaged': 3, 'Destroyed': 4}
    mask = np.zeros((r_shape[0], r_shape[1]))
    for _, row in polys.iterrows():
        color_val = value_mapp[row['damage_gra']]
        poly_mask = rio.features.rasterize(shapes=[row['geometry']], out_shape=r_shape, transform=window_transform) * color_val
        np.putmask(mask, mask==0, poly_mask)
        if mask.max() > 4:
            raise ValueError('Mask value is greater than 4')
    return mask



def process_collections(col):
    os.makedirs(root_dir + '/out_images/' + col, exist_ok=True)
    os.makedirs(root_dir + '/out_images/' + col + '/pre_images', exist_ok=True)
    os.makedirs(root_dir + '/out_images/' + col + '/post_images', exist_ok=True)
    os.makedirs(root_dir + '/out_images/' + col + '/masks', exist_ok=True)
    with rio.open(root_dir + '/merged_tifs/' + f'pre_tif.tif') as src:
        pre_profile = src.profile.copy()
        pre_meta = src.meta.copy()
        pre_bounds = box(*src.bounds)
        n_rows = src.width // 512
        n_cols = src.height // 512
        for i in range(1, n_rows-1):
            for j in range(1, n_cols-1):
                window = Window(col_off=j*512, row_off=i*512, width=512, height=512)
                window_transform = src.window_transform(window)
                win_bbox = src.window_bounds(window)
                data_pre = src.read(window=window)
                if data_pre.shape != (3, 512, 512): continue
                if not data_pre.any(): continue
                test = {tif:check_crs(root_dir + f'/data_tifs/{col}/' + tif) for tif in os.listdir(root_dir + f'/data_tifs/{col}') if tif.endswith('visual_cropped.tif')}
                tifs = [i[0] for i in test.items() if box(*win_bbox).within(i[1])]
                if len(tifs) == 0: continue
                tif_name = tifs[0]
                with rio.open(root_dir + '/data_tifs/' + f'{col}/' + tif_name) as src2:
                    read_win = rio.windows.from_bounds(*win_bbox, src2.transform)
                    data = src2.read(window=read_win)
                    if data.shape != (3, 512, 512): continue
                    if not data.any(): continue
                    polys = fps_descr.cx[win_bbox[0]:win_bbox[2], win_bbox[1]:win_bbox[3]].copy()
                    if polys[polys['damage_gra'] != 'no_label'].empty: 
                        dam = '_dam'
                    else: 
                        dam = ''
                    if len(polys) == 0: continue
                    pre_name = root_dir + '/out_images/' + col + '/pre_images/' + 'pre_' + f'{col}_{i}_{j}{dam}.png'
                    post_name = root_dir + '/out_images/' + col + '/post_images/' + 'post_' + f'{col}_{i}_{j}{dam}.png'
                    mask_name = root_dir + '/out_images/' + col + '/masks/' + 'mask_' + f'{col}_{i}_{j}{dam}.png'
                    
                    bin_mask=binary_mask(polys, window_transform, data.shape[1:])
                    Image.fromarray(bin_mask.astype('uint8')).save(mask_name)
                    Image.fromarray(data.transpose((1, 2, 0)), 'RGB').save(post_name)
                    Image.fromarray(data_pre.transpose((1, 2, 0)), 'RGB').save(pre_name)


def write_file_list(file_list, output_file):
    with open(output_file, "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")



def create_train_test_val_split(directory_path, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_seed=42):
    file_list = os.listdir(directory_path)
    dir_n = Path(directory_path).stem
    save_dir = str(Path(directory_path).parent)
    data = []
    for file in file_list: 
        if file.endswith('dam.png'):
            data.append({'y': 'dam', 'X': file})
        else:
            data.append({'y': 'no_dam', 'X': file})
            
    data_df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(data_df['X'], data_df['y'], train_size=train_ratio, random_state=random_seed,
                                                            stratify=data_df['y'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=test_ratio / (test_ratio + val_ratio),
                                                        random_state=random_seed, stratify=y_train)
    
    
    # Write file lists to text files
    write_file_list(X_train, f"{save_dir}/{dir_n}_train.txt")
    write_file_list(X_test, f"{save_dir}/{dir_n}_test.txt")
    write_file_list(X_val, f"{save_dir}/{dir_n}_val.txt")

    print("Split completed. Train files: {}, Test files: {}, Validation files: {}".format(len(X_train),
                                                                                          len(X_test),
                                                                                          len(X_val)))

def main():
    
    if not os.path.exists(args.metadata):
        metadata = pd.read_csv(args.out_dir + '/metadata.csv')
        metadata = unify_crs(metadata)
        metadata.to_file(args.metadata, driver='GeoJSON')

    metadata = gpd.GeoDataFrame.from_file(args.metadata)
    metadata['date'] = pd.to_datetime(metadata['base_url'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    metadata['file'] = [get_filename_from_url(url, vis, args.img_folder) for url, vis in zip(metadata.base_url, metadata.visual)]
    event_date = datetime.strptime('2023-02-07', '%Y-%m-%d')
    metadata['time'] = ['pre' if i < event_date else 'post' for i in metadata.date]
    
    aoi = get_aoi_from_zipfile(args.aoi_zipfile, args.extract_dir)
    # some building are on the edge of the aoi, so we need to buffer the aoi
    aoi_buf = aoi.buffer(0.0001)
    
    metadata = metadata[metadata.intersects(aoi_buf.geometry[0])]
    pres = metadata[metadata.time == 'pre'].copy()
    posts = metadata[metadata.time == 'post'].copy()
    
    fps = gpd.read_file(args.footprint_file)
    fps = fps.to_crs(32637)
    aoi_buf = aoi_buf.to_crs(32637)
    
    # crop pre image to Copernicus EMS area of interest (aoi)
    for i, row in pres.iterrows():
        with rio.open(row['file']) as src:
            profile = src.profile.copy()
            data_window = get_data_window(
                src.read(masked=True)
                )
            data_transform = transform(data_window, src.transform)
            data = src.read(window=data_window)
            profile.update(
                transform=data_transform,
                height=data_window.height,
                width=data_window.width)
        with rio.open(data_tif_dir + f'{Path(row["file"]).stem}_cropped.tif', 'w', **profile) as dst:
            dst.write(data)
        with rio.open(data_tif_dir + f'{Path(row["file"]).stem}_cropped.tif') as src:
            
            data = src.read()
            out_image, out_transform = rio.mask.mask(src, [aoi_buf.geometry[0]], crop=True)
            
    # create unified post-disaster image. Requires a lot of free memory!
    os.makedirs(root_dir + 'merged_tifs', exist_ok=True)
    
    raster_to_mosaic = []
    for p in os.listdir(data_tif_dir):
        raster = rio.open(data_tif_dir + p)
        raster_to_mosaic.append(raster)
    mosaic, output = merge(raster_to_mosaic)

    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        }
    )
    with rio.open(root_dir + 'merged_tifs/pre_tif.tif', "w", **output_meta) as m:
        m.write(mosaic)

    # process post-event images
    
    # get data_masks for post tifs (these are very large files, so difficult to process without further masking)
    data_masks = {}
    for col in posts.collection.unique():
        data_masks[col] = []
        for i, row in posts[posts.collection == col].iterrows():
            data_masks[col].append(row['quadkey'])
            
    
    for col in posts.collection.unique():
        dmask_l = []
        for qk in data_masks[col]:
            dmask = glob.glob(data_mask_dir + f'0{qk}*{col}*.gpkg')[0]
            dmask_l.append(gpd.read_file(dmask))
        data_masks[col] = unary_union(gpd.GeoDataFrame(pd.concat(dmask_l, ignore_index=True)))
        
        
    # crop post images to data_mask
    for col in tqdm(posts.collection.unique()):
        os.makedirs(data_tif_dir + col, exist_ok=True)
        col_dir = data_tif_dir + col + '/'
        for i, row in posts[posts.collection == col].iterrows():
            with rio.open(row['file']) as src:
                data = src.read()
                profile = src.profile.copy()
                out_image, out_transform = rio.mask.mask(src, [data_masks[col]], crop=True, nodata=0)
                profile.update(
                    transform=out_transform,
                    height=out_image.shape[1],
                    width=out_image.shape[2]
                )
            with rio.open(col_dir + f'/{Path(row["file"]).stem}_cropped.tif', 'w', **profile) as dst:
                dst.write(out_image)
                
                
    # with rio.open(root_dir + '/merged_tifs/' + f'pre_tif.tif') as src:
    #     pre_profile = src.profile.copy()
    #     pre_meta = src.meta.copy()
    #     pre_bounds = box(*src.bounds)
    
    # descriptivbe stats for post-event image
    
    pre_bounds = check_crs(root_dir + '/merged_tifs/' + f'pre_tif.tif')
    
    # get the building footpritns that intersect with the pre-event image
    fps_descr = fps.to_crs(32637)
    # filter for those that intersect with the pre-event image
    fps_descr_pre_img = fps_descr[fps_descr.intersects(pre_bounds)]
    
    fps_descr_pre_img.damage_gra.value_counts()
    
    # lets look at the post-event images and get some descriptive stats
    for col in tqdm(posts.collection.unique()):
        bound_list = []
        for i, row in posts[posts.collection == col].iterrows():
            bound_list.append(check_crs(data_tif_dir + col + '/' + f'{Path(row["file"]).stem}_cropped.tif'))
        bbox_covered = pre_bounds.intersection(unary_union(bound_list))
        fps_descr_pre_img = fps_descr[fps_descr.intersects(bbox_covered)]
        print(col)
        print(f"Date: {posts[posts.collection == col].date.unique()}")
        print(f"Off-Nadir: {np.nanmean(posts[posts.collection == col].off_nadir.unique())}")
        print(f"Sun Azimuth: {posts[posts.collection == col].sun_azimuth.unique()}")
        print(f"Azimuth: {posts[posts.collection == col].azimuth.unique()}")
        print(f"Sun elevation: {np.nanmean(posts[posts.collection == col].sun_elevation.unique())}")
        print(fps_descr_pre_img.damage_gra.value_counts())
        
    os.makedirs(args.out_dir + '/out_images', exist_ok=True)
    
    # creates all pre, post and mask image tiles for each collection
    cols = posts.collection.unique()
    for col in col:
        posts_crs = posts[posts['collection'] == col]
        posts_crs = posts_crs.to_crs(32637)
        process_collections(col)
    
    # create train/test/val split for all collections
    for direc in os.listdir(root_dir + "/out_images/"):
        directory_path = root_dir + "/out_images/" + direc
        for direc2 in os.listdir(directory_path):
            directory_path = root_dir + "/out_images/" + direc + "/" + direc2
            create_train_test_val_split(directory_path, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, random_seed=42)

if __name__ == '__main__':
    main() # pray for no errors
