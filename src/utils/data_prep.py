import os
import glob
import zipfile

import geopandas as gpd
from osgeo import gdal
import numpy as np

def extract_zipfile(zipfile_path, extract_dir):
    """Extract the Copernicus zipfile to a directory"""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        # Check if all files are already extracted
        all_files_extracted = all(os.path.exists(os.path.join(extract_dir, f)) for f in zip_ref.namelist())
        if not all_files_extracted:
            zip_ref.extractall(extract_dir)
            print(f'{zipfile_path} successfully extracted to {extract_dir}.')
        else:
            print(f'All files in {zipfile_path} already extracted to {extract_dir}.')

def get_aoi_from_zipfile(zipfile_path, extract_dir):
    """Extract the Copernicus zipfile to a directory and return the area of interest (aoi) as a geodataframe"""
    extract_zipfile(zipfile_path, extract_dir)
    json_aoi = glob.glob(f'{extract_dir}/*areaOfInterest*.json')
    gdf = gpd.GeoDataFrame.from_file(json_aoi[0]).to_crs(epsg=4326)
    return gdf

def write_geotiff(filename, arr, in_ds):
    """Write a geotiff from a numpy array"""
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

def write_filenames(filenames, out_dir):
    """Write the filenames to a text file. This is required for the MS4DNet model input. 
    Files will be further divided according to MS4DNet specifications by the split_files.py script.
    """
    pre, post, mask = filenames
    with open(f'{out_dir}/pre_images.txt', 'a') as f:
        f.write(f"{os.getcwd()}/{pre}"+"\n")
    with open(f'{out_dir}/post_images.txt', 'a') as f:
        f.write(f"{os.getcwd()}/{post}"+"\n")
    with open(f'{out_dir}/labels.txt', 'a') as f:
        f.write(f"{os.getcwd()}/{mask}"+"\n")

class ImageExistsExc(Exception):
    """Exception raised when an image already exists in the output folder."""
    def __init__(self, message):
        super(ImageExistsExc, self).__init__(message)


def get_filename_from_url(url, vis, img_folder):
    filename = img_folder + '/' + '-'.join(os.path.dirname(url).split('/')[-2:]) + os.path.basename(vis)
    return filename
