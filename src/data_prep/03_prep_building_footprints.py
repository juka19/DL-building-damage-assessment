import argparse
import glob
import os
import zipfile

import fiona
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from utils.data_prep import extract_zipfile
from utils.data_prep import get_aoi_from_zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--zipfile_path', type=str, default='copernicus/EMSR648_AOI04_GRA_MONIT01_r1_VECTORS_v1_vector.zip')
parser.add_argument('--extract_dir', type=str, default='data/AOI04')
parser.add_argument('--dest_dir', type=str, default="dest_buildings/hotosm_tur_destroyed_buildings_polygons_geojson.geojson")
parser.add_argument('--out_dir', type=str, default="data/footprints/AOI04.geojson")
args = parser.parse_args()

def join_datasets(fps, dmg):
    """Join the building footprints and the damage datasets to map the damage information (points) to the building footprints (polygons)"""
    # filter for buildings that are not destroyed
    if 'destroyed:building' in fps.columns:
        fps_filt = fps.loc[['way'],:][fps['destroyed:building'] != 'yes'].reset_index()
        dmg = dmg[['geometry', 'damage_gra']]
        # join to nearest footprint
        dmg_points = dmg.sjoin_nearest(fps_filt, how='left')[['damage_gra', 'osmid']]
        dmg_polys = fps_filt.reset_index()[['osmid', 'geometry']].merge(dmg_points, how='left', on='osmid').replace(np.nan, 'no_label')
        fps_rest = fps.loc[['way'],:][fps['destroyed:building'] == 'yes'].reset_index()
        # add destroyed buildings from hotosm (destroyed:buildings == 'yes') that do not intersect with any of the existing buildings
        fps_non_intersect = fps_rest[~fps_rest.intersects(dmg_polys)]
        fps_non_intersect['damage_gra'] = 'Destroyed'
        dmg_builds = pd.concat([fps_non_intersect[['osmid', 'geometry', 'damage_gra']], dmg_polys])
    else:
        fps_filt = fps.loc[['way'],:].reset_index()
        dmg = dmg[['geometry', 'damage_gra']]
        # join to nearest footprint
        dmg_points = dmg.sjoin_nearest(fps_filt, how='left')[['damage_gra', 'osmid']]
        dmg_polys = fps_filt.reset_index()[['osmid', 'geometry']].merge(dmg_points, how='left', on='osmid').replace(np.nan, 'no_label')
        # add destroyed buildings from hotosm (destroyed:buildings == 'yes') that do not intersect with any of the existing buildings

        dmg_builds = pd.concat([dmg_polys])
    
    return dmg_builds


def read_dmg_file(dmg_dir):
    """Read the damage file and convert it to a geodataframe"""
    json_dmg = glob.glob(f'{dmg_dir}/*builtUp*.json')
    gdf = gpd.GeoDataFrame.from_file(json_dmg[0]).to_crs(epsg=4326)
    return gdf

def get_destroyed_buildings(dest_dir, aoi, fps):
    """Get the destroyed buildings from the hotosm dataset"""
    dest_b = gpd.GeoDataFrame().from_file(dest_dir)
    dest_b = dest_b[dest_b.within(aoi.loc[0, 'geometry'])]
    dest_b = dest_b[dest_b.disjoint(fps['geometry'].reset_index(drop=True))]
    dest_b['damage_gra'] = 'Destroyed'
    dest_b.rename(columns={'osmid': 'osm_id'}, inplace=True)
    return dest_b[['osm_id', 'geometry', 'damage_gra']]

def main():
    aoi = get_aoi_from_zipfile(args.zipfile_path, args.extract_dir)
    fps = ox.geometries.geometries_from_polygon(aoi.geometry[0], tags={'building':True}).dropna(axis=1, how='all') # get all buildings for aoi
    dmg = read_dmg_file(args.extract_dir)
    
    dest_b = get_destroyed_buildings(args.dest_dir, aoi, fps)
    
    buildings = join_datasets(fps, dmg)
    
    buildings.drop_duplicates(inplace=True)
    filt_buildings = buildings.loc[~buildings.intersects(dest_b.unary_union)].reset_index(drop=True)
    
    # adding destroyed buildings from hotosm. OSM does not have the footprints of destroyed buildings anymore.
    buildings_f = pd.concat([filt_buildings, dest_b]) 
    buildings_f.sort_values('damage_gra').drop_duplicates(subset='geometry', inplace=True)
    buildings_f.to_file('./' + args.out_dir, driver='GeoJSON', encoding='utf-8', engine='pyogrio')

if __name__ == '__main__':
    main()