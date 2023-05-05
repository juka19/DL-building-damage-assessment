conda env create -n thesis python=3.11 --file environment.yml 
source activate thesis
# conda install -f environment.yml

# execute scripts
python src/data_prep/01_get_maxar_links.py --out_dir $1
python src/data_prep/02_download_tifs.py --metadata $1/data/metadata.csv --out_dir $1/tifs
for area in $(ls copernicus)
do
    aoi='(AOI.{2})'
    if [[ $area =~ $aoi ]]
    then
        aoi_name=${BASH_REMATCH[1]}
    fi
    python src/data_prep/03_prep_building_footprints.py --zipfile_path copernicus/$area --extract_dir $1/$aoi_name --dest_dir dest_buildings/hotosm_tur_destroyed_buildings_polygons_geojson.geojson --out_dir $1/footprints/$aoi_name.geojson
    python src/data_prep/04_raster_processing.py --metadata $1/data/metadata_gdf.geojson --aoi_zipfile copernicus/$area --footprint_file $1/data/footprints/$aoi_name.geojson --out_dir $1/$aoi_name --extract_dir $1/$aoi_name --img_folder $1/tifs/visual/ --data_mask_dir $1/masks/ --root_dir $1 --data_tifs $1/data_tifs
done
