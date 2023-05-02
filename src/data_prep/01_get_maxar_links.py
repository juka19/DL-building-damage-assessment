import argparse
import json
import os
from urllib.request import urlopen

import pandas as pd


def get_jsonparsed_data(url):
    """Receive the content of ``url``, parse it as JSON and return the object."""
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_filename_from_url(url, vis, out_dir):
    """Create a filename from the url"""
    filename = out_dir + '/' + '-'.join(os.path.dirname(url).split('/')[-2:]) + os.path.basename(vis)
    return filename

def item_data_extract(item_data):
    """Extract the relevant data from the json file"""
            item_data['assets'].setdefault('ms_analytic', {'href': None})
            item_data['assets'].setdefault('pan_analytic', {'href': None})
            item_data['assets'].setdefault('building-footprints', {'href': None})
            item_data['assets'].setdefault('cloud-mask', {'href': None})
            item_data['assets'].setdefault('data_mask', {'href': None})
            data_dic = {
                'type': item_data['type'],
                'visual': item_data['assets']['visual']['href'],
                'ms_analytic': item_data['assets']['ms_analytic']['href'],
                'pan_analytic': item_data['assets']['pan_analytic']['href'],
                'data_mask': item_data['assets']['data-mask']['href'],
                'collection': item_data['collection'],
                'utm_zone': item_data['properties']['utm_zone'],
                'quadkey': item_data['properties']['quadkey'],
                'clouds_cover': item_data['properties']['tile:clouds_percent'],
                'off_nadir': item_data['properties']['view:off_nadir'],
                'epsg': item_data['properties']['proj:epsg'],
                'coordinates': item_data['properties']['proj:geometry']['coordinates'],
                'grid:code': item_data['properties']['grid:code'],
                'bbox': item_data['properties']['proj:bbox'],
                'sun_elevation': item_data['properties']['view:sun_elevation'],
                'sun_azimuth': item_data['properties']['view:sun_azimuth'],
                'azimuth': item_data['properties']['view:azimuth'],
                'incidence_angle': item_data['properties']['view:incidence_angle'],
                'building_footprints': item_data['assets']['building-footprints']['href'],
                'cloud_mask': item_data['assets']['cloud-mask']['href']
            }
            return data_dic

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_dir", type=str, help='Path to save metadata.csv', default='data')
args = parser.parse_args()

def main():
    """Parse the metadata from the maxar website and save it to a csv file"""
    if not os.path.exists(args.out_dir + '/metadata.csv'):
        BASE_URL = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/"

        collection_links = [l['href'][2:] for l in get_jsonparsed_data(BASE_URL + "collection.json")['links'] if l['rel'] == 'child']

        data = []
        cn = 0
        for link in collection_links:
            cn += 1
            collection = get_jsonparsed_data(BASE_URL + link)
            item_links = [l['href'][2:] for l in collection['links'] if l['rel'] == 'item']
            i_n = 0
            for item_link in item_links:
                i_n += 1
                print(f"Parsing metadata for item {i_n}/{len(item_links)} of collection {cn}/{len(collection_links)}")
                try:
                    item_data = get_jsonparsed_data(BASE_URL + 'ard' + item_link)
                    data_dic = item_data_extract(item_data)
                    data_dic['base_url'] = BASE_URL + 'ard' + item_link
                    data.append(data_dic)
                except Exception as e:
                    print(f"Could not parse metadata for {BASE_URL + 'ard' + item_link}. ERROR:")
                    print(e)
                    continue
        metadata = pd.DataFrame.from_records(data)
        metadata['date'] = pd.to_datetime(metadata['base_url'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
        metadata['file'] = [get_filename_from_url(url, vis) for url, vis in zip(metadata.base_url, metadata.visual)]
        metadata.to_csv(args.out_dir + '/metadata.csv')
    else: print(f"{args.out_dir + '/metadata.csv'} already exists, links already parsed.")

if __name__ == '__main__':
    main()