import pickle
import json
from pyproj import Proj

def dump_to_file(arr, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)
        fp.close()


def read_dump_from_file(filename):
    with open(filename, 'rb') as fp:
        arr = pickle.load(fp)
        fp.close()
        return arr


def project_coordinate(coordinate, projection):
    # Use projection on coordinate
    p = Proj(projection)
    return p(coordinate[0], coordinate[1])


def write_to_GEOJSON(patches):
    data = {}
    crs = {}
    crs['type'] = 'name'
    crs['properties'] = {'name': 'EPSG:4326'}
    data['type'] = 'FeatureCollection'
    data['crs'] = crs
    data['features'] = []
    counter = 0
    for patch in patches:
        coordinates = []
        coordinates.append([patch.corners[0][1], patch.corners[0][0]])
        coordinates.append([patch.corners[2][1], patch.corners[2][0]])
        coordinates.append([patch.corners[3][1], patch.corners[3][0]])
        coordinates.append([patch.corners[1][1], patch.corners[1][0]])
        coordinates.append([patch.corners[0][1], patch.corners[0][0]])
        prop = 0
        for date in patch.dates:
            prop += date.probabilities['Steinpilz'] / len(patch.dates)
        geom = {}
        print(prop)
        props = {'color' : 'rgba(0, 255, 0, ' + str(min(0.5 * prop, 0.5)) +')'}
        geom['type'] = 'Polygon'
        geom['coordinates'] = [coordinates]
        data['features'].append({
            'type': 'Feature',
            'geometry': geom,
            'properties': props
        })
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)
    return 0

#write_to_GEOJSON()