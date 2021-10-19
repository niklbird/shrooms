import pickle
import json
from pyproj import Proj
from numba import jit
import numpy as np
import constants


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


@jit(nopython=True)
def get_lat_fac():
    # Translate angle distance to km distance latitudial
    return 110.574


@jit(nopython=True)
def get_long_fac(longitude):
    # Translate angle distance to km distance longitudinal
    return np.abs(111.32 * np.cos(longitude))


def write_to_GEOJSON(patches):
    data = {}
    crs = {}
    crs['type'] = 'name'
    crs['properties'] = {'name': 'EPSG:4326'}
    data['type'] = 'FeatureCollection'
    data['crs'] = crs
    data['features'] = []
    print("Amount of dates: " + str(len(patches) * len(patches[0].dates)))
    for patch in patches:
        corner = patch.corners[0]
        dist_x = constants.point_dist / get_long_fac(corner[0]) / 2.0
        dist_y = constants.point_dist / get_lat_fac() / 2.0

        for date in patch.dates:
            point = date.coord
            coordinates = []
            coordinates.append([point[1] + dist_y, point[0] + dist_x])
            coordinates.append([point[1] - dist_y, point[0] + dist_x])
            coordinates.append([point[1] - dist_y, point[0] - dist_x])
            coordinates.append([point[1] + dist_y, point[0] - dist_x])
            coordinates.append([point[1] + dist_y, point[0] + dist_x])

            prop = date.probabilities['Steinpilz']

            geom = {}
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
#coordinates.append([patch.corners[0][1], patch.corners[0][0]])
#coordinates.append([patch.corners[2][1], patch.corners[2][0]])
#coordinates.append([patch.corners[3][1], patch.corners[3][0]])
#coordinates.append([patch.corners[1][1], patch.corners[1][0]])
#coordinates.append([patch.corners[0][1], patch.corners[0][0]])
#for date in patch.dates:
    #prop += date.probabilities['Steinpilz'] / len(patch.dates)