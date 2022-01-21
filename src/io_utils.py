import pickle
import json
import constants
from utils import *

'''
Utilities to deal with IO-Operations. This includes writing the final data to a GEOJSON file.
'''


def dump_to_file(arr, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)
        fp.close()


def read_dump_from_file(filename):
    with open(filename, 'rb') as fp:
        arr = pickle.load(fp)
        fp.close()
        return arr


def write_to_GEOJSON(patches):
    data = {}
    crs = {'type': 'name', 'properties': {'name': 'EPSG:4326'}}
    data['type'] = 'FeatureCollection'
    data['crs'] = crs
    data['features'] = []

    patches_shape = get_patches_shape(patches)
    super_patch = create_super_patch(patches, patches_shape)

    dist_x = constants.point_dist / get_lat_fac() / 2.0

    final_shapes = shape_reduction(super_patch, dist_x, -1, patches_shape[0] * 10, patches_shape[1] * 10)
    final_shapes = remove_zero_shapes(final_shapes)

    print("Amount of Datapoints before: " + str(len(patches) * len(patches[0].dates)))
    print("Amount of Datapoints after: " + str(len(final_shapes)))

    for j in range(len(final_shapes)):
        new_cords = final_shapes[j][0]
        new_cords.append(new_cords[0])
        geom = {}
        props = {'color': 'rgba(0, 255, 0, ' + str(min(0.5 * final_shapes[j][1], 0.5)) + ')'}
        # props = {'color': 'rgba(0, 255, 0, ' + str(min(0.5 * super_patch[j][1], 0.5)) + ')'}
        geom['type'] = 'Polygon'
        # geom['coordinates'] = [coordinates]
        geom['coordinates'] = [new_cords]
        data['features'].append({
            'type': 'Feature',
            'geometry': geom,
            'properties': props
        })

    with open(constants.pwd + '/web/data.txt', 'w') as outfile:
        json.dump(data, outfile)
    return
