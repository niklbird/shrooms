import pickle
import json
import reparse_utils
from utils import *
import os
import math

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


def clear_directory(directory):
    filelist = [f for f in os.listdir(directory) if f.endswith(".dump")]
    for f in filelist:
        os.remove(os.path.join(directory, f))


def get_dumpamount_in_folder(directory):
    return len([f for f in os.listdir(directory) if f.endswith(".dump")])


def patches_to_folder(patches):
    clear_directory(constants.pwd + "/data/dumps/patches/")
    for i in range(len(patches)):
        patches_t = patches[i]
        io_utils.dump_to_file(patches_t, constants.pwd + "/data/dumps/patches/patches_weather" + str(i) + ".dump")


def generate_file_names(len_patches):
    return ["/data/dumps/patches/patches_weather" + str(i) + ".dump" for i in range(len_patches)]


def read_patches_from_folder(directory):
    filelist = [f for f in os.listdir(directory) if f.endswith(".dump")]
    patches = []
    for f in filelist:
        patches.append(io_utils.read_dump_from_file(os.path.join(directory, f)))
    return patches


def flatten_patches(patches):
    return [item for sublist in patches for item in sublist]


def make_shapes_grainy(shapes):
    max_sizes = reparse_utils.find_max_size_shapes(np.array(shapes))
    ret = []
    for i in range(len(shapes)):
        shape = shapes[i][0]
        size = max_sizes[i]
        if size > 20:

            ret.append(shapes[i])
    print("Amount of Datapoints grainy: " + str(len(ret)))
    return ret



def subdivide_patches(patches, shape_amount_sqrt):
    patches_shape = get_patches_shape(patches)

    x_split = math.ceil(patches_shape[0] / shape_amount_sqrt)
    y_split = math.ceil(patches_shape[1] / shape_amount_sqrt)

    add_x = patches_shape[0] / x_split
    add_y = patches_shape[1] / y_split

    return_arr = [[] for i in range(x_split * y_split)]
    for y in range(patches_shape[1]):
        for x in range(patches_shape[0]):
            index = x + y * patches_shape[0]

            x_target = int(x / add_x)
            y_target = int(y / add_y)

            target = x_target + y_target * x_split

            #print(f"Target: {target} index {index}")
            return_arr[target].append(patches[index])
    print(get_patches_shape(return_arr[0]))
    print(get_patches_shape(return_arr[1]))
    return return_arr

def write_to_GEOJSON(patches_a):
    print(f"Patcheese length: {len(patches_a)}")
    patches_shape = get_patches_shape(patches_a)
    patches_d = subdivide_patches(patches_a, 100)
    print(patches_shape)
    for i in range(len(patches_d)):
        patches = patches_d[i]
        #patches = patches_a[i:min(i + 5000, len(patches_a))]

        data = {}
        crs = {'type': 'name', 'properties': {'name': 'EPSG:4326'}}
        data['type'] = 'FeatureCollection'
        data['crs'] = crs
        data['features'] = []

        data_grainy = {}
        crs2 = {'type': 'name', 'properties': {'name': 'EPSG:4326'}}
        data_grainy['type'] = 'FeatureCollection'
        data_grainy['crs'] = crs2
        data_grainy['features'] = []

        patches_shape = get_patches_shape(patches)
        super_patch = create_super_patch(patches, patches_shape)

        dist_x = constants.point_dist / get_lat_fac() / 2.0

        print("Reducing Amount of Shapes")
        final_shapes = shape_reduction(super_patch, dist_x, -1, patches_shape[0] * 10, patches_shape[1] * 10)

        print("Removing Shapes with Probability 0")
        final_shapes = remove_zero_shapes(final_shapes)

        print("Amount of Datapoints before: " + str(len(patches) * len(patches[0].dates)))
        print("Amount of Datapoints after: " + str(len(final_shapes)))
        grainy_shapes = make_shapes_grainy(final_shapes)
        for j in range(len(final_shapes)):
            new_cords = final_shapes[j][0]
            new_cords.append(new_cords[0])
            geom = {}
            #props = {'color': 'rgba(0, ' + str(random.randint(0, 255)) + ', ' + str(random.randint(0, 255)) + ', ' + str(random.randint(0, 255)) + ')'}
            props = {'color': 'rgba(0, 255, 0 , ' + str(min(0.5 * final_shapes[j][1], 0.5)) + ')'}
            geom['type'] = 'Polygon'
            # geom['coordinates'] = [coordinates]
            geom['coordinates'] = [new_cords]
            data['features'].append({
                'type': 'Feature',
                'geometry': geom,
                'properties': props
            })

        for j in range(len(grainy_shapes)):
            new_cords = grainy_shapes[j][0]
            new_cords.append(new_cords[0])
            geom = {}
            #props = {'color': 'rgba(0, 255, ' + str(random.randint(0, 255)) + ', ' + str(random.randint(0, 255)) + ')'}
            props = {'color': 'rgba(0, 255, 0, ' + str(min(0.5 * grainy_shapes[j][1], 0.5)) + ')'}
            geom['type'] = 'Polygon'
            # geom['coordinates'] = [coordinates]
            geom['coordinates'] = [new_cords]
            data_grainy['features'].append({
                'type': 'Feature',
                'geometry': geom,
                'properties': props
            })

        with open(constants.pwd + f'/web/data{i}.json', 'w') as outfile:
            json.dump(data, outfile)
        with open(constants.pwd + f'/web/data_grainy{i}.json', 'w') as outfile:
            json.dump(data_grainy, outfile)
