import pickle
import json
import reparse_utils
from utils import *
import os
import math
import datetime
import json
import security_utils
import soil
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


'''
Only take shapes that have a certain size to display them at larger zoom-levels
'''
def make_shapes_grainy(shapes):
    max_sizes = reparse_utils.find_max_size_shapes(np.array(shapes))
    ret = []
    for i in range(len(shapes)):
        shape = shapes[i][0]
        size = max_sizes[i]
        if size > 9:

            ret.append(shapes[i])
    print("Amount of Datapoints grainy: " + str(len(ret)))
    return ret


'''
Remove unnecessary points of a shape, i.e. points that lie on a line
'''
def remove_points(final_shapes):
    # Remove unnecessary points from the shapes
    new_shapes = []
    for shape in final_shapes:
        excluded_points = []
        points = shape[0]
        for i in range(1, len(points) - 2):
            point = points[i]
            if point[0] == points[i + 1][0] == points[i + 2][0] or \
                    point[1] == points[i + 1][1] == points[i + 2][1]:
                excluded_points.append(i + 1)
        new_points = []
        for i in range(len(points)):
            if i not in excluded_points:
                new_points.append(points[i])
        new_shapes.append([new_points, shape[1], shape[2]])
    return new_shapes


'''
Divide one larger patch in multiple smaller once.
Not trivial as patches should remain rectangular.
'''
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

            return_arr[target].append(patches[index])

    return return_arr


'''
Update probabilities with the update-data
'''
def update_probs(patch, shapes):
    for i in range(len(shapes)):
        ind = shapes[i][2][0]
        p = patch[ind][1]
        shapes[i][1] = p
    return shapes


'''
Generate the update-file for the phone-app. 
This file data is signed.
'''
def generate_app_update(update, update_grainy):
    sig_u = security_utils.sign_data("./res/private.key", update.encode("utf8"))
    sig_g = security_utils.sign_data("./res/private.key", update_grainy.encode("utf8"))

    json_o = {"update": {
                    "data": update,
                    "signature": sig_u.decode("utf8")
                    },
                "update_grainy": {
                    "data": update_grainy,
                    "signature": sig_g.decode("utf8")
                }
              }

    json_s = json.dumps(json_o)
    return json_s


def get_short_name(db, soil):
    return db[soil.lower()].attr['short'].title()

def write_to_GEOJSON(patches_a, reparse, reduce_shapes=True):
    patches_shape = get_patches_shape(patches_a)

    soil_db = soil.read_soil_XML('../data/soil_databank.xml')

    # Subdivide patches to ease calculation
    # This is necessary as the shape-reduction algorithm works recursively
    # Calculating to many shapes at ones can lead to a stack-overflow or very long calculation-times
    patches_d = subdivide_patches(patches_a, 100)

    for i in range(len(patches_d)):
        patches = patches_d[i]

        # Create file in GEOJSON format
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
        if not reparse:
            patches_shape = get_patches_shape(patches)
            super_patch = create_super_patch(patches, patches_shape)
            shapes_tmp = read_dump_from_file(constants.pwd + f"/data/tmp/tmp-shapes{i}.dump")
            final_shapes = update_probs(super_patch, shapes_tmp)

        elif reduce_shapes:
            patches_shape = get_patches_shape(patches)
            super_patch = create_super_patch(patches, patches_shape)

            dist_x = constants.point_dist / get_lat_fac() / 2.0

            print("Reducing Amount of Shapes")
            final_shapes = shape_reduction(super_patch, dist_x, -1, patches_shape[0] * 10, patches_shape[1] * 10)

            final_shapes = remove_zero_shapes(final_shapes)

            print("Amount of Datapoints before: " + str(len(patches) * len(patches[0].dates)))
            print("Amount of Datapoints after: " + str(len(final_shapes)))

        else:
            final_shapes = patches

        final_shapes = remove_points(final_shapes)
        grainy_shapes = make_shapes_grainy(final_shapes)

        dump_to_file(final_shapes, constants.pwd + f"/data/tmp/tmp-shapes{i}.dump")
        dump_to_file(grainy_shapes, constants.pwd + f"/data/tmp/tmp-shapes-grainy{i}.dump")

    final_shapes = []
    grainy_shapes = []

    # Finaly, combine subdivided patches to a big one again
    for i in range(len(patches_d)):
        final_shapes.extend(read_dump_from_file(constants.pwd + f"/data/tmp/tmp-shapes{i}.dump"))
        grainy_shapes.extend(read_dump_from_file(constants.pwd + f"/data/tmp/tmp-shapes-grainy{i}.dump"))

    final_props = ""
    final_props_grainy = ""

    for j in range(len(final_shapes)):
        new_cords = final_shapes[j][0]
        new_cords.append(new_cords[0])
        geom = {}
        col_val = f"0,128,0,{str(min(0.8* 2 * final_shapes[j][1], 0.8))}"
        props = {'color': f'rgba({col_val})', 'trees': final_shapes[j][2][2],
                 'soil': get_short_name(soil_db, final_shapes[j][2][1])}
        geom['type'] = 'Polygon'
        geom['coordinates'] = [new_cords]
        data['features'].append({
            'type': 'Feature',
            'geometry': geom,
            'properties': props
        })
        final_props += col_val + ";"

    for j in range(len(grainy_shapes)):
        new_cords = grainy_shapes[j][0]
        new_cords.append(new_cords[0])
        geom = {}
        col_val = f"0, 128, 0, {str(min(0.5 * grainy_shapes[j][1], 0.5))}"
        props = {'color': f'rgba({col_val})'}
        geom['type'] = 'Polygon'
        geom['coordinates'] = [new_cords]
        data_grainy['features'].append({
            'type': 'Feature',
            'geometry': geom,
            'properties': props
        })
        final_props_grainy += col_val + ";"

    day = datetime.datetime.today().day
    file_name = constants.pwd + f'/web/data/data{day}.json'
    file_name_grainy = constants.pwd + f'/web/data/data_grainy{day}.json'

    update_file = generate_app_update(final_props, final_props_grainy)

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)
    with open(file_name_grainy, 'w') as outfile:
        json.dump(data_grainy, outfile)
    with open(constants.pwd + f'/web/publish/update_data.txt', 'w') as outfile:
        outfile.write(final_props)
    with open(constants.pwd + f'/web/publish/update_data_grainy.txt', 'w') as outfile:
        outfile.write(final_props_grainy)
    with open(constants.pwd + f'/web/publish/update_file.json', 'w') as outfile:
        outfile.write(update_file)
