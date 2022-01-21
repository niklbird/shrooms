import constants

import io_utils
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime
import mushroom
import time

from reparse_utils import *

'''
File for common utility functions.
'''


# For Debugging
def plt_shapefile():
    # Plot shapefile
    # Currently not used
    map = Basemap(width=1200000, height=900000, resolution=None, projection='lcc', lat_0=51, lon_0=10.5)
    # draw coastlines, country boundaries, fill continents.
    map.bluemarble()
    map.readshapefile("shapes", "ger")
    plt.title('contour lines over filled continent background')
    plt.show()


# Deprecated
def get_german_treename(latname):
    # Translate JSON name to real german tree name
    return constants.treeNames_l[latname]


# Deprecated
def get_latname_treename(germname):
    # Translate german tree name to JSON name
    return constants.treeNames_g[germname]


def filter_relevant_weather_data(weather_data):
    # Only use weather data relevant to mushrooms
    if weather_data is None:
        return

    ret = {'temperature': weather_data['temperature_max_200'], 'humidity': weather_data['humidity'],
           'rain': weather_data['precipitation_height']}
    return ret


def format_timestamp(timestamp_l):
    # Format timestamp for DWD request
    return datetime.datetime(timestamp_l.year, timestamp_l.month, timestamp_l.day, 12)


def add_weather(patches):
    # Add weather data to each patch
    for patch_l in patches:
        patch_l.weather_data = {}

        weather = patch_l.weather_data
        timestamp = datetime.datetime.today()

        # Remove old data
        for weather_ts in weather.keys():
            tdiff = (timestamp - weather_ts).days
            if (weather_ts - timestamp).days > 31:
                del weather[format_timestamp(weather_ts)]

        for i in range(2, 31):
            # Fill in all missing weather data
            ts = format_timestamp(datetime.datetime.today() - datetime.timedelta(days=i))

            if ts not in weather.keys() or weather[ts] is None:
                weather[ts] = filter_relevant_weather_data(environment_utils.get_weather_data_id(patch_l.station, ts))

        patch_l.weather_data = weather


def get_month_factors(month):
    # Factor that indicates if mushroom is in season
    ret = {}
    mushroooms = mushroom.read_XML('../data/mushrooms_databank.xml')
    for s_name in mushroooms.keys():
        ret[s_name] = int(
            int(mushroooms[s_name].attr['seasonStart']) <= month <= int(mushroooms[s_name].attr['seasonEnd']))
    return ret


def calc_dynamic_value(patches):
    # Calculate the actual mushroom probabilities
    month_factors = get_month_factors(datetime.datetime.today().month)

    for patch in patches:
        weather = patch.weather_data
        temperatures = []
        rains = []
        humidities = []

        # Look at weather of last 30 days
        for i in range(30, 1, -1):
            ts = format_timestamp(datetime.datetime.today() - datetime.timedelta(days=i))
            if weather[ts] is None or weather[ts]['temperature'] is None:
                temperatures.append(0)
                rains.append(0)
                humidities.append(50)
                continue
            temperatures.append(weather[ts]['temperature'])
            rains.append(weather[ts]['rain'])
            humidities.append(weather[ts]['humidity'])

        rain_val, temp_val, hum_val = mushroom.environment_factor(rains, temperatures, humidities)

        # Factors may have to be tweaked
        dynamic_factor = (2 * rain_val + 1 * temp_val + 0.7 * hum_val) / 3.7

        for date in patch.dates:
            for shroom in date.mushrooms.keys():
                # Base-Factor, Seasonality, Environment-Factor
                # min(date.mushrooms[shroom] * month_factors[shroom] * dynamic_factor, 1)
                date.probabilities[shroom] = min(date.mushrooms[shroom] * dynamic_factor, 1)


def calc_static_values(patches):
    # Calculate weather-independent factor for each point
    mushrooms = mushroom.read_XML('../data/mushrooms_databank.xml')
    counter = 0
    for patch in patches:
        counter += 1
        for date in patch.dates:
            trees = date.trees
            for shroom in mushrooms.values():
                date.mushrooms[shroom.attr['name']] = mushroom.tree_value_new(shroom, trees)


def preprocess_records(records):
    # Preprocess records, remove artifcats from Umlaute
    # This is hacky, better to chose correct encoding but I could not be bothered
    for i in range(len(records)):
        record = records[i]
        text = record[3]
        record[3] = str(text).replace("Ã¤", "ä").replace("Ã¶", "Ö").replace("Ã¼", "ü").replace("Ã", "Ü").replace("Ã",
                                                                                                                  "ß")
        records[i] = record


def convert_stacks_to_shapes(stacks):
    # This function is very specific to this application
    # The shape are structured in a way that 2-1-2-1-2-1-0-3-0-3-0-3 will always work
    final_shapes = []
    for stack in stacks:
        final_shape = []
        for i in range(len(stack)):
            shape = stack[i][0]
            final_shape.append(shape[2])
            final_shape.append(shape[1])
        for i in range(len(stack) - 1, -1, -1):
            shape = stack[i][0]
            final_shape.append(shape[0])
            final_shape.append(shape[3])
        final_shapes.append([final_shape, stack[0][1]])
    return final_shapes


def combine_rows(rows, column_amount):
    # Combine rows in a patch to reduce shape amount
    # Idea here is:
    # If the shapes touch -> Safe that they touch
    # We later iterate the shapes that touched and combine them into a single large one
    used_shapes = []
    l = column_amount
    # Each stack will later result in one large shape
    stack_dictionary = {}
    stacks = []
    for i in range(len(rows) - 1):
        row = rows[i][0]
        next_row = rows[i + 1][0]
        j = k = 0
        while j < len(row) and k < len(next_row):
            # If this shape and the shape have same probability -> Look if can be combined
            val1 = rows[i][1][j]
            val2 = rows[i + 1][1][k]
            if not l * (i + 1) + k in used_shapes and val1 == val2:
                if l * i + j in stack_dictionary:
                    # This shape was already used in a bigger shape -> Get the stack it lays on
                    stack_index = stack_dictionary[l * i + j]
                    stack = stacks[stack_index]
                    shape = stack[len(stack) - 1]
                    # Geometric logic -> The two middlemost points are the relevant ones to look at
                    point_0 = shape[0][len(shape) - 2]
                    point_1 = shape[0][len(shape) - 1]
                else:
                    shape = row[j]
                    point_0 = shape[0]
                    point_1 = shape[1]

                shape_new = next_row[k]
                point_2 = shape_new[2]
                point_3 = shape_new[3]
                if point_1[1] <= point_3[1] <= point_0[1] or point_1[1] <= point_2[1] <= point_0[1]:
                    # The shapes touch -> Combine to larger shape
                    if l * i + j in stack_dictionary:
                        stack_index = stack_dictionary[l * i + j]
                        stacks[stack_index].append([shape_new, val1])

                    else:
                        # If not yet in the dictionary -> Beginning of new shape
                        stack_index = len(stacks)
                        stacks.append([[shape, val1], [shape_new, val2]])

                    # Store to which larger shape this shape now belongs
                    stack_dictionary[l * (i + 1) + k] = stack_index

                    used_shapes.append(int(i * l + j))
                    used_shapes.append(int((i + 1) * l + k))

                if point_1[1] > point_3[1]:
                    k += 1
                else:
                    j += 1
            else:
                j += 1

    shapes = convert_stacks_to_shapes(stacks)

    # Now at last, also add the shapes that could not be combined
    for i in range(len(rows)):
        for j in range(len(rows[i][0])):
            if int(i * l + j) not in used_shapes:
                shapes.append([rows[i][0][j], rows[i][1][j]])

    return shapes


def shape_reduction(arr, dist_x, dist_y, row_amount, column_amount):
    # This looks at a single row
    # Combines the shapes in this row into larger ones if possible
    l = row_amount
    shapes = []
    rows = []
    for c in range(column_amount):
        row_tmp = []
        final_shapes_row = []
        for r in range(l):
            v = c * l + r

            point = arr[v][0]
            # Check if array carries distance-value
            if len(arr[v]) == 3:
                dist_y = arr[v][2]
            shape = []
            # First Element in Row is starting point
            if r == 0 or row_tmp[r - 1][1] != arr[v][1]:
                # For graphic representation, the point needs to be translated into a rectangle
                shape.append([point[0] + dist_y, point[1] + dist_x])
                shape.append([point[0] + dist_y, point[1] - dist_x])
                shape.append([point[0] - dist_y, point[1] - dist_x])
                shape.append([point[0] - dist_y, point[1] + dist_x])
                row_tmp.append([shape, arr[v][1]])
                final_shapes_row.append(r)
            else:
                # This belongs to the same shape as previous point
                # -> Remove previous point and combine to larger shape
                final_shapes_row.remove(r - 1)
                shape = row_tmp[r - 1]
                new_shape = [[point[0] + dist_y, point[1] + dist_x], shape[0][1], shape[0][2],
                             [point[0] - dist_y, point[1] + dist_x]]
                # Keep upper left and lower left, generate new upper right and lower right points
                row_tmp.append([new_shape, shape[1]])
                final_shapes_row.append(r)
        # Add all elements to list
        shapes.extend(np.array(row_tmp)[final_shapes_row][:, 0])
        rows.append([np.array(row_tmp)[final_shapes_row][:, 0], np.array(row_tmp)[final_shapes_row][:, 1]])
    # Now after reducing the shapes inside each row -> Combine Rows
    return combine_rows(rows, column_amount)


def remove_zero_shapes(shapes):
    # Remove each shape that has a probability value of 0
    ret = []
    for shape in shapes:
        if shape[1] != 0.0:
            ret.append(shape)
    return ret


def get_patches_shape(patches):
    c = 0
    while patches[c].corners[0][1] == patches[c + 1].corners[0][1]:
        c += 1
    return c + 1, int(len(patches) / (c + 1))


def create_super_patch(patches, patches_shape):
    # We created patches for better data processing
    # However now they are in the way of reducing storage space
    # So now all patches are combined into a single large one
    # This needs to take the overall shape of the patches into consideration
    # So that the points are combined in the correct order
    final_array = []
    p_a = constants.points_per_patch_sqrt

    # patches_shape[0] is the amount of patches that are in the same column
    for h in range(0, len(patches), patches_shape[0]):
        for i in range(constants.points_per_patch_sqrt):
            for j in range(h, h + patches_shape[0]):

                patch_i = patches[j]
                corner = patch_i.corners[0]
                corner_2 = patch_i.corners[2]
                dist = corner_2[1] - corner[1]
                # To make sure the graphic representation fits, we need to calc dist_y for every point
                # because the earth is unfortunately a globe :(
                dist_y = dist / (constants.points_per_patch_sqrt * 2.0)

                for k in range(constants.points_per_patch_sqrt):
                    date = patch_i.dates[i * p_a + k]
                    point = date.coord
                    prop = date.probabilities['Steinpilz']

                    final_array.append([[point[1], point[0]], prop, dist_y])
    return final_array


def reparse():
    # Recreate everything
    start = time.time()

    if COMPLETE_REPARSE:
        tree_shapes, records, lu = parse_in_shape(constants.pwd + "/data/tree_folder/trees", "EPSG:4326")
        # Changing first and second coordinate as format is inconsistent
        for i in range(len(tree_shapes)):
            my_array = np.array(tree_shapes[i].points)
            temp = np.copy(my_array[:, 0])
            my_array[:, 0] = my_array[:, 1]
            my_array[:, 1] = temp
            tree_shapes[i] = my_array
        io_utils.dump_to_file(tree_shapes, constants.pwd + "/data/trees.dump")

    # Read in Shapes and Values of Tree-Data
    records = create_records(constants.pwd + "/data/tree_folder/trees")
    tree_shapes = io_utils.read_dump_from_file(constants.pwd + "/data/trees.dump")

    # Preprocess Records to remove Encoding-Artifacts
    preprocess_records(records)

    # Create a Grid of Points
    patches = create_points(49.942287, 8.484820, 49.654197, 9.076708,
                            constants.point_dist, constants.points_per_patch_sqrt)

    # patches = create_points(50.04028803094584, 8.49786633110003, 49.679084616354025, 9.210604350500015,
    #                       constants.point_dist, constants.points_per_patch_sqrt)

    # Create a second Grid of Tree-Points to speed up Calculations later
    tree_patches = create_points_inner(49.0, 8.0, 50.0, 9.5, 1.0 / get_lat_fac(), 1.0 / get_long_fac(50.0), 1.0)

    if COMPLETE_REPARSE:
        # Find max. Size of each Tree-Shape
        tree_shape_distances = find_max_size_shapes(tree_shapes)
        io_utils.dump_to_file(tree_shape_distances, constants.pwd + "/src/dumps/tree_shape_dist.dump")

    tree_shape_distances = io_utils.read_dump_from_file(constants.pwd + "/src/dumps/tree_shape_dist.dump")

    if COMPLETE_REPARSE:
        # Preprocess Trees: Fit Tree-Shapes to the Tree-Grid
        prepro = preprocess_trees(tree_patches, tree_shapes, tree_shape_distances, 1)
        io_utils.dump_to_file(prepro, constants.pwd + "/src/dumps/prepro.dump")

    prepro = io_utils.read_dump_from_file(constants.pwd + "/src/dumps/prepro.dump")

    # Now find out which Tree-Type (Shape) each created Data-Point has
    # This requires the most calculation effort -> Speed-Up as much as possible
    patches = fit_trees_to_patches(patches, tree_shapes, records, tree_patches, prepro)

    # Lastly calculate static probabilities for each Point, depending on the Tree-Types present there
    calc_static_values(patches)

    end = time.time()

    print("Total Time for Parsing: " + str(end - start))
    print("Time per Patch: " + str((end - start) / float(len(patches))))

    io_utils.dump_to_file(patches, constants.pwd + "/src/dumps/patches_weather.dump")


COMPLETE_REPARSE = True
