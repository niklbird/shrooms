import constants

import io_utils
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime
import mushroom
import time
import sys

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


def combine_extension(shapes, extension_points):
    # This is the final combination step -> Combine all shapes that are touching

    # The shapes are indexed the same as the stacks

    # Use dictionary to keep track which shapes where combined to which ones
    # This because multiple shapes might point to the same shape for combination
    used = {}

    start_points = {}

    final_shapes = []

    a = 0
    for extension in extension_points:
        a += 1
        touch_point = extension[0]
        touch_point_new = extension[1]
        index = extension[2]
        index_new = extension[3]

        if index in start_points:
            index = start_points[index]
            while index in used:
                index = used[index]
            shp = final_shapes[index]
            used[index] = len(final_shapes)
            # Erase the current shape as we will create a new bigger one
            final_shapes[index].append(-1)
        else:
            shp = shapes[index]
            start_points[index] = len(final_shapes)

        break_outer = False
        if index_new in start_points:
            index_new = start_points[index_new]
            while index_new in used:
                if used[index_new] >= len(final_shapes):
                    del used[index_new]
                    break_outer = True
                    break
                index_new = used[index_new]
            if break_outer:
                # I think this means that this combination was already implicitly done, not sure though
                # It works so it will stay this way
                continue
            shp_new = final_shapes[index_new]
            used[index_new] = len(final_shapes)
            # Erase the current shape as we will create a new bigger one
            final_shapes[index_new].append(-1)
        else:
            shp_new = shapes[index_new]
            start_points[index_new] = len(final_shapes)

        shape = shp[0]
        shape_new = shp_new[0]

        #if shp[1] != shp_new[1]:
        #    print("Somethings wrong, I can feel it")

        touch_point_i = shape.index(touch_point)

        touch_point_new_i = shape_new.index(touch_point_new)

        final_shape = []

        for i in range(touch_point_i + 1):
            final_shape.append(shape[i])

        for i in range(touch_point_new_i, touch_point_new_i + len(shape_new) + 1):
            final_shape.append(shape_new[i % len(shape_new)])

        for i in range(touch_point_i + 1, len(shape)):
            final_shape.append(shape[i])
        final_shapes.append([final_shape, shp[1]])

    f_shapes = []
    for shape in final_shapes:
        if len(shape) == 2:
            f_shapes.append(shape)

    for i in range(len(shapes)):
        if i not in start_points:
            f_shapes.append(shapes[i])

    return f_shapes


def combine_rows(rows):
    # Combine rows in a patch to reduce shape amount
    # Idea here is:
    # If the shapes touch -> Safe that they touch
    # We later iterate the shapes that touched and combine them into a single large one
    used_shapes = []
    # Each stack will later result in one large shape
    stack_dictionary = {}
    stacks = []
    print("Combining rows")
    executions = 0
    base_counter = 0

    # Shapes or from top left clockwise: 2,1,0,3

    extension_points = []
    print(len(rows))
    for i in range(len(rows) - 1):
        row = rows[i][0]
        next_row = rows[i + 1][0]
        for j in range(len(row)):
            for k in range(len(next_row)):
                executions += 1
                # If this shape was already used -> Cant just simply add, create extension point to later combine these
                # If this shape and the shape have same probability -> Look if can be combined
                val1 = rows[i][1][j]

                if val1 == 0:
                    used_shapes.append(i)
                    continue

                val2 = rows[i + 1][1][k]
                dif = abs(val2 - val1)
                if dif < 0.001:
                    shape = row[j]
                    point_0 = shape[0]
                    point_1 = shape[1]

                    shape_new = next_row[k]
                    point_2 = shape_new[2]
                    point_3 = shape_new[3]

                    if point_1[1] <= point_3[1] <= point_0[1] or point_1[1] <= point_2[1] <= point_0[1]\
                            or point_2[1] <= point_1[1] <= point_0[1] <= point_3[1] or\
                            point_1[1] <= point_2[1] <= point_3[1] <= point_0[1]:
                        # The shapes touch -> Combine to larger shape
                        if base_counter + len(row) + k in used_shapes or base_counter + j in used_shapes:
                            if base_counter + len(row) + k in stack_dictionary:
                                index_new = stack_dictionary[base_counter + len(row) + k]
                            else:
                                index_new = len(stacks)
                                stacks.append([[shape, val1], [shape_new, val2]])
                                stack_dictionary[base_counter + len(row) + k] = index_new
                                used_shapes.append(base_counter + len(row) + k)
                            if base_counter + j in stack_dictionary:
                                index = stack_dictionary[base_counter + j]
                            else:
                                index = len(stacks)
                                stacks.append([[shape, val1], [shape_new, val2]])
                                stack_dictionary[base_counter + j] = index
                                used_shapes.append(base_counter + j)
                            extension_points.append([point_1, point_2, index, index_new])
                            continue
                        else:
                            if int(base_counter + j) in stack_dictionary:
                                stack_index = stack_dictionary[base_counter + j]
                                stacks[stack_index].append([shape_new, val1])

                            else:
                                # If not yet in the dictionary -> Beginning of new shape
                                stack_index = len(stacks)
                                stacks.append([[shape, val1], [shape_new, val2]])

                            # Store to which larger shape this shape now belongs
                            stack_dictionary[base_counter + len(row) + k] = stack_index

                        used_shapes.append(base_counter + j)
                        used_shapes.append(base_counter + len(row) + k)
        base_counter += len(row)

    print("Converting Stacks to Shapes")
    shapes = convert_stacks_to_shapes(stacks)

    print("Combining Extensions")
    shapes = combine_extension(shapes, extension_points)

    a = 0
    # Now at last, also add the shapes that could not be combined
    for i in range(len(rows)):
        for j in range(len(rows[i][0])):
            if a + j not in used_shapes:
                shapes.append([rows[i][0][j], rows[i][1][j]])
        a += len(rows[i][0])
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
            if r == 0 or row_tmp[r - 1][1] - arr[v][1]:
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
    return combine_rows(rows)


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


def split_patches(patches, patches_per_file):
    shape = get_patches_shape(patches)
    row_amount = int(patches_per_file / shape[0])
    final_shapes = []
    for i in range(0, len(patches), row_amount * shape[0]):
        final_shapes.append(patches[i:min(i + row_amount * shape[0], len(patches))])
    return final_shapes


COMPLETE_REPARSE = True
