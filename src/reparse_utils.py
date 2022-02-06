import numpy as np
import shapefile

import patch
import environment_utils
from numba import jit
from dbfread import DBF
from pyproj import Proj

import time

import datum

import math
import constants

import io_utils

'''
This File contains all operations that are related to do the geometrical processing of the data points.
This includes creating a grid of points and fitting each point with its corresponding tree-types and weather-station.
'''


@jit(nopython=True)
def get_lat_fac():
    # Translate angle distance to km distance latitudial
    return 110.574


@jit(nopython=True)
def get_long_fac(longitude):
    # Translate angle distance to km distance longitudinal
    return np.abs(111.32 * np.cos(longitude))


@jit(nopython=True)
def get_distance(x_1, y_1, x_2, y_2):
    # Get distance in km between two coordinates
    return np.sqrt(((x_1 - x_2) * get_lat_fac()) ** 2 + ((y_1 - y_2) * get_long_fac(x_1)) ** 2)


@jit(nopython=True)
def get_distance_arr(x_1, y_1, x_2, y_2):
    # Get distance for an entire array in km
    return np.sqrt(((x_1 - x_2) * get_lat_fac()) ** 2 + ((y_1 - y_2) * get_long_fac(x_1)) ** 2)


@jit(nopython=True)
def find_closest_station(coord, stations):
    # Find the closest DWD station to a coordinate
    best_dist = np.float64(100000.0)
    best_stat = np.int8(0)
    for i in range(len(stations)):
        station = stations[i]
        dist = np.float64(get_distance(np.float64(station[0]), np.float64(station[1]), coord[0], coord[1]))
        if dist < np.float64(1.0):
            return station[2]
        elif dist < best_dist:
            best_dist = np.float64(dist)
            best_stat = station[2]
    return best_stat


def create_points_inner(topx, topy, botx, boty, x_add, y_add, dist):
    # Create points from topx to boty without creating patches
    cords = []
    cord = [topx, topy]

    while cord[1] + y_add < boty:
        y_add = dist / get_long_fac(cord[0])
        cords.append(cord)

        while cord[0] + x_add < botx:
            cord = [cord[0] + x_add, cord[1]]
            cords.append(cord)
        cord = [topx, cord[1] + y_add]

    # If last coordinate is only excluded because of round-errors, include it
    if (cord[1] + y_add - boty) / y_add < 0.00001:
        cords.append(cord)
        while cord[0] + x_add < botx:
            cord = [cord[0] + x_add, cord[1]]
            cords.append(cord)
    return cords


def create_points_inner_fixed(topx, topy, x_add, y_add, dist, amount):
    # Create points from topx to boty without creating patches
    cords = []
    cord = [topx, topy]

    for i in range(amount):
        inner_counter = 0
        cords.append(cord)
        for j in range(amount - 1):
            inner_counter += 1
            cord = [cord[0] + x_add, cord[1]]
            cords.append(cord)
        cord = [topx, cord[1] + y_add]
    return cords


def create_points(topx, topy, botx, boty, dist, patch_size_sqrt):
    # Create points from topx to boty with equal distance dist
    # Combine them in patches of patch_size
    print("Starting to create points")

    x_start = min(topx, botx)
    y_start = min(topy, boty)
    x_end = max(topx, botx)
    y_end = max(topy, boty)

    # Current coordinate that is being created
    cord = [x_start, y_start]

    x_add = dist / get_lat_fac()
    y_add = dist / get_long_fac(cord[0])

    patches = []

    # Store all DWD stations
    stations = environment_utils.get_stations()
    stations_minimized = []

    # Remove unused Data about Station
    for station in stations:
        stations_minimized.append([station['geo_lat'], station['geo_lon'], station['station_id']])
    stations_minimized = np.array(stations_minimized, dtype=np.float64)

    # Iterate Grid
    while cord[1] + patch_size_sqrt * y_add < y_end:
        y_added = 0
        while cord[0] + patch_size_sqrt * x_add < x_end:
            # For each patch, create patch_size points (Standard: 100)
            patch_points = create_points_inner_fixed(cord[0], cord[1], x_add, y_add, dist, patch_size_sqrt)

            # Create additional Information about Patch
            middle = get_middle(cord[0], cord[0] + patch_size_sqrt * x_add, cord[1], cord[1] + patch_size_sqrt * y_add)
            station_id = find_closest_station(np.array(cord), stations_minimized)
            corners = create_corners(cord, patch_size_sqrt * x_add, patch_size_sqrt * y_add)

            patches.append(patch.Patch(patch_points, middle, station_id, corners))

            # Adapt coordinate for next patch
            cord = [cord[0] + patch_size_sqrt * x_add, cord[1]]
            y_added += y_add

        cord = [x_start, cord[1] + patch_size_sqrt * y_add]
        y_add = dist / get_long_fac(cord[0])

    print("Created amount of Patches: " + str(len(patches)))
    print("Created amount of Points: " + str(len(patches) * (patch_size_sqrt ** 2)))
    return patches


def create_corners(cord, x_delta, y_delta):
    cords = [cord, [cord[0] + x_delta, cord[1]], [cord[0], cord[1] + y_delta], [cord[0] + x_delta, cord[1] + y_delta]]
    return cords


def get_middle(x_start, x_end, y_start, y_end):
    return [(x_start + x_end) / 2, (y_start + y_end) / 2]


def calc_averages(shape_points):
    # Calc averages of points
    res = []
    for i in range(len(shape_points)):
        points = np.array(shape_points[i])
        res.append([np.mean(points[:, 0]), np.mean(points[:, 1])])
    return res


def find_closest_point(point, points):
    # Currently not used
    arr = np.array([abs(get_distance(point[0], point[1], points[i][0], points[i][1])) for i in range(len(points))])
    min_ind = np.argmin(arr)
    return min_ind, points[min_ind]


def project_coordinate_inverse(coordinate, projection):
    # Use inverse projection on coordinate
    p = Proj(projection)
    point = p(coordinate[0], coordinate[1], inverse=True)
    return [point[1], point[0]]


def project_coordinate(coordinate, projection):
    # Use projection on coordinate
    p = Proj(projection)
    return p(coordinate[0], coordinate[1])


def project_shapes(shapes: list, projection: str):
    # If shapes are not stored in correct coordinate system
    # Project them to common coordinate system
    # We are currently using EPSG:4326
    re = []
    finished_shapes = 0
    print("Start projecting %f Shapes", len(shapes))

    for i in range(len(shapes)):
        points = shapes[i].points
        projected_points = []

        for point in points:
            projected_points.append(project_coordinate_inverse(point, projection))
            if finished_shapes % 1000 == 0:
                print(str(finished_shapes) + " of " + str(len(points)))
            finished_shapes += 1

        finished_shapes = 0
        re.append(projected_points)
    return re


def create_lookup(shape_folder):
    for record in DBF(shape_folder + '.dbf', encoding="iso-8859-1"):
        return list(record.keys())


def create_records(shape_folder):
    # Create list of records (data about each shape)
    sf = shapefile.Reader(shape_folder, encoding="iso-8859-1")
    return sf.records()


def parse_in_shape(shape_folder, projection):
    # Parse in a shape file
    sf = shapefile.Reader(shape_folder, encoding="iso-8859-1")

    # If projection is already the desired one -> Skip projecting it
    if projection == "EPSG:4326":
        return sf.shapes(), sf.records(), create_lookup(shape_folder)
    return project_shapes(sf.shapes(), projection), sf.records(), create_lookup(shape_folder)


@jit(nopython=True)
def shape_contains_points(shape, points):
    ctn = False
    for point in points:
        if shape_contains_point(shape, point):
            ctn = True
    return ctn


def patch_in_shape(shape, patch):
    # If one of the patch corners not contained in shape -> Check all
    # Requires sufficiently smooth shape to work
    if not shape_contains_points(np.array(shape), np.array(patch.corners)):
        contained_points = []

        for p in patch.points:
            if shape_contains_point(np.array(shape), np.array(p)):
                contained_points.append(p)
                print("Kept point")

        patch.points = contained_points
        return


def cut_patches(patches, shape):
    # Ensure that patches only contain points in shape
    # Used to remove all points that are not in a shape
    # E.g. remove points that are not in Germany from all patches that lie on the border
    for patch in patches:
        patch_in_shape(np.array(shape), patch)


def find_max_size_shape(shape):
    # Approximate distance between the two most distant points in shape
    middle_point = shape[0]  # This is only an approximation, but will always over-estimate distance in the end
    dist_arr = np.array(np.abs(get_distance_arr(shape[:, 0], shape[:, 1], middle_point[0], middle_point[1])))
    return 2 * np.max(dist_arr)


def find_max_size_shapes(shapes):
    # Find the maximal distance between two shape points
    dist_arr = []
    for i in range(len(shapes)):
        dist_arr.append(find_max_size_shape(shapes[i][0]))
    return dist_arr


@jit(nopython=True)
def approximate_point_in_shapes(shapes, point):
    # Currently not used anymore, but maybe useful later
    # This approximates the shape as a square
    # Then finds all shape_squares that contain this point
    # May find more than one "fitting" shape
    found_shapes = []
    for i in range(len(shapes)):
        shape = shapes[i]
        upper_left = [np.max(shape[:, 0]), np.max(shape[:, 1])]
        lower_right = [np.min(shape[:, 0]), np.min(shape[:, 1])]
        upper_right = [lower_right[0], upper_left[1]]
        lower_left = [upper_left[0], lower_right[1]]
        reduced_shape = np.array([upper_left, upper_right, lower_right, lower_left, upper_left])
        if shape_contains_point(reduced_shape, point):
            found_shapes.append(shape)
    return found_shapes


def fit_trees_to_point(tree_shapes_points_arr, point, start_point):
    # Find the shape that actually contains point
    # May not find anything
    # Start at index of last found shape (Speedup)
    for j in range(start_point, len(tree_shapes_points_arr) + start_point):
        arr = tree_shapes_points_arr[np.mod(j, len(tree_shapes_points_arr))]
        if shape_contains_point(arr[0], point):
            if len(arr) == 1:
                return int(np.mod(j, len(tree_shapes_points_arr)))
            else:
                # Check that none of the inner (excluded shapes) contain the point
                # As this means that the shape is not in the shape after all
                inner_contains = False
                for i in range(1, len(arr)):
                    if shape_contains_point(arr[i], point):
                        inner_contains = True
                        break
                if not inner_contains:
                    return int(np.mod(j, len(tree_shapes_points_arr)))
    print("ERROR, DID NOT FIND ANY SHAPE THAT CONTAINS THIS POINT")
    return None


def fit_trees_to_points(tree_shapes_points, points):
    # Find the correct shape for each point
    # If no shape is found -> None
    ret = []
    tree_shapes_points_np = np.array(tree_shapes_points)
    re = 0
    for i in range(len(points)):
        re = fit_trees_to_point(tree_shapes_points_np, points[i], re)
        ret.append(re)
        if re is None:
            re = 0
    return ret


def get_fitting_shapes(tree_patches, middle, tree_preprocessed, tree_shapes_points_np, tree_records_np, patch,
                       tree_bool, closest_points=4):
    # Get all shapes that may fit to a point (All shapes that are near the 4 closest tree-patches-points)
    indices = find_n_closest_points(tree_patches, middle, closest_points)

    possible_shape_indices = []
    for index in indices:
        possible_shape_indices = possible_shape_indices + tree_preprocessed[index]

    possible_shapes = tree_shapes_points_np[possible_shape_indices]
    if tree_bool:
        possible_records = tree_records_np[possible_shape_indices][:, 3]
    else:
        possible_records = tree_records_np[possible_shape_indices][:, 3]

    # Now look for each point which shape is the correct one
    # This fails surprisingly often...
    fitting_shapes = fit_trees_to_points(possible_shapes, patch.points)

    return fitting_shapes, possible_records


@jit(nopython=True)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
@jit(nopython=True)
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


@jit(nopython=True)
def extend_back(shapes_1, shapes_2):
    # This is used to extend back the reduced array
    for i in range(len(shapes_1)):
        if shapes_1[i][0][0] == shapes_2[0][0] and shapes_1[i][0][1] == shapes_2[0][1]:
            return i


@jit(nopython=True)
def remove_doubles_array(shapes):
    # Remove shapes that are contained more than once
    # -> Return unique shapes
    shapes_reduced = []
    for i in range(len(shapes)):
        new_val = True
        for j in range(len(shapes_reduced)):
            if shapes[i][0][0] == shapes_reduced[j][0][0]:
                new_val = False
                break
        if new_val:
            shapes_reduced.append(shapes[i][0])

    return shapes_reduced


def shapes_from_dist(shapes, points):
    # Currently not used anymore but may be useful later?
    # Find all shapes that may contain this point
    dist = 1000.0
    best_ind = 0

    possible_shapes = []
    pos_shapes_indices = []

    # Look which shape has the closest point to this point
    # This should be the shape the point is in (or the one directly neighboring)
    for i in range(len(shapes)):
        shape = shapes[i][0]
        m = np.min(np.abs(get_distance_arr(shape[:, 0], shape[:, 1], points[0], points[1])))
        if np.abs(dist - m) < 0.0001:
            possible_shapes.append(shape)
            pos_shapes_indices.append(i)
        elif dist > m:
            dist = m
            best_ind = i

    possible_shapes.append(shapes[best_ind])
    pos_shapes_indices.append(best_ind)
    return possible_shapes, pos_shapes_indices, dist


def create_dates(patch, fitting_shapes, possible_records, trees_bool):
    # Use "while" instead of "for" to be able to manipulate the index j
    for j in range(len(patch.points)):
        value = possible_records[fitting_shapes[j]]
        patch.dates[j].set_env(value, trees_bool)
    return patch


def fit_values_to_patches(patches, value_shapes_points, value_records, value_patches, value_preprocessed, trees_bool):
    # This function iterates all patches and finds the correct tree-types at each point in each patch
    print("Started fitting with amount: " + str(len(patches)))
    value_shapes_points_np = np.array(value_shapes_points)
    value_records_np = np.array(value_records)
    for i in range(len(patches)):
        if i % 200 == 0:
            print("Progress: " + str(i))
        patch = patches[i]
        middle = patch.middle
        # Find all shapes that could be used in this patch
        fitting_shapes, possible_records = get_fitting_shapes(value_patches, middle, value_preprocessed,
                                                                               value_shapes_points_np, value_records_np,
                                                                               patch, trees_bool)
        # Calculate the actual shape for each point in the patch
        patches[i] = create_dates(patch, fitting_shapes, possible_records, trees_bool)
    return patches


def preprocess_values(points, value_shapes, value_shape_distances, dist):
    # For each point, find all tree-shapes that may be as close as dist/2 to it
    ret = [[] for j in range(len(points))]
    dist_half = dist / np.sqrt(2)
    points = np.array(points)
    for i in range(len(value_shapes)):
        shape = value_shapes[i][0]
        indices = np.where((get_distance_arr(points[:, 0], points[:, 1], shape[0][0], shape[0][1]) -
                            value_shape_distances[i]) < dist_half * 2)[0]
        for index in indices:
            ret[index].append(i)
    return ret


def find_n_closest_points(points, point, n):
    # Find the n closest points in "points" to the given point
    points = np.array(points)
    distances = get_distance_arr(points[:, 0], points[:, 1], point[0], point[1])
    partitioned = np.partition(distances, n - 1)[0:n]
    indices = []
    # Two options, not sure which one is faster
    for i in range(len(partitioned)):
        indices.append(np.where(distances == partitioned[i])[0][0])
    return indices


def middle_points(points):
    return (points[:, 0] + points[:, 1] + points[:, 2] + points[:, 3]) / 4


@jit(nopython=True)
def shape_contains_point(shape, point):
    # Find out if a Shape contains a Point
    # CAREFUL: This is not robust!
    nvert = len(shape)
    c = False
    j = nvert - 1
    # This code section is taken from stackoverflow
    for i in range(0, nvert):
        if ((shape[i][1] > point[1]) != (shape[j][1] > point[1])) and (
                point[0] < ((shape[j][0] - shape[i][0]) * (point[1] - shape[i][1])
                            / (shape[j][1] - shape[i][1]) + shape[i][0])):
            c = not c
        j = i
    return c


def preprocess_records(records):
    # Preprocess records, remove artifcats from Umlaute
    # This is hacky, better to chose correct encoding but I could not be bothered
    for i in range(len(records)):
        record = records[i]
        text = record[3]
        record[3] = str(text).replace("Ã¤", "ae").replace("Ã¶", "Oe").replace("Ã¼", "ue").replace("Ã", "Ue").replace(
            "Ã",
            "ss")
        records[i] = record


def get_patches_shape(patches):
    # Get the rectangle of patches
    c = 0
    while patches[c].corners[0][1] == patches[c + 1].corners[0][1]:
        c += 1
    return c + 1, int(len(patches) / (c + 1))


def convert_shapes_to_format(shapes):
    # Convert the shapes to the format more suited for processing
    for i in range(len(shapes)):
        parts = shapes[i].parts
        my_array = np.array(shapes[i].points)
        temp = np.copy(my_array[:, 0])
        my_array[:, 0] = my_array[:, 1]
        my_array[:, 1] = temp
        if len(parts) > 1:
            arr = []
            for j in range(len(parts) - 1):
                arr.append(my_array[parts[j]: parts[j + 1]])
            shapes[i] = arr
        else:
            shapes[i] = [my_array]
    return shapes


def soil_parse(patches, start_cord, end_cord, complete_reparse):
    # Parse in Soil Data
    if complete_reparse:
        soil_shapes, records, lu = parse_in_shape(constants.pwd + "/data/soil_folder/Bodenarten_new_new", "EPSG:4326")
        # Changing first and second coordinate as format is inconsistent
        soil_shapes = convert_shapes_to_format(soil_shapes)
        io_utils.dump_to_file(soil_shapes, constants.pwd + "/data/dumps/soils.dump")

    soil_shapes = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/soils.dump")

    # Preprocess Records to remove Encoding-Artifacts
    records = create_records(constants.pwd + "/data/soil_folder/Bodenarten_new_new")
    preprocess_records(records)

    soil_patches = create_points_inner(start_cord[0], start_cord[1], end_cord[0], end_cord[1], 1.0 / get_lat_fac(), 1.0 / get_long_fac(end_cord[0]), 1.0)

    if complete_reparse:
        soil_shape_distances = find_max_size_shapes(soil_shapes)
        io_utils.dump_to_file(soil_shape_distances, constants.pwd + "/data/dumps/soil_shape_dist.dump")

    soil_shape_distances = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/soil_shape_dist.dump")

    if complete_reparse:
        prepro = preprocess_values(soil_patches, soil_shapes, soil_shape_distances, 1.0)
        io_utils.dump_to_file(prepro, constants.pwd + "/data/dumps/soil_prepro.dump")

    prepro = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/soil_prepro.dump")

    print("Started fitting soils")
    patches = fit_values_to_patches(patches, soil_shapes, records, soil_patches, prepro, False)
    return patches


def fill_patches_with_empy_dates(patches):
    for j in range(len(patches)):
        for i in range(len(patches[j].points)):
            patches[j].dates.append(datum.Datum(patches[j].points[i]))
    return patches


def reparse(patches, corners):
    # Recreate everything
    start = time.time()

    start_cord = [math.floor(min(corners[0], corners[2])), math.floor(min(corners[1], corners[3]))]
    end_cord = [math.ceil(max(corners[0], corners[2])), math.ceil(max(corners[1], corners[3]))]

    COMPLETE_REPARSE = True

    if COMPLETE_REPARSE:
        # Ensure that shape is in ESPG
        tree_shapes, records, lu = parse_in_shape(constants.pwd + "/data/tree_folder/trees", "EPSG:4326")
        # Changing first and second coordinate as format is inconsistent and split shape into its parts
        tree_shapes = convert_shapes_to_format(tree_shapes)
        io_utils.dump_to_file(tree_shapes, constants.pwd + "/data/dumps/trees.dump")

    # Read in Shapes and Values of Tree-Data
    records = create_records(constants.pwd + "/data/tree_folder/trees")
    tree_shapes = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/trees.dump")

    # Preprocess Records to remove Encoding-Artifacts
    preprocess_records(records)

    fill_patches_with_empy_dates(patches)

    patches = soil_parse(patches, start_cord, end_cord, COMPLETE_REPARSE)

    # Create a second Grid of Tree-Points to speed up Calculations later
    tree_patches = create_points_inner(start_cord[0], start_cord[1], end_cord[0], end_cord[1], 1.0 / get_lat_fac(), 1.0 / get_long_fac(end_cord[0]), 1.0)

    if COMPLETE_REPARSE:
        # Find max. Size of each Tree-Shape
        tree_shape_distances = find_max_size_shapes(tree_shapes)
        io_utils.dump_to_file(tree_shape_distances, constants.pwd + "/data/dumps/tree_shape_dist.dump")

    tree_shape_distances = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/tree_shape_dist.dump")

    if COMPLETE_REPARSE:
        # Preprocess Trees: Fit Tree-Shapes to the Tree-Grid
        prepro = preprocess_values(tree_patches, tree_shapes, tree_shape_distances, 1)
        io_utils.dump_to_file(prepro, constants.pwd + "/data/dumps/prepro_trees.dump")

    prepro = io_utils.read_dump_from_file(constants.pwd + "/data/dumps/prepro_trees.dump")

    # Now find out which Tree-Type (Shape) each created Data-Point has
    # This requires the most calculation effort -> Speed-Up as much as possible
    patches = fit_values_to_patches(patches, tree_shapes, records, tree_patches, prepro, True)
    end = time.time()
    print("Total Time for Parsing: " + str(end - start))
    print("Time per Patch: " + str((end - start) / float(len(patches))))
    return patches