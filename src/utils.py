import constants
import numpy as np
import shapefile
import patch
import environment_utils
from numba import jit, njit
from dbfread import DBF
from pyproj import Proj
import io_utils
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datum
import datetime
import mushroom
import time
import matplotlib.path as mpltPath


# All common utility functions

# p = Proj("EPSG:5683")
# p2 = Proj("EPSG:3034")


# For Debugging
def plt_shapefile(shapes):
    # Plot shapefile
    # Currently not used
    map = Basemap(width=1200000, height=900000, resolution=None, projection='lcc', lat_0=51, lon_0=10.5)
    # draw coastlines, country boundaries, fill continents.
    map.bluemarble()
    map.readshapefile("shapes", "ger")
    plt.title('contour lines over filled continent background')
    plt.show()


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


# Deprecated
def get_german_treename(latname):
    # Translate JSON name to real german tree name
    return constants.treeNames_l[latname]


# Deprecated
def get_latname_treename(germname):
    # Translate german tree name to JSON name
    return constants.treeNames_g[germname]


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
        while cord[0] + patch_size_sqrt * x_add < x_end:
            # For each patch, create patch_size points (Standard: 100)
            patch_points = create_points_inner(cord[0], cord[1], cord[0] +
                                               patch_size_sqrt * x_add, cord[1]
                                               + patch_size_sqrt * y_add, x_add, y_add, dist)

            # Create additional Information about Patch
            middle = get_middle(cord[0], cord[0] + patch_size_sqrt * x_add, cord[1], cord[1] + patch_size_sqrt * y_add)
            station_id = find_closest_station(np.array(cord), stations_minimized)
            corners = create_corners(cord, patch_size_sqrt * x_add, patch_size_sqrt * y_add)

            patches.append(patch.Patch(patch_points, middle, station_id, corners))

            # Adapt coordinate for next patch
            cord = [cord[0] + patch_size_sqrt * x_add, cord[1]]
            y_add = dist / get_long_fac(cord[0])

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


# Deprecated
def extract_tree_info_for_point(point, records, patch_ll):
    # Extract all tree data relevant for mushrooms
    trees = {'hardwood': records['pDecid'], 'softwood': records['pConifer'], 'coverage': records['pAll'],
             'buche': records['pFagus'], 'kiefer': records['pPinus'], 'fichte': records['pPicea'],
             'eiche': records['pQuerus'], 'birke': records['pBetula']}
    patch_ll.dates.append(datum.Datum(point, trees, 0))


# Deprecated
def fit_trees_to_patch_old(tree_middlepoints, tree_records, patches, patch_size):
    # Put all tree shapes into patch that have a middle-point inside of this patch
    for patch_l in patches:
        points = np.array(patch_l.points)
        middle = np.array(patch_l.middle)
        dst = get_distance(np.array(tree_middlepoints[:, 0]), np.array(tree_middlepoints[:, 1]),
                           middle[0], middle[1])
        args = np.argwhere(dst < patch_size * 2 * 1.4)
        if len(args) != 0:
            # Only look at patches that contain trees
            trees = tree_middlepoints[args][0]
            for point in points:
                dist = get_distance_arr(trees[:, 0], trees[:, 1], point[0], point[1])
                # [0][0] required as this returns a nested array
                smallest = np.argwhere(dist == np.amin(dist))[0][0]
                record = tree_records[args[smallest][0]]
                extract_tree_info_for_point(point, record, patch_l)


def middle_points(points):
    return (points[:, 0] + points[:, 1] + points[:, 2] + points[:, 3]) / 4


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


def find_max_size_shape(shape):
    # Approximate distance between the two most distant points in shape
    middle_point = shape[0]  # This is only an approximation, but will always over-estimate distance in the end
    dist_arr = np.array(np.abs(get_distance_arr(shape[:, 0], shape[:, 1], middle_point[0], middle_point[1])))
    return 2 * np.max(dist_arr)


def find_max_size_shapes(shapes):
    #
    dist_arr = []
    for i in range(len(shapes)):
        dist_arr.append(find_max_size_shape(shapes[i]))
    return dist_arr


def fit_trees_to_point(tree_shapes_points, point, start_point):
    # Find the shape that actually contains point
    # May not find anything
    # Start at index of last found shape (Speedup)
    for j in range(start_point, len(tree_shapes_points) + start_point):
        if shape_contains_point(tree_shapes_points[np.mod(j, len(tree_shapes_points))], point):
            return int(np.mod(j, len(tree_shapes_points)))


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
                       closest_points=4):
    # Get all shapes that may fit to a point (All shapes that are near the 4 closest tree-patches-points)
    indeces = find_n_closest_points(tree_patches, middle, closest_points)

    possible_shape_indeces = []
    for index in indeces:
        possible_shape_indeces = possible_shape_indeces + tree_preprocessed[index]

    possible_shapes = tree_shapes_points_np[possible_shape_indeces]
    possible_records = tree_records_np[possible_shape_indeces][:, 3]

    # Now look for each point which shape is the correct one
    # This fails surprisingly often...
    fitting_shapes = fit_trees_to_points(possible_shapes, patch.points)

    return fitting_shapes, possible_records, possible_shapes


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
            shapes_reduced.append(shapes[i])

    return shapes_reduced


@jit(nopython=True)
def approximate_point_in_shapes(shapes, point):
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


@jit(nopython=True)
def try_shifted_point(red_shapes, point, dist, shapes):
    # Shift the point a bit to try if it is now found in the shape
    # TODO combine to single loop
    new_point = [point[0] + (dist - 0.001), point[1]]
    for shape in red_shapes:
        if shape_contains_point(shape, new_point):
            return extend_back(shapes, shape)
    new_point = [point[0], point[1] + (dist - 0.001)]
    for shape in red_shapes:
        if shape_contains_point(shape, new_point):
            return extend_back(shapes, shape)
    new_point = [point[0] - (dist - 0.001), point[1]]
    for shape in red_shapes:
        if shape_contains_point(shape, new_point):
            return extend_back(shapes, shape)
    new_point = [point[0], point[1] - (dist - 0.001)]
    for shape in red_shapes:
        if shape_contains_point(shape, new_point):
            return extend_back(shapes, shape)
    return -1


@jit(nopython=True)
def try_line_intersections(red_shapes, point, shapes):
    # Last resort: Draw lines from point in all directions
    # The correct shape should be the only one which all lines intersect
    target_point1 = [point[0], 1000.0]
    target_point2 = [point[0], -1000.0]
    target_point3 = [1000.0, point[1]]
    target_point4 = [-1000.0, point[1]]

    shape_intersections = []
    for b in range(len(red_shapes)):
        shape_intersections.append([0])

    target_points = [target_point1, target_point2, target_point3, target_point4]
    for i in range(len(red_shapes)):
        shape = red_shapes[i]
        for j in range(len(shape) - 1):
            for k in range(len(target_points)):
                if intersect(point, target_points[k], shape[j], shape[j + 1]) and k not in shape_intersections[i]:
                    shape_intersections[i].append(k)
    for i in range(len(shape_intersections)):
        if len(shape_intersections[i]) == 4:
            return extend_back(shapes, red_shapes[i])
    return -1


@jit(nopython=True)
def find_shape_for_point_backup(shapes, point, min_dist):
    # This is used for times were the shape_contains_point algorithm fails
    # It can be used to reliably find out in which shape the point lies
    # Remove doubles
    shapes_reduced = remove_doubles_array(shapes)

    # First attempt -> Approximate shape as square
    found_shapes = approximate_point_in_shapes(shapes_reduced, point)

    if len(found_shapes) == 1:
        # It worked
        return extend_back(shapes, found_shapes[0])

    ind = try_shifted_point(shapes_reduced, point, min_dist, shapes)
    if ind != -1:
        # It worked
        return ind

    ind2 = try_line_intersections(shapes_reduced, point, shapes)

    if ind2 != -1:
        # It worked
        return ind2

    # If everything failed -> Return any shape
    # print("No fitting Shape was found -> Using random Shape")
    return 0


# TODO JIT THIS
def shapes_from_dist(shapes, points):
    # Find all shapes that may contain this point
    dist = 1000.0
    best_ind = 0

    possible_shapes = []
    pos_shapes_indices = []

    # Look which shape has the closest point to this point
    # This should be the shape the point is in (or the one directly neighboring)
    for i in range(len(shapes)):
        shape = shapes[i]
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


def no_fitting_shape(points, possible_shapes, possible_records):
    # If no shape was found in the first step -> Invest in heavier calculation to find correct shape

    possible_shapes, pos_shapes_indices, dist = shapes_from_dist(possible_shapes, points)

    best_tmp = find_shape_for_point_backup(possible_shapes, points, dist)

    best_index = pos_shapes_indices[best_tmp]
    trees = possible_records[best_index]

    return trees, best_index


def create_dates(patch, fitting_shapes, possible_records, possible_shapes):
    last_fit = -1

    # The fail counter allows to reduce recalculation
    # If no fitting shape was found for a point, only recalculate every tenth point
    fail_counter = 0
    skip_ind = -1
    trees_before = ""

    cur_recalc = False
    j = 0

    # Use "while" instead of "for" to be able to manipulate the index j
    while j < len(patch.points):
        if j == skip_ind:
            # Recalc is used to improve approximation
            # Idea: Skip 10 values -> If value change then -> Recalc the 9 previous values
            skip_ind = -1
            cur_recalc = False
            j += 1
            continue

        # If the contains algorithm failed, no fitting tree shape was found
        if fitting_shapes[j] is None:
            if cur_recalc:
                trees, fitted_in = no_fitting_shape(np.array(patch.points[j]), possible_shapes, possible_records)
                patch.dates[j].trees = trees
                j += 1
                continue

            if last_fit != -1:
                # If 10 points have been skipped -> Calculate a new value
                if fail_counter == 10:
                    fail_counter = 0
                    trees, last_fit = no_fitting_shape(np.array(patch.points[j]), possible_shapes, possible_records)

                    patch.dates.append(datum.Datum(patch.points[j], trees, 0))
                    skip_ind = j
                    # If the value changed to 10 points ago -> Calculate the skipped points
                    if trees != trees_before:
                        trees_before = trees
                        j -= 10
                        cur_recalc = True
                    j += 1
                    continue
                else:
                    patch.dates.append(datum.Datum(patch.points[j], possible_records[last_fit], 0))
                    fail_counter += 1
                    j += 1
                    continue

            else:
                # This only occurs for the first fail in the patch or after a recalculation
                trees, last_fit = no_fitting_shape(np.array(patch.points[j]), possible_shapes, possible_records)
                patch.dates.append(datum.Datum(patch.points[j], trees, 0))
                j += 1
                continue
        if not cur_recalc:
            trees = possible_records[fitting_shapes[j]]
            patch.dates.append(datum.Datum(patch.points[j], trees, 0))
        j += 1


def fit_trees_to_patches(patches, tree_shapes_points, tree_records, tree_patches, tree_preprocessed):
    # This function iterates all patches and finds the correct tree-types at each point in each patch
    print("Started fitting with amount: " + str(len(patches)))
    tree_shapes_points_np = np.array(tree_shapes_points)
    tree_records_np = np.array(tree_records)
    for i in range(len(patches)):
        if i % 200 == 0:
            print("Progress: " + str(i))
        patch = patches[i]
        middle = patch.middle
        # Find all shapes that could be used in this patch
        fitting_shapes, possible_records, possible_shapes = get_fitting_shapes(tree_patches, middle, tree_preprocessed,
                                                                               tree_shapes_points_np, tree_records_np,
                                                                               patch)
        possible_shapes_np = np.array(possible_shapes)
        # Calculate the actual shape for each point in the patch
        create_dates(patch, fitting_shapes, possible_records, possible_shapes_np)
    return patches


def preprocess_trees(points, tree_shapes, tree_shape_distances, dist):
    # For each point, find all tree-shapes that may be as close as dist/2 to it
    ret = [[] for j in range(len(points))]
    dist_half = dist / np.sqrt(2)
    points = np.array(points)
    for i in range(len(tree_shapes)):
        shape = tree_shapes[i]
        indices = np.where((get_distance_arr(points[:, 0], points[:, 1], shape[0][0], shape[0][1]) -
                            tree_shape_distances[i]) < dist_half * 2)[0]
        for index in indices:
            ret[index].append(i)
    return ret


def find_n_closest_points(points, point, n):
    # Find the n closest points in "points" to the given point
    points = np.array(points)
    distances = get_distance_arr(points[:, 0], points[:, 1], point[0], point[1])
    partitioned = np.partition(distances, n - 1)[0:n]
    # sorte = np.sort(distances)
    indeces = []
    # Two options, not sure which one is faster
    for i in range(len(partitioned)):
        indeces.append(np.where(distances == partitioned[i])[0][0])
    # for i in range(n):
    #    indeces.append(np.where(distances == sorte[i])[0][0])
    return indeces


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


def matlab_shape_contains_point(shape, point):
    # For testing purposes
    path = mpltPath.Path(shape)
    inside2 = path.contains_point(point)
    return inside2


def preprocess_records(records):
    # Preprocess records, remove artifcats from Umlaute
    for i in range(len(records)):
        record = records[i]
        text = record[3]
        record[3] = str(text).replace("Ã¤", "ä").replace("Ã¶", "Ö").replace("Ã¼", "ü").replace("Ã", "Ü").replace("Ã",
                                                                                                                  "ß")
        records[i] = record


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
    patches = create_points(50.013876, 8.654014, 49.686693, 9.196464,
                            constants.point_dist, constants.points_per_patch_sqrt)

    # patches = create_points(50.04028803094584, 8.49786633110003, 49.679084616354025, 9.210604350500015,
    #                       constants.point_dist, constants.points_per_patch_sqrt)

    # Create a second Grid of Tree-Points to speed up Calculations later
    tree_patches = create_points_inner(49.0, 8.0, 51.0, 10.0, 1.0 / get_lat_fac(), 1.0 / get_long_fac(51.0), 1.0)

    if COMPLETE_REPARSE:
        # Find max. Size of each Tree-Shape
        tree_shape_distances = find_max_size_shapes(tree_shapes)
        io_utils.dump_to_file(tree_shape_distances, constants.pwd + "/data/tree_shape_dist.dump")

    tree_shape_distances = io_utils.read_dump_from_file(constants.pwd + "/data/tree_shape_dist.dump")

    if COMPLETE_REPARSE:
        # Preprocess Trees: Fit Tree-Shapes to the Tree-Grid
        prepro = preprocess_trees(tree_patches, tree_shapes, tree_shape_distances, 1)
        io_utils.dump_to_file(prepro, constants.pwd + "/data/prepro.dump")

    prepro = io_utils.read_dump_from_file(constants.pwd + "/data/prepro.dump")

    # Now find out which Tree-Type (Shape) each created Data-Point has (Most Effort)
    patches = fit_trees_to_patches(patches, tree_shapes, records, tree_patches, prepro)

    # Lastly calculate static probabilities for each Point, depending on the Tree-Types present there
    calc_static_values(patches)

    end = time.time()

    print("Total Time for Parsing: " + str(end - start))
    print("Time per Patch: " + str((end - start) / float(len(patches))))

    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_weather.dump")


COMPLETE_REPARSE = False
