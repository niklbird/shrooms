import constants
import numpy as np
import shapefile
import patch
import environment_utils
from numba import jit
from dbfread import DBF
from pyproj import Proj
import io_utils
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datum
import datetime
import mushroom

# All common utility functions

p = Proj("EPSG:5683")
p2 = Proj("EPSG:3034")


def plt_shapefile(shapes):
    # Plot shapefile
    # Currently not used
    map = Basemap(width=1200000, height=900000, resolution=None, projection='lcc', lat_0=51, lon_0=10.5)
    # draw coastlines, country boundaries, fill continents.
    map.bluemarble()
    map.readshapefile("shapes", "ger")
    plt.title('contour lines over filled continent background')
    plt.show()


def translate_epsg_to_utm(coord):
    # Projection function with target coordinate system ESPG 5683 - 3-degree Gauss-Kruger zone 3
    return p(coord[0], coord[1], inverse=True)


def translate_utm_to_espg(coord):
    # Translate from german coordinate system to gps coordinates
    return p(coord[0], coord[1])


def translate_epsg2_to_utm(coord):
    # Projection function with target coordinate system ESPG 5683 - 3-degree Gauss-Kruger zone 3
    point = p2(coord[0], coord[1], inverse=True)
    return [point[1], point[0]]


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


def get_german_treename(latname):
    # Translate JSON name to real german tree name
    return constants.treeNames_l[latname]


def get_latname_treename(germname):
    # Translate german tree name to JSON name
    return constants.treeNames_g[germname]


def create_points_inner(topx, topy, botx, boty, x_add, y_add, dist):
    # Create points inside of a patch
    cords = []
    cord = [topx, topy]
    while cord[1] + y_add < boty:
        while cord[0] + x_add < botx:
            cord = [cord[0] + x_add, cord[1]]
            cords.append(cord)
        cord = [topx, cord[1] + y_add]
        cords.append(cord)
        y_add = dist / get_long_fac(cord[0])
    return cords


def create_points(topx, topy, botx, boty, dist, patch_size_sqrt):
    # Create points from topx to boty with equal distance dist
    # Combine them in patches of batch_size
    print("Starting to create points")
    x_start = min(topx, botx)
    y_start = min(topy, boty)
    x_end = max(topx, botx)
    y_end = max(topy, boty)

    cord = [x_start, y_start]

    x_add = dist / get_lat_fac()
    y_add = dist / get_long_fac(cord[0])

    patches = []

    stations = environment_utils.get_stations()
    stations_minimized = []
    for station in stations:
        stations_minimized.append([station['geo_lat'], station['geo_lon'], station['station_id']])
    stations_minimized = np.array(stations_minimized, dtype=np.float64)

    while cord[1] + patch_size_sqrt * y_add < y_end:
        while cord[0] + patch_size_sqrt * x_add < x_end:
            batch = create_points_inner(cord[0], cord[1], cord[0] +
                                        patch_size_sqrt * x_add, cord[1]
                                        + patch_size_sqrt * y_add, x_add, y_add, dist)

            middle = get_middle(cord[0], cord[0] + patch_size_sqrt * x_add, cord[1], cord[1] + patch_size_sqrt * y_add)

            station_id = find_closest_station(np.array(cord), stations_minimized)

            corners = create_corners(cord, patch_size_sqrt * x_add, patch_size_sqrt * y_add)

            patches.append(patch.Patch(batch, middle, station_id, corners))
            cord = [cord[0] + patch_size_sqrt * x_add, cord[1]]

        cord = [x_start, cord[1] + patch_size_sqrt * y_add]

    print("Created amount of batches: " + str(len(patches)))
    print("Created amount of points: " + str(len(patches) * (patch_size_sqrt ** 2)))
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


def find_n_closest_points(point, points):
    # Currently not used
    arr = np.array([abs(get_distance(point[0], point[1], points[i][0], points[i][1])) for i in range(len(points))])
    arr_sorted = np.sort(arr)
    return np.where(arr == arr_sorted[0]), np.where(arr == arr_sorted[1]), \
           np.where(arr == arr_sorted[2]), np.where(arr == arr_sorted[3])


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
    re = []
    finished_shapes = 0
    print("Start projecting %f shapes", len(shapes))
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
    return project_shapes(sf.shapes(), projection), sf.records(), create_lookup(shape_folder)


@jit(nopython=True)
def shape_contains_point(shape, point):
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


def extract_tree_info_for_point(point, records, patch_ll):
    # Extract all tree data relevant for mushrooms
    trees = {'hardwood': records['pDecid'], 'softwood': records['pConifer'], 'coverage': records['pAll'],
             'buche': records['pFagus'], 'kiefer': records['pPinus'], 'fichte': records['pPicea'],
             'eiche': records['pQuerus'], 'birke': records['pBetula']}
    patch_ll.dates.append(datum.Datum(point, trees, 0))


def fit_trees_to_patch(tree_middlepoints, tree_records, patches, patch_size):
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
    if weather_data == None:
        return
    ret = {}
    ret['temperature'] = weather_data['temperature_max_200']
    ret['humidity'] = weather_data['humidity']
    ret['rain'] = weather_data['precipitation_height']
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

            if not ts in weather.keys():
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
    month_facs = get_month_factors(datetime.datetime.today().month)
    for patch in patches:
        weather = patch.weather_data
        temperatures = []
        rains = []
        humidities = []
        for i in range(30, 1, -1):
            ts = format_timestamp(datetime.datetime.today() - datetime.timedelta(days=i))
            temperatures.append(weather[ts]['temperature'])
            rains.append(weather[ts]['rain'])
            humidities.append(weather[ts]['humidity'])
        rain_val, temp_val, hum_val = mushroom.environment_factor(rains, temperatures, humidities)
        # Factors may have to be tweeked
        dynamic_factor = (2 * rain_val + 1 * temp_val + 0.7 * hum_val) / 3.7
        for date in patch.dates:
            for shroom in date.mushrooms.keys():
                # Basefactor, seasonality, environment factor
                date.probabilities[shroom] = min(date.mushrooms[shroom] * month_facs[shroom] * dynamic_factor, 1)


def calc_static_values(patches):
    mushrooms = mushroom.read_XML('../data/mushrooms_databank.xml')
    for patch in patches:
        for date in patch.dates:
            trees = date.trees
            for shroom in mushrooms.values():
                date.mushrooms[shroom.attr['name']] = mushroom.tree_value(shroom, trees)


def reparse(cut_out_germany):
    patches = create_points(50.00520532919058, 8.646406510673339, 49.767632303668734, 9.118818592516165, 0.1, 10)
    # If some of the points can be outside of Germany -> Only use points inside
    if cut_out_germany:
        germany_shape = io_utils.read_dump_from_file(constants.pwd + "/data/ger_folder/ger_points_proc.dump")[1]
        cut_patches(patches, germany_shape)
    trees = io_utils.read_dump_from_file(constants.pwd + "/data/trees_folder/trees_points_proc.dump")
    records = create_records(constants.pwd + "/data/trees_folder/trees")
    mp = middle_points(np.array(trees))
    fit_trees_to_patch(np.array(mp), records, np.array(patches), 1)
    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_proc.dump")
    calc_static_values(patches)
    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_shrooms.dump")
    add_weather(patches)
    io_utils.dump_to_file(patches, constants.pwd + "/data/patches_weather.dump")
