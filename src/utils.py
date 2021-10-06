from pyproj import Proj
import math
import constants
import numpy as np
import shapefile
from shapely.geometry import LineString, Point, LinearRing
import patch
import environment_utils
from numba import jit


def preprocess(data):
    # Find middle point for all 5 point surfaces
    new_data = []
    for i in range(0, len(data), 5):
        try:
            point = data[i]
            x_1 = data[i]["geometry"]["coordinates"][0]
            y_1 = data[i]["geometry"]["coordinates"][1]
            x_2 = data[i + 2]["geometry"]["coordinates"][0]
            y_2 = data[i + 1]["geometry"]["coordinates"][1]
            x_mid = x_1 + 0.5 * (x_2 - x_1)
            y_mid = y_1 + 0.5 * (y_2 - y_1)
            point["geometry"]["coordinates"] = (x_mid, y_mid)
            new_data.append(point)
        except Exception:
            print("Exception")
    return new_data


p = Proj("EPSG:5683")
p2 = Proj("EPSG:3034")


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
    # translate angle distance to km distance latitudial
    return 110.574

@jit(nopython=True)
def get_long_fac(longitude):
    # translate angle distance to km distance longitudinal
    return 111.32 * np.cos(longitude)

@jit(nopython=True)
def get_distance(x_1, y_1, x_2, y_2):
    return np.sqrt(((x_1 - x_2) * get_lat_fac()) ** 2 + ((y_1 - y_2) * get_long_fac(x_1)) ** 2)

@jit(nopython=True)
def find_closest_station(coord: list, stations: list):
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
    # translate german treename to JSON name
    return constants.treeNames_g[germname]


def create_points_inner(topx, topy, botx, boty, x_add, y_add, dist):
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


def create_points(topx, topy, botx, boty, dist, batch_size_sqrt):
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
    stations_minimized = np.array(stations_minimized,  dtype=np.float64)
    while cord[1] + batch_size_sqrt * y_add < y_end:
        while cord[0] + batch_size_sqrt * x_add < x_end:
            batch = create_points_inner(cord[0], cord[1], cord[0] +
                                        batch_size_sqrt * x_add, cord[1]
                                        + batch_size_sqrt * y_add, x_add, y_add, dist)

            middle = get_middle(cord[0], cord[0] + batch_size_sqrt * x_add, cord[1], cord[1] + batch_size_sqrt * y_add)
            station_id = find_closest_station(cord, stations_minimized)  # environment_utils.closest_station2(middle)
            corners = create_corners(cord, batch_size_sqrt * x_add, batch_size_sqrt * y_add)
            patches.append(patch.Patch(batch, middle, station_id, corners))
            cord = [cord[0] + batch_size_sqrt * x_add, cord[1]]
        cord = [x_start, cord[1] + batch_size_sqrt * y_add]
    print("Created amount of batches: " + str(len(patches)))
    print("Created amount of points: " + str(len(patches) * (batch_size_sqrt ** 2)))
    return patches


def create_corners(cord, x_delta, y_delta):
    cords = [cord, [cord[0] + x_delta, cord[1]], [cord[0], cord[1] + y_delta], [cord[0] + x_delta, cord[1] + y_delta]]
    return cords


def get_middle(x_start, x_end, y_start, y_end):
    return [(x_start + x_end) / 2, (y_start + y_end) / 2]


def calc_averages(shape_points):
    res = []
    for i in range(len(shape_points)):
        points = np.array(shape_points[i])
        res.append([np.mean(points[:, 0]), np.mean(points[:, 1])])
    return res


def find_closest_point(point, points):
    arr = np.array([abs(get_distance(point[0], point[1], points[i][0], points[i][1])) for i in range(len(points))])
    min_ind = np.argmin(arr)
    return min_ind, points[min_ind]


def find_n_closest_points(point, points):
    arr = np.array([abs(get_distance(point[0], point[1], points[i][0], points[i][1])) for i in range(len(points))])
    arr_sorted = np.sort(arr)
    return np.where(arr == arr_sorted[0]), np.where(arr == arr_sorted[1]), \
           np.where(arr == arr_sorted[2]), np.where(arr == arr_sorted[3])


def get_distance_point_line(point, line):
    p = Point(point)
    l = LineString(line)
    return p.distance(l)


def find_closest_line(point, lines):
    # Correct distance, as it gives back distance of coordinates, but we want km
    arr = np.array([abs(get_distance_point_line(point, lines[i]) * 109.0) for i in range(len(lines))])
    min_ind = np.argmin(arr)
    # Return index of closest line and its distance
    return min_ind, arr[min_ind]


def find_closest_line_segments(point, lines):
    # Find clostest line when using segments
    l = np.array(lines)
    # Find clostest line if data is segmented
    # First find correct segments
    index = list(find_n_closest_points(point, l[:, 0]))
    # Find closest line in the 4 closest segments
    arr = []
    for i in range(4):
        lines_at_node = lines[int(index[i][0])][1]
        minimal_ind, dist = find_closest_line(point, lines_at_node)
        arr.append([minimal_ind, dist])
    arr = np.array(arr)
    # Find the index of the overall closest line in arr
    # This index is also the index of the index in index lololololololol
    min_ind = np.argmin(arr[:, 1])
    index_in_lines = int(index[min_ind][0])
    index_in_arr = int(arr[min_ind][0])
    best_line = lines[index_in_lines][1][index_in_arr]
    return best_line, np.min(arr[:, 1])


points = create_points(50.000071, 8.541154, 49.578922, 9.441947, 0.1, 10)
#o = find_closest_station([50.000071, 8.541154], environment_utils.get_stations())
z = environment_utils.closest_station2([50.000071, 8.541154])
a = 0
