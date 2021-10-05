from pyproj import Proj
import math
import constants
import numpy as np
import shapefile
from shapely.geometry import LineString, Point, LinearRing

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

def get_lat_fac():
    # translate angle distance to km distance latitudial
    return 110.574


def get_long_fac(longitude):
    # translate angle distance to km distance longitudinal
    return 111.32 * math.cos(longitude)


def get_distance(x_1, y_1, x_2, y_2):
    return math.sqrt(((x_1 - x_2) * get_lat_fac()) ** 2 + ((y_1 - y_2) * get_long_fac(x_1)) ** 2)


def get_german_treename(latname):
    # Translate JSON name to real german tree name
    return constants.treeNames_l[latname]


def get_latname_treename(germname):
    # translate german treename to JSON name
    return constants.treeNames_g[germname]


def create_points(topx, topy, botx, boty, dist, batch_size_sqrt):
    x_start = min(topx, botx)
    y_start = min(topy, boty)
    x_end = max(topx, botx)
    y_end = max(topy, boty)

    curcord = [x_start, y_start]

    x_add = dist / get_lat_fac()
    y_add = dist / get_long_fac(curcord[0])

    cords = [curcord]

    batch_dist_x = (x_end - x_start) / batch_size_sqrt
    batch_dist_y = (y_end - y_start) / batch_size_sqrt
    cur_batch_x = x_start
    cur_batch_y = y_start
    batches = []

    batch_counter_x = 0
    batch_counter_y = 0

    r =  (x_end-x_start) / x_add

    max_batch_counter_x = int(((x_end-x_start) / x_add) / batch_size_sqrt)
    max_batch_counter_y = int(((y_end - y_start) / y_add) / batch_size_sqrt)

    while batch_counter_y < max_batch_counter_y:
        while batch_counter_x < max_batch_counter_x:

            cord = [cur_batch_x - batch_dist_x, cur_batch_y - batch_dist_y]
            while cord[1] + y_add < cur_batch_y:
                while cord[0] + x_add < cur_batch_x:
                    cord = [cord[0] + x_add, cord[1]]
                    cords.append(cord)
                cord = [cur_batch_x - batch_dist_x, cord[1] + y_add]
                cords.append(curcord)
                y_add = dist / get_long_fac(curcord[0])
            batches.append(cords)
            cords = []
            cur_batch_x += batch_dist_x
            batch_counter_x += 1
        cur_batch_y += batch_dist_y
        batch_counter_y += 1


    #while curcord[1] + y_add < y_end:
    #    while curcord[0] + x_add < x_end:
    #        curcord = [curcord[0] + x_add, curcord[1]]
    #        cords.append(curcord)
    #    curcord = [curcord[0], curcord[1] + y_add]
    #    cords.append(curcord)
    #    y_add = dist / get_long_fac(curcord[0])
    print("Created amount of datapoints: " + str(len(cords)))
    return batches

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


def create_patch(start_coord_x, start_coord_y, end_coord_x, end_coord_y, distance_m):

    return 0


def create_patches():
    return 0

points = create_points(49.95424784938799, 8.681484215611686, 49.781966423998526, 9.029333148681545, 0.1, 2)
o = 2