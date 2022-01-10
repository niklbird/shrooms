import pickle
import json
from pyproj import Proj
from numba import jit
import numpy as np
import constants


def dump_to_file(arr, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)
        fp.close()


def read_dump_from_file(filename):
    with open(filename, 'rb') as fp:
        arr = pickle.load(fp)
        fp.close()
        return arr


def project_coordinate(coordinate, projection):
    # Use projection on coordinate
    p = Proj(projection)
    return p(coordinate[0], coordinate[1])


@jit(nopython=True)
def get_lat_fac():
    # Translate angle distance to km distance latitudial
    return 110.574


@jit(nopython=True)
def get_long_fac(longitude):
    # Translate angle distance to km distance longitudinal
    return np.abs(111.32 * np.cos(longitude))


@jit(nopython=True)
def get_neighbors(i, length):
    ret = []
    max_ind = length ** 2
    top = i + 1
    bot = i - 1
    left = i - length
    right = i + length
    if top < max_ind:
        ret.append(top)
    if bot >= 0:
        ret.append(bot)
    if left >= 0:
        ret.append(left)
    if right < max_ind:
        ret.append(right)
    return ret


def linear_path_finder(points, length):
    # solution = [cur_point]
    points = points[0]
    final_shapes = []
    used = 0
    solution_index = [0]
    cur_shape = [0]
    while used < len(points):
        for i in range(1, len(points)):
            pos_neighbors = []

            for j in range(i, len(points)):
                if j in solution_index:
                    continue
                if is_neighbor(j, solution_index[-1], length):
                    pos_neighbors.append(j)

            if len(pos_neighbors) == 0 and j not in solution_index:
                solution_index.append(j)
                cur_shape.append(j)
                used += 1
                break

            elif len(pos_neighbors) == 0:
                continue

            best_j = pos_neighbors[0]
            if len(pos_neighbors) > 1:
                best_dist = 10000
                for neighbor in pos_neighbors:
                    dist_tmp = dist_to_middle(ind_to_cord(neighbor, length), length)
                    if dist_tmp < best_dist:
                        best_dist = dist_tmp
                        best_j = neighbor

            solution_index.append(best_j)
            cur_shape.append(best_j)
            used += 1

        final_shapes.append(cur_shape)
        cur_shape = []
    return final_shapes


def unite_shapes(arr):
    already_handled = []
    leng = int(np.sqrt(len(arr)))
    shapes = []
    for i in range(len(arr)):
        if i in already_handled:
            continue

        cur_shape = [i]
        already_handled.append(i)
        neighbors = []
        neighbors_tmp = get_neighbors(i, leng)

        for k in neighbors_tmp:
            if k not in already_handled:
                neighbors.append(k)

        cur_val = arr[i][1]
        while len(neighbors) > 0:
            j = neighbors[0]

            if arr[j][1] != cur_val:
                pass

            else:
                cur_shape.append(j)
                already_handled.append(j)
                neighbors_tmp_2 = get_neighbors(j, leng)

                append = 0
                for k in neighbors_tmp_2:
                    if k not in already_handled and k not in neighbors:
                        neighbors.append(k)
                    if arr[k][1] == cur_val:
                        append += 1
                if append == 4:
                    print("Removing " + str(j))
                    cur_shape.remove(j)
            neighbors.remove(j)
        shapes.append(cur_shape)
    return shapes


def ind_to_cord(i, l):
    return np.mod(i, l), int(np.square(l) / (i + 0.1))


def dist_to_middle(cord, length):
    middle = ((length - 1) / 2, (length - 1) / 2)
    dist = np.square(middle[0] - cord[0]) + np.square(middle[1] - cord[1])
    return dist


@jit(nopython=True)
def is_neighbor(i, j, length):
    return j == i + 1 or j == i - 1 or j == i + length or j == i - length


def combine_rows(rows):
    used_shapes = []
    shapes = []
    l = 20
    for i in range(len(rows) - 1):
        row = rows[i][0]
        next_row = rows[i + 1][0]
        j = k = 0
        while j < len(row) and k < len(next_row):
            if not l*i + j in used_shapes and not l*(i+1) + k in used_shapes and rows[i][1][j] == rows[i + 1][1][k]:
                shape = row[j]
                shape_new = next_row[k]
                point_0 = shape[1]
                point_1 = shape[0]
                point_2 = shape_new[1]
                point_3 = shape_new[0]
                if point_0[1] < point_2[1] < point_1[1] or point_0[1] < point_3[1] < point_1[1]:
                    # The shapes touch -> Combine to larger shape
                    final_shape = [[shape[2], shape[1], shape_new[2], shape_new[1],
                                   shape_new[0], shape_new[3], shape[0], shape[3]], rows[i][1][j]]
                    used_shapes.append(int(i * l + j))
                    used_shapes.append(int((i + 1) * l + k))
                    shapes.append(final_shape)
                    print("Reduced")
                if point_1[1] > point_3[1]:
                    k += 1
                else:
                    j += 1
            else:
                j += 1
    for i in range(len(rows)):
        for j in range(len(rows[i][0])):
            if int(i*l + j) not in used_shapes:
                shapes.append([rows[i][0][j], rows[i][1][j]])
            else:
                print("already used")
    return shapes

def algorithm4(arr, dist_x, dist_y):
    l = 10
    shapes = []
    rows = []
    for c in range(l):
        row_tmp = []

        final_shapes_row = []
        for r in range(l):
            v = c * l + r
            point = arr[v][0]
            shape = []
            # First Element in Row is starting point
            if r == 0 or row_tmp[r - 1][1] !=  arr[v][1]:
                shape.append([point[0] + dist_y, point[1] + dist_x])
                shape.append([point[0] + dist_y, point[1] - dist_x])
                shape.append([point[0] - dist_y, point[1] - dist_x])
                shape.append([point[0] - dist_y, point[1] + dist_x])
                row_tmp.append([shape, arr[v][1]])
                final_shapes_row.append(r)
            else:
                # This belongs to the same shape as previous point
                final_shapes_row.remove(r - 1)
                shape = row_tmp[r - 1]
                new_shape = [[point[0] + dist_y, point[1] + dist_x], shape[0][1], shape[0][2],
                             [point[0] - dist_y, point[1] + dist_x]]
                # Keep upper left and lower left, generate new upper right and lower right points
                row_tmp.append([new_shape, shape[1]])
                final_shapes_row.append(r)
        shapes.extend(np.array(row_tmp)[final_shapes_row][:,0])
        rows.append([np.array(row_tmp)[final_shapes_row][:,0], np.array(row_tmp)[final_shapes_row][:,1]])
    return combine_rows(rows)
    #return shapes




def write_to_GEOJSON(patches):
    data = {}
    crs = {'type': 'name', 'properties': {'name': 'EPSG:4326'}}
    data['type'] = 'FeatureCollection'
    data['crs'] = crs
    data['features'] = []
    print("Amount of dates: " + str(len(patches) * len(patches[0].dates)))
    for patch in patches:
        corner = patch.corners[0]

        dist_x = constants.point_dist / get_lat_fac() / 2.0
        dist_y = constants.point_dist / get_long_fac(corner[0]) / 2.0

        to_reduce = []

        for date in patch.dates:
            point = date.coord

            prop = date.probabilities['Steinpilz']

            to_reduce.append([[point[1], point[0]], prop])

            #to_reduce.append([[point[1] + dist_y, point[0] + dist_x], prop])
            #to_reduce.append([[point[1] + dist_y, point[0] - dist_x], prop])
            #to_reduce.append([[point[1] - dist_y, point[0] - dist_x], prop])
            #to_reduce.append([[point[1] - dist_y, point[0] + dist_x], prop])
            #to_reduce.append([[point[1] + dist_y, point[0] + dist_x], prop])

            #coordinates = [[point[1] + dist_y, point[0] + dist_x], [point[1] - dist_y, point[0] + dist_x],
            #               [point[1] - dist_y, point[0] - dist_x], [point[1] + dist_y, point[0] - dist_x],
            #               [point[1] + dist_y, point[0] + dist_x]]



            #to_reduce.append((coordinates, prop))
        unified = algorithm4(to_reduce, dist_x, dist_y)

        print("Before: " + str(len(to_reduce)))
        print("Combination: " + str(len(unified)))

        for j in range(len(unified)):
            new_cords = unified[j][0]
            new_cords.append(new_cords[0])
            geom = {}
            props = {'color': 'rgba(0, 255, 0, ' + str(min(0.5 * unified[j][1], 0.5)) + ')'}
            geom['type'] = 'Polygon'
            # geom['coordinates'] = [coordinates]
            geom['coordinates'] = [new_cords]
            data['features'].append({
                'type': 'Feature',
                'geometry': geom,
                'properties': props
            })

    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)
    return
