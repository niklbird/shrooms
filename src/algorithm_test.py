import random
from numba import jit
import numpy as np

'''
The sole purpose of this file is to test algorithm ideas.
'''


size = 20


def create_array():
    global size
    ret = []
    for i in range(size):
        for j in range(size):
            ret.append([i, j, max(random.randint(0, 3) - 2, 0)])
    return ret


def print_arr(arr):
    global size
    for i in range(size ** 2):
        if i % size == 0:
            print("")
        if arr[i][2] == 0:
            print("O", end='')
        else:
            print("X", end='')


def print_shapes(arr):
    global size
    for i in range(size ** 2):
        printed = False
        if i % size == 0:
            print("")
        for j in range(len(arr)):
            if i in arr[j][0]:
                if arr[j][1] == 0:
                    print("O", end='')
                else:
                    print("X", end='')
                printed = True
                break
        if not printed:
            print("-", end='')


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


@jit(nopython=True)
def get_cross_neighbors(i, length):
    ret = []
    max_ind = length ** 2
    top = i + 1
    bot = i - 1
    left_1 = top + length
    left_2 = top - length
    right_1 = bot + length
    right_2 = bot - length
    if left_1 < max_ind:
        ret.append(left_1)
    if left_2 >= 0:
        ret.append(left_2)
    if right_2 >= 0:
        ret.append(right_2)
    if right_1 < max_ind:
        ret.append(right_1)
    return ret


@jit(nopython=True)
def is_neighbor(i, j, length):
    return j == i + 1 or j == i - 1 or j == i + length or j == i - length


def ind_to_cord(i, l):
    return np.mod(i, l), int(np.square(l) / (i + 0.1))


def dist_to_middle(cord, length):
    middle = ((length - 1) / 2, (length - 1) / 2)
    dist = middle - cord
    return np.square(dist[0]) + np.square(dist[1])


def linear_path_finder(points, length):
    # solution = [cur_point]
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
                if is_neighbor(j, solution_index[-1]):
                    pos_neighbors.append(j)
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
    return cur_shape


def algorithm(arr):
    global size
    already_handled = []
    shapes = []
    for i in range(len(arr)):
        if i in already_handled:
            continue

        cur_shape = [i]
        already_handled.append(i)
        neighbors = []
        neighbors_tmp = get_neighbors(i, size)

        for k in neighbors_tmp:
            if k not in already_handled:
                neighbors.append(k)

        cur_val = arr[i][2]
        while len(neighbors) > 0:
            j = neighbors[0]

            if arr[j][2] != cur_val:
                pass

            else:
                cur_shape.append(j)
                already_handled.append(j)
                neighbors_tmp_2 = get_neighbors(j, size)

                append = 0
                for k in neighbors_tmp_2:
                    if k not in already_handled and k not in neighbors:
                        if arr[k][2] == cur_val:
                            neighbors.append(k)
                    if arr[k][2] != cur_val:
                        append = 1
                neighbors_tmp_3 = get_cross_neighbors(j, size)
                for k in neighbors_tmp_3:
                    if arr[k][2] != cur_val:
                        append = 1
                # If no shape border
                if append == 0 and 20 < j < 20 * 19 and j % 20 != 0 and j % 20 != 20-1:
                    cur_shape.remove(j)
            neighbors.remove(j)
        shapes.append([cur_shape, cur_val])
    return shapes


def algorithm3(arr):
    l = 20
    output = {}
    for i in range(len(arr)):
        shape = arr[i]
        for j in range(len(shape)):
            point = shape[j][0]
            out = ""
            required_points = [0, 0, 0, 0]
            # Up
            if not point - l >= 0 and not point - l in shape[0]:
                required_points[0] = 1
                required_points[1] = 1

            # Right
            if not (point + 1) % 20 != 0 and not point + 1 in shape[0]:
                required_points[1] = 1
                required_points[2] = 1

            # Down
            if not point + l < l * l and not point + l in shape[0]:
                required_points[2] = 1
                required_points[3] = 1

            # Left
            if not point % l != 0 and not point - 1 in shape[0]:
                required_points[0] = 1
                required_points[3] = 1
            output[point] = required_points
    return output

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
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if int(i*l + j) not in used_shapes:
                shapes.append([rows[i][0][j], rows[i][1][j]])
            else:
                print("already used")
    print(len(shapes))




def algorithm4(arr):
    l = 20
    shapes = []
    rows = []
    for c in range(l):
        row_tmp = []

        final_shapes_row = []
        for r in range(l):
            v = c * l + r
            point = arr[v]
            shape = []
            # First Element in Row is starting point
            if r == 0 or row_tmp[r - 1][1] != point[2]:
                shape.append([point[0] + 0.5, point[1] + 0.5])
                shape.append([point[0] + 0.5, point[1] - 0.5])
                shape.append([point[0] - 0.5, point[1] - 0.5])
                shape.append([point[0] - 0.5, point[1] + 0.5])
                row_tmp.append([shape, point[2]])
                final_shapes_row.append(r)
            else:
                # This belongs to the same shape as previous point
                final_shapes_row.remove(r - 1)
                shape = row_tmp[r - 1]
                new_shape = [[point[0] + 0.5, point[1] + 0.5], shape[0][1], shape[0][2],
                             [point[0] - 0.5, point[1] + 0.5]]
                # Keep upper left and lower left, generate new upper right and lower right points
                row_tmp.append([new_shape, shape[1]])
                final_shapes_row.append(r)
        shapes.extend(np.array(row_tmp)[final_shapes_row][:,0])
        rows.append([np.array(row_tmp)[final_shapes_row][:,0], np.array(row_tmp)[final_shapes_row][:,1]])
    combine_rows(rows)
    return shapes

def algorithm2(arr):
    l = 20
    output = []
    shape_dic = {}
    for i in range(len(arr)):
        shape = arr[i]
        came_from = 0
        used = []
        point = shape[0][0]
        while point != -1:
            # Go up
            if point - l >= 0 and point - l in shape[0] and point - l not in used:
                used.append(point)
                point = point - l
            # Go to right
            elif (point + 1) % 20 != 0 and point + 1 in shape[0] and point + 1 not in used:
                used.append(point)
                point = point + 1

            # Go down
            elif point + l < l*l and point + l in shape[0] and point + l not in used:
                used.append(point)
                point = point + l

            # Go to left
            elif point % l != 0 and point - 1 in shape[0] and point - 1 not in used:
                used.append(point)
                point = point - 1

            else:
                pass

        if i < l or i > 1  * (l - 1):
            continue



def main():
    arr = create_array()
    print_arr(arr)
    shapes = algorithm(arr)
    print("")
    print("")
    print_shapes(shapes)
    out = algorithm4(arr)
    a = 0


if __name__ == '__main__':
    main()
