import random
from numba import jit


def create_array():
    ret = []
    for i in range(10):
        for j in range(10):
            ret.append([i, j, random.randint(0, 1)])
    return ret



def print_arr(arr):
    for i in range(10**2):
        if i % 20 == 0:
            print("")
        if arr[i][2] == 0:
            print("O", end='')
        else:
            print("X", end='')


@jit(nopython=True)
def get_neighbors(i, length):
    ret = []
    max_ind = length**2
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


def algorithm(arr):
    already_handled = []
    shapes = []
    for i in range(len(arr)):
        if i in already_handled:
            continue

        cur_shape = [i]
        already_handled.append(i)
        neighbors = []
        neighbors_tmp = get_neighbors(i, 10)

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
                neighbors_tmp_2 = get_neighbors(j, 10)

                for k in neighbors_tmp_2:
                    if k not in already_handled and k not in neighbors:
                        neighbors.append(k)

            neighbors.remove(j)
        shapes.append(cur_shape)
    return shapes


def main():
    arr = create_array()
    print_arr(arr)
    shapes = algorithm(arr)
    a = 0


if __name__ == '__main__':
    main()
