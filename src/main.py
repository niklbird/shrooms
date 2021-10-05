def load_tree_data():
    # Load in tree data
    file = open("test.geojson", "r")
    js = preprocess(geojson.load(file)["features"])
    return js


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
    return treeNames_l[latname]


def get_latname_treename(germname):
    # translate german treename to JSON name
    return treeNames_g[germname]

def dump_to_file(arr, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)
        fp.close()


def read_dump_from_file(filename):
    with open(filename, 'rb') as fp:
        arr = pickle.load(fp)
        fp.close()
        return arr