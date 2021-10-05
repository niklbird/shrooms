def dump_to_file(arr, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)
        fp.close()


def read_dump_from_file(filename):
    with open(filename, 'rb') as fp:
        arr = pickle.load(fp)
        fp.close()
        return arr

def load_tree_data():
    # Load in tree data
    file = open("test.geojson", "r")
    js = preprocess(geojson.load(file)["features"])
    return js