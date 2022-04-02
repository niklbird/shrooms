import json
import datetime

def check_format(value: str):
    values = value.split(",")
    if len(values) != 4:
        return False
    for v in values:
        try:
            f = float(v)
            if f > 255:
                return False
        except Exception:
            return False
    return True


day = datetime.datetime.today().day

DATA_DIR = f"/var/www/server_shrooms/data/data{day}.json"
UPDATE_DIR = "/home/niklas/shroom_data/update_data.txt"

DATA_GRAINY_DIR = f"/var/www/server_shrooms/data/data_grainy{day}.json"
UPDATE_GRAINY_DIR = "/home/niklas/shroom_data/update_data_grainy.txt"
def update_data():
    with open(DATA_DIR, 'r') as outfile:
        data = json.load(outfile)

    with open(UPDATE_DIR, 'r') as outfile:
        file_data = outfile.read()

    with open(DATA_GRAINY_DIR, 'r') as outfile:
        data_g = json.load(outfile)

    with open(UPDATE_GRAINY_DIR, 'r') as outfile:
        file_data_g = outfile.read()

    up_data = file_data.split(";")
    up_data = up_data[:len(up_data) - 1]

    up_data_g = file_data_g.split(";")
    up_data_g = up_data_g[:len(up_data_g) - 1]


    if len(data["features"]) != len(up_data):
        return False

    if len(data_g["features"]) != len(up_data_g):
        return False

    for i in range(len(data["features"])):
        if not check_format(up_data[i]):
            return False

        feature = data["features"][i]
        feature['properties']["color"] = f"rgba({up_data[i]})"

    for i in range(len(data_g["features"])):
        if not check_format(up_data_g[i]):
            return False

        feature = data_g["features"][i]
        feature['properties']["color"] = f"rgba({up_data_g[i]})"

    with open(DATA_DIR, 'w') as outfile:
        json.dump(data, outfile)

    with open(DATA_GRAINY_DIR, 'w') as outfile:
        json.dump(data_g, outfile)

if __name__ == "__main__":
    update_data()

