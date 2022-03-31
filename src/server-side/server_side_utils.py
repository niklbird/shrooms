import json
import constants

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

def new_file():
    with open(constants.pwd + f'/../web/data0.json', 'r') as outfile:
        data = json.load(outfile)

    ret = ""
    for i in range(len(data["features"])):
        ret += "0,132,0,0.5;"

    ret = ret[:len(ret) - 1]
    with open(constants.pwd + f'/../web/update_data.txt', 'w') as outfile:
        outfile.write(ret)

DATA_DIR = "/var/www/shrooms/web/data.json"
UPDATE_DIR = "/home/niklas/shroom_data/update_data.txt"

def update_data():
    with open(DATA_DIR, 'r') as outfile:
        data = json.load(outfile)

    with open(UPDATE_DIR, 'r') as outfile:
        file_data = outfile.read()

    up_data = file_data.split(";")
    up_data = up_data[:len(up_data) - 1]

    if len(data["features"]) != len(up_data):
        return False

    for i in range(len(data["features"])):
        if not check_format(up_data[i]):
            return False

        feature = data["features"][i]
        feature['properties']["color"] = f"rgba({up_data[i]})"

    with open(DATA_DIR, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    update_data()

