import datetime
from dwdweather import DwdWeather
import csv
import numpy as np
from numpy import genfromtxt


dwd = DwdWeather(resolution="daily")
querried = {};

def get_stations():
    stations = []
    my_data = genfromtxt("C:/Users/Niklas/Desktop/GIT/shrooms/data/ha_messnetz.csv", delimiter=',')[1:]
    for data in my_data:
        dic = {}
        dic['station_id'] = int(data[0])
        dic['geo_lat'] = data[9]
        dic['geo_lon'] = data[10]
        dic['name'] = data[1]
        stations.append(dic)
    return stations


def get_weather_data_cords(coordinates):
    # Query DWD station closest to coordinates
    closest = dwd.nearest_station(lon=coordinates[1], lat=coordinates[0])
    timestamp = datetime.now()
    timestamp = datetime(timestamp.year, timestamp.month, timestamp.day - 1, 12)
    station_id = closest["station_id"]
    if station_id in querried:
        result = querried[station_id]
        print("Using stored result")
    else:
        result = dwd.query(station_id=station_id, timestamp=timestamp)
        querried[station_id] = result
        print("New query")
    return result


def get_weather_data_id(station_id, timestamp):
    # Query DWD for weather data at station id
    #timestamp = datetime.now()
    #timestamp = datetime(timestamp.year, timestamp.month, timestamp.day - 1, 12)
    result = 0
    if station_id in querried:
        result = querried[station_id]
        print("Using stored result")
    else:
        result = dwd.query(station_id=int(station_id), timestamp=timestamp)
        querried[station_id] = result
        print("New query")
    return result


def read_dwd_stations(filename):
    # Read all stations from file
    stations = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                stations.append([row[0], float(row[9].replace(",", ".")), float(row[10].replace(",", "."))])
        stations = np.array(stations)
        return np.array(stations)


def closest_station2(coord):
    return dwd.nearest_station(coord[1], coord[0])