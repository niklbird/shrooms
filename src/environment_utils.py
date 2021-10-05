import datetime
from dwdweather import DwdWeather
from utils import get_distance
import csv
import numpy as np
import os.path
import io_utils

dwd = DwdWeather(resolution="daily")
querried = {};


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


def get_weather_data_id(station_id):
    # Query DWD for weather data at station id
    timestamp = datetime.now()
    timestamp = datetime(timestamp.year, timestamp.month, timestamp.day - 1, 12)
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


def find_closest_station(stations, coord):
    best_dist = 100000
    best_stat = 0
    for i in range(len(stations)):
        station = stations[i]
        dist = get_distance(float(station[1]), float(station[2]), coord[0], coord[1])
        if dist < best_dist:
            best_dist = dist
            best_stat = station[0]
    return best_stat

o = dwd.stations()
oa = 0