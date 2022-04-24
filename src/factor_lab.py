'''
This file provides utilities to experiment with the factor calculations
'''

import csv
import numpy as np
from matplotlib import pyplot as plt
import factor_calculations


def data_to_months(arr, factors):
    i = 1
    years = []
    while i < len(arr):
        cur_year = arr[i][1][0:4]
        a = 0
        year = []
        while i < len(arr) and arr[i][1][0:4] == cur_year:
            cur_month = arr[i][1][4:6]
            month = []
            while i < len(arr) and arr[i][1][4:6] == cur_month:
                month.append([arr[i], factors[i]])
                i += 1
            year.append(month)
        print("new year")
        years.append(year)

    return np.array(years)

def calc_factors(arr):
    arr_rev = list(reversed(arr))
    ret = [(0,0,0) for i in range(28)]
    for i in range(len(arr_rev) - 28):
        values = np.array(arr_rev[i: i + 28])
        rain = values[:,6]
        temp = values[:,13]
        humidity = values[:,14]
        factor = factor_calculations.environment_factor(rain, temp, humidity)
        ret.append(factor)
    return list(reversed(ret))


def load_archive_data():
    '''
    0: Station-ID
    1: Date
    6: Reciprocation
    13: Average tmp
    14: Relative Humidity
    15: Max tmp
    :return:
    '''
    arr = []
    with open('../data/produkt_klima.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            r = []
            for val in row:
                r.append(val.strip())
            arr.append(r)
    return arr[1:]


def calc_archive_weather_factors():
    pass


def plot_period():
    pass


def extract_month():
    pass


'Calculate characteristic values of the distribution like average, stand. deviation, expected value'


def calculate_characteristic_values():
    pass


def plot_month(data, month_i):
    month_i -= 1
    values = np.array(data[:, month_i])
    day_avg = [0 for i in range(len(values[0]))]
    day_max = [0 for i in range(len(values[0]))]
    day_min = [100 for i in range(len(values[0]))]
    all_vals = []
    for month in values:
        for i in range(len(month)):
            day = month[i][0]
            # Average tmp is at index 13
            all_vals.append([day[6], day[13], day[14], month[i][1]])
            temp = float(day[14])
            if i >= len(day_avg):
                day_avg.append(0)
                day_max.append(0)
                day_min.append(0)
            day_avg[i] += temp
            day_max[i] = max(day_max[i], temp)
            day_min[i] = min(day_min[i], temp)
    for i in range(len(day_avg)):
        day_avg[i] = day_avg[i] / len(data)
    #plt.plot(day_avg)
    #plt.plot(day_max)
    #plt.plot(day_min)
    #plt.show()
    return all_vals

def processing(all_vals):
    rain = []
    temperature = []
    humidity = []
    for i in range(len(all_vals)):
        rain.append(all_vals[i][3][0])
        temperature.append(all_vals[i][3][1])
        humidity.append(all_vals[i][3][2])
    #plt.hist(humidity)
    #plt.show()

    d_f = []
    for i in range(len(all_vals)):
        dynamic_factor = ((1 - (1 - rain[i])**2) * 1 * (1 - (1 - temperature[i])**2) * 1 * (1 - (1 - humidity[i])**2))
        d_f.append(dynamic_factor)
    plt.hist(d_f)
    plt.show()

    return np.average(rain), np.average(temperature), np.average(humidity)


ret = load_archive_data()
fac = calc_factors(ret)
data = data_to_months(ret, fac)
#print(fac[:10])

rains = []
temps = []
hums = []
dy_fac = []
for i in range(12):
    print(i)
    all_val = plot_month(data, i)
    rain, temp, hum = processing(all_val)
    dynamic_factor = (2 * rain + 1 * temp + 1 * hum) / 4
    rains.append(rain)
    temps.append(temp)
    hums.append(hum)
    dy_fac.append(dynamic_factor)

#plt.plot(dy_fac)
#plt.show()
