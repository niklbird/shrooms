import mushroom
import datetime
import utils
import soil
from numba import jit


def tree_value(mushroom, tree_type: str):
    com_fac = 1
    if mushroom.attr['commonness'] == "Selten":
        com_fac = 0.33
    hardwood = 0
    if tree_type == "Mischwaelder" or tree_type == "Laubwaelder":
        hardwood = 1
    softwood = 0
    if tree_type == "Mischwaelder" or tree_type == "Nadelwaelder":
        softwood = 1
    wt = mushroom.attr['woodtype']

    wood_type_factor = min(wt[0] * hardwood + wt[1] * softwood, 1)

    wiesen = ["Wiesen und Weiden", "Natuerliches Gruenland", "Heiden und Moorheiden", "Wald-Strauch-Uebergangsstadien"]
    if "wiese" in mushroom.attr['habitat'].lower() and tree_type in wiesen:
        wood_type_factor = 1.0

    # In the future, this could also consider specific trees
    return wood_type_factor * com_fac


def soil_value(mushroom, soil):
    mushroom_ph = mushroom.attr['ph'].lower()
    ph_val = 2.0
    if mushroom_ph != "All":
        ph_val = int(mushroom_ph in soil.attr["ph"])*2
    if ph_val == 0:
        ph_val = 0.5

    soil_val = 1.5
    sv = mushroom.attr["soil"]
    if mushroom.attr["soil"] != "All" and not mushroom.attr["soil"].lower() in soil.attr["typ"]:
        soil_val = 1

    soil_score = float(soil.attr["score"])
    return ph_val * soil_val #* soil_score

def get_month_factors(month):
    # Factor that indicates if mushroom is in season
    ret = {}
    mushroooms = mushroom.read_mushroom_XML('../data/mushrooms_databank.xml')
    for s_name in mushroooms.keys():
        ret[s_name] = int(
            int(mushroooms[s_name].attr['seasonStart']) <= month <= int(mushroooms[s_name].attr['seasonEnd']))
    return ret


def calc_static_values(patches):
    # Calculate weather-independent factor for each point
    mushrooms = mushroom.read_mushroom_XML('../data/mushrooms_databank.xml')
    soils = soil.read_soil_XML('../data/soil_databank.xml')
    counter = 0
    for patch in patches:
        counter += 1
        for date in patch.dates:
            trees = date.trees
            soiL_t = date.soil.lower()
            for shroom in mushrooms.values():
                #print("-->")
                #print(date.soil)
                #print(soil.soil_value(shroom, soils[soiL_t]))
                tree_val = tree_value(shroom, trees)
                if soiL_t not in soils:
                    print(soiL_t)
                    soil_val = 0.0
                else:
                    soil_val = soil_value(shroom, soils[soiL_t])
                if tree_val == 0 or soil_val == 0:
                    date.mushrooms[shroom.attr['name']] = 0.0
                else:
                    date.mushrooms[shroom.attr['name']] = (tree_val + soil_val) / 2


def calc_dynamic_value(patches):
    # Calculate the actual mushroom probabilities
    month_factors = get_month_factors(datetime.datetime.today().month)

    for patch in patches:
        weather = patch.weather_data
        temperatures = []
        rains = []
        humidities = []

        # Look at weather of last 30 days
        for i in range(30, 1, -1):
            ts = utils.format_timestamp(datetime.datetime.today() - datetime.timedelta(days=i))
            if weather[ts] is None or weather[ts]['temperature'] is None or weather[ts]['rain'] is None:
                temperatures.append(0)
                rains.append(0)
                humidities.append(50)
                continue
            temperatures.append(weather[ts]['temperature'])
            rains.append(weather[ts]['rain'])
            humidities.append(weather[ts]['humidity'])

        rain_val, temp_val, hum_val = environment_factor(rains, temperatures, humidities)

        # Factors may have to be tweaked
        dynamic_factor = (2 * rain_val + 1 * temp_val + 0.7 * hum_val) / 3.7

        for date in patch.dates:
            for shroom in date.mushrooms.keys():
                # Base-Factor, Seasonality, Environment-Factor
                # min(date.mushrooms[shroom] * month_factors[shroom] * dynamic_factor, 1)
                date.probabilities[shroom] = min(date.mushrooms[shroom] * dynamic_factor, 1)
                #date.probabilities[shroom] = min(date.mushrooms[shroom] * dynamic_factor * 100, 1)


def temp_deviation(temp, opt_val):
    if temp < opt_val:
        return temp / opt_val
    elif temp > opt_val + 5:
        return opt_val / temp
    else:
        return 1.0


def environment_factor(rain, temperature, humidity):
    # The factorization of the values can be tweeked, it's just a gross estimation
    # First look at 28 days ago to 14 days ago
    ra = 0
    temp = 0
    hum = 0
    optimal_temp = 15
    optimal_rain = 0.3
    optimal_humidty = 90
    for j in range(0, 14):
        # If 10mm is perfect amount, this measures the normalized contribution
        ra += 0.5 * min(rain[j], 25) / 14
        temp += 0.3 * temp_deviation(temperature[j], optimal_temp) / 14
        if humidity[j] is None:
            humidity[j] = 60
        hum += humidity[j] / optimal_humidty / 14
    # Emphasize 2-1 week ago
    for j in range(14, 21):
        ra += 3 * min(rain[j], 25) / 7
        temp += 0.75 * temp_deviation(temperature[j], optimal_temp) / 7
        if humidity[j] is None:
            humidity[j] = 60
        hum += humidity[j] / optimal_humidty / 7
    for j in range(21, 28):
        ra += 0.75 * min(rain[j], 25) / 7
        temp += 2 * temp_deviation(temperature[j], optimal_temp) / 7
        if humidity[j] is None:
            humidity[j] = 60
        hum += humidity[j] / optimal_humidty / 7
    norm_rain = 0.5 * 14 + 3 * 7 + 7 * 0.75
    norm_temp = 3
    norm_hum = 2.0
    return min(ra / norm_rain / optimal_rain, 3), min(temp / norm_temp, 3), hum / norm_hum