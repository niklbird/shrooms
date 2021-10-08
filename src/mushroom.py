import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt

class Mushroom:

    def __init__(self, attributes: dict):
        self.attr = attributes



    def soil_value(self, ph, soils):
        ph_val = 1
        if self.attr["ph"] != "All":
            ph_val = int(self.attr["ph"] == ph)
        soil_val = 1
        if self.attr["soil"][0] != "All":
            soil_val = 0.2
            for soil in soils:
                if soil in self.attr["soil"]:
                    soil_val = 1
                    break
        return ph_val * soil_val

    def humidity_value(self, humidity, temperature):
        # Humidity of last 30 days
        # First look at 28 days ago to 14 days ago
        val = 0
        for i in range(0, 14):
            # If 10mm is perfect amount, this measures the normalized contribution
            val += 0.5 * max(humidity[i], 25) / (10 * 14)
        for i in range(14, 21):
            val += max(humidity[i], 25) / (10 * 7)
        for i in range(21, 28):
            val += 0.75 * max(humidity[i], 25) / (10 * 7)
        norm = 0.5 * 14 + 7 + 7 * 0.75
        return 0

    def time_value(self, cur_month):
        return self.attr["seasonStart"] <= cur_month <= self.attr["seasonEnd"]


def tree_value(mushroom, trees: dict):
    p_all = trees['coverage']
    if p_all == 0:
        return 0
    wood_type_factor = (mushroom.attr['woodtype'][0] * trees['hardwood'] +
                        mushroom.attr['woodtype'][1] * trees['softwood'])
    if mushroom.attr["trees"][0] == "ALL":
        return wood_type_factor / p_all
    val = wood_type_factor / 2
    # This could be adapted for exclusive trees
    # Give benefit
    for tree in trees.keys():
        if tree in mushroom.attr["trees"]:
            val += trees[tree]
    return min(val / p_all, 1)


def readXML():
    mushrooms = {}
    root = ET.parse('../data/mushrooms_databank.xml').getroot()
    for type_tag in root.findall('mushroom'):
        shroom = {};
        for child in type_tag:
            if "Knollen" in child.text:
                a = 0
            if child.tag == "woodtype":
                shroom[child.tag] = (int("Hardwood" in child.text), int("Softwood" in child.text))
            elif child.tag == "trees" or child.tag == "habitat":
                shroom[child.tag] = child.text.lower().split(",")
            else:
                shroom[child.tag] = child.text
        mushrooms[shroom["name"]] = Mushroom(shroom)

    return mushrooms

def humidity_value(humidity, temperature):
    # The factorization of the values can be tweeked, it's just a gross estimation
    # First look at 28 days ago to 14 days ago
    val = 0
    temp = 0
    for j in range(0, 14):
        # If 10mm is perfect amount, this measures the normalized contribution
        val += 0.5 * min(humidity[j], 25) / 14
        temp += 0.3 * (temperature[j] / 20) / 14
    # Emphazise 2-1 week ago
    for j in range(14, 21):
        val += 1.5*min(humidity[j], 25) / 7
        temp += 0.75 * (temperature[j] / 20) / 7
    for j in range(21, 28):
        val += 0.75 * min(humidity[j], 25) / 7
        temp += 2 * (temperature[j] / 20) / 7

    norm_hum = 0.3 * (0.5 * 14 + 1.5*7 + 7 * 0.75)
    norm_temp = 2.5
    return min(val / norm_hum, 3), min(temp / norm_temp, 3)

def sanity_test():
    with open('rain.txt.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        rowcounter = 0
        val = []
        val2 = []
        ns = 0
        c = 0
        for row in spamreader:
            if c == 0:
                c += 1
                continue
            ns += float(row[3])
            rowcounter += 1
            c += 1
            if (rowcounter % 24) == 0:
                val.append((str(row[1])[0: len(str(row[1])) - 2], ns))
                val2.append(ns)
                rowcounter = 0
                ns = 0
        hum_res = []
        for i in range(40, len(val2)):
            hum_res.append(humidity_value(val2[i - 30:i], 0))
        curMonth = '10'
        res = []
        i = 0
        while i < len(val):
            con = 0
            while i < len(val) and val[i][0][4:6] == curMonth:
                con += float(val[i][1])
                i += 1
            if i >= len(val):
                break
            res.append((val[i][0][0:4] + '_' + curMonth, con))
            curMonth = val[i][0][4:6]
        yo = 0
        plt.plot(hum_res)
        plt.show()
        cnt = 0
        for i in range(len(hum_res)):
            if (hum_res[i] > 1):
                print(val[i][0])
                cnt += 1
        print(str(cnt) + " of " + str(len(hum_res)))

mushrooms = readXML()
good = [10, 10, 10, 10, 10, 10,10, 10, 10,10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 21, 22, 23, 24, 25, 25, 25]
bad = good[::-1]
print(len(good))
a, e = humidity_value(good, good)
c, d = humidity_value(bad, bad)
o = 0