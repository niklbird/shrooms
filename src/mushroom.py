import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt

class Mushroom:

    def __init__(self, attributes: dict):
        self.attr = attributes

    def tree_value(self, hard_soft_wood, trees: dict):
        if self.attr['woodtype'][0] * hard_soft_wood[0] + self.attr['woodtype'][1] * hard_soft_wood[1] == 0:
            return 0
        if self.attr["trees"][0] == "All":
            return 0.8
        val = 0.8
        # This could be adapted for exclusive trees
        # Give small benefit to special trees
        for tree in trees.keys():
            if tree in self.attr["trees"]:
                val += 0.2e-3 * trees[tree]
        return val

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


def readXML():
    mushrooms = {}
    root = ET.parse('../data/mushrooms_databank.xml').getroot()
    for type_tag in root.findall('mushroom'):
        shroom = {};
        for child in type_tag:
            if child.tag == "woodtype":
                shroom[child.tag] = ("Hardwood" in child.text, "Softwood" in child.text)
            if child.tag == "trees" or child.tag == "habitat":
                shroom[child.tag] = child.tag.split(",")
            else:
                shroom[child.tag] = child.text
        mushrooms[shroom["name"]] = Mushroom(shroom)
    return mushrooms

def humidity_value(humidity, temperature):
    # Humidity of last 30 days
    # First look at 28 days ago to 14 days ago
    val = 0
    for j in range(0, 14):
        # If 10mm is perfect amount, this measures the normalized contribution
        val += 0.5 * min(humidity[j], 25) / 14
    # Emphazise 2-1 week ago
    for j in range(14, 21):
        val += 1.5*min(humidity[j], 25) / 7
    for j in range(21, 28):
        val += 0.75 * min(humidity[j], 25) / 7
    # This normalization value should be tweeked
    norm = 0.3 * (0.5 * 14 + 1.5*7 + 7 * 0.75)
    return min(val / norm, 3)

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
o = 0