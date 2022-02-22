import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt

'''
This class represents one Mushroom Type.
'''


class Mushroom:

    def __init__(self, attributes: dict):
        self.attr = attributes

    def time_value(self, cur_month):
        return self.attr["seasonStart"] <= cur_month <= self.attr["seasonEnd"]




def read_mushroom_XML(url):
    mushrooms = {}
    root = ET.parse(url).getroot()
    for type_tag in root.findall('mushroom'):
        shroom = {};
        for child in type_tag:
            if child.tag == "woodtype":
                shroom[child.tag] = (int("Hardwood" in child.text), int("Softwood" in child.text))
            elif child.tag == "trees":
                shroom[child.tag] = child.text.lower().split(",")
            else:
                shroom[child.tag] = child.text
        mushrooms[shroom["name"]] = Mushroom(shroom)
    return mushrooms





def sanity_test():
    # Deprecated, currently unused
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
        # for i in range(40, len(val2)):
        # hum_res.append(humidity_value(val2[i - 30:i], 0))
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
