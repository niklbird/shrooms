import xml.etree.ElementTree as ET

'''
This file is not used yet -> May later be used to also include soil information in the processing.
'''
class Soil:
    def __init__(self, attributes: dict):
        self.attr = attributes


def read_soil_XML(uri):
    soils = {}
    root = ET.parse(uri).getroot()
    for type_tag in root.findall('soil'):
        soil = {}
        for child in type_tag:
            #if child.tag == "ph":
            #    soil[child.tag] = ["Basisch" in child.text, "Neutral" in child.text, "Sauer" in child.text]
            if child.tag == "typ":
                soil[child.tag] = child.text.lower().split(",")
            else:
                soil[child.tag] = child.text.lower()
        soils[soil["name"]] = Soil(soil)
    return soils


def soil_value(mushroom, soil):
    mushroom_ph = mushroom.attr['ph'].lower()
    ph_val = 2.0
    if mushroom_ph != "All":
        ph_val = int(mushroom_ph in soil.attr["ph"])*2
    if ph_val == 0:
        ph_val = 0.5

    soil_val = 1.5
    sv = mushroom.attr["soil"]
    a = sv[0]
    if mushroom.attr["soil"] != "All" and not mushroom.attr["soil"].lower() in soil.attr["typ"]:
        soil_val = 1

    soil_score = float(soil.attr["score"])
    return ph_val * soil_val #* soil_score

