import xml.etree.ElementTree as ET

'''
This file is not used yet -> May later be used to also include soil information in the processing.
'''
class Soil:
    def __init__(self, attributes: dict):
        self.attr = attributes


def readXML():
    soils = {}
    root = ET.parse('../data/soil_databank.xml').getroot()
    for type_tag in root.findall('soil'):
        soil = {}
        for child in type_tag:
            if child.tag == "ph":
                soil[child.tag] = ["Basisch" in child.text, "Neutral" in child.text, "Sauer" in child.text]
            elif child.tag == "typ":
                soil[child.tag] = child.text.split(",")
            else:
                soil[child.tag] = child.text
        soils[soil["name"]] = Soil(soil)
    return soils

