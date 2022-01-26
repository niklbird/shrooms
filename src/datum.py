"""
This class represents one single data point on the map.
"""
class Datum:

    def __init__(self, coord):
        self.coord = coord
        self.trees = 0
        self.soil = 0
        self.mushrooms = {}
        self.probabilities = {}

    def set_trees(self, trees):
        self.trees = trees

    def set_soil(self, soil):
        self.soil = soil

    def set_env(self, value, trees_bool):
        if trees_bool:
            self.trees = value
        else:
            self.soil = value
