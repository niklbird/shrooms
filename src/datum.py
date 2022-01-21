"""
This class represents one single data point on the map.
"""
class Datum:

    def __init__(self, coord, trees, soils):
        self.coord = coord
        self.trees = trees
        self.soils = soils
        self.mushrooms = {}
        self.probabilities = {}
