"""
Collection of all points that belong to an area with same weather (1km x 1km) for easier processing
"""
class Patch:

    def __init__(self, points, middle, station, corners):
        self.points = points
        self.middle = middle
        self.station = station
        self.corners = corners
        self.dates = []
        self.weather_data = {}
