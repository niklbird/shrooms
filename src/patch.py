class Patch:

    def __init__(self, points, station, id):
        self.points = points
        self.middle = points[int(len(points) / 2)]
        self.station = station
        self.id = id