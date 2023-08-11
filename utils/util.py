# positions
class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# agent data
class Agent:
    def __init__(self, inspire, coord, heading):
        self.inspire = inspire
        self.coord = coord
        self.heading = heading
