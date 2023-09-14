# positions
from copy import deepcopy


class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Pos(self.x, self.y)


# agent data
class Agent:
    def __init__(self, inspire, coord, heading):
        self.inspire = inspire
        self.coord = coord
        self.heading = heading

    def copy(self):
        return deepcopy(self)
