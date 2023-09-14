from parameters.STD14 import *


class Sensors:
    def __init__(self, size_x, size_y):
        self.size_X = size_x
        self.size_Y = size_y

    def adjustXPosition(self, x):
        """
            Usage: Adjusts position for movement on torus
            Ensure that coordinate x is smaller than SIZE_X
            :param: x
            :return: updated x value
        """
        if x < 0:
            x += self.size_X
        elif x > self.size_X - 1:
            x -= self.size_X
        return x

    def adjustYPosition(self, y):
        """
            Usage: Adjusts position for movement on torus
            Ensure that coordinate y is smaller than SIZE_Y
            :param: y
            :return: updated y value
        """
        if y < 0:
            y += self.size_Y
        elif y > self.size_Y - 1:
            y -= self.size_Y
        return y

    # STD14: 14 Sensors
    #   Toward west
    #   5    3   4
    #   2    0   1
    #   7    X   6
    #   10   8   9
    #   13   11  12
    def sensorModelSTDL(self, i, grid, intensity, p):
        """
            Usage: sensor model with 14 sensors surrounding the agent
            :param i: index of current agent
            :param grid: array with agents positions [0 --> Empty]
            :param p: agent
            :return : sensor value
        """
        sensors = [0] * SENSORS

        # short range forward
        dx = int(self.adjustXPosition(p[i].coord.x + p[i].heading.x))       # -x OR x
        dy = int(self.adjustYPosition(p[i].coord.y + p[i].heading.y))       # -y OR y

        # # long range forward
        # dxl = int(self.adjustXPosition(p[i].coord.x + 2 * p[i].heading.x))  # -2 / 2 * x
        # dyl = int(self.adjustYPosition(p[i].coord.y + 2 * p[i].heading.y))  # -2 / 2 * Y

        # points for left and right sensors
        dyplus = int(self.adjustYPosition(p[i].coord.y + 1))    # RIGHT
        dymin = int(self.adjustYPosition(p[i].coord.y - 1))     # LEFT
        dxplus = int(self.adjustXPosition(p[i].coord.x + 1))    # RIGHT
        dxmin = int(self.adjustXPosition(p[i].coord.x - 1))     # LEFT

        # short range backwards
        dxb = int(self.adjustXPosition(p[i].coord.x - p[i].heading.x))
        dyb = int(self.adjustYPosition(p[i].coord.y - p[i].heading.y))

        # # long range backwards
        # dxbl = int(self.adjustXPosition(p[i].coord.x - 2 * p[i].heading.x))
        # dybl = int(self.adjustYPosition(p[i].coord.y - 2 * p[i].heading.y))

        sensors[S0] = intensity[p[i].coord.x][p[i].heading.x]          # FORWARD SHORT
        sensors[S1] = intensity[dx][dy]        # BACKWARD SHORT
        sensors[S2] = intensity[dxb][dyb]  # BACKWARD SHORT

        sensors[S5] = grid[dx][dy]             # 1 Forward proximity sensor

        if p[i].heading.y == 0:
            if p[i].heading.x == 1:     # Toward North
                sensors[S3] = intensity[int(p[i].coord.x)][dyplus]
                sensors[S4] = intensity[int(p[i].coord.x)][dymin]
            else:                       # Toward South
                sensors[S4] = intensity[int(p[i].coord.x)][dyplus]
                sensors[S3] = intensity[int(p[i].coord.x)][dymin]
        elif p[i].heading.x == 0:
            if p[i].heading.y == 1:     # Toward East
                sensors[S4] = intensity[dxplus][int(p[i].coord.y)]
                sensors[S3] = intensity[dxmin][int(p[i].coord.y)]
            else:                       # Toward West
                sensors[S3] = intensity[dxplus][int(p[i].coord.y)]
                sensors[S4] = intensity[dxmin][int(p[i].coord.y)]

        return sensors