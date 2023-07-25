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

    def sensorModelSTD(self, i, grid, p):
        """
            Usage sensor model with 6 binary values in heading direction
            :param i: index of current agent
            :param grid: array with agents positions [0 --> Empty]
            :param p: agent
            :return : sensor value
        """
        sensors = [0] * SENSORS

        # short range forward
        dx = self.adjustXPosition(p[i].coord.x + p[i].heading.x)
        dy = self.adjustYPosition(p[i].coord.y + p[i].heading.y)

        # long range forward
        dxl = self.adjustXPosition(p[i].coord.x + 2 * p[i].heading.x)
        dyl = self.adjustYPosition(p[i].coord.y + 2 * p[i].heading.y)

        # points for left and right sensors
        dyplus = self.adjustYPosition(p[i].coord.y + 1)
        dymin = self.adjustYPosition(p[i].coord.y - 1)

        dxplus = self.adjustXPosition(p[i].coord.x + 1)
        dxmin = self.adjustXPosition(p[i].coord.x - 1)

        # forward looking sensor / in direction of heading
        sensors[S0] = grid[dx][dy]
        sensors[S3] = grid[dxl][dyl]

        # headings in x-direction (i.e., y equals 0)
        if p[i].heading.y == 0:
            if p[i].heading.x == 1:
                sensors[S2] = grid[dx][dyplus]
                sensors[S5] = grid[dxl][dyplus]
                sensors[S1] = grid[dxl][dymin]
                sensors[S4] = grid[dxl][dymin]
            else:
                sensors[S1] = grid[dx][dyplus]
                sensors[S4] = grid[dxl][dyplus]
                sensors[S2] = grid[dx][dymin]
                sensors[S5] = grid[dxl][dymin]

        elif p[i].heading.x == 0:
            if p[i].heading.y == 1:
                sensors[S1] = grid[dxplus][dy]
                sensors[S4] = grid[dxplus][dyl]
                sensors[S2] = grid[dxmin][dy]
                sensors[S5] = grid[dxmin][dyl]
            else:
                sensors[S2] = grid[dxplus][dy]
                sensors[S5] = grid[dxplus][dyl]
                sensors[S1] = grid[dxmin][dy]
                sensors[S4] = grid[dxmin][dyl]

        return sensors

    # STD14: 14 Sensors
    #   Toward west
    #   5    3   4
    #   2    0   1
    #   7    X   6
    #   10   8   9
    #   13   11  12
    def sensorModelSTDL(self, i, grid, p):
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

        # long range forward
        dxl = int(self.adjustXPosition(p[i].coord.x + 2 * p[i].heading.x))  # -2 / 2 * x
        dyl = int(self.adjustYPosition(p[i].coord.y + 2 * p[i].heading.y))  # -2 / 2 * Y

        # points for left and right sensors
        dyplus = int(self.adjustYPosition(p[i].coord.y + 1))    # RIGHT
        dymin = int(self.adjustYPosition(p[i].coord.y - 1))     # LEFT
        dxplus = int(self.adjustXPosition(p[i].coord.x + 1))    # RIGHT
        dxmin = int(self.adjustXPosition(p[i].coord.x - 1))     # LEFT

        # short range backwards
        dxb = int(self.adjustXPosition(p[i].coord.x - p[i].heading.x))
        dyb = int(self.adjustYPosition(p[i].coord.y - p[i].heading.y))

        # long range backwards
        dxbl = int(self.adjustXPosition(p[i].coord.x - 2 * p[i].heading.x))
        dybl = int(self.adjustYPosition(p[i].coord.y - 2 * p[i].heading.y))

        sensors[S0] = grid[dx][dy]          # FORWARD SHORT
        sensors[S3] = grid[dxl][dyl]        # FORWARD LONG
        sensors[S8] = grid[dxb][dyb]        # BACKWARD SHORT
        sensors[S11] = grid[dxbl][dybl]     # BACKWARD LONG
        sensors[S_T] = grid[dx][dy]         # Temperature sensor FORWARD

        if p[i].heading.y == 0:
            if p[i].heading.x == 1:     # Toward North
                sensors[S2] = grid[dx][dyplus]
                sensors[S1] = grid[dx][dymin]
                sensors[S5] = grid[dxl][dyplus]
                sensors[S4] = grid[dxl][dymin]
                sensors[S7] = grid[int(p[i].coord.x)][dyplus]
                sensors[S6] = grid[int(p[i].coord.x)][dymin]
                sensors[S10] = grid[dxb][dyplus]
                sensors[S9] = grid[dxb][dymin]
                sensors[S13] = grid[dxbl][dyplus]
                sensors[S12] = grid[dxbl][dymin]
            else:                       # Toward South
                sensors[S1] = grid[dx][dyplus]
                sensors[S2] = grid[dx][dymin]
                sensors[S4] = grid[dxl][dyplus]
                sensors[S5] = grid[dxl][dymin]
                sensors[S6] = grid[int(p[i].coord.x)][dyplus]
                sensors[S7] = grid[int(p[i].coord.x)][dymin]
                sensors[S9] = grid[dxb][dyplus]
                sensors[S10] = grid[dxb][dymin]
                sensors[S12] = grid[dxbl][dyplus]
                sensors[S13] = grid[dxbl][dymin]
        elif p[i].heading.x == 0:
            if p[i].heading.y == 1:     # Toward East
                sensors[S1] = grid[dxplus][dy]
                sensors[S4] = grid[dxplus][dyl]
                sensors[S2] = grid[dxmin][dy]
                sensors[S5] = grid[dxmin][dyl]
                sensors[S6] = grid[dxplus][int(p[i].coord.y)]
                sensors[S7] = grid[dxmin][int(p[i].coord.y)]
                sensors[S10] = grid[dxmin][dyb]
                sensors[S9] = grid[dxplus][dyb]
                sensors[S13] = grid[dxmin][dybl]
                sensors[S12] = grid[dxplus][dybl]
            else:                       # Toward West
                sensors[S2] = grid[dxplus][dy]
                sensors[S5] = grid[dxplus][dyl]
                sensors[S1] = grid[dxmin][dy]
                sensors[S4] = grid[dxmin][dyl]
                sensors[S7] = grid[dxplus][int(p[i].coord.y)]
                sensors[S6] = grid[dxmin][int(p[i].coord.y)]
                sensors[S9] = grid[dxmin][dyb]
                sensors[S10] = grid[dxplus][dyb]
                sensors[S12] = grid[dxmin][dybl]
                sensors[S13] = grid[dxplus][dybl]

        return sensors

    def sensorModelSTDSL(self, i, grid, p):
        """
            Usage: sensor model with 8 sensors covering Moore neighborhood
            :param i: index of current agent
            :param grid: array with agents positions [0 --> Empty]
            :param p: agent
            :return : sensor value
        """
        sensors = [0] * SENSORS

        dx = self.adjustXPosition(p[i].coord.x + p[i].heading.x)
        dy = self.adjustYPosition(p[i].coord.y + p[i].heading.y)

        # points for left and right sensors
        dyplus = self.adjustYPosition(p[i].coord.y + 1)
        dymin = self.adjustYPosition(p[i].coord.y - 1)

        dxplus = self.adjustXPosition(p[i].coord.x + 1)
        dxmin = self.adjustXPosition(p[i].coord.x - 1)

        # short range backwards
        dxb = self.adjustXPosition(p[i].coord.x - p[i].heading.x)
        dyb = self.adjustYPosition(p[i].coord.y - p[i].heading.y)

        # forward looking sensor --> in direction of heading
        sensors[S0] = grid[dx][dy]      # Forward short
        sensors[S5] = grid[dxb][dyb]    # Backward short

        if p[i].heading.y == 0:
            if p[i].heading.x == 1:
                sensors[S2] = grid[dx][dyplus]
                sensors[S1] = grid[dx][dymin]
                sensors[S4] = grid[p[i].coord.x][dyplus]
                sensors[S3] = grid[p[i].coord.x][dymin]
                sensors[S7] = grid[dxb][dyplus]
                sensors[S6] = grid[dxb][dymin]
            else:
                sensors[S1] = grid[dx][dyplus]
                sensors[S2] = grid[dx][dymin]
                sensors[S3] = grid[p[i].coord.x][dyplus]
                sensors[S4] = grid[p[i].coord.x][dymin]
                sensors[S6] = grid[dxb][dyplus]
                sensors[S7] = grid[dxb][dymin]
        elif p[i].heading.x == 0:
            if p[i].heading.y == 1:
                sensors[S1] = grid[dxplus][dy]
                sensors[S2] = grid[dxmin][dy]
                sensors[S3] = grid[dxplus][p[i].coord.y]
                sensors[S4] = grid[dxmin][p[i].coord.y]
                sensors[S7] = grid[dxmin][dyb]
                sensors[S6] = grid[dxplus][dyb]
            else:
                sensors[S2] = grid[dxplus][dy]
                sensors[S1] = grid[dxmin][dy]
                sensors[S4] = grid[dxplus][p[i].coord.y]
                sensors[S3] = grid[dxmin][p[i].coord.y]
                sensors[S6] = grid[dxmin][dyb]
                sensors[S7] = grid[dxplus][dyb]

        return sensors