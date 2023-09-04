import numpy as np
import matplotlib.pyplot as plt

# Constants
HIGH, MEDIUM, LOW = 3, 2, 1
TARGET_SIZE = 1
Heat_alpha = {HIGH: 0.05, MEDIUM: 0.7, LOW: 0.25}

def is_on_square_border(x, y, target_x, target_y, dist):
    return max(abs(x - target_x), abs(y - target_y)) == dist

def heatmap_intensity(x, y, sizeX=15, sizeY=15, target_x=7, target_y=7):
    dist = max(abs(x - target_x), abs(y - target_y))

    if is_on_square_border(x, y, target_x, target_y, 1):
        intensity_level = HIGH
    elif is_on_square_border(x, y, target_x, target_y, 2) or is_on_square_border(x, y, target_x, target_y, 3):
        intensity_level = MEDIUM
    else:
        intensity_level = LOW

    alpha_rate = Heat_alpha[intensity_level]
    upper = np.abs(sizeX - dist) * np.abs(sizeY - dist)
    lower = sizeX * sizeY
    new_intensity = alpha_rate * upper / lower

    return new_intensity

class HeatMap:
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.heat_intensity = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]
        self.heatmap = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]

    def update_heat_intensity(self, targets):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.heat_intensity[x][y] = heatmap_intensity(x, y, self.sizeX, self.sizeY, targets[0][0], targets[0][1])

    def plot_2D_heatmap_version2(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.heat_intensity, cmap='Reds', interpolation='nearest')
        fig.colorbar(cax, label='Intensity')
        ax.set_title("2D Heat Intensity Map, Target at (7, 7)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.savefig('2d_heatmap.jpg')
        plt.show()

if __name__ == "__main__":
    heatmap = HeatMap(15, 15)
    heatmap.update_heat_intensity([(7, 7)])
    heatmap.plot_2D_heatmap_version2()
