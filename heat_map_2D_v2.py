import numpy as np
import matplotlib.pyplot as plt

# Constants
HIGH, MEDIUM, LOW = 3, 2, 1
TARGET_SIZE = 1
Heat_alpha = {HIGH: 0.05, MEDIUM: 0.7, LOW: 0.25}

class HeatMap:
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.heat_intensity = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]
        self.heatmap = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]

    def update_heat_intensity(self, targets):
        self.heatmap = [[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        for tgt in targets:
            tx, ty = int(tgt[0]), int(tgt[1])
            for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    x, y = tx + dx, ty + dy
                    if 0 <= x < self.sizeX and 0 <= y < self.sizeY:
                        dist = max(np.abs(dx), np.abs(dy))
                        if dist < 1 + TARGET_SIZE:
                            self.heatmap[x][y] = HIGH
                        elif dist < 3 + TARGET_SIZE:
                            self.heatmap[x][y] = MEDIUM

        self.update_heat_intensity_values()

    def update_heat_intensity_values(self):
        for x in range(15):
            for y in range(15):
                if self.heatmap[x][y] == HIGH:
                    intensity_level = HIGH
                    cal_dis = 1
                elif self.heatmap[x][y] == MEDIUM:
                    intensity_level = MEDIUM
                    cal_dis = 2
                else:
                    intensity_level = LOW
                    cal_dis = 4

                alpha_rate = Heat_alpha[intensity_level]
                upper = np.abs(self.sizeX - cal_dis) * np.abs(self.sizeY - cal_dis)
                lower = self.sizeX * self.sizeY
                new_intensity = alpha_rate * upper / lower
                self.heat_intensity[x][y] = max(self.heat_intensity[x][y], new_intensity)

    def plot_2D_heatmap_version2(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.heat_intensity, cmap='Reds', interpolation='nearest')
        fig.colorbar(cax, label='Intensity')
        ax.set_title("2D Heat Intensity Map with Same dist, Target at (7, 7)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.savefig('2d_heatmap_v2.jpg')
        plt.show()

if __name__ == "__main__":
    heatmap = HeatMap(15, 15)
    heatmap.update_heat_intensity([(7, 7)])
    heatmap.plot_2D_heatmap_version2()
