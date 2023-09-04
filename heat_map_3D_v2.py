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

    def update_heat_intensity(self, targets):
        # Initialize heat_intensity to zero
        self.heat_intensity = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]
        for tgt in targets:
            tx, ty = int(tgt[0]), int(tgt[1])
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    # Calculate the Euclidean distance
                    dx, dy = x - tx, y - ty
                    dist = np.sqrt(dx ** 2 + dy ** 2)

                    # Define intensity based on distance
                    if dist < 1 + TARGET_SIZE:
                        intensity_level = HIGH
                        cal_dis = 1
                    elif dist < 2 + TARGET_SIZE:
                        intensity_level = MEDIUM
                        cal_dis = 2
                    elif dist < 4 + TARGET_SIZE:
                        intensity_level = LOW
                        cal_dis = 4
                    else:
                        continue

                    alpha_rate = Heat_alpha[intensity_level]
                    upper = np.abs(self.sizeX - cal_dis) * np.abs(self.sizeY - cal_dis)
                    lower = self.sizeX * self.sizeY
                    new_intensity = alpha_rate * upper / lower

                    # Set the intensity to the maximum of the current value and the new computed value
                    self.heat_intensity[x][y] = max(self.heat_intensity[x][y], new_intensity)

    def plot_3D_surface_version2(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, self.sizeX - 1, self.sizeX)
        y = np.linspace(0, self.sizeY - 1, self.sizeY)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, np.array(self.heat_intensity), cmap='viridis') # using viridis colormap for clearer contrast
        ax.set_title("3D Heat Intensity Map with Same dist, Target at (7, 7)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        plt.savefig('3d_heatmap_v2.jpg')
        plt.show()

if __name__ == "__main__":
    heatmap = HeatMap(15, 15)
    heatmap.update_heat_intensity([(7, 7)])
    heatmap.plot_3D_surface_version2()
