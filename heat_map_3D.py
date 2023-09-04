import numpy as np
import matplotlib.pyplot as plt
from parameters.STD14 import *
# Assuming these values are imported from parameters.STD14
# LOW, MEDIUM, HIGH, AIM, Heat_alpha = ...

def is_on_square_border(x, y, target_x, target_y, dist):
    return (
                   x == target_x - dist or x == target_x + dist or
                   y == target_y - dist or y == target_y + dist
           ) and (
                   target_x - dist <= x <= target_x + dist and
                   target_y - dist <= y <= target_y + dist
           )

def cal_heatmap(sizeX=15, sizeY=15, target_x=7, target_y=7):
    heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]
    TARGET_SIZE = 1

    heatmap[target_x][target_y] = AIM

    for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
        for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
            x, y = target_x + dx, target_y + dy

            # Ensure (x, y) is within bounds
            if 0 <= x < sizeX and 0 <= y < sizeY:
                dist = max(np.abs(dx), np.abs(dy))

                if dist < 1 + TARGET_SIZE:  # High intensity
                    heatmap[x][y] = max(heatmap[x][y], HIGH)
                elif dist < 3 + TARGET_SIZE:  # Medium intensity
                    heatmap[x][y] = max(heatmap[x][y], MEDIUM)

    return heatmap

def heatmap_intensity(x, y, sizeX=15, sizeY=15, target_x=7, target_y=7):
    heatmap = cal_heatmap(sizeX, sizeY, target_x, target_y)
    dist = max(abs(x - target_x), abs(y - target_y))

    # Define intensity based on distance
    if is_on_square_border(x, y, target_x, target_y, 1):
        intensity_level = HIGH
    elif is_on_square_border(x, y, target_x, target_y, 2) or is_on_square_border(x, y, target_x, target_y, 3):
        intensity_level = MEDIUM
    else:
        intensity_level = LOW

    boundary_dist = dist

    alpha_rate = Heat_alpha[intensity_level]
    upper = np.abs(sizeX - boundary_dist) * np.abs(sizeY - boundary_dist)
    lower = sizeX * sizeY
    new_intensity = alpha_rate * upper / lower

    return new_intensity

def plot_3d_heatmap():
    sizeX, sizeY = 15, 15
    X, Y = np.meshgrid(range(sizeX), range(sizeY))
    Z = np.array([[heatmap_intensity(x, y) for y in range(sizeY)] for x in range(sizeX)])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_title("3D Heat Intensity Distribution, Target at (7, 7)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Intensity")

    plt.savefig('3d_heatmap.jpg')
    plt.show()

if __name__ == "__main__":
    plot_3d_heatmap()
