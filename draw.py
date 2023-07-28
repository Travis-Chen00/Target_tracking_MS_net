import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import random
import pickle
import numpy as np


def random_location(p, p_next, target, sizeX, sizeY, gen, fitness):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 设置画布大小

    for ax in axs:
        ax.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4, linestyle="dashed")
        for i in range(1, sizeY):
            ax.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
        for i in range(1, sizeX):
            ax.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")
        ax.set_xlim(0, sizeX + 1)
        ax.set_ylim(0, sizeY + 1)

    # Add a sphere into the center of the grid
    x = int(target[0]) + 1     # 小网格中心的x坐标
    y = int(target[1]) + 1     # 小网格中心的y坐标

    # 创建一个稍大的2D数组，颜色值为蓝色的RGB
    color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 浅蓝色

    # # 将目标周围的一圈（9个格子）设为红色
    # for i in range(sizeX):
    #     for j in range(sizeY):
    #         if heatmap[i][j] == HIGH:
    #             color_grid[i + 1, j + 1] = np.array([222, 71, 71]) / 255.0  # 浅红色
    #         elif heatmap[i][j] == MEDIUM:
    #             color_grid[i + 1, j + 1] = np.array([255, 165, 100]) / 255.0  # 浅橘色

    # 将目标周围的一圈（9个格子）设为红色
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            color_grid[x + dx, y + dy] = np.array([222, 71, 71]) / 255.0  # 浅红色

    # 将目标周围的外圈（24个格子）设为橘色
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if abs(dx) < 2 and abs(dy) < 2:  # 跳过内圈
                continue
            color_grid[x + dx, y + dy] = np.array([255, 165, 100]) / 255.0  # 浅橘色

    filename = "Generation: " + str(int(gen) + 1) + " Fitness: "+ str(fitness)
    axs[0].set_title('Original Figure')
    axs[1].set_title(filename)

    # 在小网格中心添加圆形
    for ax in axs:
        circle = Circle((x, y), 0.28, color='red')
        ax.add_patch(circle)

    # 在每个子图中添加颜色背景
    for ax in axs:
        ax.imshow(color_grid, origin='lower')

    # 其他绘图代码...
    data = [p, p_next]
    for idx, ax in enumerate(axs):
        for i in range(len(data[idx])):
            x = int(data[idx][i].coord.x) + 1
            y = int(data[idx][i].coord.y) + 1

            if int(data[idx][i].heading.x) == 1 and int(data[idx][i].heading.y) == 0:  # Face east
                vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
            elif int(data[idx][i].heading.x) == -1 and int(data[idx][i].heading.y) == 0:  # Face west
                vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
            elif int(data[idx][i].heading.x) == 0 and int(data[idx][i].heading.y) == 1:  # Face North
                vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
            elif int(data[idx][i].heading.x) == 0 and int(data[idx][i].heading.y) == -1:  # Face south
                vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

            triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
            ax.add_patch(triangle)

    plt.show()
# End Location Initialisation


def grid_with_shape(p_initial, target, sizeX, sizeY):
    # 绘制矩形边界
    plt.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4, linestyle="dashed")

    # 绘制垂直线
    for i in range(1, sizeY):
        plt.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")

    # 绘制水平线
    for i in range(1, sizeX):
        plt.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")

    # Add a sphere into the center of the grid
    x = target[0]     # 小网格中心的x坐标
    y = target[1]    # 小网格中心的y坐标

    # 在小网格中心添加圆形
    circle = Circle((x, y), 0.28, color='red')
    plt.gca().add_patch(circle)

    for i in range(len(p_initial)):
        x = int(p_initial[i].coord.x) + 1
        y = int(p_initial[i].coord.y) + 1

        if p_initial[i].heading.x == 1 and p_initial[i].heading.y == 0:  # Face east
            vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
        elif p_initial[i].heading.x == -1 and p_initial[i].heading.y == 0:  # Face west
            vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
        elif p_initial[i].heading.x == 0 and p_initial[i].heading.y == 1:  # Face North
            vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
        elif p_initial[i].heading.x == 0 and p_initial[i].heading.y == -1:  # Face south
            vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

        triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
        plt.gca().add_patch(triangle)

    # 返回当前图形对象
    plt.show()


if __name__ == "__main__":
    file = "results/agents_Agents_100_TargetX_7_TargetY_7"
    sizeX = 15
    sizeY = 15

    p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    #
    tmp_X = []
    tmp_Y = []
    tmp_head_X = []
    tmp_head_Y = []

    begining_tmp_X = []
    begining_tmp_Y = []
    begining_tmp_head_X = []
    begining_tmp_head_Y = []

    target = []
    generation = 0
    fitness = 0
    # agent = 0
    f = open(file, 'r')
    for line in f:
        if line[0] == 'G' and line[1] == 'e':
            gen = re.split('[,.: \n]', line)
            generation = gen[2]
        if line[0] == 'F':
            gen = re.split('[,: \n]', line)
            fitness = gen[2]
        if line[0] == 'T':
            gen = re.split('[,.: \n]', line)
            target.append(int(gen[2]))
            target.append(int(gen[4]))
        if line[0] != 'G' and line[0] != 'F' and line[0] != ' ' and line[0] != 'T':
            gen = gen = re.split('[,.: \n]', line)
            if len(gen) > 10:
                tmp_X.append(gen[0])
                tmp_Y.append(gen[2])
                tmp_head_X.append(gen[8])
                tmp_head_Y.append(gen[10])

                begining_tmp_X.append(gen[4])   # Max generation agents' original position
                begining_tmp_Y.append(gen[6])
                begining_tmp_head_X.append(gen[12])
                begining_tmp_head_Y.append(gen[14])

    for i in range(NUM_AGENTS):
        p_next[i].coord.x = tmp_X[i]
        p_next[i].coord.y = tmp_Y[i]
        p_next[i].heading.x = tmp_head_X[i]
        p_next[i].heading.y = tmp_head_Y[i]

        p_initial[i].coord.x = begining_tmp_X[i]
        p_initial[i].coord.y = begining_tmp_Y[i]
        p_initial[i].heading.x = begining_tmp_head_X[i]
        p_initial[i].heading.y = begining_tmp_head_Y[i]

    # random_location(p_next, p, 15, 15)

    # Reset the grid
    # grid = [[0] * int(sizeY) for _ in range(int(sizeX))]
    #
    # target = [int(sizeX / 2), int(sizeY / 2)]   # Coordinate of the target
    # grid[target[0]][target[1]] = 1  # Set this cell unavailable
    #
    # # generate agent positions
    # # In each repeat, all agent will be initialized
    # for i in range(NUM_AGENTS):
    #     # initialisation of starting positions
    #     block = True
    #
    #     # Find an unoccupied location
    #     while block:
    #         # Randomise a position for each agent
    #         p_initial[i].coord.x = random.randint(0, sizeX - 1)
    #         p_initial[i].coord.y = random.randint(0, sizeY - 1)
    #
    #         if grid[p_initial[i].coord.x][p_initial[i].coord.y] == 0:  # not occupied
    #             print("Agent", i, ": ", "X: ", p_initial[i].coord.x, "Y:", p_initial[i].coord.y,
    #                   "Grid situation:", grid[p_initial[i].coord.x][p_initial[i].coord.y])
    #
    #             block = False
    #             grid[p_initial[i].coord.x][p_initial[i].coord.y] = 1  # set grid cell occupied
    #
    #         if p_initial[i].coord.x == target[0] and p_initial[i].coord.y == target[1]:
    #             # Randomise a position for each agent
    #             p_initial[i].coord.x = random.randint(0, sizeX - 1)
    #             p_initial[i].coord.y = random.randint(0, sizeY - 1)
    #
    #     # Set agent heading values randomly (north, south, west, east)
    #     directions = [1, -1]
    #     randInd = random.randint(0, 1)
    #     if random.random() < 0.5:  # West & East
    #         p_initial[i].heading.x = directions[randInd]
    #         p_initial[i].heading.y = 0
    #     else:  # North & South
    #         p_initial[i].heading.x = 0
    #         p_initial[i].heading.y = directions[randInd]

    random_location(p_initial, p_next, target, sizeX, sizeY, generation, fitness)

