import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import random
import pickle


def random_location(p, p_next, sizeX, sizeY):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 设置画布大小

    for ax in axs:
        ax.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4, linestyle="dashed")
        for i in range(1, sizeY):
            ax.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
        for i in range(1, sizeX):
            ax.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")
        ax.set_xlim(0, sizeX + 1)
        ax.set_ylim(0, sizeY + 1)

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
    x = target[0] + 1      # 小网格中心的x坐标
    y = target[1] + 1     # 小网格中心的y坐标

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
    # file = "data/agents0_0_100"
    sizeX = 15
    sizeY = 15

    p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    # p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    #
    # tmp_X = []
    # tmp_Y = []
    # tmp_head_X = []
    # tmp_head_Y = []
    #
    # next_tmp_X = []
    # next_tmp_Y = []
    # next_tmp_head_X = []
    # next_tmp_head_Y = []
    #
    # agent = 0
    # f = open(file, 'r')
    # for line in f:
    #     if line[0] != 'G' and line[0] != 'F' and line[0] != ' ':
    #         gen = line.split(', ')
    #         tmp_X.append(gen[0])
    #         tmp_Y.append(gen[1])
    #         tmp_head_X.append(gen[4])
    #         tmp_head_Y.append(gen[5])
    #
    #         next_tmp_X.append(gen[2])
    #         next_tmp_Y.append(gen[3])
    #         next_tmp_head_X.append(gen[6])
    #         next_tmp_head_Y.append(gen[7])
    #
    # for i in range(NUM_AGENTS):
    #     p[i].coord.x = tmp_X[i]
    #     p[i].coord.y = tmp_Y[i]
    #     p[i].heading.x = tmp_head_X[i]
    #     p[i].heading.y = tmp_head_Y[i]
    #
    #     p_next[i].coord.x = next_tmp_X[i]
    #     p_next[i].coord.y = next_tmp_Y[i]
    #     p_next[i].heading.x = next_tmp_head_X[i]
    #     p_next[i].heading.y = next_tmp_head_Y[i]
    #
    # random_location(p_next, p, 15, 15)

    # Reset the grid
    grid = [[0] * int(sizeY) for _ in range(int(sizeX))]

    target = [int(sizeX / 2), int(sizeY / 2)]   # Coordinate of the target
    grid[target[0]][target[1]] = 1  # Set this cell unavailable

    # generate agent positions
    # In each repeat, all agent will be initialized
    for i in range(NUM_AGENTS):
        # initialisation of starting positions
        block = True

        # Find an unoccupied location
        while block:
            # Randomise a position for each agent
            p_initial[i].coord.x = random.randint(0, sizeX - 1)
            p_initial[i].coord.y = random.randint(0, sizeY - 1)

            if grid[p_initial[i].coord.x][p_initial[i].coord.y] == 0:  # not occupied
                print("Agent", i, ": ", "X: ", p_initial[i].coord.x, "Y:", p_initial[i].coord.y,
                      "Grid situation:", grid[p_initial[i].coord.x][p_initial[i].coord.y])

                block = False
                grid[p_initial[i].coord.x][p_initial[i].coord.y] = 1  # set grid cell occupied

            if p_initial[i].coord.x == target[0] and p_initial[i].coord.y == target[1]:
                # Randomise a position for each agent
                p_initial[i].coord.x = random.randint(0, sizeX - 1)
                p_initial[i].coord.y = random.randint(0, sizeY - 1)

        # Set agent heading values randomly (north, south, west, east)
        directions = [1, -1]
        randInd = random.randint(0, 1)
        if random.random() < 0.5:  # West & East
            p_initial[i].heading.x = directions[randInd]
            p_initial[i].heading.y = 0
        else:  # North & South
            p_initial[i].heading.x = 0
            p_initial[i].heading.y = directions[randInd]

    grid_with_shape(p_initial, target, sizeX, sizeY)

