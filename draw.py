import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils.util import Agent, Pos
from parameters.STD14 import *
import random



def random_location(p_initial, sizeX, sizeY):
        for i in range(len(p_initial)):
            print("Agent ", i, ":",
                      "X:", p_initial[i].coord.x,
                  "Y:", p_initial[i].coord.y,
                  "Head x:", p_initial[i].heading.x,
                  "Head y:", p_initial[i].heading.y)
        # 绘制矩形
        plt.figure(figsize=(6, 6))  # 设置画布大小
        plt.gca().set_aspect('equal')  # 设置坐标轴比例为1:1，保证矩形为正方形

        # 绘制矩形边界
        plt.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4, linestyle="dashed")

        # 绘制垂直线
        for i in range(1, sizeY):
            plt.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")

        # 绘制水平线
        for i in range(1, sizeX):
            plt.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")

        for i in range(len(p_initial)):
            x = p_initial[i].coord.x + 1
            y = p_initial[i].coord.y + 1

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

        # 设置坐标轴范围
        plt.xlim(0, sizeX + 1)
        plt.ylim(0, sizeY + 1)

        plt.show()
# End Location Initialisation


