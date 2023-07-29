import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import numpy as np
import math


def random_location(p, agents, target, sizeX, sizeY, move):
    # 计算 N 的值
    num_targets = len(target)
    N = math.ceil(math.sqrt(num_targets))  # 向上取整

    # 创建 NxN 子图
    fig, axs = plt.subplots(N, N, figsize=(12, 12))

    # 为了方便处理，如果 axs 不是二维数组，那么我们将其转化为二维数组
    if len(axs.shape) < 2:
        axs = np.array([[axs]])
    move_num = 1
    for ge in range(N):
        for j in range(N):
            # 计算当前目标的索引
            idx = ge * N + j

            # 如果当前索引大于目标的数量，那么就没有其他的子图需要画，所以直接退出
            if idx >= num_targets:
                axs[ge][j].axis('off')  # 隐藏没有使用的子图
                continue

            x = int(target[idx][0]) + 1  # 小网格中心的x坐标
            y = int(target[idx][1]) + 1  # 小网格中心的y坐标

            # 创建一个稍大的2D数组，颜色值为蓝色的RGB
            color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 浅蓝色

            # 将目标周围的一圈（9个格子）设为红色
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)
                    if dist < 2:
                        color_grid[y + dy, x + dx] = np.array([222, 71, 71]) / 255.0  # 浅红色
                    elif dist < 5:
                        color_grid[y + dy, x + dx] = np.array([255, 165, 100]) / 255.0  # 浅橘色

            # 清除之前的内容
            axs[ge][j].clear()

            # 添加新的内容
            if idx == 0:
                filename = "Initial Figure/" + "Target: " + str(target[idx][0]) + ", " + str(target[idx][1])
                axs[ge][j].set_title(filename)
            else:
                filename = "Target: " + str(target[idx][0]) + ", " + str(target[idx][1]) + " Move " + str(move_num)
                move_num += 1
                axs[ge][j].set_title(filename)

            circle = Circle((x, y), 0.28, color='red')
            axs[ge][j].add_patch(circle)
            axs[ge][j].imshow(color_grid, origin='lower')

            # 绘制网格
            axs[ge][j].plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4,
                            linestyle="dashed")
            for i in range(1, sizeY):
                axs[ge][j].plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
            for i in range(1, sizeX):
                axs[ge][j].plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")

            axs[ge][j].set_xlim(0, sizeX + 1)
            axs[ge][j].set_ylim(0, sizeY + 1)

    for ge in range(N):
        for j in range(N):
            # 计算当前目标的索引
            idx = ge * N + j
            if idx >= num_targets:
                continue

            if idx == 0:
                for i in range(NUM_AGENTS):
                    x = int(p[i].coord.x) + 1
                    y = int(p[i].coord.y) + 1

                    if int(p[i].heading.x) == 1 and int(p[i].heading.y) == 0:  # Face east
                        vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
                    elif int(p[i].heading.x) == -1 and int(p[i].heading.y) == 0:  # Face west
                        vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
                    elif int(p[i].heading.x) == 0 and int(p[i].heading.y) == 1:  # Face North
                        vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
                    elif int(p[i].heading.x) == 0 and int(p[i].heading.y) == -1:  # Face south
                        vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

                    triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
                    axs[ge][j].add_patch(triangle)

            elif idx > 0:
                for i in range(len(agents)):
                    for k in range(NUM_AGENTS):
                        x = int(agents[i][k].coord.x) + 1
                        y = int(agents[i][k].coord.y) + 1

                        if int(agents[i][k].heading.x) == 1 and int(agents[i][k].heading.y) == 0:  # Face east
                            vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
                        elif int(agents[i][k].heading.x) == -1 and int(agents[i][k].heading.y) == 0:  # Face west
                            vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
                        elif int(agents[i][k].heading.x) == 0 and int(agents[i][k].heading.y) == 1:  # Face North
                            vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
                        elif int(agents[i][k].heading.x) == 0 and int(agents[i][k].heading.y) == -1:  # Face south
                            vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

                        triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
                        axs[ge][j].add_patch(triangle)

    plt.show()


if __name__ == "__main__":
    file = "results/agents_Agents_50_TargetX_7_TargetY_7_moving"
    sizeX = 15
    sizeY = 15

    p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

    tmp_X = []
    tmp_Y = []
    tmp_head_X = []
    tmp_head_Y = []

    begining_tmp_X = []
    begining_tmp_Y = []
    begining_tmp_head_X = []
    begining_tmp_head_Y = []

    total_target = []

    init_gen = False

    target = []
    generation = 0
    fitness = 0
    line_count = 0
    f = open(file, 'r')
    total_gen = 0
    for line in f:
        if line[0] == 'G' and line[1] == 'e':
            gen = re.split('[,.: \n]', line)
            if int(gen[2]) == 0:
                init_gen = True
            total_gen += 1

        if line[0] == 'T':
            gen = re.split('[,.: \n]', line)
            target = [int(gen[2]), int(gen[4])]

        if line[0] != 'G' and line[0] != 'F' and line[0] != ' ' and line[0] != 'T':
            gen = gen = re.split('[,.: \n]', line)
            if len(gen) > 10:
                if not init_gen:
                    tmp_X.append(gen[0])
                    tmp_Y.append(gen[2])
                    tmp_head_X.append(gen[8])
                    tmp_head_Y.append(gen[10])
                else:
                    begining_tmp_X.append(gen[4])  # Max generation agents' original position
                    begining_tmp_Y.append(gen[6])
                    begining_tmp_head_X.append(gen[12])
                    begining_tmp_head_Y.append(gen[14])

        if line_count == NUM_AGENTS + 4:
            total_target.append(target)
            init_gen = False
            line_count = 0
            continue
        line_count += 1

    agents_p = []

    for t in range(total_gen):
        for i in range(NUM_AGENTS):
            if t > 0:
                p_next[i].coord.x = tmp_X[i]
                p_next[i].coord.y = tmp_Y[i]
                p_next[i].heading.x = tmp_head_X[i]
                p_next[i].heading.y = tmp_head_Y[i]

            if t == 0:
                p_initial[i].coord.x = begining_tmp_X[i]
                p_initial[i].coord.y = begining_tmp_Y[i]
                p_initial[i].heading.x = begining_tmp_head_X[i]
                p_initial[i].heading.y = begining_tmp_head_Y[i]

        agents_p.append(p_next)
    random_location(p_initial, agents_p, total_target, sizeX, sizeY, total_gen)