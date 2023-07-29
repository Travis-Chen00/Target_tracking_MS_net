import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import numpy as np
import math
import time

def random_location(final, target, sizeX, sizeY, move):
    num_targets = len(target)
    N = math.ceil(math.sqrt(num_targets))  # 向上取整
    prev_x, prev_y = None, None  # 初始化上一次的目标位置
    move_num = 1

    plt.ion()  # 开启交互模式
    fig = plt.figure()  # 创建新的figure

    for ge in range(N):
        for j in range(N):
            idx = ge * N + j  # 计算当前目标的索引

            if idx >= num_targets:  # 如果当前索引大于目标的数量，那么就没有其他的子图需要画，所以直接退出
                continue

            x = int(target[idx][0]) + 1  # 小网格中心的x坐标
            y = int(target[idx][1]) + 1  # 小网格中心的y坐标

            if prev_x is not None and prev_y is not None and (prev_x != x or prev_y != y):  # 如果上一次的目标位置已经被定义，并且和当前目标位置不同，那么重置 move_num
                move_num = 1
            prev_x, prev_y = x, y  # 记录当前目标位置，用于下一次循环

            color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 创建一个稍大的2D数组，颜色值为蓝色的RGB

            # 将目标周围的一圈（9个格子）设为红色
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)
                    if dist < 2:
                        color_grid[y + dy, x + dx] = np.array([222, 71, 71]) / 255.0  # 浅红色
                    elif dist < 5:
                        color_grid[y + dy, x + dx] = np.array([255, 165, 100]) / 255.0  # 浅橘色

            plt.clf()  # 清除当前的figure
            ax = fig.add_subplot(111)  # 将当前子图添加到figure中

            # 添加新的内容
            if idx == 0:
                filename = "Initial " + "Target: " + str(target[idx][0]) + ", " + str(target[idx][1])
                ax.set_title(filename)
            else:
                filename = "Target: " + str(target[idx][0]) + ", " + str(target[idx][1]) + " Move " + str(move_num)
                move_num += 1
                ax.set_title(filename)

            circle = Circle((x, y), 0.28, color='red')
            ax.add_patch(circle)
            ax.imshow(color_grid, origin='lower')

            # 绘制网格
            ax.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4,
                            linestyle="dashed")
            for i in range(1, sizeY):
                ax.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
            for i in range(1, sizeX):
                ax.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")

            ax.set_xlim(0, sizeX + 1)
            ax.set_ylim(0, sizeY + 1)

            # 下面的代码块是处理绘制 agents 的逻辑，我们需要将它移到每个子图的绘制部分
            for i in range(NUM_AGENTS):
                x = int(final[idx][i].coord.x) + 1
                y = int(final[idx][i].coord.y) + 1

                if int(final[idx][i].heading.x) == 1 and int(final[idx][i].heading.y) == 0:  # Face east
                    vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
                elif int(final[idx][i].heading.x) == -1 and int(final[idx][i].heading.y) == 0:  # Face west
                    vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
                elif int(final[idx][i].heading.x) == 0 and int(final[idx][i].heading.y) == 1:  # Face North
                    vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
                elif int(final[idx][i].heading.x) == 0 and int(final[idx][i].heading.y) == -1:  # Face south
                    vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

                triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
                ax.add_patch(triangle)

            plt.draw()  # 绘制图像
            plt.pause(1.5)  # 等待0.5秒

    plt.ioff()  # 关闭交互模式
    plt.show()


if __name__ == "__main__":
    file = "results/agents_Agents_50_TargetX_7_TargetY_7"
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

    agents_p = []
    original = []

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
                tmp_X.append(gen[0])
                tmp_Y.append(gen[2])
                tmp_head_X.append(gen[8])
                tmp_head_Y.append(gen[10])

                begining_tmp_X.append(gen[4])  # Max generation agents' original position
                begining_tmp_Y.append(gen[6])
                begining_tmp_head_X.append(gen[12])
                begining_tmp_head_Y.append(gen[14])

        if line_count == NUM_AGENTS + 4:
            if total_gen == 1:
                total_target.append(target)
            total_target.append(target)
            init_gen = False
            line_count = 0

            for i in range(NUM_AGENTS):
                p_next[i].coord.x = tmp_X[i]
                p_next[i].coord.y = tmp_Y[i]
                p_next[i].heading.x = tmp_head_X[i]
                p_next[i].heading.y = tmp_head_Y[i]

                p_initial[i].coord.x = begining_tmp_X[i]
                p_initial[i].coord.y = begining_tmp_Y[i]
                p_initial[i].heading.x = begining_tmp_head_X[i]
                p_initial[i].heading.y = begining_tmp_head_Y[i]

            agents_p.append(p_next)
            original.append(p_initial)
            # Reset
            tmp_X, tmp_Y, tmp_head_X, tmp_head_Y = [], [], [], []
            begining_tmp_X, begining_tmp_Y, begining_tmp_head_X, begining_tmp_head_Y = [], [], [], []
            p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
            p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

            continue
        line_count += 1

    final_pos = []
    for i in range(len(agents_p)):
        if i == 0:
            final_pos.append(original[i])
            final_pos.append(agents_p[i])
        else:
            final_pos.append(agents_p[i])

    random_location(final_pos, total_target, sizeX, sizeY, total_gen)