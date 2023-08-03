import csv
import re
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import numpy as np
import math
import time
import os


def count_files_in_dir(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def moving(agents, generation, target, pop, sizeX, sizeY):
    num_generations = len(agents)  # 总共的代数

    # 根据代数确定子图的布局，N x N
    N = int(np.ceil(np.sqrt(num_generations)))

    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(8, 8))  # 创建新的figure，你可以根据需要修改figure的大小
    prev_x, prev_y = None, None  # 初始化上一次的目标位置
    move_num = 1

    # 对每一代进行迭代
    for gen in range(num_generations):
        ax = fig.add_subplot(N, N, gen + 1)  # 将当前子图添加到figure中

        # 对于当前代的每个时间点进行迭代
        for t in range(MAX_TIME + 1):
            # 清除当前子图
            ax.clear()

            x = int(target[gen][0]) + 1  # 小网格中心的x坐标
            y = int(target[gen][1]) + 1  # 小网格中心的y坐标

            if prev_x is not None and prev_y is not None and (
                    prev_x != x or prev_y != y):  # 如果上一次的目标位置已经被定义，并且和当前目标位置不同，那么重置 move_num
                move_num = 1
            prev_x, prev_y = x, y  # 记录当前目标位置，用于下一次循环

            color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 创建一个稍大的2D数组，颜色值为蓝色的RGB

            # 将目标周围的一圈（9个格子）设为红色
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)
                    if dist < 2:
                        color_grid[y + dy, x + dx] = np.array([222, 71, 71]) / 255.0  # 浅红色
                    elif dist < 4:
                        color_grid[y + dy, x + dx] = np.array([255, 165, 100]) / 255.0  # 浅橘色

            plt.clf()  # 清除当前的figure
            ax = fig.add_subplot(111)  # 将当前子图添加到figure中

            # 添加新的内容
            if gen == 0:
                filename = "Initial " + "Target: " + str(target[gen][0]) + ", " + str(target[gen][1]) + " Time:" + str(t)
                ax.set_title(filename)
            else:
                filename = "Target: " + str(target[gen][0]) + ", " + str(target[gen][1]) + " Time:" + str(t)
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
                x = int(agents[gen][t][i].coord.x) + 1
                y = int(agents[gen][t][i].coord.y) + 1

                if int(agents[gen][t][i].heading.x) == 1 and int(agents[gen][t][i].heading.y) == 0:  # Face east
                    vertices = [(x - 0.28, y - 0.28), (x - 0.28, y + 0.28), (x + 0.28, y)]
                elif int(agents[gen][t][i].heading.x) == -1 and int(agents[gen][t][i].heading.y) == 0:  # Face west
                    vertices = [(x + 0.28, y - 0.28), (x + 0.28, y + 0.28), (x - 0.28, y)]
                elif int(agents[gen][t][i].heading.x) == 0 and int(agents[gen][t][i].heading.y) == 1:  # Face North
                    vertices = [(x - 0.28, y - 0.28), (x + 0.28, y - 0.28), (x, y + 0.28)]
                elif int(agents[gen][t][i].heading.x) == 0 and int(agents[gen][t][i].heading.y) == -1:  # Face south
                    vertices = [(x - 0.28, y + 0.28), (x + 0.28, y + 0.28), (x, y - 0.28)]

                triangle = Polygon(vertices, closed=True, edgecolor='black', facecolor='black')
                ax.add_patch(triangle)

            # 绘制图像并暂停
            plt.draw()
            plt.pause(0.05)  # 暂停0.15秒

    plt.ioff()  # 关闭交互模式
    plt.show()


if __name__ == "__main__":
    directory = "moving/1_layer_All_Medium/"
    file_num = count_files_in_dir(directory)

    agent_p = [[[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for i in range(MAX_TIME + 1)] for _ in
               range(int(file_num))]
    total_gen, total_target, total_pop = [], [], []
    for gen in range(int(file_num)):
        para = False
        generation = None
        filename = directory + "Agents_" + str(NUM_AGENTS) + "_TimeStep_" + str(MAX_TIME) + "_Gen_" + str(gen)

        with open(filename, 'r') as f:
            generation = None
            for line in f:
                line = line.strip()
                if line.startswith('Gen'):
                    generation = int(line.split(":")[1].strip())
                    total_gen.append(generation)
                elif line.startswith('Pop'):
                    total_pop.append(int(line.split(":")[1].strip()))
                elif line.startswith('Target'):
                    total_target.append(list(map(int, line.split(":")[1].strip().split(","))))
                elif line.startswith('Fitness'):
                    continue  # you can handle 'Fitness' here if necessary
                else:  # handle data lines
                    gen = re.split('[,.: \n]', line)
                    if len(gen) > 10:
                        agent_p[generation][int(gen[0])][int(gen[2])].coord.x = int(gen[4])
                        agent_p[generation][int(gen[0])][int(gen[2])].coord.y = int(gen[6])
                        agent_p[generation][int(gen[0])][int(gen[2])].heading.x = int(gen[8])
                        agent_p[generation][int(gen[0])][int(gen[2])].heading.y = int(gen[10])

    # All heatmap is low
    num_file = f"Num_for_Gen_{file_num}.csv"
    csv_data = []
    heatmap = [[LOW for _ in range(15)] for _ in range(15)]
    for gen in range(int(file_num)):
        # Set target
        heatmap[total_target[gen][0]][total_target[gen][1]] = AIM

        # Set target
        hori = total_target[gen][0] - 15
        vertical = total_target[gen][1] - 15
        if np.abs(hori) >= 4 and np.abs(vertical) >= 4:
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if dist < 2:  # High
                        heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = HIGH
                    elif dist < 4:  # Medium
                        heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = MEDIUM
        elif np.abs(hori) < 4 and np.abs(vertical) >= 4:
            for dx in range(-np.abs(hori), np.abs(hori) + 1):
                for dy in range(-np.abs(hori), np.abs(hori) + 1):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if total_target[gen][0] == 15:
                        heatmap[total_target[gen][0] - 1][total_target[gen][1] + dy] = HIGH
                    else:
                        if dist < 2:  # High
                            heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = HIGH
                        elif dist < np.abs(hori):  # Medium
                            heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = MEDIUM
        elif np.abs(hori) >= 4 and np.abs(vertical) < 4:
            for dx in range(-np.abs(vertical), np.abs(vertical) + 1):
                for dy in range(-np.abs(vertical), np.abs(vertical) + 1):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if total_target[gen][1] == 15:
                        heatmap[total_target[gen][0] - 1][total_target[gen][1] + dy] = HIGH
                    else:
                        if dist < 2:  # High
                            heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = HIGH
                        elif dist < np.abs(hori):  # Medium
                            heatmap[total_target[gen][0] + dx][total_target[gen][1] + dy] = MEDIUM

        for t in range(MAX_TIME):
            low_number_new = 0
            medium_number_new = 0
            high_number_new = 0

            for i in range(NUM_AGENTS):
                if heatmap[int(agent_p[gen][t][i].coord.x)][int(agent_p[gen][t][i].coord.y)] == LOW:
                    low_number_new += 1

                if heatmap[int(agent_p[gen][t][i].coord.x)][int(agent_p[gen][t][i].coord.y)] == MEDIUM:
                    medium_number_new += 1

                if heatmap[int(agent_p[gen][t][i].coord.x)][int(agent_p[gen][t][i].coord.y)] == HIGH:
                    high_number_new += 1

            # Append a dictionary to csv_data
            csv_data.append(
                {"Timestep": t, "LOW": low_number_new, "MEDIUM": medium_number_new, "HIGH": high_number_new})

        # Write the data to csv file
        csv_file = os.path.join(directory, 'output.csv')
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ["Timestep", "LOW", "MEDIUM", "HIGH"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)

        # Read the CSV file using pandas
        df = pd.read_csv('moving/1_layer_All_Medium/output.csv')

        # Transpose the DataFrame
        df_T = df.set_index('Timestep').T

        # Save the transposed DataFrame to a new CSV file
        df_T.to_csv('moving/1_layer_All_Medium/transposed_output.csv')

        heatmap = [[LOW for _ in range(15)] for _ in range(15)]  # Reset

    moving(agent_p, total_gen, total_target, total_pop, 15, 15)
