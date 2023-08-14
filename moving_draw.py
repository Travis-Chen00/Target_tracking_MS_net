import csv
import re
import imageio as imageio
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon, Circle
from utils.util import Agent, Pos
from parameters.STD14 import *
import numpy as np
import os


def count_files_in_dir(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def moving(agents, target, sizeX, sizeY):
    num_generations = len(agents)  # 总共的代数

    # 根据代数确定子图的布局，N x N
    N = int(np.ceil(np.sqrt(num_generations)))

    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(8, 8))  # 创建新的figure，你可以根据需要修改figure的大小

    prev_x, prev_y = None, None  # 初始化上一次的目标位置
    move_num = 1

    image_files = []  # 用于保存所有的图像路径

    # 对每一代进行迭代
    for gen in range(num_generations):
        ax = fig.add_subplot(N, N, gen + 1)  # 为每一代定义子图

        # 对于当前代的每个时间点进行迭代
        for t in range(MAX_TIME + 1):
            # 清除当前子图
            ax.clear()
            ax.clear()

            plt.clf()  # 清除当前的figure
            ax = fig.add_subplot(111)  # 为每一代定义子图
            color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 创建一个稍大的2D数组，颜色值为蓝色的RGB

            RED_COLOR = np.array([222, 71, 71]) / 255.0  # 浅红色
            ORANGE_COLOR = np.array([255, 165, 100]) / 255.0  # 浅橘色

            for tgt in range(len(target[gen])):
                x = int(target[gen][tgt][0]) + 1
                y = int(target[gen][tgt][1]) + 1

                # 如果目标位置是(0, 0)（请注意，由于您在代码中为所有坐标加了1，所以检查(1, 1)）
                if x == 0 and y == 0:
                    color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0  # 重新设置为全蓝色
                    ax.imshow(color_grid, origin='lower')  # 显示全蓝色图像
                    continue  # 跳过当前循环，进入下一个tgt

                if prev_x is not None and prev_y is not None and (prev_x != x or prev_y != y):
                    move_num = 1
                prev_x, prev_y = x, y  # 记录当前目标位置，用于下一次循环

                # 将目标周围的一圈（9个格子）设为红色
                for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                        # Check bounds before accessing color_grid
                        if 0 <= y + dy < color_grid.shape[0] and 0 <= x + dx < color_grid.shape[1]:
                            current_color = color_grid[y + dy, x + dx]

                            # 如果当前颜色为红色，保留红色
                            if np.array_equal(current_color, RED_COLOR):
                                continue

                            dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)
                            if dist < 1 + TARGET_SIZE:
                                color_grid[y + dy, x + dx] = RED_COLOR
                            elif dist < 3 + TARGET_SIZE:
                                color_grid[y + dy, x + dx] = ORANGE_COLOR

                # 设置标题
                if gen == 0:
                    filename = "Initial " + "Target: " + str(target[gen][tgt][0]) + ", " + str(
                        target[gen][tgt][1]) + " Time:" + str(t)
                    ax.set_title(filename)
                else:
                    filename = "Target: " + str(target[gen][tgt][0]) + ", " + str(target[gen][tgt][1]) + " Time:" + str(
                        t)
                    move_num += 1
                    ax.set_title(filename)

                circle = Circle((x, y), TARGET_SIZE - 0.78, color='red')
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
            # 将当前图像保存为文件
            filename = f"temp_file/temp_frame_{gen}_{t}.png"
            plt.savefig(filename)
            image_files.append(filename)
            plt.pause(0.05)  # 暂停0.15秒

    # 关闭matplotlib的窗口
    plt.close()

    # 使用imageio生成视频
    with imageio.get_writer('moving/8_13/test_data/movie.mp4', fps=15) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    # 删除临时的图像文件
    for filename in image_files:
        os.remove(filename)


def update_heatmap_position(heatmap, target, hori, vertical, sizeX, sizeY):
    for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
        for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
            x, y = target[0] + dx, target[1] + dy

            if 0 <= x < sizeX and 0 <= y < sizeY:  # Boundary check
                dist = max(np.abs(dx), np.abs(dy))

                if dist < 1 + TARGET_SIZE:
                    heatmap[x][y] = max(heatmap[x][y], HIGH)
                elif dist < 3 + TARGET_SIZE:
                    heatmap[x][y] = max(heatmap[x][y], MEDIUM)

    if np.abs(hori) < 3 + TARGET_SIZE:
        limit = np.abs(hori)
        for dx in range(-limit, limit + 1):
            x, y = target[0] + dx, target[1]
            if 0 <= x < sizeX:
                heatmap[x][y] = max(heatmap[x][y], HIGH) if x == 14 else max(heatmap[x][y], MEDIUM)

    if np.abs(vertical) < 3 + TARGET_SIZE:
        limit = np.abs(vertical)
        for dy in range(-limit, limit + 1):
            x, y = target[0], target[1] + dy
            if 0 <= y < sizeY:
                heatmap[x][y] = max(heatmap[x][y], HIGH) if y == 14 else max(heatmap[x][y], MEDIUM)


if __name__ == "__main__":
    directory = "moving/8_13/test_data/"
    file_num = count_files_in_dir(directory)

    sizeX = 19
    sizeY = 19

    agent_p = [[[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for i in range(MAX_TIME + 1)] for _ in
               range(int(file_num))]
    total_target, total_gen, total_pop = [[[0, 0] for _ in range(AIM_NUM)] for _ in range(int(file_num))], [], []
    for f_num in range(int(file_num)):
        para = False
        generation = None
        filename = directory + "Agents_" + str(NUM_AGENTS) + "_TimeStep_" + str(MAX_TIME) + "_Gen_" + str(f_num)

        with open(filename, 'r') as f:
            target_count = 0
            for line in f:
                line = line.strip()
                if line.startswith('Gen'):
                    generation = int(line.split(":")[1].strip())
                    # total_gen.append(generation)
                elif line.startswith('Pop'):
                    total_pop.append(int(line.split(":")[1].strip()))
                elif line.startswith('Target'):
                    gen = re.split('[,.: \n]', line)
                    total_target[f_num][target_count][0] = int(gen[3])
                    total_target[f_num][target_count][1] = int(gen[5])
                    target_count += 1
                    # total_target[f_num].append(list(map(int, line.split(":")[1].strip().split(","))))
                elif line.startswith('Fitness'):
                    continue  # you can handle 'Fitness' here if necessary
                else:  # handle data lines
                    gen = re.split('[,.: \n]', line)
                    if len(gen) > 10:
                        agent_p[generation][int(gen[0])][int(gen[2])].coord.x = int(gen[4])
                        agent_p[generation][int(gen[0])][int(gen[2])].coord.y = int(gen[6])
                        agent_p[generation][int(gen[0])][int(gen[2])].heading.x = int(gen[8])
                        agent_p[generation][int(gen[0])][int(gen[2])].heading.y = int(gen[10])
            tmp_target = []

    heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]
    csv_data = []

    for gen in range(int(file_num)):
        for t in range(AIM_NUM):
            target1 = total_target[gen][t]
            target2 = total_target[gen][AIM_NUM - t - 1]  # 假设第二个target紧跟在第一个target后面

            hori1 = target1[0] - sizeX
            vertical1 = target1[1] - sizeY

            hori2 = target2[0] - sizeX
            vertical2 = target2[1] - sizeY

            heatmap[target1[0]][target1[1]] = AIM
            heatmap[target2[0]][target2[1]] = AIM

            update_heatmap_position(heatmap, target1, hori1, vertical1, sizeX, sizeY)
            update_heatmap_position(heatmap, target2, hori2, vertical2, sizeX, sizeY)

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
        df = pd.read_csv('moving/8_13/test_data/output.csv')

        # Transpose the DataFrame
        df_T = df.set_index('Timestep').T

        # Save the transposed DataFrame to a new CSV file
        df_T.to_csv('moving/8_13/test_data/transposed_output.csv')

        heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]  # Reset

    moving(agent_p, total_target, sizeX, sizeY)
