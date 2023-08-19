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
    num_generations = len(agents)

    N = int(np.ceil(np.sqrt(num_generations)))

    plt.ion()
    fig = plt.figure(figsize=(8, 8))

    prev_x, prev_y = None, None
    move_num = 1

    image_files = []

    for gen in range(num_generations):
        ax = fig.add_subplot(N, N, gen + 1)

        for t in range(MAX_TIME + 1):
            ax.clear()
            ax.clear()

            plt.clf()
            ax = fig.add_subplot(111)
            color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0

            RED_COLOR = np.array([222, 71, 71]) / 255.0
            ORANGE_COLOR = np.array([255, 165, 100]) / 255.0

            for tgt in range(len(target[gen])):
                x = int(target[gen][tgt][0]) + 1
                y = int(target[gen][tgt][1]) + 1

                if x == 0 and y == 0:
                    color_grid = np.full((sizeX + 2, sizeY + 2, 3), [188, 216, 235]) / 255.0
                    ax.imshow(color_grid, origin='lower')
                    continue

                if prev_x is not None and prev_y is not None and (prev_x != x or prev_y != y):
                    move_num = 1
                prev_x, prev_y = x, y

                for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                        # Check bounds before accessing color_grid
                        if 0 <= y + dy < color_grid.shape[0] and 0 <= x + dx < color_grid.shape[1]:
                            current_color = color_grid[y + dy, x + dx]

                            if np.array_equal(current_color, RED_COLOR):
                                continue

                            dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)
                            if dist < 1 + TARGET_SIZE:
                                color_grid[y + dy, x + dx] = RED_COLOR
                            elif dist < 3 + TARGET_SIZE:
                                color_grid[y + dy, x + dx] = ORANGE_COLOR

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

            ax.plot([1, sizeX, sizeY, 1, 1], [1, 1, sizeX, sizeY, 0], color='black', linewidth=0.4,
                    linestyle="dashed")
            for i in range(1, sizeY):
                ax.plot([i, i], [1, sizeY], color='black', linewidth=0.5, linestyle="dashed")
            for i in range(1, sizeX):
                ax.plot([1, sizeX], [i, i], color='black', linewidth=0.5, linestyle="dashed")

            ax.set_xlim(0, sizeX + 1)
            ax.set_ylim(0, sizeY + 1)

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

            plt.draw()
            filename = f"temp_file/temp_frame_{gen}_{t}.png"
            plt.savefig(filename)
            image_files.append(filename)
            plt.pause(0.05)  # 暂停0.15秒

    plt.close()

    # Create videos
    with imageio.get_writer('moving/8_13/test_data/movie.mp4', fps=15) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    # delete files
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


def parse_colon_separated_line(line):
    return int(line.split(":")[1].strip())


def parse_data_line(line):
    return re.split('[,.: \n]', line)


def process_file(filename, total_target, total_pop, agent_p):
    with open(filename, 'r') as f:
        target_count = 0
        generation = None
        for line in f:
            line = line.strip()
            if line.startswith('Gen'):
                generation = parse_colon_separated_line(line)
            elif line.startswith('Pop'):
                total_pop.append(parse_colon_separated_line(line))
            elif line.startswith('Target'):
                data = parse_data_line(line)
                total_target[f_num][target_count] = [int(data[3]), int(data[5])]
                target_count += 1
            elif not line.startswith('Fitness'):  # Handle data lines
                data = parse_data_line(line)
                if len(data) > 10:
                    agent = agent_p[generation][int(data[0])][int(data[2])]
                    agent.coord.x, agent.coord.y = int(data[4]), int(data[6])
                    agent.heading.x, agent.heading.y = int(data[8]), int(data[10])


def get_agent_value(agent, heatmap):
    return heatmap[int(agent.coord.x)][int(agent.coord.y)]


def write_csv_data(directory, csv_data):
    csv_file = os.path.join(directory, 'output.csv')
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ["Timestep", "LOW", "MEDIUM", "HIGH"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    df = pd.read_csv(csv_file)
    df_T = df.set_index('Timestep').T
    df_T.to_csv(os.path.join(directory, 'transposed_output.csv'))


if __name__ == "__main__":
    directory = "moving/8_13/test_data/"
    sizeX = 15
    sizeY = 15
    file_num = count_files_in_dir(directory)

    agent_p = [[[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(MAX_TIME + 1)] for _ in
               range(file_num)]
    total_target = [[[0, 0] for _ in range(AIM_NUM)] for _ in range(file_num)]
    total_gen, total_pop = [], []

    for f_num in range(file_num):
        filename = f"{directory}Agents_{NUM_AGENTS}_TimeStep_{MAX_TIME}_Gen_{f_num}"
        process_file(filename, total_target, total_pop, agent_p)

        heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]
        csv_data = []

        for gen in range(file_num):
            targets = [(total_target[gen][t], total_target[gen][AIM_NUM - t - 1]) for t in range(AIM_NUM)]
            for target1, target2 in targets:
                heatmap[target1[0]][target1[1]], heatmap[target2[0]][target2[1]] = AIM, AIM
                for target, size in zip([target1, target2], [sizeX, sizeY]):
                    update_heatmap_position(heatmap, target, target[0] - size, target[1] - size, sizeX, sizeY)

            csv_data.extend([{
                "Timestep": t,
                "LOW": sum(get_agent_value(agent, heatmap) == LOW for agent in agent_p[gen][t]),
                "MEDIUM": sum(get_agent_value(agent, heatmap) == MEDIUM for agent in agent_p[gen][t]),
                "HIGH": sum(get_agent_value(agent, heatmap) == HIGH for agent in agent_p[gen][t])
            } for t in range(MAX_TIME)])

            write_csv_data(directory, csv_data)

        heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]  # Reset

    moving(agent_p, total_target, sizeX, sizeY)
