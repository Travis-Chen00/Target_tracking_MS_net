import csv
import os
import re

import pandas as pd

from utils.util import Agent, Pos
import numpy as np
from parameters.STD14 import *


def parse_colon_separated_line(line):
    return int(line.split(":")[1].strip())


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


def generate_heatmap_for_gen(total_target, gen, sizeX, sizeY):
    heatmap = [[LOW for _ in range(sizeY)] for _ in range(sizeX)]
    targets = [(total_target[gen][t], total_target[gen][AIM_NUM - t - 1]) for t in range(AIM_NUM)]
    for target1, target2 in targets:
        heatmap[target1[0]][target1[1]], heatmap[target2[0]][target2[1]] = AIM, AIM
        for target, size in zip([target1, target2], [sizeX, sizeY]):
            update_heatmap_position(heatmap, target, target[0] - size, target[1] - size, sizeX, sizeY)
    return heatmap


def parse_data_line(line):
    return re.split('[,.: \n]', line)


def process_file(filename, total_target, total_pop, agent_p):
    with open(filename, 'r') as f:
        target_count = 0
        generation = None
        for line in f:
            line = line.strip()
            if re.match(r'^Gen(eration)?: \d+$', line):
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
                    agent.inspire = int(data[12])


def check_agent_inspiration_compatibility(agent, heatmap):
    """
    Check if the agent is correctly situated based on its inspire value.
    """
    position_value = heatmap[int(agent.coord.x)][int(agent.coord.y)]
    if position_value == HIGH:
        return False
    else:
        return position_value == agent.inspire


def record_inspiration_compatibility(directory, agent_p, total_target, file_num):
    """
    Record the compatibility of agents' inspiration with their actual position in a new file.
    """
    result_data = []
    timestep_counts_data = []

    for gen in range(file_num):

        heatmap = generate_heatmap_for_gen(total_target, gen, sizeX, sizeY)

        # Initialize the timestep count for this generation
        timestep_true_count = {t: 0 for t in range(MAX_TIME)}

        for t in range(MAX_TIME):
            for i in range(NUM_AGENTS):
                agent = agent_p[gen][t][i]
                is_compatible = check_agent_inspiration_compatibility(agent, heatmap)
                result_data.append({
                    "Generation": gen,
                    "Timestep": t,
                    "Agent": i,
                    "Inspire": agent.inspire,
                    "Is_Correctly_Situated": is_compatible
                })

                if is_compatible:
                    timestep_true_count[t] += 1

            # Append timestep normalized count data for this generation
            timestep_counts_data.append({
                "Generation": gen,
                "Timestep": t,
                "Normalized_True_Count": timestep_true_count[t]
            })

    dir = "F://self//evaluation//csv_files//results/"

    # Write the normalized counts to a new CSV file
    csv_file_counts = os.path.join(dir, 'TAMS_timestep_counts.csv')
    with open(csv_file_counts, 'w', newline='') as f:
        fieldnames = ["Generation", "Timestep", "Normalized_True_Count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timestep_counts_data)


def count_files_in_dir(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


if __name__ == "__main__":
    directory = "F://self//Evaluation//test_data/"
    sizeX = 15
    sizeY = 15
    file_num = count_files_in_dir(directory)

    agent_p = [[[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(MAX_TIME + 1)] for _ in
               range(file_num)]
    total_target = [[[0, 0] for _ in range(AIM_NUM)] for _ in range(file_num)]
    total_gen, total_pop = [], []

    for f_num in range(1, file_num):
        filename = f"{directory}Agents_{NUM_AGENTS}_TimeStep_{MAX_TIME}_Gen_{f_num}"
        process_file(filename, total_target, total_pop, agent_p)

    record_inspiration_compatibility(directory, agent_p, total_target, file_num)
