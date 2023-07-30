import copy
import os

import numpy as np
import random
from utils.util import Pos, Agent
from parameters.STD14 import *
from minimal_surprise import MinimalSurprise
from utils.sensors import *
from draw import random_location


class SelfAssembly:
    def __init__(self, p, p_next, manipulation, size_x, size_y):
        self.p = p  # current position
        self.p_next = p_next  # next position
        self.manipulation = manipulation
        self.sizeX = size_x
        self.sizeY = size_y

        self.fit = 0
        self.move = 0  # A variable to determine the target move

        self.heatmap = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        # Set the coordinates of target / embedded into swarms
        self.target = [int(self.sizeX) // 2, int(self.sizeY) // 2]

        # Evolution count
        self.count = 0

        self.tmp_fit = 0

        # Network initialisation
        self.minimalSurprise = MinimalSurprise(INPUTA, INPUTP, HIDDENA, HIDDENP,
                                               OUTPUTA, OUTPUTP, self.manipulation, self.sizeX, self.sizeY)

    """
        Usage:  Execute agents, let them move
                Count the prediction
        Input:
            gen: Generation index
            ind: Index of population
            p_initial: Initial position of agent
            maxTime: 500
            log: index of log file
            noagent: Number of agents
    """

    def execute(self, gen, ind, p_initial, maxTime, log, noagents):
        timeStep = 0  # Current timestep
        fit = 0
        predReturn = [0] * SENSORS
        tmp_agent_next = Pos(0, 0)

        inputA = np.zeros(INPUTA)
        inputP = np.zeros(INPUTP)

        max_p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

        moving = f"Agents_{NUM_AGENTS}_TimeStep_{maxTime}_Gen_{gen}"
        # file names
        directory = "moving"
        if not os.path.exists(directory):
            os.makedirs(directory)

        moving_file = os.path.join(directory, moving)

        self.p = p_initial.copy()

        # 初始化init_p的位置移动到循环外部，并在每次循环开始都记录初始位置

        print("Gen: ", gen, "Pop", ind, "Initial p: ",
              p_initial[0].coord.x, p_initial[0].coord.y)
        # random_location(p_initial, self.p, self.target, self.sizeX, self.sizeY, 1, 0, heatmap)

        storage_p = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(maxTime + 1)]

        while timeStep < maxTime:
            # determine occupied grid cells (0 - unoccupied, 1 - occupied)
            # Locate all agents into the grid
            # Location with agent equals to 1
            grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

            for i in range(noagents):
                grid[int(self.p[i].coord.x)][int(self.p[i].coord.y)] = 1

            grid[self.target[0]][self.target[1]] = 1  # Set target location

            # Iterate all agents
            # Execute agents one by one in each timeStep
            for i in range(noagents):
                # Determine current sensor values (S of t)
                sensor = Sensors(self.sizeX, self.sizeY)
                temp = self.heatmap[self.p[i].coord.x][self.p[i].coord.y]

                if SENSOR_MODEL == STDL:
                    sensors = sensor.sensorModelSTDL(i, grid, self.p)
                elif SENSOR_MODEL == STDS:
                    sensors = sensor.sensorModelSTD(i, grid, self.p)
                elif SENSOR_MODEL == STDSL:
                    sensors = sensor.sensorModelSTDSL(i, grid, self.p)

                # Get all sensor values from 15 * 15 grid
                # Shape Line S0, S3, S8, S11 equals to 1
                # At least 3 agents in one line
                # Set sensor values to both networks
                for j in range(SENSORS):
                    # set sensor values as ANN input values
                    if j == 14:
                        inputA[j] = temp
                        inputP[j] = temp
                    else:
                        inputA[j] = sensors[j]
                        inputP[j] = sensors[j]

                # Propagate action network
                # Input: current sensor values + last action
                # Output: next action --> 0 or 1
                if timeStep <= 0:
                    inputA[SENSORS] = STRAIGHT
                else:  # Last time action
                    inputA[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep - 1]

                action_output = self.minimalSurprise.action.propagate_action_net(
                    self.minimalSurprise.action.weight_actionNet_layer0[ind],
                    self.minimalSurprise.action.weight_actionNet_layer1[ind],
                    self.minimalSurprise.action.weight_actionNet_layer2[ind], inputA)

                # print("Action output for agent", i, ": ", action_output[0])
                self.minimalSurprise.action.current_action[i][timeStep] = action_output[0]

                # Propagate prediction network Call it after *Action*
                # Input: current sensor values + next action [Returned by ANN]
                # Output: Prediction of sensor value (per agent)
                # Action from the *Action* network
                inputP[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep]

                # Feed input values into the prediction Network
                if self.manipulation != PRE:
                    self.minimalSurprise.prediction.propagate_prediction_network(
                        self.minimalSurprise.prediction.weight_predictionNet_layer0[ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer1[ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer2[ind], i, inputP, self.p)

                # Calculate the fitness
                # No need to compare the temperature
                for j in range(SENSORS - 1):
                    if sensors[j] == self.minimalSurprise.prediction.predictions[i][j]:
                        fit += 1
                    predReturn[j] += self.minimalSurprise.prediction.predictions[i][j]
                    # End Sensor loops

                # Check next action
                # 0 == move straight; 1 == turn
                if self.minimalSurprise.action.current_action[i][timeStep] == STRAIGHT:
                    # movement only possible when cell in front is not occupied (sensor S0)
                    # move in heading direction (i.e. straight)
                    tmp_agent_next.x = sensor.adjustXPosition(self.p[i].coord.x + self.p[i].heading.x)
                    tmp_agent_next.y = sensor.adjustYPosition(self.p[i].coord.y + self.p[i].heading.y)

                    # Front sensor and check next grid is available
                    if sensors[S0] == 0 and grid[tmp_agent_next.x][tmp_agent_next.y] == 0 \
                            and self.heatmap[tmp_agent_next.x][tmp_agent_next.y] != HIGH and \
                            (tmp_agent_next.x != self.sizeX or tmp_agent_next.y != self.sizeY):
                        # check if next cell is already occupied by agent
                        # next agent positions as far as updated (otherwise positions already checked via sensors)
                        # Agent move
                        grid[self.p[i].coord.x][self.p[i].coord.y] = 0  # Set current cell available
                        grid[tmp_agent_next.x][tmp_agent_next.y] = 1  # Set next cell unavailable

                        self.p_next[i].coord.x = tmp_agent_next.x
                        self.p_next[i].coord.y = tmp_agent_next.y
                        self.p_next[i].heading.x = self.p[i].heading.x
                        self.p_next[i].heading.y = self.p[i].heading.y

                    else:
                        # print("Can't move")
                        self.p_next[i].coord.x = self.p[i].coord.x
                        self.p_next[i].coord.y = self.p[i].coord.y
                        self.p_next[i].heading.x = self.p[i].heading.x
                        self.p_next[i].heading.y = self.p[i].heading.y

                # agent Turn --> Update heading
                elif self.minimalSurprise.action.current_action[i][timeStep] == TURN:
                    # Keep same coordinate
                    self.p_next[i].coord.x = self.p[i].coord.x
                    self.p_next[i].coord.y = self.p[i].coord.y

                    # calculate current orientation
                    angle = np.arctan2(self.p[i].heading.y, self.p[i].heading.x)
                    self.p_next[i].heading.x = round(np.cos(angle + action_output[1] * (PI / 2)))
                    self.p_next[i].heading.y = round(np.sin(angle + action_output[1] * (PI / 2)))

                # High score scenarios:
                # 1. When the agent is moving to the LOW zone,
                #    Compare the distance, if the distance decreases, which means the agent moves towards the target.
                # 2. When the agent is located in the HIGH zone,
                #    If the agent moves away the target, the action is good.
                loc_original = np.array([self.p[i].coord.x, self.p[i].coord.y])
                loc_next = np.array([self.p_next[i].coord.x, self.p_next[i].coord.y])
                distance = np.linalg.norm(np.array(self.target) - loc_next) \
                           - np.linalg.norm(np.array(self.target) - loc_original)
                if self.heatmap[self.p[i].coord.x][self.p[i].coord.y] == HIGH and \
                        self.heatmap[self.p_next[i].coord.x][self.p_next[i].coord.y] == MEDIUM:
                    self.fit += 1  # 逃避高温区
                elif self.heatmap[self.p_next[i].coord.x][self.p_next[i].coord.y] == LOW:
                    if distance < 0 and \
                            self.heatmap[self.p_next[i].coord.x][self.p_next[i].coord.y] == MEDIUM:  # 朝目标移动
                        self.fit += 1
                elif self.heatmap[self.p[i].coord.x][self.p[i].coord.y] == MEDIUM:
                    if self.heatmap[self.p_next[i].coord.x][self.p_next[i].coord.y] != HIGH:
                        self.fit += 1  # 朝目标移动,且不进入高温区 / 保持在中温区

                storage_p[timeStep] = self.p_next.copy()

            # End Agent Iterations
            # random_location(self.p, self.p_next, self.target, self.sizeX, self.sizeY, 0, 0)
            timeStep += 1

            # Update positions
            temp = self.p_next
            self.p = temp
            self.p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]  # Reset p_next
        # End while loop

        # prediction counter
        # Pred_return = 1 / T * N
        # for i in range(SENSORS):
        #     self.minimalSurprise.prediction.pred_return[i] = float(predReturn[i]) / (maxTime * noagents)

        # F = 1 / T * N * R * (1 - |S - P|) * HOT_PARAMETER
        fit_return = ((float(fit) / float(noagents * maxTime * SENSORS))
                      + float(self.fit) / float(noagents * maxTime)) * 1 / 2
        # fit_return = float(self.fit) / float(noagents * maxTime)
        self.fit = 0  # Reset

        if self.tmp_fit <= fit_return:
            max_p = self.p.copy()
            self.tmp_fit = fit_return
            f = open(moving_file, "w")
            f.write(f"Gen: {gen} \n")
            f.write(f"Pop: {ind} \n")
            f.write(f"Target: {self.target[0]}, {self.target[1]} \n")
            f.write(f"Fitness: {self.tmp_fit} \n")
            f.write(f"\n")

            for t in range(maxTime + 1):
                for i in range(NUM_AGENTS):
                    if t == 0:
                        f.write(f"{t}, {i}: {p_initial[i].coord.x}, {p_initial[i].coord.y}, "
                                f"{p_initial[i].heading.x}, {p_initial[i].heading.y}\n")
                    else:
                        f.write(f"{t}, {i}: {storage_p[t - 1][i].coord.x}, {storage_p[t - 1][i].coord.y}, "
                                f"{storage_p[t - 1][i].heading.x}, {storage_p[t - 1][i].heading.y}\n")
                f.write(f"\n")

            f.close()

        return fit_return, max_p

    """
        Usage: Do evolution
    """

    def evolution(self):
        print("Evolution count: ", self.count)

        # Store fitness for all population
        fitness = np.zeros(POP_SIZE, dtype=float)

        # store agent movement
        agent_maxfit = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        tmp_agent_maxfit_final = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        temp_p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

        # file names
        directory = "result_1"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file = f"_Agents_{NUM_AGENTS}_TargetX_{self.target[0]}_TargetY_{self.target[1]}"

        # fit_file = os.path.join(directory, "fitness" + file)
        agent_file = os.path.join(directory, file)

        # initialise weights of neural nets in range [-0.5, 0.5]
        # Shape (50, 3, 224)
        for ind in range(POP_SIZE):
            for j in range(LAYERS):
                if j == 0:
                    for k in range(ACT_CONNECTIONS):
                        self.minimalSurprise.action.weight_actionNet_layer0[ind][k] = random.uniform(-0.5, 0.5)
                    for k in range(PRE_CONNECTIONS):
                        self.minimalSurprise.prediction.weight_predictionNet_layer0[ind][k] = random.uniform(-0.5, 0.5)
                    continue

                elif j == 1:  # 15 * 8 for action || 15 * 14 for prediction
                    # print("Layer 2:", INPUTA * HIDDENA, INPUTP * HIDDENP)
                    for k in range(INPUTA * HIDDENA):
                        self.minimalSurprise.action.weight_actionNet_layer1[ind][k] = random.uniform(-0.5, 0.5)
                    for k in range((INPUTP + 1) * HIDDENP):
                        self.minimalSurprise.prediction.weight_predictionNet_layer1[ind][k] = random.uniform(-0.5, 0.5)
                    continue

                elif j == 2:
                    # print("Layer 3:", HIDDENA * (OUTPUTA + 1), (HIDDENP * (OUTPUTP + 1)))
                    for k in range(HIDDENA * OUTPUTA):
                        self.minimalSurprise.action.weight_actionNet_layer2[ind][k] = random.uniform(-0.5, 0.5)
                    for k in range(HIDDENP * OUTPUTP):
                        self.minimalSurprise.prediction.weight_predictionNet_layer2[ind][k] = random.uniform(-0.5, 0.5)
                    continue

        p_initial = self.location_init()
        # random_location(p_initial, p_initial, self.target, self.sizeX, self.sizeY, 1, 0, heatmap)

        # evolutionary runs
        # For one generation:
        # 50 Agents * 10 times repetitions ==> 500 individuals
        # Total: 50 * 10 * 100 ==> 50000 generations
        for gen in range(MAX_GENS):
            max = 0.0
            avg = 0.0
            maxID = -1

            for i in range(NUM_AGENTS):
                temp_p[i].coord.x = p_initial[i].coord.x
                temp_p[i].coord.y = p_initial[i].coord.y
                temp_p[i].heading.x = p_initial[i].heading.x
                temp_p[i].heading.y = p_initial[i].heading.y
                temp_p[i].type = p_initial[i].type
            # population level (iterate through individuals)
            # POP_SIZE = 50
            # Each generation have 50 population, 100 agents
            for ind in range(POP_SIZE):
                # fitness evaluation - initialisation based on case
                fitness[ind] = 0.0
                tmp_fitness = 0.0

                store = False

                tmp_fitness, max_p = self.execute(gen, ind, temp_p, MAX_TIME, 0, NUM_AGENTS)
                print("Fitness for population:", ind + 1, "Score:", tmp_fitness)

                # max fitness of Repetitions kept
                if FIT_EVAL == MAX:
                    if tmp_fitness > fitness[ind]:
                        fitness[ind] = tmp_fitness
                        store = True

                # store best fitness + id of repetition
                if store:
                    for i in range(NUM_AGENTS):  # store agent end positions
                        tmp_agent_maxfit_final[i].coord.x = max_p[i].coord.x
                        tmp_agent_maxfit_final[i].coord.y = max_p[i].coord.y
                        tmp_agent_maxfit_final[i].heading.x = max_p[i].heading.x
                        tmp_agent_maxfit_final[i].heading.y = max_p[i].heading.y
                        tmp_agent_maxfit_final[i].type = max_p[i].type

                # Average fitness of generation
                avg += fitness[ind]

                # store individual with maximum fitness
                if fitness[ind] > max:
                    max = fitness[ind]
                    maxID = ind

                    # store initial and final agent positions
                    for i in range(NUM_AGENTS):
                        agent_maxfit[i].coord.x = tmp_agent_maxfit_final[i].coord.x
                        agent_maxfit[i].coord.y = tmp_agent_maxfit_final[i].coord.y
                        agent_maxfit[i].heading.x = tmp_agent_maxfit_final[i].heading.x
                        agent_maxfit[i].heading.y = tmp_agent_maxfit_final[i].heading.y
                        agent_maxfit[i].type = tmp_agent_maxfit_final[i].type

                    tmp_initial = agent_maxfit.copy()
                # End Fitness store
                print("Score for generation: ", gen + 1, "Score: ", max)
            # End population loop

            print("Next gen: ", p_initial[0].coord.x, p_initial[0].coord.y)
            p_initial = tmp_initial.copy()  # Update Initial pos

            print(f"#{gen} {max} ({maxID})")
            with open(agent_file, "a") as f:
                f.write(f"Gen: {gen}\n")
                f.write(f"Grid: {self.sizeX}, {self.sizeY}\n")
                f.write(f"Fitness: {max}\n")
                f.write(f"Target: {self.target[0]}, {self.target[1]}\n")
                for i in range(NUM_AGENTS):
                    f.write(f"{agent_maxfit[i].coord.x}, {agent_maxfit[i].coord.y}, ")
                    f.write(f"{temp_p[i].coord.x}, {temp_p[i].coord.y}, ")
                    f.write(f"{agent_maxfit[i].heading.x}, {agent_maxfit[i].heading.y}, ")
                    f.write(f"{temp_p[i].heading.x}, {temp_p[i].heading.y}, ")
                    f.write("\n")
                f.write("\n")

            self.tmp_fit = 0

            # Do selection & mutation per generation
            self.minimalSurprise.select_mutate(maxID, fitness)

            "Do target moving HERE"
            # Move the target when the score is larger than the threshold
            # Make sure there is no obstacle around the target
            if max >= Threshold:
                move = True
                for i in range(NUM_AGENTS):
                    if self.heatmap[agent_maxfit[i].coord.x][agent_maxfit[i].coord.y] == HIGH:
                        move = False

                if move:
                    self.update_heatmap(agent_maxfit)

            # End evolution runs loop

        self.execute(gen, ind, p_initial, MAX_TIME, 1, NUM_AGENTS)

    def location_init(self):
        p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        # initialisation of starting positions
        # (all genomes have same set of starting positions)
        grid[self.target[0]][self.target[1]] = 1  # Set target location
        self.heatmap[self.target[0]][self.target[1]] = AIM

        # All heatmap is low
        self.heatmap = [[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        # Set target
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                # Calculate the distance
                dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                if dist < 2:  # High
                    self.heatmap[self.target[0] + dx][self.target[1] + dy] = HIGH
                    grid[self.target[0] + dx][self.target[1] + dy] = 1  # Set positions unavailable
                elif dist < 5:  # Medium
                    self.heatmap[self.target[0] + dx][self.target[1] + dy] = MEDIUM

        # generate agent positions
        # In each repeat, all agent will be initialized
        for i in range(NUM_AGENTS):
            # initialisation of starting positions
            block = True

            # Find an unoccupied location
            while block:
                # Randomise a position for each agent
                p_initial[i].coord.x = random.randint(0, self.sizeX - 1)
                p_initial[i].coord.y = random.randint(0, self.sizeY - 1)

                if grid[p_initial[i].coord.x][p_initial[i].coord.y] == 0:  # not occupied
                    block = False
                    grid[p_initial[i].coord.x][p_initial[i].coord.y] = 1  # set grid cell occupied

                # Set agent heading values randomly (north, south, west, east)
                directions = [1, -1]
                randInd = random.randint(0, 1)
                if random.random() < 0.5:  # West & East
                    p_initial[i].heading.x = directions[randInd]
                    p_initial[i].heading.y = 0
                else:  # North & South
                    p_initial[i].heading.x = 0
                    p_initial[i].heading.y = directions[randInd]

        # End Location Initialisation
        return p_initial

    def target_move(self, agent):
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        for i in range(NUM_AGENTS):
            grid[agent[i].coord.x][agent[i].coord.y] = 1  # set grid cell occupied

        block = True
        while block:
            randInd = random.randint(0, 1)
            direction = [-1, 1]
            if random.random() < 0.5:  # West & East
                tmp_X = self.target[0] + direction[randInd]
                tmp_Y = self.target[1]
            else:  # North & South
                tmp_X = self.target[0]
                tmp_Y = self.target[1] + direction[randInd]

            if grid[tmp_X][tmp_Y] == 0 and 0 <= tmp_X <= self.sizeX and 0 <= tmp_Y <= self.sizeY:
                block = False  # Move the target and Avoid walls

        self.target[0] = tmp_X
        self.target[1] = tmp_Y

    def update_heatmap(self, agent):
        self.heatmap[self.target[0]][self.target[1]] = HIGH  # Change old Pos to HIGH zone
        self.target_move(agent)
        self.heatmap[self.target[0]][self.target[1]] = AIM  # Update new aim

        # All heatmap is low
        self.heatmap = [[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        # Set target
        hori = self.target[0] - self.sizeX
        vertical = self.target[1] - self.sizeY
        if np.abs(hori) >= 5 and np.abs(vertical) >= 5:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if dist < 2:  # High
                        self.heatmap[self.target[0] + dx][self.target[1] + dy] = HIGH
                    elif dist < 5:  # Medium
                        self.heatmap[self.target[0] + dx][self.target[1] + dy] = MEDIUM
        elif np.abs(hori) < 5 and np.abs(vertical) >= 5:
            for dx in range(-np.abs(hori), np.abs(hori) + 1):
                for dy in range(-5, 6):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if self.target[0] == 15:
                        self.heatmap[self.target[0] - 1][self.target[1] + dy] = HIGH
                    else:
                        if dist < 2:  # High
                            self.heatmap[self.target[0] + dx][self.target[1] + dy] = HIGH
                        elif dist < np.abs(hori):  # Medium
                            self.heatmap[self.target[0] + dx][self.target[1] + dy] = MEDIUM
        elif np.abs(hori) >= 5 and np.abs(vertical) < 5:
            for dx in range(-5, 6):
                for dy in range(-np.abs(vertical), np.abs(vertical) + 1):
                    # Calculate the distance
                    dist = np.abs(dx) if np.abs(dx) > np.abs(dy) else np.abs(dy)

                    if self.target[1] == 15:
                        self.heatmap[self.target[0] - 1][self.target[1] + dy] = HIGH
                    else:
                        if dist < 2:  # High
                            self.heatmap[self.target[0] + dx][self.target[1] + dy] = HIGH
                        elif dist < np.abs(hori):  # Medium
                            self.heatmap[self.target[0] + dx][self.target[1] + dy] = MEDIUM
