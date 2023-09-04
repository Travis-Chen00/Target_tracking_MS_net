import os
from copy import copy, deepcopy
import numpy as np
import random
from utils.util import Pos, Agent
from minimal_surprise import MinimalSurprise
from utils.sensors import *


class SelfAssembly:
    def __init__(self, p, p_next, manipulation, size_x, size_y):
        self.p = p  # current position
        self.p_next = p_next  # next position
        self.manipulation = manipulation
        self.sizeX = size_x
        self.sizeY = size_y

        self.move = 0  # A variable to determine the target move

        self.heatmap = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]
        self.heat_intensity = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        # Set the coordinates of target / embedded into swarms
        self.target = [[0, 0] for _ in range(AIM_NUM)]
        self.tmp_target = [[0, 0] for _ in range(AIM_NUM)]

        # Evolution count
        self.count = 0

        # Index to determine the catastrophe
        self.cata = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

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
        fit_individual = 0
        total_fit = []
        diff = 0.0

        tmp_agent_next = Pos(0, 0)

        inputA = np.zeros(INPUTA)
        inputP = np.zeros(INPUTP)

        max_p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

        moving = f"Agents_{NUM_AGENTS}_TimeStep_{maxTime}_Gen_{gen}"
        # file names
        directory = "moving/8_15/new_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        moving_file = os.path.join(directory, moving)

        self.p = p_initial.copy()

        # random_location(p_initial, self.p, self.target, self.sizeX, self.sizeY, 1, 0, heatmap)
        storage_p = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(maxTime + 1)]

        while timeStep < maxTime:
            # determine occupied grid cells (0 - unoccupied, 1 - occupied)
            # Locate all agents into the grid
            # Location with agent equals to 1
            grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

            for i in range(noagents):
                grid[int(self.p[i].coord.x)][int(self.p[i].coord.y)] = 1

            for t in range(AIM_NUM):
                grid[self.target[t][0]][self.target[t][1]] = 1  # Set target location

            # Iterate all agents
            # Execute agents one by one in each timeStep
            for i in range(noagents):
                # Determine current sensor values (S of t)
                sensor = Sensors(self.sizeX, self.sizeY)

                if SENSOR_MODEL == STDL:
                    sensors = sensor.sensorModelSTDL(i, grid, self.heatmap, self.p)

                for j in range(SENSORS):
                    # set sensor values as ANN input values
                    inputA[j] = sensors[j]
                    inputP[j] = sensors[j]

                    # Calculate the difference
                    if j < 5:
                        diff += 1 - np.abs(sensors[j] - self.minimalSurprise.prediction.predictions[i][j])

                # Propagate action network
                if timeStep <= 0:
                    inputA[SENSORS] = STRAIGHT
                else:  # Last time action
                    inputA[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep - 1]

                action_output = self.minimalSurprise.action.propagate_action_net(
                    self.minimalSurprise.action.weight_actionNet_layer0[ind],
                    self.minimalSurprise.action.weight_actionNet_layer1[ind],
                    self.minimalSurprise.action.weight_actionNet_layer2[ind], inputA)

                self.minimalSurprise.action.current_action[i][timeStep] = action_output[0]

                # Propagate prediction network Call it after *Action*
                inputP[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep]

                # Feed input values into the prediction Network
                if self.manipulation != PRE:
                    self.minimalSurprise.prediction.propagate_prediction_network(
                        self.minimalSurprise.prediction.weight_predictionNet_layer0[ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer1[ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer2[ind], i, inputP,
                        self.p[i].inspire)

                # Check next action
                # 0 == move straight; 1 == turn
                current_action = self.minimalSurprise.action.current_action[i][timeStep]
                if current_action == STRAIGHT:
                    # movement only possible when cell in front is not occupied (sensor S0)
                    # move in heading direction (i.e. straight)
                    tmp_agent_next.x = sensor.adjustXPosition(self.p[i].coord.x + self.p[i].heading.x)
                    tmp_agent_next.y = sensor.adjustYPosition(self.p[i].coord.y + self.p[i].heading.y)

                    # Front sensor and check next grid is available
                    if sensors[S0] == 0 and grid[tmp_agent_next.x][tmp_agent_next.y] == 0 \
                            and self.heatmap[tmp_agent_next.x][tmp_agent_next.y] != HIGH:
                        # check if next cell is already occupied by agent
                        # next agent positions as far as updated (otherwise positions already checked via sensors)
                        # Agent move
                        grid[self.p[i].coord.x][self.p[i].coord.y] = 0  # Set current cell available
                        grid[tmp_agent_next.x][tmp_agent_next.y] = 1  # Set next cell unavailable

                        self.p_next[i].coord.x = tmp_agent_next.x
                        self.p_next[i].coord.y = tmp_agent_next.y
                        self.p_next[i].heading.x = self.p[i].heading.x
                        self.p_next[i].heading.y = self.p[i].heading.y
                        self.p_next[i].inspire = self.p[i].inspire

                    else:
                        # print("Can't move")
                        self.p_next[i].coord.x = self.p[i].coord.x
                        self.p_next[i].coord.y = self.p[i].coord.y

                        angle = np.arctan2(self.p[i].heading.y, self.p[i].heading.x)
                        self.p_next[i].heading.x = round(np.cos(angle + action_output[1] * (PI / 2)))
                        self.p_next[i].heading.y = round(np.sin(angle + action_output[1] * (PI / 2)))

                        self.p_next[i].inspire = self.p[i].inspire

                # agent Turn --> Update heading
                elif self.minimalSurprise.action.current_action[i][timeStep] == TURN:
                    # Keep same coordinate
                    self.p_next[i].coord.x = self.p[i].coord.x
                    self.p_next[i].coord.y = self.p[i].coord.y
                    self.p_next[i].inspire = self.p[i].inspire

                    # calculate current orientation
                    angle = np.arctan2(self.p[i].heading.y, self.p[i].heading.x)
                    self.p_next[i].heading.x = round(np.cos(angle + action_output[1] * (PI / 2)))
                    self.p_next[i].heading.y = round(np.sin(angle + action_output[1] * (PI / 2)))

                storage_p[timeStep] = self.p_next.copy()
                sensor_fit = float(diff) / float(SENSORS - 1)
                total_fit.append(sensor_fit)

            # End Agent Iterations
            timeStep += 1

            # Update positions
            temp = self.p_next
            self.p = temp
            self.cata = temp.copy()
            self.p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]  # Reset p_next
        # End while loop

        sum_num = sum(total_fit)
        # Length = 100, MAX_TIME = 10, noagents = 10
        fit_return = sum_num / float(len(total_fit) * MAX_TIME * noagents)

        if self.tmp_fit <= fit_return:
            max_p = self.p.copy()
            self.tmp_fit = fit_return
            f = open(moving_file, "w")
            f.write(f"Gen: {gen} \n")
            f.write(f"Pop: {ind} \n")

            for t in range(AIM_NUM):
                f.write(f"Target {t}: {self.target[t][0]}, {self.target[t][1]} \n")

            f.write(f"Fitness: {self.tmp_fit} \n")
            f.write(f"\n")

            for t in range(maxTime + 1):
                for i in range(NUM_AGENTS):
                    if t == 0:
                        f.write(f"{t}, {i}: {int(p_initial[i].coord.x)}, {int(p_initial[i].coord.y)}, "
                                f"{int(p_initial[i].heading.x)}, {int(p_initial[i].heading.y)}, "
                                f"{int(p_initial[i].inspire)}\n")
                    else:
                        f.write(f"{t}, {i}: {int(storage_p[t - 1][i].coord.x)}, {int(storage_p[t - 1][i].coord.y)}, "
                                f"{int(storage_p[t - 1][i].heading.x)}, {int(storage_p[t - 1][i].heading.y)}, "
                                f"{int(p_initial[i].inspire)}\n")
                f.write(f"\n")

        return fit_return, max_p

    """
        Usage: Do evolution
    """

    def evolution(self):
        # Store fitness for all population
        fitness = np.zeros(POP_SIZE, dtype=float)

        # store agent movement
        agent_maxfit = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        tmp_agent_maxfit_final = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        temp_p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        tmp_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

        # initialise weights of neural nets in range [-0.5, 0.5]
        # Define layers and their attributes to avoid repetitive code
        layers_info = {
            0: {
                'action': {
                    'weight': 'weight_actionNet_layer0',
                    'connections': ACT_CONNECTIONS
                },
                'prediction': {
                    'weight': 'weight_predictionNet_layer0',
                    'connections': PRE_CONNECTIONS
                }
            },
            1: {
                'action': {
                    'weight': 'weight_actionNet_layer1',
                    'connections': INPUTA * HIDDENA
                },
                'prediction': {
                    'weight': 'weight_predictionNet_layer1',
                    'connections': (INPUTP + 1) * HIDDENP
                }
            },
            2: {
                'action': {
                    'weight': 'weight_actionNet_layer2',
                    'connections': HIDDENA * OUTPUTA
                },
                'prediction': {
                    'weight': 'weight_predictionNet_layer2',
                    'connections': HIDDENP * OUTPUTP
                }
            }
        }

        for ind in range(POP_SIZE):
            for j in range(LAYERS):
                if j in layers_info:
                    for net_type, net_info in layers_info[j].items():
                        net_weights = getattr(self.minimalSurprise, net_type)
                        weights = getattr(net_weights, net_info['weight'])

                        for k in range(net_info['connections']):
                            weights[ind][k] = random.uniform(-0.5, 0.5)

        p_initial = self.location_init()

        asd_prev = 0

        # evolutionary runs
        for gen in range(MAX_GENS):
            max_gen = 0.0
            avg = 0.0
            maxID = -1
            fitness_count = 0

            # for i in range(NUM_AGENTS):
            temp_p = deepcopy(p_initial)

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
                    # for i in range(NUM_AGENTS):  # store agent end positions
                    tmp_agent_maxfit_final = deepcopy(max_p)

                # Average fitness of generation
                avg += fitness[ind]

                # store individual with maximum fitness
                if fitness[ind] > max_gen:
                    max_gen = fitness[ind]
                    maxID = ind
                    # store initial and final agent positions
                    agent_maxfit = deepcopy(tmp_agent_maxfit_final)
                    tmp_initial = agent_maxfit.copy()
                else:
                    fitness_count += 1

                # End Fitness store
                if fitness_count == 300:
                    self.minimalSurprise.catastrophe(ind)
                    fitness_count = 0

                print("Score for generation: ", gen + 1, "Pop: ", ind, "Score: ", max_gen)

            # End population loop

            p_initial = tmp_initial.copy()  # Update Initial pos
            self.cata = tmp_initial.copy()
            self.tmp_fit = 0

            if gen == 0:
                asd_prev = self.minimalSurprise.dynamic_mutate(maxID, fitness, gen, asd_prev)
            else:
                asd_prev = self.minimalSurprise.dynamic_mutate(maxID, fitness, gen, asd_prev)
            # self.minimalSurprise.select_mutate(maxID, fitness)

            "Do target moving HERE"
            # if gen % 2 == 0:
            #     if gen >= 2:
            #         for t in range(AIM_NUM):
            #             tmp_x, tmp_y = self.tmp_target[t][0], self.tmp_target[t][1]
            #
            #             self.target[t][0], self.target[t][1] = tmp_x, tmp_y
            #
            #             self.tmp_target[t][0], self.tmp_target[t][1] = 0, 0
            #
            #     self.update_heatmap(tmp_initial)
            # else:
            #     self.target_disappear()

            # End evolution runs loop
            # if gen > 0 and gen % 2 == 0:
            #     self.update_heatmap(tmp_initial)
            self.update_heatmap(tmp_initial)

        self.execute(gen, ind, p_initial, MAX_TIME, 1, NUM_AGENTS)

    def target_init(self):
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]
        for i in range(AIM_NUM):
            block = True
            while block:
                temp_x = random.randint(5, 6)
                temp_y = random.randint(5, 6)

                # Check if this spot is already occupied
                if grid[temp_x][temp_y] == 1:
                    continue

                valid_position = True

                # Check the distance with all previous targets
                for j in range(i):
                    target_dis = np.linalg.norm(
                        np.array([self.target[j][0], self.target[j][1]]) - np.array([temp_x, temp_y]))
                    if target_dis <= 3:
                        valid_position = False
                        break

                # If the position is valid, break the loop and finalize the position of this target
                if valid_position:
                    block = False
                    self.target[i][0] = temp_x
                    self.target[i][1] = temp_y
                    grid[temp_x][temp_y] = 1  # set grid cell occupied

    def location_init(self):
        self.target_init()
        p_initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        # Set entire heatmap to LOW
        self.heatmap = [[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        # Set target locations and nearby heat values
        for t in range(AIM_NUM):
            target_x, target_y = self.target[t]
            self.heatmap[target_x][target_y] = AIM

            for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    x, y = target_x + dx, target_y + dy

                    # Ensure (x, y) is within bounds
                    if 0 <= x < self.sizeX and 0 <= y < self.sizeY:
                        dist = max(np.abs(dx), np.abs(dy))

                        if dist < 1 + TARGET_SIZE:  # High intensity
                            self.heatmap[x][y] = max(self.heatmap[x][y], HIGH)
                            grid[x][y] = 1  # Mark the position as occupied
                        elif dist < 3 + TARGET_SIZE:  # Medium intensity
                            self.heatmap[x][y] = max(self.heatmap[x][y], MEDIUM)

        self.update_heat_intensity(self.target)

        # generate agent positions
        for i in range(NUM_AGENTS):
            # Find an unoccupied and low intensity location
            while True:
                x = random.randint(0, self.sizeX - 1)
                y = random.randint(0, self.sizeY - 1)

                if grid[x][y] == 0 and self.heatmap[x][y] == MEDIUM:
                    p_initial[i].coord.x, p_initial[i].coord.y = x, y
                    p_initial[i].inspire = MEDIUM
                    grid[x][y] = 1  # Mark the position as occupied

                    # Set agent heading randomly
                    if random.random() < 0.5:  # West & East
                        p_initial[i].heading.x = random.choice([-1, 1])
                        p_initial[i].heading.y = 0
                    else:  # North & South
                        p_initial[i].heading.x = 0
                        p_initial[i].heading.y = random.choice([-1, 1])

                    break  # Exit the while loop when a suitable position is found

        return p_initial

    def target_move(self, agent, x, y):
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        for i in range(NUM_AGENTS):
            grid[agent[i].coord.x][agent[i].coord.y] = 1  # set grid cell occupied

        block = True

        while block:
            randInd = random.randint(0, 1)
            direction = [-1, 1]

            if random.random() < 0.5:  # West & East
                tmp_X = x + direction[randInd]
                tmp_Y = y
            else:  # North & South
                tmp_X = x
                tmp_Y = y + direction[randInd]

            if 0 <= tmp_X < self.sizeX and 0 <= tmp_Y < self.sizeY and grid[int(tmp_X)][int(tmp_Y)] == 0:
                block = False  # Move the target and Avoid walls

        return tmp_X, tmp_Y

    def update_heat_intensity(self, targets):
        # Initialize heat_intensity to zero
        self.heat_intensity = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        for tgt in targets:
            tx, ty = int(tgt[0]), int(tgt[1])

            for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):  # Assuming a radius of 4 as in the previous code
                for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    x, y = tx + dx, ty + dy

                    # Ensure (x,y) is within bounds
                    if 0 <= x < self.sizeX and 0 <= y < self.sizeY:
                        dist = max(np.abs(dx), np.abs(dy))
                        upper = 0.0

                        if dist < 1 + TARGET_SIZE:
                            intensity_level = HIGH
                        elif dist < 3 + TARGET_SIZE:
                            intensity_level = MEDIUM
                        else:
                            intensity_level = LOW

                        alpha_rate = Heat_alpha[intensity_level]

                        if intensity_level == HIGH:
                            upper = np.abs(self.sizeX - 2) * np.abs(self.sizeY - 2)
                        elif intensity_level == MEDIUM:
                            upper = np.abs(self.sizeX - 3) * np.abs(self.sizeY - 3)
                        elif intensity_level == LOW:
                            upper = np.abs(self.sizeX - 5) * np.abs(self.sizeY - 5)

                        lower = self.sizeX * self.sizeY
                        new_intensity = alpha_rate * upper / lower

                        # Set the intensity to the maximum of the current value and the new computed value
                        self.heat_intensity[x][y] = max(self.heat_intensity[x][y], new_intensity)

    def update_heatmap(self, agent):
        # Clear entire heatmap to LOW
        self.heatmap = [[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)]

        for t in range(AIM_NUM):
            self.target[t][0], self.target[t][1] = self.target_move(agent, self.target[t][0], self.target[t][1])

            target_x = self.target[t][0]
            target_y = self.target[t][1]

            # Set target surroundings heatmap values
            for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    # Ensure we're within bounds
                    if 0 <= target_x + dx < self.sizeX and 0 <= target_y + dy < self.sizeY:
                        # Calculate distance
                        dist = max(np.abs(dx), np.abs(dy))

                        # Update heatmap value based on distance
                        if dist < 1 + TARGET_SIZE:
                            self.heatmap[target_x + dx][target_y + dy] = HIGH
                        elif dist < 3 + TARGET_SIZE:
                            self.heatmap[target_x + dx][target_y + dy] = MEDIUM

            # Handle the target itself
            self.heatmap[target_x][target_y] = AIM

        self.update_heat_intensity(self.target)
