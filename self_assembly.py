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

        self.heatmap = [[[0] * int(self.sizeY) for _ in range(int(self.sizeX))] for _ in range(REPETITION)]
        self.heat_intensity = [[[0] * int(self.sizeY) for _ in range(int(self.sizeX))] for _ in range(REPETITION)]

        # Set the coordinates of target / embedded into swarms
        self.target = [[[0, 0] for _ in range(AIM_NUM)] for _ in range(REPETITION)]
        self.tmp_target = [[0, 0] for _ in range(AIM_NUM)]

        # Evolution count
        self.count = 0

        # Index to determine the catastrophe
        self.cata = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(REPETITION)]

        self.tmp_fit = [0.0] * REPETITION

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

    def execute(self, gen, ind, p_initial, maxTime, log, noagents, rep):
        timeStep = 0  # Current timestep

        total_fit = []
        diff = 0.0

        tmp_agent_next = Pos(0, 0)

        inputA = np.zeros(INPUTA)
        inputP = np.zeros(INPUTP)

        max_p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

        moving = f"Agents_{NUM_AGENTS}_TimeStep_{maxTime}_Gen_{gen}_Rep_{rep}"
        # file names
        directory = "moving/9_14/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        moving_file = os.path.join(directory, moving)

        self.p = p_initial[rep].copy()

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
                grid[self.target[rep][t][0]][self.target[rep][t][1]] = 1  # Set target location

            # Iterate all agents
            # Execute agents one by one in each timeStep
            for i in range(noagents):
                # Determine current sensor values (S of t)
                sensor = Sensors(self.sizeX, self.sizeY)

                if SENSOR_MODEL == STDL:
                    # print(self.p[i].coord.x)
                    sensors = sensor.sensorModelSTDL(i, grid, self.heatmap[rep], self.p)

                for j in range(SENSORS):
                    # set sensor values as ANN input values
                    inputA[j] = sensors[j]
                    inputP[j] = sensors[j]

                    # Calculate the difference
                    if j < 5:
                        diff += 1 - np.abs(sensors[j] - self.minimalSurprise.prediction.predictions[rep][i][j])

                # Propagate action network
                if timeStep <= 0:
                    inputA[SENSORS] = STRAIGHT
                else:  # Last time action
                    inputA[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep - 1]

                action_output = self.minimalSurprise.action.propagate_action_net(
                    self.minimalSurprise.action.weight_actionNet_layer0[rep][ind],
                    self.minimalSurprise.action.weight_actionNet_layer1[rep][ind],
                    self.minimalSurprise.action.weight_actionNet_layer2[rep][ind], inputA, rep)

                self.minimalSurprise.action.current_action[i][timeStep] = action_output[0]

                # Propagate prediction network Call it after *Action*
                inputP[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep]

                # Feed input values into the prediction Network
                if self.manipulation != PRE:
                    self.minimalSurprise.prediction.propagate_prediction_network(
                        self.minimalSurprise.prediction.weight_predictionNet_layer0[rep][ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer1[rep][ind],
                        self.minimalSurprise.prediction.weight_predictionNet_layer2[rep][ind], i, inputP,
                        self.p[i].inspire, rep)

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
                            and self.heatmap[rep][tmp_agent_next.x][tmp_agent_next.y] != HIGH:
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

        if self.tmp_fit[rep] <= fit_return:
            max_p = self.p.copy()
            self.tmp_fit[rep] = fit_return
            f = open(moving_file, "w")
            f.write(f"Gen: {gen} \n")
            f.write(f"Pop: {ind} \n")

            for t in range(AIM_NUM):
                f.write(f"Target {t}: {self.target[rep][t][0]}, {self.target[rep][t][1]} \n")

            f.write(f"Fitness: {self.tmp_fit[rep]} \n")
            f.write(f"\n")

            for t in range(maxTime + 1):
                for i in range(NUM_AGENTS):
                    if t == 0:
                        f.write(f"{t}, {i}: {int(p_initial[rep][i].coord.x)}, {int(p_initial[rep][i].coord.y)}, "
                                f"{int(p_initial[rep][i].heading.x)}, {int(p_initial[rep][i].heading.y)}, "
                                f"{int(p_initial[rep][i].inspire)}\n")
                    else:
                        f.write(f"{t}, {i}: {int(storage_p[t - 1][i].coord.x)}, {int(storage_p[t - 1][i].coord.y)}, "
                                f"{int(storage_p[t - 1][i].heading.x)}, {int(storage_p[t - 1][i].heading.y)}, "
                                f"{int(p_initial[rep][i].inspire)}\n")
                f.write(f"\n")

        return fit_return, max_p

    """
        Usage: Do evolution
    """

    def evolution(self):
        # Store fitness for all population
        fitness = np.zeros((REPETITION, POP_SIZE), dtype=float)
        tmp_fitness = [0.0 for _ in range(REPETITION)]

        # store agent movement
        tmp_agent_maxfit_final = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(REPETITION)]
        tmp_initial = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(REPETITION)]
        max_p = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in
                                  range(REPETITION)]
        agent_maxfit = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in
                                  range(REPETITION)]
        best_position_for_current_rep = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in
                                  range(REPETITION)]

        # initialise weights of neural nets in range [-0.5, 0.5]
        # Define layers and their attributes to avoid repetitive code
        # layers_info = {
        #     0: {
        #         'action': {
        #             'weight': 'weight_actionNet_layer0',
        #             'connections': ACT_CONNECTIONS * REPETITION
        #         },
        #         'prediction': {
        #             'weight': 'weight_predictionNet_layer0',
        #             'connections': PRE_CONNECTIONS * REPETITION
        #         }
        #     },
        #     1: {
        #         'action': {
        #             'weight': 'weight_actionNet_layer1',
        #             'connections': INPUTA * HIDDENA * REPETITION
        #         },
        #         'prediction': {
        #             'weight': 'weight_predictionNet_layer1',
        #             'connections': (INPUTP + 1) * HIDDENP * REPETITION
        #         }
        #     },
        #     2: {
        #         'action': {
        #             'weight': 'weight_actionNet_layer2',
        #             'connections': HIDDENA * OUTPUTA * REPETITION
        #         },
        #         'prediction': {
        #             'weight': 'weight_predictionNet_layer2',
        #             'connections': HIDDENP * OUTPUTP * REPETITION
        #         }
        #     }
        # }

        # for rep in range(REPETITION):
        #     for ind in range(POP_SIZE):
        #         for j in range(LAYERS):
        #             if j in layers_info:
        #                 for net_type, net_info in layers_info[j].items():
        #                     net_weights = getattr(self.minimalSurprise, net_type)
        #                     weights = getattr(net_weights, net_info['weight'])
        #
        #                     if len(weights) <= rep or len(weights[rep]) <= ind or len(weights[rep][ind]) < net_info[
        #                         'connections']:
        #                         print(f"Error at: rep={rep}, ind={ind}, j={j}, net_type={net_type}")
        #                         continue
        #
        #                     for k in range(net_info['connections']):
        #                         weights[rep][ind][k] = random.uniform(-0.5, 0.5)
        for rep in range(REPETITION):
            for ind in range(POP_SIZE):
                for j in range(LAYERS):
                    if j == 0:
                        for k in range(PRE_CONNECTIONS):
                            self.minimalSurprise.prediction.weight_predictionNet_layer0[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)
                        for k in range(ACT_CONNECTIONS):
                            self.minimalSurprise.action.weight_actionNet_layer0[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)
                    if j == 1:
                        for k in range((INPUTP + 1) * HIDDENP):
                            self.minimalSurprise.prediction.weight_predictionNet_layer1[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)
                        for k in range(INPUTA * HIDDENA):
                            self.minimalSurprise.action.weight_actionNet_layer1[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)
                    if j == 2:
                        for k in range(OUTPUTP * HIDDENP):
                            self.minimalSurprise.prediction.weight_predictionNet_layer2[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)
                        for k in range(HIDDENA * OUTPUTA):
                            self.minimalSurprise.action.weight_actionNet_layer2[rep][ind][k] \
                                = random.uniform(-0.5, 0.5)

        p_initial = self.location_init()

        asd_prev = 0

        # evolutionary runs
        for gen in range(MAX_GENS):
            max_gen = [0.0] * REPETITION
            avg = [0.0] * REPETITION
            maxID = [0] * REPETITION
            fitness_count = 0

            temp_p = deepcopy(p_initial)
            best_positions_for_reps = [None] * REPETITION  # 为每个rep存储最佳位置

            for rep in range(REPETITION):
                best_position_for_current_rep = None

                for ind in range(POP_SIZE):
                    # fitness evaluation - initialisation based on case
                    fitness[rep][ind] = 0.0
                    store = False

                    tmp_fitness[rep], max_p[rep] = self.execute(gen, ind, temp_p, MAX_TIME, 0, NUM_AGENTS, rep)
                    print("Fitness for population:", ind + 1, "Score:", tmp_fitness[rep])

                    # max fitness of Repetitions kept
                    if FIT_EVAL == MAX:
                        if tmp_fitness[rep] > fitness[rep][ind]:
                            fitness[rep][ind] = tmp_fitness[rep]
                            store = True

                    # store best fitness + id of repetition
                    if store:
                        tmp_agent_maxfit_final[rep] = max_p[rep].copy()

                    # Average fitness of generation
                    avg[rep] += fitness[rep][ind]

                    # store individual with maximum fitness
                    if fitness[rep][ind] > max_gen[rep]:
                        max_gen[rep] = fitness[rep][ind]
                        maxID[rep] = ind
                        agent_maxfit[rep] = tmp_agent_maxfit_final[rep].copy()
                        best_positions_for_reps[rep] = tmp_agent_maxfit_final[rep].copy()
                    else:
                        fitness_count += 1

                    # End Fitness store
                    if fitness_count == 300:
                        self.minimalSurprise.catastrophe(ind, rep)
                        fitness_count = 0

                    print("Score for generation: ", gen + 1, " Repetition: ", rep, "Pop: ", ind, "Score: ", max_gen[rep])

                # 使用当前重复的最佳位置更新p_initial
                if best_positions_for_reps[rep] is not None:
                    p_initial = best_positions_for_reps.copy()

            # End population loop

            self.cata = p_initial.copy()
            self.tmp_fit = [0.0] * REPETITION

            if gen == 0:
                asd_prev = self.minimalSurprise.dynamic_mutate(maxID, fitness, gen, asd_prev)
            else:
                asd_prev = self.minimalSurprise.dynamic_mutate(maxID, fitness, gen, asd_prev)

            "Movement option for the target"
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

            # if gen > 0 and gen % 2 == 0:
            #     self.update_heatmap(tmp_initial)
            self.update_heatmap(p_initial)

    def target_init(self, rep):
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
                        np.array([self.target[rep][j][0], self.target[rep][j][1]]) - np.array([temp_x, temp_y]))
                    if target_dis <= 3:
                        valid_position = False
                        break

                # If the position is valid, break the loop and finalize the position of this target
                if valid_position:
                    block = False
                    self.target[rep][i][0] = temp_x
                    self.target[rep][i][1] = temp_y
                    grid[temp_x][temp_y] = 1  # set grid cell occupied

    def location_init(self):
        for rep in range(REPETITION):
            self.target_init(rep)
        p_initial = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(REPETITION)]
        grid = [[[0] * int(self.sizeY) for _ in range(int(self.sizeX))] for _ in range(REPETITION)]

        # Set entire heatmap to LOW
        self.heatmap = [[[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)] for _ in range(REPETITION)]

        # Set target locations and nearby heat values
        for rep in range(REPETITION):
            for t in range(AIM_NUM):
                target_x, target_y = self.target[rep][t]
                self.heatmap[rep][target_x][target_y] = AIM

                for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                        x, y = target_x + dx, target_y + dy

                        # Ensure (x, y) is within bounds
                        if 0 <= x < self.sizeX and 0 <= y < self.sizeY:
                            dist = max(np.abs(dx), np.abs(dy))

                            if dist < 1 + TARGET_SIZE:  # High intensity
                                self.heatmap[rep][x][y] = max(self.heatmap[rep][x][y], HIGH)
                                grid[rep][x][y] = 1  # Mark the position as occupied
                            elif dist < 3 + TARGET_SIZE:  # Medium intensity
                                self.heatmap[rep][x][y] = max(self.heatmap[rep][x][y], MEDIUM)

        self.update_heat_intensity(self.target)

        # generate agent positions
        for rep in range(REPETITION):
            for i in range(NUM_AGENTS):
                # Find an unoccupied and low intensity location
                while True:
                    x = random.randint(0, self.sizeX - 1)
                    y = random.randint(0, self.sizeY - 1)

                    if grid[rep][x][y] == 0 and self.heatmap[rep][x][y] == LOW:
                        p_initial[rep][i].coord.x, p_initial[rep][i].coord.y = x, y
                        p_initial[rep][i].inspire = MEDIUM
                        grid[rep][x][y] = 1  # Mark the position as occupied

                        # Set agent heading randomly
                        if random.random() < 0.5:  # West & East
                            p_initial[rep][i].heading.x = random.choice([-1, 1])
                            p_initial[rep][i].heading.y = 0
                        else:  # North & South
                            p_initial[rep][i].heading.x = 0
                            p_initial[rep][i].heading.y = random.choice([-1, 1])

                        break  # Exit the while loop when a suitable position is found

        return p_initial

    """Target move option:
        1. Randomly jump in and out.
        2. Continuely moving 
    """
    # def target_move(self, agent, x, y):
    #     grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]
    #
    #     for i in range(NUM_AGENTS):
    #         grid[int(agent[i].coord.x)][int(agent[i].coord.y)] = 1  # set grid cell occupied
    #
    #     block = True
    #
    #     while block:
    #         tmp_X = random.randint(0, self.sizeX - 1)
    #         tmp_Y = random.randint(0, self.sizeY - 1)
    #
    #         if 0 <= tmp_X < self.sizeX and 0 <= tmp_Y < self.sizeY and grid[int(tmp_X)][int(tmp_Y)] == 0:
    #             block = False  # Move the target and Avoid walls
    #
    #     return tmp_X, tmp_Y

    def target_move(self, agent, x, y):
        grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

        # for i in range(NUM_AGENTS):
        #     grid[agent[i].coord.x][agent[i].coord.y] = 1  # set grid cell occupied

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
        self.heat_intensity = [[[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)] for _ in range(REPETITION)]

        for rep in range(REPETITION):
            for tgt in targets[rep]:
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
                                upper = np.abs(self.sizeX - 1) * np.abs(self.sizeY - 1)
                            elif intensity_level == MEDIUM:
                                upper = np.abs(self.sizeX - 2) * np.abs(self.sizeY - 2)
                            elif intensity_level == LOW:
                                upper = np.abs(self.sizeX - 4) * np.abs(self.sizeY - 4)

                            lower = self.sizeX * self.sizeY
                            new_intensity = alpha_rate * upper / lower

                            # Set the intensity to the maximum of the current value and the new computed value
                            self.heat_intensity[rep][x][y] = max(self.heat_intensity[rep][x][y], new_intensity)

    # def is_on_square_border(self, x, y, target_x, target_y, dist):
    #     return (
    #                    x == target_x - dist or x == target_x + dist or
    #                    y == target_y - dist or y == target_y + dist
    #            ) and (
    #                    target_x - dist <= x <= target_x + dist and
    #                    target_y - dist <= y <= target_y + dist
    #            )
    #
    # def update_heat_intensity(self, targets):
    #     # Initialize heat_intensity to zero
    #     self.heat_intensity = [[0 for _ in range(self.sizeY)] for _ in range(self.sizeX)]
    #
    #     for tgt in targets:
    #         tx, ty = int(tgt[0]), int(tgt[1])
    #
    #         for x in range(self.sizeX):
    #             for y in range(self.sizeY):
    #                 dist = max(abs(x - tx), abs(y - ty))
    #
    #                 # Define intensity based on distance
    #                 if self.heatmap[x][y] == HIGH:
    #                     intensity_level = HIGH
    #                 elif self.heatmap[x][y] == MEDIUM:
    #                     intensity_level = MEDIUM
    #                 else:
    #                     intensity_level = LOW
    #
    #                 alpha_rate = Heat_alpha[intensity_level]
    #                 upper = np.abs(self.sizeX - dist) * np.abs(self.sizeY - dist)
    #                 lower = self.sizeX * self.sizeY
    #                 new_intensity = alpha_rate * upper / lower
    #
    #                 # Set the intensity to the maximum of the current value and the new computed value
    #                 self.heat_intensity[x][y] = max(self.heat_intensity[x][y], new_intensity)

    def update_heatmap(self, agent):
        # Clear entire heatmap to LOW
        self.heatmap = [[[LOW for _ in range(self.sizeY)] for _ in range(self.sizeX)] for _ in range(REPETITION)]

        for rep in range(REPETITION):
            for t in range(AIM_NUM):
                self.target[rep][t][0], self.target[rep][t][1] = self.target_move(agent[rep], self.target[rep][t][0], self.target[rep][t][1])

                target_x = self.target[rep][t][0]
                target_y = self.target[rep][t][1]

                # Set target surroundings heatmap values
                for dx in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                    for dy in range(-3 - TARGET_SIZE, 4 + TARGET_SIZE):
                        # Ensure we're within bounds
                        if 0 <= target_x + dx < self.sizeX and 0 <= target_y + dy < self.sizeY:
                            # Calculate distance
                            dist = max(np.abs(dx), np.abs(dy))

                            # Update heatmap value based on distance
                            if dist < 1 + TARGET_SIZE:
                                self.heatmap[rep][target_x + dx][target_y + dy] = HIGH
                            elif dist < 3 + TARGET_SIZE:
                                self.heatmap[rep][target_x + dx][target_y + dy] = MEDIUM

                # Handle the target itself
                self.heatmap[rep][target_x][target_y] = AIM

            self.update_heat_intensity(self.target)
