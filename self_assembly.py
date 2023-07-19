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

        # Evolution count
        self.count = 0

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
        timeStep = 0    # Current timestep
        fit = 0
        predReturn = [0] * SENSORS
        tmp_agent_next = Pos(0, 0)

        inputA = np.zeros(INPUTA)
        inputP = np.zeros(INPUTP)

        trajectory_file = f"agent_trajectory"
        if log == 1:
            f = open(trajectory_file, "a")
            f.write(f"Gen: {gen}\n")
            f.write(f"Grid: {self.sizeX}, {self.sizeY}\n")
            f.write(f"Agents: {noagents}\n")
            f.close()

        # Initialise agents
        for i in range(noagents):
            self.p[i].coord.x = p_initial[i].coord.x
            self.p[i].coord.y = p_initial[i].coord.y
            self.p[i].heading.x = p_initial[i].heading.x
            self.p[i].heading.y = p_initial[i].heading.y

            self.p_next[i].coord.x = p_initial[i].coord.x
            self.p_next[i].coord.y = p_initial[i].coord.y
            self.p_next[i].heading.x = p_initial[i].heading.x
            self.p_next[i].heading.y = p_initial[i].heading.y

            if self.manipulation is not None:
                if self.p[i].type == LINE:
                    if SENSOR_MODEL == STDSL:
                        self.minimalSurprise.prediction.predictions[i][S0] = 1
                        self.minimalSurprise.prediction.predictions[i][S5] = 1
                    else:   # Forward & Backward will have agents
                        self.minimalSurprise.prediction.predictions[i][S0] = 1
                        self.minimalSurprise.prediction.predictions[i][S3] = 1

                        if SENSOR_MODEL == STDL:    # 14 sensors
                            self.minimalSurprise.prediction.predictions[i][S8] = 1
                            self.minimalSurprise.prediction.predictions[i][S11] = 1
                elif self.p[i].type == SQUARE:
                    if SENSOR_MODEL == STDL:
                        self.minimalSurprise.prediction.predictions[i][S3] = 1
                        self.minimalSurprise.prediction.predictions[i][S11] = 1

                elif self.p[i].type == DIAMOND:
                    if SENSOR_MODEL == STDL:
                        self.minimalSurprise.prediction.predictions[i][S1] = 1
                        self.minimalSurprise.prediction.predictions[i][S2] = 1
                        self.minimalSurprise.prediction.predictions[i][S3] = 1
                        self.minimalSurprise.prediction.predictions[i][S9] = 1
                        self.minimalSurprise.prediction.predictions[i][S10] = 1
                        self.minimalSurprise.prediction.predictions[i][S11] = 1

                elif self.p[i].type == PAIR:
                    self.minimalSurprise.prediction.predictions[i][S0] = 1

                elif self.p[i].type == AGGREGATION:
                    self.minimalSurprise.prediction.predictions[i] = [1] * SENSORS

        while timeStep < maxTime:

            # determine occupied grid cells (0 - unoccupied, 1 - occupied)
            # Locate all agents into the grid
            # Location with agent equals to 1
            grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]
            for i in range(noagents):
                grid[int(self.p[i].coord.x)][int(self.p[i].coord.y)] = 1

            # Iterate all agents
            # Execute agents one by one in each timeStep
            for i in range(noagents):
                # store agent trajectory
                if log == 1:
                    f = open(trajectory_file, "a")
                    f.write(f"{timeStep}: {self.p[i].coord.x}, {self.p[i].coord.y}, "
                            f"{self.p[i].heading.x}, {self.p[i].heading.y}\n")
                    f.close()

                # Determine current sensor values (S of t)
                sensor = Sensors(self.sizeX, self.sizeY)

                if SENSOR_MODEL == STDL:
                    sensors = sensor.sensorModelSTDL(i, grid, self.p)
                elif SENSOR_MODEL == STDS:
                    sensors = sensor.sensorModelSTD(i, grid, self.p)
                elif SENSOR_MODEL == STDSL:
                    sensors = sensor.sensorModelSTDSL(i, grid, self.p)

                print("Agent ", i, " Sensors: ", sensors)
                print("Agent ", i, " Prediction: ", self.minimalSurprise.prediction.predictions[i])
                # Set sensor values to both networks
                for j in range(SENSORS):
                    # set sensor values as ANN input values
                    inputA[j] = sensors[j]
                    inputP[j] = sensors[j]

                    # count correct predictions

                    if sensors[j] == self.minimalSurprise.prediction.predictions[i][j]:
                        fit += 1
                    # Count correct shape

                    # set prediction counters
                    predReturn[j] += self.minimalSurprise.prediction.predictions[i][j]
                # End Sensor loops

                """
                    Propagate action network 
                    Input: current sensor values + last action 
                    Output: next action --> 0 or 1
                """
                if timeStep <= 0:
                    inputA[SENSORS] = STRAIGHT
                else:  # Last time action
                    inputA[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep - 1]

                action_output = self.minimalSurprise.action.propagate_action_net(
                    self.minimalSurprise.action.weight_actionNet[ind], inputA)
                print("Action output for agent", i, ": ", action_output)
                self.minimalSurprise.action.current_action[i][timeStep] = action_output[0]

                """
                     Propagate prediction network Call it after *Action*
                     Input: current sensor values + next action [Returned by ANN] 
                     Output: Prediction of sensor value (per agent)
                 """
                # Action from the *Action* network
                inputP[SENSORS] = self.minimalSurprise.action.current_action[i][timeStep]
                # Feed input values into the prediction Network
                if self.manipulation != PRE:
                    self.minimalSurprise.prediction.propagate_prediction_network(
                        self.minimalSurprise.prediction.weight_predictionNet[ind], i, inputP)

                # Update the values
                self.p_next[i] = self.p[i]

                # Check next action
                # 0 == move straight; 1 == turn
                if self.minimalSurprise.action.current_action[i][timeStep] == STRAIGHT:
                    occupied = 0

                    # movement only possible when cell in front is not occupied (sensor S0)
                    if sensors[S0] == 0:
                        # move in heading direction (i.e. straight)
                        tmp_agent_next.x = sensor.adjustXPosition(self.p[i].coord.x + self.p[i].heading.x)
                        tmp_agent_next.y = sensor.adjustYPosition(self.p[i].coord.y + self.p[i].heading.y)

                        # check if next cell is already occupied by agent
                        # next agent positions as far as updated (otherwise positions already checked via sensors)
                        for k in range(i):
                            if self.p_next[k].coord.x == tmp_agent_next.x \
                                    and self.p_next[k].coord.y == tmp_agent_next.y:
                                occupied = 1
                                break

                        # Agent move
                        if not occupied:
                            self.p_next[i].coord.x = tmp_agent_next.x
                            self.p_next[i].coord.y = tmp_agent_next.y

                # agent Turn --> Update heading
                elif self.minimalSurprise.action.current_action[i][timeStep] == TURN:
                    # calculate current orientation
                    angle = np.arctan2(self.p[i].heading.y, self.p[i].heading.x)
                    self.p_next[i].heading.x = round(np.cos(angle + action_output[1] * (PI / 2)))
                    self.p_next[i].heading.y = round(np.sin(angle + action_output[1] * (PI / 2)))
                # End action determination
            # End Agent Iterations
            # random_location(p_initial, self.sizeX, self.sizeY)

            timeStep += 1
            # random_location(p_initial[k], self.sizeX, self.sizeY)
            # Update positions
            temp = self.p
            self.p = self.p_next
            self.p_next = temp
        # End while loop

        # prediction counter
        # Pred_return = 1 / T * N
        for i in range(SENSORS):
            self.minimalSurprise.prediction.pred_return[i] = float(predReturn[i]) / (maxTime * noagents)

        if log == 1:
            f = open(trajectory_file, "a")
            for i in range(noagents):
                f.write(f"{maxTime}: {self.p[i].coord.x}, {self.p[i].coord.y}, "
                        f"{self.p[i].heading.x}, {self.p[i].heading.y}\n")
            f.close()

        # F = 1 / T * N * R (1 - |S - P|)
        fit_return = float(fit) / float(noagents * maxTime * SENSORS)
        return fit_return   # Return fitness score

    """
        Usage: Do evolution
    """
    def evolution(self):
        print("Evolution count: ", self.count)
        p_initial = [[Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)] for _ in range(REPETITIONS)]

        # Store fitness for all population
        fitness = np.zeros(POP_SIZE, dtype=float)

        # action values of stored run
        actionValues = [[0] * MAX_TIME for _ in range(NUM_AGENTS)]

        # store agent movement
        agent_maxfit = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        agent_maxfit_beginning = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        tmp_agent_maxfit_final = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
        tmp_action = [[0] * MAX_TIME for _ in range(NUM_AGENTS)]

        # file names
        file = f"{self.count}_0_{NUM_AGENTS}"
        fit_file = "fitness" + file
        predGen_file = "prediction_genomes" + file
        actGen_file = "action_genomes" + file
        actVal_file = "actionValues" + file
        agent_file = "agents" + file

        # initialise weights of neural nets in range [-0.5, 0.5]
        # Shape (50, 3, 224)
        for ind in range(POP_SIZE):
            for j in range(LAYERS):
                for k in range(CONNECTIONS):
                    self.minimalSurprise.action.weight_actionNet[ind][j][k] = random.uniform(-0.5, 0.5)
                    self.minimalSurprise.prediction.weight_predictionNet[ind][j][k] = random.uniform(-0.5, 0.5)


        # print("Action weight:", self.minimalSurprise.action.weight_actionNet)
        # print("Prediction weight:", self.minimalSurprise.prediction.weight_predictionNet)

        # evolutionary runs
        # For one generation:
        # 50 Agents * 10 times repetitions ==> 500 individuals
        # Total: 50 * 10 * 100 ==> 50000 generations
        for gen in range(MAX_GENS):
            # max. fitness of per generation
            # Average fitness of per generation
            # Index of best individual
            max = 0.0
            avg = 0.0
            maxID = - 1

            # initialisation of starting positions
            # (all genomes have same set of starting positions)
            for k in range(REPETITIONS):
                # Reset the grid
                grid = [[0] * int(self.sizeY) for _ in range(int(self.sizeX))]

                # generate agent positions
                # In each repeat, all agent will be initialized
                for i in range(NUM_AGENTS):
                    # initialisation of starting positions
                    block = True

                    # Find an unoccupied location
                    while block:
                        # Randomise a position for each agent
                        p_initial[k][i].coord.x = random.randint(0, self.sizeX - 1)
                        p_initial[k][i].coord.y = random.randint(0, self.sizeY - 1)

                        # print("Agent ", i, ":", p_initial[k][i].coord.x, p_initial[k][i].coord.y)

                        # print("No.", i, "Locate:", p_initial[k][i].coord.x, p_initial[k][i].coord.y)

                        if grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] == 0:  # not occupied
                            block = False
                            grid[p_initial[k][i].coord.x][p_initial[k][i].coord.y] = 1  # set grid cell occupied

                    # Set agent heading values randomly (north, south, west, east)
                    directions = [1, -1]
                    randInd = random.randint(0, 1)
                    if random.random() < 0.5:   # West & East
                        p_initial[k][i].heading.x = directions[randInd]
                        p_initial[k][i].heading.y = 0
                    else:                       # North & South
                        p_initial[k][i].heading.x = 0
                        p_initial[k][i].heading.y = directions[randInd]

                # random_location(p_initial[k], self.sizeX, self.sizeY)
            # End Location Initialisation

            # population level (iterate through individuals)
            # POP_SIZE = 50
            # Each generation have 50 population, 100 agents
            for ind in range(POP_SIZE):

                # fitness evaluation - initialisation based on case
                if FIT_EVAL == MIN:  # MIN - initialise to value higher than max
                    fitness[ind] = SENSORS + 1.0
                    tmp_fitness = SENSORS + 1.0
                else:  # MAX, AVG - initialise to zero
                    fitness[ind] = 0.0
                    tmp_fitness = 0.0

                # reset prediction storage
                pred = [0.0] * SENSORS

                for rep in range(REPETITIONS):
                    store = False

                    tmp_fitness = self.execute(gen, ind, p_initial[rep], MAX_TIME, 0, NUM_AGENTS)
                    print("Fitness for population:", ind + 1, "rep:", rep + 1, "Score:", tmp_fitness)
                    # Min fitness of repetitions
                    if FIT_EVAL == MIN:
                        if tmp_fitness < fitness[ind]:
                            fitness[ind] = tmp_fitness
                            pred = self.minimalSurprise.prediction.pred_return.copy()
                            store = True

                    # max fitness of Repetitions kept
                    elif FIT_EVAL == "MAX":
                        if tmp_fitness > fitness[ind]:
                            fitness[ind] = tmp_fitness
                            pred = self.minimalSurprise.prediction.pred_return.copy()
                            store = True

                    # average fitness of Repetitions
                    elif FIT_EVAL == "AVG":
                        fitness[ind] += tmp_fitness / REPETITIONS
                        pred = [pred[s] + self.minimalSurprise.prediction.pred_return[s] /
                                REPETITIONS for s in range(SENSORS)]

                        if rep == REPETITIONS - 1:  # store data of last repetition
                            store = True

                    # store best fitness + id of repetition
                    if store:
                        max_rep = rep  # Store the index of best repetition

                        for i in range(NUM_AGENTS):  # store agent end positions
                            tmp_agent_maxfit_final[i].coord.x = self.p[i].coord.x
                            tmp_agent_maxfit_final[i].coord.y = self.p[i].coord.y
                            tmp_agent_maxfit_final[i].heading.x = self.p[i].heading.x
                            tmp_agent_maxfit_final[i].heading.y = self.p[i].heading.y
                            tmp_agent_maxfit_final[i].type = self.p[i].type

                        for i in range(NUM_AGENTS):  # store action values of best try of repetition
                            for j in range(MAX_TIME):   # MAX_TIME = POP_SIZE * REPETITION
                                tmp_action[i][j] = self.minimalSurprise.action.current_action[i][j]
                # End repetitions loop

                # Average fitness of generation
                avg += fitness[ind]
                print("Average for generation: ", gen + 1, "Score: ", avg)

                # store individual with maximum fitness
                if fitness[ind] > max:
                    max = fitness[ind]
                    maxID = ind

                    # store agent predictions
                    agentPrediction = pred.copy()

                    # store initial and final agent positions
                    for i in range(NUM_AGENTS):
                        agent_maxfit[i].coord.x = tmp_agent_maxfit_final[i].coord.x
                        agent_maxfit[i].coord.y = tmp_agent_maxfit_final[i].coord.y
                        agent_maxfit[i].heading.x = tmp_agent_maxfit_final[i].heading.x
                        agent_maxfit[i].heading.y = tmp_agent_maxfit_final[i].heading.y
                        agent_maxfit[i].type = tmp_agent_maxfit_final[i].type

                        agent_maxfit_beginning[i].coord.x = p_initial[max_rep][i].coord.x
                        agent_maxfit_beginning[i].coord.y = p_initial[max_rep][i].coord.y
                        agent_maxfit_beginning[i].heading.x = p_initial[max_rep][i].heading.x
                        agent_maxfit_beginning[i].heading.y = p_initial[max_rep][i].heading.y

                    # store action values of best run in generation
                    for j in range(NUM_AGENTS):
                        for k in range(MAX_TIME):
                            actionValues[j][k] = tmp_action[j][k]
                # End Fitness store
                print("Max for population: ", ind + 1, "Score: ", max)
            # End population loop

            print(f"#{gen} {max} ({maxID})")

            with open(fit_file, "a") as f:
                f.write(f"{self.sizeX} {self.sizeY} {gen} {max} {avg / POP_SIZE} ({maxID}) ")
                f.write(" ".join(str(val) for val in agentPrediction))
                f.write("\n")

            with open(agent_file, "a") as f:
                f.write(f"Gen: {gen}\n")
                f.write(f"Grid: {self.sizeX}, {self.sizeY}\n")
                f.write(f"Fitness: {max}\n")
                for i in range(NUM_AGENTS):
                    f.write(f"{agent_maxfit[i].coord.x}, {agent_maxfit[i].coord.y}, ")
                    f.write(f"{agent_maxfit_beginning[i].coord.x}, {agent_maxfit_beginning[i].coord.y}, ")
                    f.write(f"{agent_maxfit[i].heading.x}, {agent_maxfit[i].heading.y}, ")
                    f.write(f"{agent_maxfit_beginning[i].heading.x}, {agent_maxfit_beginning[i].heading.y}, ")
                    f.write(f"{agent_maxfit[i].type}\n")
                f.write("\n")

            with open(actVal_file, "a") as f:
                f.write(f"Gen: {gen}\n")
                f.write(f"Grid: {self.sizeX}, {self.sizeY}\n")
                f.write(f"Fitness: {max}\n")
                for i in range(NUM_AGENTS):
                    f.write(f"Agent: {i}\n")
                    f.write("[")
                    f.write(", ".join(str(actionValues[i][j]) for j in range(MAX_TIME)))
                    f.write("]\n")
                f.write("\n")

            with open(actGen_file, "a") as f:
                for i in range(LAYERS):
                    for j in range(CONNECTIONS):
                        f.write(f"{self.minimalSurprise.action.weight_actionNet[maxID][i][j]} ")
                    f.write("\n")
                f.write("\n")

            with open(predGen_file, "a") as f:
                for i in range(LAYERS):
                    for j in range(CONNECTIONS):
                        f.write(f"{self.minimalSurprise.prediction.weight_predictionNet[maxID][i][j]} ")
                    f.write("\n")
                f.write("\n")

            for j in range(LAYERS):
                for k in range(CONNECTIONS):
                    print("Max:", self.minimalSurprise.prediction.weight_predictionNet[maxID][j][k])

            # Do selection & mutation per generation
            self.minimalSurprise.select_mutate(maxID, fitness)
            # End evolution runs loop
            self.count += 1

        self.execute(gen, ind, p_initial[rep], MAX_TIME, 1, NUM_AGENTS)