import sys
import numpy as np
import random
from self_assembly import *
from minimal_surprise import *

FIT_FUN = None
MANIPULATION = None
EVOL = False
SIZE_X = 0.0
SIZE_Y = 0.0

if __name__ == "__main__":
    """
        Evolution or Re-run of genome
    """
    type_1 = NOTYPE

    # Positions of agents
    p = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]
    p_next = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS)]

    # if sys.argv[1] == "EVOL":
    #     print("EVOLUTION")
    #     EVOL = True
    #
    #     if len(sys.argv) != 7:
    #         print("Please specify 6 input values: [REPLAY/EVOL] "
    #               "[Grid size x direction] [Grid size y direction] "
    #               "[Fitness Function] [Manipulation] [Agent Type (Manipulation)]")
    # else:
    #     print("Please specify either REPLAY or EVOL as the first option.\n")
    #     exit(0)
    EVOL = True

    if EVOL:  # Train the network

        # Get grid size

        # SIZE_X = float(sys.argv[2])
        # SIZE_Y = float(sys.argv[3])
        SIZE_X = 20
        SIZE_Y = 20

        # manipulation_arg = sys.argv[5]
        manipulation_arg = "MAN"
        if manipulation_arg == "NONE":
            MANIPULATION = NONE
        elif manipulation_arg == "MAN":
            MANIPULATION = MAN
        elif manipulation_arg == "PRE":
            MANIPULATION = PRE
        else:
            print("No valid manipulation option specified.")
            exit(0)

        if MANIPULATION != NONE:
            # agent_type_arg = sys.argv[6]
            agent_type_arg = "LINE"
            if agent_type_arg == "LINE":
                type_1 = LINE
            elif agent_type_arg == "PAIR":
                type_1 = PAIR
            elif agent_type_arg == "DIAMOND":
                type_1 = DIAMOND
            elif agent_type_arg == "SQUARE":
                type_1 = SQUARE
            elif agent_type_arg == "AGGREGATION":
                type_1 = AGGREGATION
            elif agent_type_arg == "DISPERSION":
                type_1 = DISPERSION
            else:
                print("Unknown Agent Type.")
                exit(0)

        for i in range(NUM_AGENTS):
            p[i].type = type_1
            p_next[i].type = type_1

        self_assembly = SelfAssembly(p, p_next, MANIPULATION, SIZE_X, SIZE_Y)
        # print(self_assembly.sizeX, self_assembly.sizeY)
        self_assembly.evolution()
    # End EVOL loop
    # else:
    #     NUM_AGENTS_OLD = 10
    #     initial = [Agent(NOTYPE, Pos(0, 0), Pos(0, 0)) for _ in range(NUM_AGENTS_OLD)]
    #
    #     if sys.argv[5] == "PRED":
    #         print("Fitness function: prediction")
    #         FIT_FUN = PRED
    #     else:
    #         print("No valid fitness function specified")
    #         exit(0)
    #
    #     if sys.argv[6] == "NONE":
    #         MANIPULATION = NONE
    #     elif sys.argv[6] == "MAN":
    #         MANIPULATION = MAN
    #     elif sys.argv[6] == "PRE":
    #         MANIPULATION = PRE
    #     else:
    #         print("No valid manipulation option specified.")
    #         exit(0)
    #
    #     if MANIPULATION != NONE:
    #         # agent_type_arg = sys.argv[6]
    #         agent_type_arg = "LINE"
    #         if agent_type_arg == "LINE":
    #             type_1 = LINE
    #         elif agent_type_arg == "PAIR":
    #             type_1 = PAIR
    #         elif agent_type_arg == "DIAMOND":
    #             type_1 = DIAMOND
    #         elif agent_type_arg == "SQUARE":
    #             type_1 = SQUARE
    #         elif agent_type_arg == "AGGREGATION":
    #             type_1 = AGGREGATION
    #         elif agent_type_arg == "DISPERSION":
    #             type_1 = DISPERSION
    #         else:
    #             print("Unknown Agent Type.")
    #             exit(0)
    #
    #     print("LOADING INITIAL AGENT POSITIONS...")
    #     file = open(sys.argv[2], "r")
    #
    #     net = MinimalSurprise(INPUTA, INPUTP, HIDDENA, HIDDENP,
    #                           OUTPUTA, OUTPUTP, MANIPULATION, 15.0, 15.0)
    #
    #     if file:
    #         # skip first line
    #         file.readline()
    #         # get grid size
    #         line = file.readline()
    #         pt = line.split(": ")[1].split(", ")
    #         SIZE_X = int(pt[0])
    #         SIZE_Y = int(pt[1])
    #
    #         # compare agent number with the set one
    #         line = file.readline()
    #         pt = line.split(": ")[1].rstrip()
    #         if int(pt) != NUM_AGENTS_OLD:
    #             print("Please adjust NUM_AGENTS_OLD.")
    #             exit(0)
    #
    #         # get agent positions and headings
    #         for i in range(NUM_AGENTS_OLD):
    #             line = file.readline().strip()
    #             pt = line.split(": ")[1].split(", ")
    #             initial[i].coord.x = int(pt[0])
    #             initial[i].coord.y = int(pt[1])
    #             initial[i].heading.x = int(pt[2])
    #             initial[i].heading.y = int(pt[3])
    #
    #         print("COMPLETED.")
    #         file.close()
    #     else:
    #         print("PATH ERROR.")
    #         exit(0)
    #
    #     if MANIPULATION != PRE:
    #         print("LOADING PREDICTION GENOME...\n")
    #
    #         with open(sys.argv[3], "r") as file:
    #             i = 0
    #             for line in file:
    #                 i += 1
    #                 if i == 397:
    #                     break
    #
    #             # first line weights
    #             pt = line.split(": ")[1]
    #             weights = pt.split(": ")
    #             for i in range(CONNECTIONS):
    #                 net.prediction.weight_predictionNet[0][0][i] = float(weights[i])
    #
    #             # second line weights
    #             line = file.readline()
    #             pt = line.split(": ")[1]
    #             weights = pt.split(": ")
    #             for i in range(CONNECTIONS):
    #                 net.prediction.weight_predictionNet[0][1][i] = float(weights[i])
    #
    #             # third line weights
    #             line = file.readline()
    #             pt = line.split(": ")[1]
    #             weights = pt.split(": ")
    #             for i in range(CONNECTIONS):
    #                 net.prediction.weight_predictionNet[0][2][i] = float(weights[i])
    #
    #         print("COMPLETED.\n")
    #     else:
    #         print("PATH ERROR.")
    #         exit(0)
    #
    #     print("LOADING ACTION GENOME...\n")
    #     with open(sys.argv[4], "r") as file:
    #         i = 0
    #         for line in file:
    #             i += 1
    #             if i == 397:
    #                 break
    #
    #         # first line weights
    #         pt = line.split(": ")[1]
    #         weights = pt.split(": ")
    #         for i in range(CONNECTIONS):
    #             net.action.weight_actionNet[0][0][i] = float(weights[i])
    #
    #         # second line weights
    #         line = file.readline()
    #         pt = line.split(": ")[1]
    #         weights = pt.split(": ")
    #         for i in range(CONNECTIONS):
    #             net.action.weight_actionNet[0][1][i] = float(weights[i])
    #
    #         # third line weights
    #         line = file.readline()
    #         pt = line.split(": ")[1]
    #         weights = pt.split(": ")
    #         for i in range(CONNECTIONS):
    #             net.action.weight_actionNet[0][2][i] = float(weights[i])
    #
    #     print("COMPLETED.\n")
    #
    #     # Set agents to chosen type - set to NOTYPE if no manipulation
    #     for i in range(NUM_AGENTS):
    #         p[i].type = type_1
    #         p_next[i].type = type_1
    #
    #     self_assembly = SelfAssembly(p, p_next, MANIPULATION, 15.0, 15.0)
    #     # Write fitness values to file
    #     with open("replay_fitness", "a") as f:
    #         f.write(f"Fitness of re-run: {self_assembly.execute(0, 0, initial, MAX_TIME, 1, NUM_AGENTS_OLD)}\n")
    #         for val in net.prediction.pred_return:
    #             f.write(f"{val} ")
    #         f.write("\n")
    #
    #     COUNT = -1  # change count number to indicate self-repair run
    #
    #     REMOVE = int(sys.argv[8])
    #     DESTROY = int(sys.argv[9])
    #     new_agent_no = int(sys.argv[10])
    #     copy_agent_no = int(sys.argv[11])
    #     grid = [[0] * SIZE_Y for _ in range(SIZE_X)]
    #     x_min = 0
    #     x_max = 0
    #     y_min = 0
    #     y_max = 0
    #
    #     if DESTROY:  # define area in which agents will be removed
    #         x_min = int(sys.argv[12])
    #         x_max = int(sys.argv[13])
    #         y_min = int(sys.argv[14])
    #         y_max = int(sys.argv[15])
    #
    #         copy_agent_no = NUM_AGENTS_OLD  # only agents in defined area won't be copied
    #
    #     replay = []
    #
    #     if not DESTROY:
    #         # copy last position of agents
    #         # random starting position for copying agents
    #         start = random.randint(0, NUM_AGENTS_OLD - 1)
    #         for i in range(copy_agent_no):
    #             if start + i >= NUM_AGENTS_OLD:
    #                 start = -i
    #
    #             replay.append({
    #                 'coord': {
    #                     'x': p[start + i].coord.x,
    #                     'y': p[start + i].coord.y
    #                 },
    #                 'heading': {
    #                     'x': p[start + i].heading.x,
    #                     'y': p[start + i].heading.y
    #                 }
    #             })
    #
    #             grid[replay[i]['coord']['x']][replay[i]['coord']['y']] = 1
    #
    #         # initialise agent positions to random discrete x & y values
    #         # min = 0 (rand()%(max + 1 - min) + min)
    #         # no plus 1 as starting from 0 and size = SIZE_X/Y
    #         for i in range(copy_agent_no, new_agent_no):
    #             b = 1
    #
    #             while b:
    #                 b = 0
    #
    #                 coord_x = random.randint(0, SIZE_X - 1)
    #                 coord_y = random.randint(0, SIZE_Y - 1)
    #
    #                 if grid[coord_x][coord_y] == 1:
    #                     b = 1
    #                 else:
    #                     grid[coord_x][coord_y] = 1
    #
    #             # set agent heading values randomly (north, south, west, east possible)
    #             directions = [1, -1]
    #             randInd = random.randint(0, 1)
    #
    #             if random.random() < 0.5:
    #                 heading_x = directions[randInd]
    #                 heading_y = 0
    #             else:
    #                 heading_x = 0
    #                 heading_y = directions[randInd]
    #
    #             replay.append({
    #                 'coord': {
    #                     'x': coord_x,
    #                     'y': coord_y
    #                 },
    #                 'heading': {
    #                     'x': heading_x,
    #                     'y': heading_y
    #                 }
    #             })
    #     else:
    #         print("DESTROY.")
    #
    #         # Remove agents in a certain area
    #
    #         # set all grid values as 1 where agents should be removed to not position new agents there when adding agents randomly
    #         for i in range(x_min, x_max + 1):
    #             for j in range(y_min, y_max + 1):
    #                 grid[i][j] = 1
    #
    #         j = 0
    #
    #         # copy agents
    #         for i in range(copy_agent_no):
    #
    #             # copy only if agent not positioned within part to be removed
    #             if not (x_min <= p[i].coord.x <= x_max and y_min <= p[i].coord.y <= y_max):
    #                 replay.append({
    #                     'coord': {
    #                         'x': p[i].coord.x,
    #                         'y': p[i].coord.y
    #                     },
    #                     'heading': {
    #                         'x': p[i].heading.x,
    #                         'y': p[i].heading.y
    #                     }
    #                 })
    #
    #                 grid[replay[j]['coord']['x']][replay[j]['coord']['y']] = 1
    #
    #                 j += 1
    #
    #         if not REMOVE:  # if not remove - replace agents to new positions
    #             for j in range(j, new_agent_no):
    #                 b = 1
    #
    #                 while b:
    #                     b = 0
    #
    #                     coord_x = random.randint(0, SIZE_X - 1)
    #                     coord_y = random.randint(0, SIZE_Y - 1)
    #
    #                     if grid[coord_x][coord_y] == 1:
    #                         b = 1
    #                     else:
    #                         grid[coord_x][coord_y] = 1
    #
    #                 # set agent heading values randomly (north, south, west, east possible)
    #                 directions = [1, -1]
    #                 randInd = random.randint(0, 1)
    #
    #                 if random.random() < 0.5:
    #                     heading_x = directions[randInd]
    #                     heading_y = 0
    #                 else:
    #                     heading_x = 0
    #                     heading_y = directions[randInd]
    #
    #                 replay.append({
    #                     'coord': {
    #                         'x': coord_x,
    #                         'y': coord_y
    #                     },
    #                     'heading': {
    #                         'x': heading_x,
    #                         'y': heading_y
    #                     }
    #                 })
    #
    #         # adjust agent number
    #         new_agent_no = j
    #
    #     with open("replay_fitness", "a") as f:
    #         f.write(f"Number of agents: {new_agent_no}\n")
    #         if not DESTROY:
    #             f.write(f"Number of agents on same starting position: {copy_agent_no}\n")
    #
    #         f.write(f"Fitness of self-repair: {self_assembly.execute(0, 0, replay, MAX_TIME, 1, new_agent_no)}\n")
    #
    #         for val in net.prediction.pred_return:
    #             f.write(f"{val} ")
    #         f.write("\n")
    #
    #     p = []
    #     p_next = []

