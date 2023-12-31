# experimental setup
MUTATION = 0.3          # 0.1 - mutation rate
CATASTROPHE = 0.4       # Catastrophe rate

POP_SIZE = 5000  # population
REPETITION = 10

MAX_TIME = 10   # time per run
MAX_GENS = 100  # maximum generations

NUM_AGENTS = 10
AIM_NUM = 1

# movement
STRAIGHT = 0
TURN = 1
UP = 1
DOWN = -1

# sensors
S0 = 0  # forward
S1 = 1  # forward right
S2 = 2  # forward left
S3 = 3  # 2 cells forward
S4 = 4  # 2 cells forward right
S5 = 5  # 2 cells forward left
S6 = 6  # right of agent

PI = 3.14159265

# Temperature setting
AIM = 3     # Target has the highest priority
HIGH = 2    # Around the target is the high temperature
MEDIUM = 1  # Outer is medium temp
LOW = 0     # Others are normal temp

# fitness evaluation
MIN = 0
MAX = 1
AVG = 2
FIT_EVAL = MAX

# define fitness function
PRED = 0  # prediction

# define sensor model
STDS = 0  # standard: 6 sensors - in heading direction forward / right / left (1 & 2 blocks ahead)
STDL = 2  # standard large: 14 sensors - 6 in heading direction (like STD), 2 next to agent, 6 behind agent
STDSL = 3  # Moore Neighborhood

# define manipulation models
NONE = 0  # none
PRE = 1  # prediction
MAN = 2  # manipulation

# agent types
NOTYPE = -1
LINE = 1
AGGREGATION = 2
DISPERSION = 3
DIAMOND = 4
SQUARE = 5
PAIR = 6
