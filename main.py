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
    EVOL = True

    if EVOL:  # Train the network
        SIZE_X = 15
        SIZE_Y = 15

        manipulation_arg = "MAN"

        self_assembly = SelfAssembly(p, p_next, MANIPULATION, SIZE_X, SIZE_Y)
        self_assembly.evolution()
    # End EVOL loop

