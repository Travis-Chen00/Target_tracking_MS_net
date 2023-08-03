from parameters.STD14 import *
import random


def movement(x, y):
    randInd = random.randint(-1, 1)
    if random.random() < 0.5:  # West & East
        x += randInd
    else:  # North & South
        y += randInd

    return x, y


if __name__ == '__main__':
    x, y = movement(7, 7)

    print(x, y)