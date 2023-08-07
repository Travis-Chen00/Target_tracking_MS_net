import numpy as np
import random
from parameters.STD14 import *
from nets.action import Action
from nets.prediction import Prediction


class MinimalSurprise:
    def __init__(self, amountInAction, amountInPrediction, amountHiddenAction,
                 amountHiddenPrediction, amountOutAction, amountOutPrediction, manipulation, size_x, size_y):
        self.amountInAction = amountInAction
        self.amountInPrediction = amountInPrediction
        self.amountHiddenAction = amountHiddenAction
        self.amountHiddenPrediction = amountHiddenPrediction
        self.amountOutAction = amountOutAction
        self.amountOutPrediction = amountOutPrediction
        self.manipulation = manipulation
        self.sizeX = size_x
        self.sizeY = size_y

        # Action & Prediction network
        self.action = Action(self.amountInAction, self.amountHiddenAction, self.amountOutAction)
        self.prediction = Prediction(self.amountInPrediction, self.amountHiddenPrediction,
                                     self.amountOutPrediction, self.manipulation)

    def catastrophe(self, ind):
        print("Doing catastrophe")
        # Mutate network
        for k in range(PRE_CONNECTIONS):
            if random.random() < CATASTROPHE:  # prediction network
                self.prediction.weight_predictionNet_layer0[ind][k] \
                    += 0.8 * random.random() - 0.4  # 0.8 * [0, 1] - 0.4 --> [-0.4, 0.4]
        for k in range(ACT_CONNECTIONS):
            if random.random() < CATASTROPHE:  # action network
                self.action.weight_actionNet_layer0[ind][k] \
                    += 0.8 * random.random() - 0.4
        for k in range((self.amountInPrediction + 1) * self.amountHiddenPrediction):
            if random.random() < CATASTROPHE:
                self.prediction.weight_predictionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
        for k in range(self.amountInAction * self.amountHiddenAction):
            if random.random() < CATASTROPHE:
                self.action.weight_actionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
        for k in range(self.amountHiddenAction * self.amountOutAction):
            if random.random() < CATASTROPHE:
                self.action.weight_actionNet_layer2[ind][k] += 0.8 * random.random() - 0.4
        for k in range(self.amountOutPrediction * self.amountHiddenPrediction):
            if random.random() < CATASTROPHE:
                self.prediction.weight_predictionNet_layer2[ind][k] += 0.8 * random.random() - 0.4

    def dynamic_mutate(self, maxID, fitness, gen, ASD_prev):
        print("Doing dynamic mutate")

        avg_fitness = np.sum(fitness) / POP_SIZE  # The average fitness value for generation T with all individuals.

        tmp, sum_tmp = 0.0, 0.0
        max_fit = fitness[maxID]
        for i in range(len(fitness)):
            tmp = np.power(fitness[i] - avg_fitness, 2)
            sum_tmp += tmp
        ASD_now = np.sqrt(sum_tmp) / POP_SIZE

        print("Average_Fitness for Generation ", gen, ": ", ASD_now,
              "Average_Fitness for Generation ", gen - 1, ": ", ASD_prev)

        if gen == 0:
            d_mutation = MUTATION
        else:
            d_mutation = MUTATION * (1 + (max_fit - ASD_prev) / (max_fit + ASD_prev))

        print("Mutation rate for Generation ", gen, ": ", MUTATION,
              "d_mutation for Generation ", gen - 1, ": ", d_mutation)

        # Select and mutate
        for ind in range(POP_SIZE):
            # individuals out of the maximum
            # Keep that weight for next generation
            if ind == maxID:
                for k in range(PRE_CONNECTIONS):
                    self.prediction.weight_predictionNet_layer0[ind][k] \
                        = self.prediction.weight_predictionNet_layer0[maxID][k]
                for k in range(ACT_CONNECTIONS):
                    self.action.weight_actionNet_layer0[ind][k] \
                        = self.action.weight_actionNet_layer0[maxID][k]
                for k in range((self.amountInPrediction + 1) * self.amountHiddenPrediction):
                    self.prediction.weight_predictionNet_layer1[ind][k] \
                        = self.prediction.weight_predictionNet_layer1[maxID][k]
                for k in range(self.amountInAction * self.amountHiddenAction):
                    self.action.weight_actionNet_layer1[ind][k] \
                        = self.action.weight_actionNet_layer1[maxID][k]
                for k in range(self.amountHiddenAction * self.amountOutAction):
                    self.action.weight_actionNet_layer2[ind][k] \
                        = self.action.weight_actionNet_layer2[maxID][k]
                for k in range(self.amountOutPrediction * self.amountHiddenPrediction):
                    self.prediction.weight_predictionNet_layer2[ind][k] \
                        = self.prediction.weight_predictionNet_layer2[maxID][k]

            # Mutation and selection
            else:
                # Mutate network
                for k in range(PRE_CONNECTIONS):
                    if random.random() < d_mutation:  # prediction network
                        self.prediction.weight_predictionNet_layer0[ind][k] \
                            += 0.8 * random.random() - 0.4  # 0.8 * [0, 1] - 0.4 --> [-0.4, 0.4]
                for k in range(ACT_CONNECTIONS):
                    if random.random() < d_mutation:  # action network
                        self.action.weight_actionNet_layer0[ind][k] \
                            += 0.8 * random.random() - 0.4
                for k in range((self.amountInPrediction + 1) * self.amountHiddenPrediction):
                    if random.random() < d_mutation:
                        self.prediction.weight_predictionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountInAction * self.amountHiddenAction):
                    if random.random() < d_mutation:
                        self.action.weight_actionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountHiddenAction * self.amountOutAction):
                    if random.random() < d_mutation:
                        self.action.weight_actionNet_layer2[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountOutPrediction * self.amountHiddenPrediction):
                    if random.random() < d_mutation:
                        self.prediction.weight_predictionNet_layer2[ind][k] += 0.8 * random.random() - 0.4

        return ASD_now

    def select_mutate(self, maxID, fitness):
        sum1 = 0.0
        sum2 = 0.0
        pr = np.zeros(POP_SIZE, dtype=float)

        # Total fitness value
        for i in range(POP_SIZE):
            sum1 += fitness[i]

        # relative fitness over individuals
        for i in range(POP_SIZE):
            sum2 += fitness[i]
            pr[i] = sum2 / sum1

        # Select and mutate
        for ind in range(POP_SIZE):
            # individuals out of the maximum
            # Keep that weight for next generation
            if ind == maxID:
                for k in range(PRE_CONNECTIONS):
                    self.prediction.weight_predictionNet_layer0[ind][k] \
                        = self.prediction.weight_predictionNet_layer0[maxID][k]
                for k in range(ACT_CONNECTIONS):
                    self.action.weight_actionNet_layer0[ind][k] \
                        = self.action.weight_actionNet_layer0[maxID][k]
                for k in range((self.amountInPrediction + 1) * self.amountHiddenPrediction):
                    self.prediction.weight_predictionNet_layer1[ind][k] \
                        = self.prediction.weight_predictionNet_layer1[maxID][k]
                for k in range(self.amountInAction * self.amountHiddenAction):
                    self.action.weight_actionNet_layer1[ind][k] \
                        = self.action.weight_actionNet_layer1[maxID][k]
                for k in range(self.amountHiddenAction * self.amountOutAction):
                    self.action.weight_actionNet_layer2[ind][k] \
                        = self.action.weight_actionNet_layer2[maxID][k]
                for k in range(self.amountOutPrediction * self.amountHiddenPrediction):
                    self.prediction.weight_predictionNet_layer2[ind][k] \
                        = self.prediction.weight_predictionNet_layer2[maxID][k]

            # Mutation and selection
            else:
                # Mutate network
                for k in range(PRE_CONNECTIONS):
                    if random.random() < MUTATION:  # prediction network
                        self.prediction.weight_predictionNet_layer0[ind][k] \
                            += 0.8 * random.random() - 0.4  # 0.8 * [0, 1] - 0.4 --> [-0.4, 0.4]
                for k in range(ACT_CONNECTIONS):
                    if random.random() < MUTATION:  # action network
                        self.action.weight_actionNet_layer0[ind][k] \
                            += 0.8 * random.random() - 0.4
                for k in range((self.amountInPrediction + 1) * self.amountHiddenPrediction):
                    if random.random() < MUTATION:
                        self.prediction.weight_predictionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountInAction * self.amountHiddenAction):
                    if random.random() < MUTATION:
                        self.action.weight_actionNet_layer1[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountHiddenAction * self.amountOutAction):
                    if random.random() < MUTATION:
                        self.action.weight_actionNet_layer2[ind][k] += 0.8 * random.random() - 0.4
                for k in range(self.amountOutPrediction * self.amountHiddenPrediction):
                    if random.random() < MUTATION:
                        self.prediction.weight_predictionNet_layer2[ind][k] += 0.8 * random.random() - 0.4

