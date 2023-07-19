import numpy as np
from parameters.STD14 import *


def activation(x):
    """
        Activation function for action network
        Usage:
            Input: x
            Output: tanh(x)
    """
    return 2 / (1 + np.exp(x * -2)) - 1


class Action:
    def __init__(self, input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.weight_actionNet = np.zeros((POP_SIZE, LAYERS, CONNECTIONS), dtype=float)
        self.newWeight_actionNet = np.zeros((POP_SIZE, LAYERS, CONNECTIONS), dtype=float)
        self.current_action = np.zeros((NUM_AGENTS, MAX_TIME), dtype=int)

    def set_weight(self, weight_action):
        # Set the weight to the action network
        self.weight_actionNet = weight_action

    def propagate_action_net(self, weight_action, input):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        hidden_net = np.zeros(self.hidden, dtype=float)
        i_layer = np.zeros(self.input, dtype=float)
        net = np.zeros(self.output, dtype=float)
        action_output = np.zeros(self.output, dtype=int)

        # calculate activation of input neurons
        # input layer to hidden layer
        # INPUTA == 15
        for i in range(self.input):
            i_layer[i] = activation(float(input[i]) * weight_action[0][2 * i]
                                         - weight_action[0][2 * i + 1])

        # hidden layer - 4 hidden neurons
        # HIDDENA == 8
        for i in range(self.hidden):
            hidden_net[i] = 0.0
            for j in range(self.input):
                hidden_net[i] += i_layer[j] * weight_action[1][self.input * i + j]
            # Activation value
            hidden_net[i] = activation(hidden_net[i])

        # calculate input of output layer
        # OUTPUTA == 2
        for i in range(self.output):
            net[i] = 0.0
            for j in range(self.hidden):
                net[i] += hidden_net[j] * weight_action[2][self.hidden * i + j]
            net[i] = activation(net[i])

        # map outputs to binary values
        # first output: action value
        # STRAIGHT == 0, TURN == 1
        if net[0] < 0.0:
            action_output[0] = STRAIGHT
        else:
            action_output[0] = TURN

        # second output: turn direction
        # UP == 1, DOWN == -1
        if net[1] < 0.0:
            action_output[1] = UP
        else:
            action_output[1] = DOWN

        return action_output

# if __name__ == '__main__':
#     action = Action(INPUTA, HIDDENA, OUTPUTA)
