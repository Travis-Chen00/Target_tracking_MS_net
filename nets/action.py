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

        self.weight_actionNet_layer0 = np.zeros((POP_SIZE, CONNECTIONS), dtype=float)
        self.weight_actionNet_layer1 = np.zeros((POP_SIZE, self.input * self.hidden), dtype=float)
        self.weight_actionNet_layer2 = np.zeros((POP_SIZE, self.hidden * self.output), dtype=float)

        self.current_action = np.zeros((NUM_AGENTS, MAX_TIME), dtype=int)

    def propagate_action_net(self, layer_0, layer_1, layer_2, input):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        action_output = np.zeros(self.output, dtype=int)

        # Reshape all lists into matrix
        input = np.array(input)[:, np.newaxis]                              # (15, 1)
        input_2_hidden_1 = np.array(layer_0)[np.newaxis, :]        # (1, 112)
        input_2_hidden_2 = np.array(layer_1).reshape((self.input, self.hidden))      # (15, 8)
        hidden_2_output = np.array(layer_2).reshape((self.hidden, self.output))        # (8, 2)

        i_layer = np.tanh(input * input_2_hidden_1[:, ::2] - input_2_hidden_1[:, 1::2])
        i_layer = np.squeeze(i_layer)
        i_layer = i_layer[:, 0]

        hidden_net = np.tanh(np.dot(i_layer.T, input_2_hidden_2.reshape(self.input, self.hidden)))

        net = np.tanh(np.dot(hidden_net.T, hidden_2_output.reshape(self.hidden, self.output)))

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

