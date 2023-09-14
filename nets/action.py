import numpy as np
from parameters.STD14 import *


class Action:
    def __init__(self, input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output

        self.weight_actionNet_layer0 = np.zeros((REPETITION, POP_SIZE, ACT_CONNECTIONS), dtype=float)
        self.weight_actionNet_layer1 = np.zeros((REPETITION, POP_SIZE, self.input * self.hidden), dtype=float)
        self.weight_actionNet_layer2 = np.zeros((REPETITION, POP_SIZE, self.hidden * self.output), dtype=float)

        self.current_action = np.zeros((NUM_AGENTS, MAX_TIME), dtype=int)

    def propagate_action_net(self, layer_0, layer_1, layer_2, input, rep):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        action_output = np.zeros(self.output, dtype=int)

        # Reshape all lists into matrix
        input = np.array(input)[:, np.newaxis]
        input_2_hidden_1 = np.array(layer_0)[np.newaxis, :]
        input_2_hidden_2 = np.array(layer_1).reshape((self.input, self.hidden))
        hidden_2_output = np.array(layer_2).reshape((self.hidden, self.output))

        # print(input.shape, input_2_hidden_1.shape, input_2_hidden_2.shape, hidden_2_output.shape)
        i_layer = np.tanh(input * input_2_hidden_1[:, ::2] - input_2_hidden_1[:, 1::2])
        i_layer = np.squeeze(i_layer)
        i_layer = i_layer[:, 0]

        hidden_net = np.tanh(np.dot(i_layer.T, input_2_hidden_2.reshape(self.input, self.hidden)))

        net = np.tanh(np.dot(hidden_net.T, hidden_2_output.reshape(self.hidden, self.output)))

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

