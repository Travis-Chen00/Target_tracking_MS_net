import numpy as np
from parameters.STD14 import *


class Prediction:
    def __init__(self, input, hidden, output, manipulation):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.manipulation = manipulation

        self.weight_predictionNet_layer0 = np.zeros((POP_SIZE, PRE_CONNECTIONS), dtype=float)
        self.weight_predictionNet_layer1 = np.zeros((POP_SIZE, (self.input + 1) * self.hidden), dtype=float)
        self.weight_predictionNet_layer2 = np.zeros((POP_SIZE, self.output * self.hidden), dtype=float)

        # hidden states prediction network
        self.hiddenBefore_predictionNet = np.zeros((NUM_AGENTS, hidden), dtype=float)

        # predictions of agents FOR Proximity sensors
        self.predictions = np.zeros((NUM_AGENTS, SENSORS - 1), dtype=float)

        self.heat_next = np.zeros(NUM_AGENTS, dtype=int)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def activation(self, net):
        for i in range(len(net)):
            net[i] = self.sigmoid(net[i])  # sigmoid function
            if i < len(net):
                if net[i] > 0.5:
                    net[i] = 1
                else:
                    net[i] = 0
        return net

    def propagate_prediction_network(self, layer0, layer1, layer2, agent, input, zone):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                agent: index of agent
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        # Reshape all lists into matrix
        input = np.array(input)[:, np.newaxis]                              # (15, 1)
        input_2_hidden_1 = np.array(layer0)[np.newaxis, :]
        input_2_hidden_2 = np.array(layer1).reshape(self.input + 1, self.hidden)
        hidden_2_output = np.array(layer2).reshape(self.hidden, self.output)

        # calculate activation of input neurons
        # Activate the input neurons
        i_layer = self.sigmoid(input * input_2_hidden_1[:, ::2] - input_2_hidden_1[:, 1::2])
        i_layer = np.squeeze(i_layer)
        i_layer = i_layer[:, 0]

        # Hidden layer
        hidden_net = np.dot(i_layer.T, input_2_hidden_2[:self.input, :])  # Input to hidden connections
        hidden_net += self.hiddenBefore_predictionNet[agent] * input_2_hidden_2[self.input, :]  # Recurrent connections

        # Update hiddenBefore_predictionNet with activated values
        self.hiddenBefore_predictionNet[agent] = self.sigmoid(hidden_net)
        hidden_net = self.sigmoid(hidden_net)  # Activate values to be used in the next layer

        # Output layer
        tmp = np.dot(hidden_net.T, hidden_2_output.reshape(self.hidden, self.output))
        net = self.sigmoid(tmp)

        for j in range(SENSORS - 1):
            # print("Before: ", zone, self.predictions[agent][j])
            if zone == MEDIUM:
                result = Heat_int[MEDIUM] * Heat_alpha[MEDIUM]
                self.predictions[agent][j] = net[j] if -0.01 <= float(result) - net[j] <= 0.01 else float(result)
            elif zone == LOW:
                result = Heat_int[LOW] * Heat_alpha[LOW]
                self.predictions[agent][j] = net[j] if -0.01 <= float(result) - net[j] <= 0.01 else float(result)
            elif zone == HIGH:
                result = Heat_int[MEDIUM] * Heat_alpha[MEDIUM]
                self.predictions[agent][j] = net[j] if -0.01 <= float(result) - net[j] <= 0.01 else float(result)
