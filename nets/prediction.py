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

        # average predictions during run
        self.pred_return = [0.0] * SENSORS

        # predictions of agents
        self.predictions = np.zeros((NUM_AGENTS, SENSORS), dtype=int)

    def prediction_output(self, value):
        """
            Output of prediction network:
                Values of all sensor for next time
            Return 0/1
        """
        return 0 if value < 0 else 1

    def activation(self, net):
        for i in range(len(net)):
            net[i] = 1 / (1 + np.exp(-net[i]))  # sigmoid function
            if i < len(net):
                if net[i] > 0.5:
                    net[i] = 1
                else:
                    net[i] = 0
        return net

    def propagate_prediction_network(self, layer0, layer1, layer2, agent, input, p):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                agent: index of agent
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        # Reshape all lists into matrix
        # print("Input:", input)
        input = np.array(input)[:, np.newaxis]                              # (15, 1)
        input_2_hidden_1 = np.array(layer0)[np.newaxis, :]
        input_2_hidden_2 = np.array(layer1).reshape(self.input + 1, self.hidden)
        hidden_2_output = np.array(layer2).reshape(self.hidden, self.output)

        # calculate activation of input neurons
        # Activate the input neurons
        i_layer = np.tanh(input * input_2_hidden_1[:, ::2] - input_2_hidden_1[:, 1::2])
        i_layer = np.squeeze(i_layer)
        i_layer = i_layer[:, 0]

        # Hidden layer
        hidden_net = np.dot(i_layer.T, input_2_hidden_2[:self.input, :])  # Input to hidden connections
        hidden_net += self.hiddenBefore_predictionNet[agent] * input_2_hidden_2[self.input, :]  # Recurrent connections
        # Update hiddenBefore_predictionNet with activated values
        self.hiddenBefore_predictionNet[agent] = np.tanh(hidden_net)
        hidden_net = np.tanh(hidden_net)  # Activate values to be used in the next layer

        # Output layer
        net = self.activation(np.dot(hidden_net.T, hidden_2_output.reshape(self.hidden, self.output)))

        # print(net)

        for i in range(self.output):
            self.predictions[agent][i] = net[i]

