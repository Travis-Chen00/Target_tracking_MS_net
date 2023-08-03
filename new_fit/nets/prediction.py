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

        self.weight_temperatureNet_layer0 = np.zeros((POP_SIZE, TEMP_CONNECTIONS), dtype=float)
        self.weight_temperatureNet_layer1 = np.zeros((POP_SIZE, (INPUTTEMP + 1) * HIDDENTEMP), dtype=float)
        self.weight_temperatureNet_layer2 = np.zeros((POP_SIZE, OUTPUTTEMP * HIDDENTEMP), dtype=float)

        # hidden states prediction network
        self.hiddenBefore_predictionNet = np.zeros((NUM_AGENTS, hidden), dtype=float)

        # hidden states temperature network
        self.hiddenBefore_temperatureNet = np.zeros((NUM_AGENTS, HIDDENTEMP), dtype=float)

        # predictions of agents
        self.predictions = np.zeros((NUM_AGENTS, SENSORS), dtype=int)

        self.heat_next = np.zeros(NUM_AGENTS, dtype=int)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def prediction_output(self, value):
        """
            Output of prediction network:
                Values of all sensor for next time
            Return 0/1
        """
        return 0 if value < 0 else 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def propagate_prediction_network(self, layer0, layer1, layer2, agent, input):
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
        net = self.sigmoid(np.dot(hidden_net.T, hidden_2_output.reshape(self.hidden, self.output)))

        # print(net)

        for i in range(self.output):
            self.predictions[agent][i] = net[i]

    def propagate_heat_network(self, layer0, layer1, layer2, agent, heat):
        heat = np.array(heat)[:, np.newaxis]
        input_layer = np.array(layer0)[np.newaxis, :]
        hidden_layer = np.array(layer1).reshape((INPUTTEMP + 1), HIDDENTEMP)
        output_layer = np.array(layer2).reshape(HIDDENTEMP, OUTPUTTEMP)

        # print(input.shape, input_2_hidden_1.shape, input_2_hidden_2.shape, hidden_2_output.shape)
        i_layer = np.tanh(heat * input_layer)
        i_layer = np.squeeze(i_layer)
        i_layer = i_layer[:, 0]

        # Hidden layer
        hidden_net = np.tanh(np.dot(i_layer.T, hidden_layer[:self.input, :]))  # Input to hidden connections
        hidden_net += self.hiddenBefore_predictionNet[agent] * hidden_layer[self.input, :]  # Recurrent connections
        # Update hiddenBefore_predictionNet with activated values
        self.hiddenBefore_predictionNet[agent] = np.tanh(hidden_net)
        hidden_net = np.tanh(hidden_net)  # Activate values to be used in the next layer
        # print(hidden_net)

        net = self.softmax(np.dot(hidden_net.T, output_layer.reshape(HIDDENTEMP, OUTPUTTEMP)))
        # print(net)
        temp = np.argmax(net, axis=0)

        # print(temp)
        self.heat_next[agent] = max(temp, MEDIUM)

        # def propagate_heat_network(self, layer0, layer1, layer2, layer3, agent, heat):
        #     heat = np.array(heat)[:, np.newaxis]
        #     input_layer = np.array(layer0)[np.newaxis, :]
        #     hidden_layer1 = np.array(layer1).reshape(INPUTTEMP, HIDDENTEMP_1)
        #     hidden_layer2 = np.array(layer2).reshape(HIDDENTEMP_1, HIDDENTEMP_2)
        #     output_layer = np.array(layer3).reshape(HIDDENTEMP_2, OUTPUTTEMP)
        #
        #     # print(input.shape, input_2_hidden_1.shape, input_2_hidden_2.shape, hidden_2_output.shape)
        #     i_layer = self.sigmoid(heat * input_layer)
        #     i_layer = np.squeeze(i_layer)
        #     i_layer = i_layer[:, 0]
        #
        #     hidden_net = self.sigmoid(np.dot(i_layer.T, hidden_layer1.reshape(INPUTTEMP, HIDDENTEMP_1)))
        #     hidden_net = self.sigmoid(np.dot(hidden_net.T, hidden_layer2.reshape(HIDDENTEMP_1, HIDDENTEMP_2)))
        #     # print(hidden_net)
        #
        #     net = self.softmax(np.dot(hidden_net.T, output_layer.reshape(HIDDENTEMP_2, OUTPUTTEMP)))
        #     # print(net)
        #     self.heat_next[agent] = net