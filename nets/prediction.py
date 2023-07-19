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


class Prediction:
    def __init__(self, input, hidden, output, manipulation):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.manipulation = manipulation
        self.weight_predictionNet = np.zeros((POP_SIZE, LAYERS, CONNECTIONS), dtype=float)
        self.newWeight_predictionNet = np.zeros((POP_SIZE, LAYERS, CONNECTIONS), dtype=float)

        # hidden states prediction network
        self.hiddenBefore_predictionNet = np.zeros((NUM_AGENTS, hidden), dtype=float)

        # average predictions during run
        self.pred_return = [0.0] * SENSORS

        # predictions of agents
        self.predictions = np.zeros((NUM_AGENTS, SENSORS), dtype=int)
        print(self.predictions)

    def prediction_output(self, value):
        """
            Output of prediction network:
                Values of all sensor for next time
            Return 0/1
        """
        return 0 if value < 0 else 1

    def propagate_prediction_network(self, weight_prediction, agent, input):
        """
            Propagation of the action network
            Usage:
                weight_action: The weight of each layer and all neurons
                agent: index of agent
                Input: [Sensor, last Action]
                Output: [Action, Turn direction]
        """
        hidden_net = np.zeros(self.hidden, dtype=float)
        i_layer = np.zeros(self.input, dtype=float)
        net = np.zeros(self.output, dtype=float)

        # calculate activation of input neurons
        # input layer to hidden layer
        # INPUTP == 15
        for i in range(self.input):
            i_layer[i] = activation(float(input[i]) * weight_prediction[0][2 * i] - weight_prediction[0][2 * i + 1])

        # hidden layer - 4 hidden neurons
        # HIDDENP == 14
        # Recurrent neurons
        for i in range(self.hidden):
            hidden_net[i] = 0.0
            for j in range(self.input):
                # inputs: InputP + 1 recurrent
                hidden_net[i] += i_layer[j] * weight_prediction[1][(INPUTP+1)*i+j]

            hidden_net[i] += self.hiddenBefore_predictionNet[agent][i] \
                             * weight_prediction[1][(self.input+1) * i+self.input]
            self.hiddenBefore_predictionNet[agent][i] = activation(hidden_net[i])
            hidden_net[i] = activation(hidden_net[i])

        for i in range(self.output):
            net[i] = 0.0
            for j in range(self.hidden):
                net[i] += hidden_net[j] * weight_prediction[2][i*self.hidden+j]
            net[i] = activation(net[i])

        if self.manipulation is None:
            for i in range(self.output):
                self.predictions[agent][i] = self.prediction_output((net[i]))
        elif self.manipulation == MAN:
            # predefined
            self.predictions[agent][S0] = 1
            self.predictions[agent][S3] = 1

            # learned from output
            self.predictions[agent][S1] = self.prediction_output(net[0])
            self.predictions[agent][S2] = self.prediction_output(net[1])

            self.predictions[agent][S4] = self.prediction_output(net[2])
            self.predictions[agent][S5] = self.prediction_output(net[3])

            if SENSOR_MODEL == STDL:  # 14 sensors

                # predefined
                self.predictions[agent][S8] = 1
                self.predictions[agent][S11] = 1

                # learned
                self.predictions[agent][S6] = self.prediction_output(net[4])
                self.predictions[agent][S7] = self.prediction_output(net[5])

                self.predictions[agent][S9] = self.prediction_output(net[6])
                self.predictions[agent][S10] = self.prediction_output(net[7])

                self.predictions[agent][S12] = self.prediction_output(net[8])
                self.predictions[agent][S13] = self.prediction_output(net[9])