from parameters.STD import *

# ANN parameter
LAYERS = 3  # ANN layers

INPUTA = 15  # input action network (14 sensors + 1 action value)
HIDDENA = 8  # hidden action network
OUTPUTA = 2  # output action network

INPUTP = 15  # input prediction network (14 sensors + 1 action value)
HIDDENP = 14  # hidden prediction network
OUTPUTP = 14  # output prediction network (14 sensor predictions)

CONNECTIONS = 224  # maximum connections (224 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)

SENSORS = 14

SENSOR_MODEL = STDL

# * STD14: 14 sensors
#
#         . . .
#         . . .
#         . X .
#         . . .
#         . . .
