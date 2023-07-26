from parameters.STD import *

# ANN parameter
LAYERS = 3  # ANN layers

INPUTA = 16  # input action network (14 sensors + 1 action value + 1 temp sensor)
HIDDENA = 8  # hidden action network
OUTPUTA = 2  # output action network

INPUTP = 16  # input prediction network (15 sensors + 1 action value + 1 temp sensor)
HIDDENP = 14  # hidden prediction network
OUTPUTP = 15  # output prediction network (14 sensor predictions + 1 temperature)

ACT_CONNECTIONS = 128   # INPUA * HIDDENA
PRE_CONNECTIONS = 238  # maximum connections
# (224 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)

SENSORS = 15    # 14 agent sensors + 1 temperature sensor

SENSOR_MODEL = STDL

# * STD14: 14 sensors
#
#         . . .
#         . . .
#         . X .
#         . . .
#         . . .
