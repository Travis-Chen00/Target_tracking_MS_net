from parameters.STD import *

# ANN parameter
LAYERS = 3  # ANN layers

INPUTA = 7  # input action network (14 sensors + 1 action value + 1 temp sensor)
HIDDENA = 8  # hidden action network
OUTPUTA = 2  # output action network

INPUTP = 7  # input prediction network (14 sensors + 1 action value)
HIDDENP = 10  # hidden prediction network
OUTPUTP = 5  # output prediction network (14 sensor predictions + 1 temperature)

ACT_CONNECTIONS = 56   # INPUA * HIDDENA
PRE_CONNECTIONS = 78   # maximum connections

SENSORS = 6    # 14 agent sensors

Heat_alpha = [0.25, 0.7, 0.05]    # Heat rate for different zones
Heat_int = [0.54, 0.75, 0.87]

TARGET_SIZE = 1

SENSOR_MODEL = STDL
