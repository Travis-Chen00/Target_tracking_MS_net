from parameters.STD import *

# ANN parameter
LAYERS = 3  # ANN layers

INPUTA = 16  # input action network (14 sensors + 1 action value + 1 temp sensor)
HIDDENA = 14  # hidden action network
OUTPUTA = 2  # output action network

INPUTP = 16  # input prediction network (14 sensors + 1 action value)
HIDDENP = 14  # hidden prediction network
OUTPUTP = 14  # output prediction network (14 sensor predictions + 1 temperature)

INPUTTEMP = 16
HIDDENTEMP = 14
OUTPUTTEMP = 3          # 4 kinds temperatures

ACT_CONNECTIONS = 224   # INPUA * HIDDENA
PRE_CONNECTIONS = 224   # maximum connections
TEMP_CONNECTIONS = 224   # Input_temp * Hidden_temp
# (224 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)

SENSORS = 14    # 14 agent sensors

TEMP_SENSORS = 1

SENSOR_MODEL = STDL

Net_para = 0.1
# * STD14: 14 sensors
#
#         . . .
#         . . .
#         . X .
#         . . .
#         . . .
#
# from parameters.STD import *
#
# # ANN parameter
# LAYERS = 4  # ANN layers
#
# INPUTA = 19  # input action network (14 sensors + 1 action value + 4 temp sensor)
# HIDDENA = 8  # hidden action network
# OUTPUTA = 2  # output action network
#
# INPUTP = 15  # input prediction network (14 sensors + 1 action value)
# HIDDENP = 14  # hidden prediction network
# OUTPUTP = 14  # output prediction network (14 sensor predictions + 1 temperature)
#
# INPUTTEMP = 5
# HIDDENTEMP_1 = 8
# HIDDENTEMP_2 = 4
# OUTPUTTEMP = 1
#
# ACT_CONNECTIONS = 152   # INPUA * HIDDENA
# PRE_CONNECTIONS = 224   # maximum connections
# TEMP_CONNECTIONS = 40   # Input_temp * Hidden_temp
# # (224 per layer: 1 per INPUTP for each HIDDENP (15x14) + recurrent --> 14 extra values)
#
# SENSORS = 14    # 14 agent sensors
#
# TEMP_SENSORS = 4
#
# SENSOR_MODEL = STDL
#
# Net_para = 0.4
#
# # * STD14: 14 sensors
# #
# #         . . .
# #         . . .
# #         . X .
# #         . . .
# #         . . .

