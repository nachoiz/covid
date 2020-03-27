#########################################################################
##
## 	main.py
##
## 	27/03/2020
##
#########################################################################
'''
## MODULES
from urllib.request import urlopen

from datetime import date, timedelta
import datetime
import time

import json
import ast
import re

# plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# other
import math
import numpy as np
import scipy.optimize as opt
'''

# Utilizar interprete python3
# Evitar acentos
# No usar rutas relativas
# -----------------------------------------------------------------------


from badfunctions import req_data, plot_death_last_x_days, plot_heat_map, plot_forecast


# Raspberry path
path = '/home/pi/Documents/telegram/covid/'

# Telegram token
with open('/home/pi/Documents/telegram/util/token.txt') as f:
    token = f.readline()


# Extract data
num_countries = 6
[data_dict, countries] = req_data(num_countries)

# Plots
plot_death_last_x_days(data_dict, countries, 10, 30, path)

plot_heat_map(data_dict, countries, path)

plot_forecast(data_dict['Spain']['Deaths'], 5, path)
