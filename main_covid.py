#########################################################################
##
## 	main.py
##
## 	27/03/2020
## 
#########################################################################

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



# Evitar acentos
# No usar rutas relativas
# -----------------------------------------------------------------------


from badfunctions import req_data 


# Raspberry path
path = '/home/pi/Documents/telegram/covid/'

# Telegram token
with open(path+'util/token.txt') as f:
    token = f.readline()


[data_dict, countries] = req_data()

