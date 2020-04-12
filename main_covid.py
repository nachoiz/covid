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

import telebot
from datetime import datetime
import time

from badfunctions import req_data, plot_death_last_x_days, plot_heat_map, plot_forecast, obtain_message, global_contagios_3d, stacket_plot_deaths_and_cases, evolution_R0, plot_cases_last_x_days


# Raspberry path
path = '/home/pi/Documents/telegram/util/'


# Telegram token
with open(path+'token.txt') as f:
    token = f.readline()
    token = token.replace('\n','')

# Channel and group IDs
GROUP_ID = -1001199015924
CHANNEL_ID = -1001470969008

# Extract data
num_countries = 12
[data_dict, countries] = req_data(num_countries)

# Plots
plot_death_last_x_days(data_dict, countries, 10, 30, path)
plot_cases_last_x_days(data_dict, countries, 10, 100, path)

# usar 'deaths' en vez de 'dailydeaths'
#plot_heat_map(data_dict, countries, path)
#plot_forecast(data_dict['Spain']['Deaths'], 5, path)

global_contagios_3d(data_dict, countries, path)
stacket_plot_deaths_and_cases(data_dict, countries, num_countries, path)
evolution_R0(data_dict, countries, path)

# timestamp
stamp = datetime.now()

# bot token
bot = telebot.TeleBot(str(token))

# Create message
[message, message_markdown] = obtain_message(data_dict, countries, path)
print(message)

if(stamp.time().hour > 8):
  bot.send_message(CHANNEL_ID, text=message_markdown, parse_mode = 'Markdown')
  if(stamp.time().hour == 22):
    figure_1 = open(path+'figures/death_last_10_threshold_30.png', 'rb')
    figure_1b = open(path+'figures/case_last_10_threshold_100.png', 'rb')
    #figure_2 = open(path+'figures/deaths_daily_square.png', 'rb')
    bot.send_photo(CHANNEL_ID, figure_1)
    bot.send_photo(CHANNEL_ID, figure_1b)
    #bot.send_photo(CHANNEL_ID, figure_2)
  elif(stamp.time().hour == 12):
    print("")
    figure_3 = open(path+'figures/stacked_cases.png', 'rb')
    figure_4 = open(path+'figures/stacked_deaths.png', 'rb')
    bot.send_photo(CHANNEL_ID, figure_3)
    bot.send_photo(CHANNEL_ID, figure_4)
  elif(stamp.time().hour == 18):
    figure_5 = open(path+'figures/ro_casos.png', 'rb')
    bot.send_photo(CHANNEL_ID, figure_5)
  elif(stamp.time().hour == 21):
    figure_6 = open(path+'figures/global_contagios_3d.png', 'rb')
    bot.send_photo(CHANNEL_ID, figure_6)

#bot.send_message(GROUP_ID, text=message_markdown, parse_mode = 'Markdown')
