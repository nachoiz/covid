import telebot
import time

GROUP_ID = -1001199015924
CHANNEL_ID = -1001470969008

# bot token
bot = telebot.TeleBot()

# path (crontab does not execute in the current path ./)
path = '/home/pi/Documents/telegram/covid/'

# create plot
execfile(path+"plot.py")

# select images
figure_infected = open(path+'figures/spain_infected.png', 'rb')
figure_deaths = open(path+'figures/spain_deaths.png', 'rb')
figure_top5_infected = open(path+'figures/top5_totalInfected_dailyInfected.png', 'rb')
figure_top5_deaths = open(path+'figures/top5_totalInfected_dailyDeaths.png', 'rb')

# send images
bot.send_photo(CHANNEL_ID, figure_infected)
time.sleep(2)
bot.send_photo(CHANNEL_ID, figure_deaths)
'''
time.sleep(2)
bot.send_message(CHANNEL_ID, "Infectados diarios en los 5 paises con mas casos.")
bot.send_photo(CHANNEL_ID, figure_top5_infected)
time.sleep(2)
bot.send_message(CHANNEL_ID, "Muertes diarias en los 5 paises con mas casos.")
bot.send_photo(CHANNEL_ID, figure_top5_deaths)
'''
print("plot sent")
