import telebot
from lxml import html
import requests
from datetime import datetime
import time

GROUP_ID = -1001199015924
CHANNEL_ID = -1001470969008

try:
	# timestamp
	stamp = datetime.now()

	# bot token
	bot = telebot.TeleBot()

	# obtain last values from .txt
	with open('/home/pi/Documents/telegram/covid/log.txt', 'r') as f:
  		lines = f.read().splitlines()
  		last_line = lines[-1]
  		data = last_line.split('.')[1]
  		data = data.split(',')
  		_infected = float(data[1])
  		_deaths = float(data[2])
  		_recovered = float(data[3])

	print(_recovered)
	# spain values
	page = requests.get('https://www.worldometers.info/coronavirus/country/spain/')
	tree = html.fromstring(page.content)
	casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
	infected = float(casos[0].text.replace(',',''))
	deaths = float(casos[1].text.replace(',',''))
	recovered = float(casos[2].text.replace(',',''))
	#msg = "test"
	msg = "*INFO COVID-19*\n -infectados: {:.0f} ({:+.0f})\n\
 -muertos: {:.0f} ({:+.0f})\n\
 -ratio: {:.2f}%\n\
 -recuperados: {:.0f} ({:+.0f})".format(infected, infected-_infected, deaths, deaths-_deaths, (deaths/infected)*100, recovered, recovered-_recovered)

	# NOT WORKING append the last values
	with open("/home/pi/Documents/telegram/covid/log2.txt", "a") as f:
		f.write("{},{:.0f},{:.0f},{:.0f}\n".format(stamp, infected, deaths, recovered))
		f.close()

	# italy values
	page_ita = requests.get('https://www.worldometers.info/coronavirus/country/italy/')
	tree_ita = html.fromstring(page_ita.content)
	casos_ita = tree_ita.xpath('//*[@id="maincounter-wrap"]/div/span')
	infected_ita = float(casos_ita[0].text.replace(',',''))
	deaths_ita = float(casos_ita[1].text.replace(',',''))
	recovered_ita = float(casos_ita[2].text.replace(',',''))
	msg_ita = "**Informazioni COVID-19**\n -infettati: {:.0f}\n -morti: {:.0f}\n -rapporto: {:.2f}%\n -recuperati: {:.0f}".format(infected_ita,deaths_ita,(deaths_ita/infected_ita)*100,recovered_ita)

	if((recovered == _recovered) and (deaths==_deaths)):
		if(stamp.time().hour < 8):
			print("Outside the time range")
		else:
			bot.send_message(GROUP_ID, "No updates :(")
			bot.send_message(CHANNEL_ID, "No hay actualizaciones")
	else:
		bot.send_message(GROUP_ID, msg)
		bot.send_message(CHANNEL_ID, msg)
		time.sleep(2)
		bot.send_message(GROUP_ID, msg_ita)

except:
	print("sth happened")
