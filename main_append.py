import telebot
from lxml import html
import requests
from datetime import datetime
import time


stamp = datetime.now()

# spain
page = requests.get('https://www.worldometers.info/coronavirus/country/spain/')
tree = html.fromstring(page.content)
casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
infected = float(casos[0].text.replace(',',''))
deaths = float(casos[1].text.replace(',',''))
recovered = float(casos[2].text.replace(',',''))
#print("preparado para escribir")
with open("/home/pi/Documents/telegram/covid/log.txt", "a") as myfile:
	myfile.write("{},{:.0f},{:.0f},{:.0f}\n".format(stamp, infected, deaths, recovered))
	myfile.close()

