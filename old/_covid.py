import telebot
from lxml import html
import requests

bot = telebot.TeleBot()

@bot.message_handler(commands=['start'])
def send_start(message):
        bot.reply_to(message, "Bienvenido.")

@bot.message_handler(commands=['help'])
def send_help(message):
        msg = "/start Welcome\n\
/spain   Info form Spain\n\
/italy   Info from Italy\n\
/figure  Send the last plot\n\
/plot    Updates the plot"
        bot.reply_to(message, msg)

@bot.message_handler(commands=['spain'])
def send_spain(message):
        page = requests.get('https://www.worldometers.info/coronavirus/country/spain/')
        tree = html.fromstring(page.content)
        casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
        infected = float(casos[0].text.replace(',',''))
        deaths = float(casos[1].text.replace(',',''))
        recovered = float(casos[2].text.replace(',',''))
        msg = "Actualizacion COVID-19\n -infectados: {:.0f}\n -muertos: {:.0f}\n -ratio: {:.2f}%\n -recuperados: {:.0f}".format(infected,deaths,(deaths/infected)*100,recovered)
        bot.reply_to(message, msg)

@bot.message_handler(commands=['italy'])
def send_italy(message):
        page = requests.get('https://www.worldometers.info/coronavirus/country/italy/')
        tree = html.fromstring(page.content)
        casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
        infected = float(casos[0].text.replace(',',''))
        deaths = float(casos[1].text.replace(',',''))
        recovered = float(casos[2].text.replace(',',''))
        msg = "Aggiornamento COVID-19\n -infettati: {:.0f}\n -morti: {:.0f}\n -rapporto: {:.2f}%\n -recuperati: {:.0f}".format(infected,deaths,(deaths/infected)*100,recovered)
        bot.reply_to(message, msg)

@bot.message_handler(commands=['figure'])
def send_fig(message):
        photo = open('figure.png', 'rb')
	#bot.send_photo(message, photo)
        bot.send_photo(-1001199015924, photo)

@bot.message_handler(commands=['plot'])
def send_plot(message):
        execfile("/home/pi/Documents/telegram/covid/plot.py")
	bot.reply_to(message, "plot.py executed")

@bot.message_handler(commands=['ratio'])
def send_ratio(message):
        page = requests.get('https://www.worldometers.info/coronavirus/country/spain/')
        tree = html.fromstring(page.content)
        casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
        bot.reply_to(message, int(casos[0].text.replace(',',''))/int(casos[1].text.replace(',','')))


@bot.message_handler(commands=['muertos'])
def send_muertos(message):
        page = requests.get('https://www.worldometers.info/coronavirus/country/spain/')
        tree = html.fromstring(page.content)
        casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
	bot.reply_to(message, casos[1].text)
	#bot.send_message(-482239607,casos[1].text)


bot.polling()
