# Based on David-Lor telebot_polling_template.py
import os
import telebot
import threading
from time import sleep
from lxml import html
import requests

BOT_TOKEN = ""
BOT_INTERVAL = 3
BOT_TIMEOUT = 30

bot = telebot.TeleBot(BOT_TOKEN)

def bot_polling():
    #global bot #Keep the bot object as global variable if needed
    #bot = telebot.TeleBot(BOT_TOKEN) #Generate new bot instance
    print("Starting bot polling now")
    while True:
        try:
            print("New bot instance started")
            botactions() #If bot is used as a global variable, remove bot as an input param
            bot.polling(none_stop=True, interval=BOT_INTERVAL, timeout=BOT_TIMEOUT)
        except Exception as ex: #Error in polling
            print("Bot polling failed, restarting in {}sec. Error:\n{}".format(BOT_TIMEOUT, ex))
            bot.stop_polling()
            sleep(BOT_TIMEOUT)
        else: #Clean exit
            bot.stop_polling()
            print("Bot polling loop finished")
            break #End loop


def botactions():
    #Set all your bot handlers inside this function
    #If bot is used as a global variable, remove bot as an input param
    @bot.message_handler(commands=['start'])
    def send_start(message):
	bot.reply_to(message, "Bienvenido.")

    @bot.message_handler(commands=['pid'])
    def send_pid(message):
	print("entra0")
	pid = os.getpid()
	bot.reply_to(message, pid)
	#os.system('kill {}'.format(pid))

    @bot.message_handler(commands=['help'])
    def send_help(message):
	msg = "/start Welcome\n/spain Info Spain\n/italy Info Italy\n/figure Send the last plot\n/plot Updates the plot"
	bot.reply_to(message, msg)

'''
    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
	data=message.text
	data = data.encode('ascii')
	if (len(data.split()) == 1):
	    try:
		country = data.lower()
		print(country)
		print(message.chat.id)
		info_country(country,message.chat.id)
	    except:
		bot.reply_to(message, "Country not found :(")
'''

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


    def info_country(country,chat_id):
	print("legoooo")
	link = 'https://www.worldometers.info/coronavirus/country/'
	link = link + country + '/'
	print(link)
	page = requests.get(link)
	tree = html.fromstring(page.content)
	casos = tree.xpath('//*[@id="maincounter-wrap"]/div/span')
	infected = float(casos[0].text.replace(',',''))
	deaths = float(casos[1].text.replace(',',''))
	recovered = float(casos[2].text.replace(',',''))
	msg = "Info COVID-19\n -infected: {:.0f}\n -deaths: {:.0f}\n -ratio: {:.2f}%\n -recovered: {:.0f}".format(infected,deaths,(deaths/infected)*100,recovered)
	#bot.reply_to(chat_id, msg)
	bot.send_message(chat_id, msg)


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




polling_thread = threading.Thread(target=bot_polling)
polling_thread.daemon = True
polling_thread.start()


#Keep main program running while bot runs threaded
if __name__ == "__main__":
    while True:
        try:
            sleep(120)
        except KeyboardInterrupt:
            break

