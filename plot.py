import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

path = '/home/pi/Documents/telegram/covid/'

with open(path+'log.txt') as f:
	lines = f.readlines()
	date = [line.split('.')[0] for line in lines]
	rest = [line.split('.')[1] for line in lines]

	infected = []
	deaths = []
	recovered = []
	dtime = []
	for elem in rest:
		elem = elem.replace('\n','')
		infected.append(elem.split(',')[1])
		deaths.append(elem.split(',')[2])
		recovered.append(elem.split(',')[3])

# datetime struct
for d in date:
	dtime.append(datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))

# str to int
infected = list(map(int, infected))
deaths = list(map(int, deaths))
recovered = list(map(int, recovered))

dates = mpl.dates.date2num(dtime)

# SPAIN
# deaths
fig = plt.figure()
plt.title("COVID-19 deaths [Spain]")
plt.plot_date(dates, deaths, linestyle='-', color='r')
#fig.suptitle('test title', fontsize=8)
plt.xlabel('Time', fontsize=10)
plt.xticks(fontsize=8)
plt.grid()
#plt.xticks(rotation=50)
plt.ylabel('Deaths', fontsize=10)
plt.tight_layout()
fig.savefig(path+'figures/spain_deaths.png')

# infected
fig = plt.figure()
plt.title("COVID-19 infected [Spain]")
plt.plot_date(dates, infected, linestyle='-', color='g')
plt.xlabel('Time', fontsize=10)
plt.xticks(fontsize=8)
plt.grid()
plt.ylabel('Infected', fontsize=10)
plt.tight_layout()
fig.savefig(path+'figures/spain_infected.png')

print("plot.py executed!")
