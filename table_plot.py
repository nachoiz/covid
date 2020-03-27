import urllib.request, json
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
import time
import math

consulta = 'Spain' #Comando que llega desde el bot
consulta_comparativa = ['Spain', 'Italy', 'China']
consulta = 'Todos'

# data is ordered from the highest to the lowest values of cases
with urllib.request.urlopen("https://corona-stats.online/?format=json") as url:
    data = json.loads(url.read().decode())

# define main vectors
countries = []
confirmed = []
recovered = []
deaths = []
day_confirmed = []
day_recovered = []
day_deaths = []
mortality = []

# define auxiliary vectors
c_aux = []
r_aux = []
d_aux = []
da_c_aux = []
da_r_aux = []
m_aux = []
dea_aux = []

for i in range(0, 100):
    countries.append(data[i]['country'])
    c_aux.append(data[i]['confirmed'])
    r_aux.append(data[i]['recovered'])
    d_aux.append(data[i]['deaths'])
    da_c_aux.append(data[i]['confirmedByDay'])
    da_r_aux.append(data[i]['recoveredByDay'])
    m_aux.append(data[i]['mortalityPer'])
    dea_aux.append(data[i]['deathsByDay'])

    confirmed.append(c_aux)
    recovered.append(r_aux)
    deaths.append(d_aux)
    day_confirmed.append(da_c_aux)
    day_recovered.append(da_r_aux)
    mortality.append(m_aux)
    day_deaths.append(dea_aux)

    # empty vectors
    c_aux = []
    r_aux = []
    d_aux = []
    da_c_aux = []
    da_r_aux = []
    m_aux = []

if 'Todos' in consulta:
    # gather the last 7 days of confirmed cases
    days_ago = 9
    data_raw =[day_confirmed[0][0][len(day_confirmed[0][0])-days_ago:len(day_confirmed[0][0])-1], # country 1
            day_confirmed[1][0][len(day_confirmed[1][0])-days_ago:len(day_confirmed[1][0])-1],
            day_confirmed[2][0][len(day_confirmed[2][0])-days_ago:len(day_confirmed[2][0])-1],
            day_confirmed[3][0][len(day_confirmed[3][0])-days_ago:len(day_confirmed[3][0])-1],
            day_confirmed[4][0][len(day_confirmed[4][0])-days_ago:len(day_confirmed[4][0])-1]]     # country n

    # Matrix filled up with cases
    data_delta_raw = []
    for i in range(0,5):
        data_delta_raw_aux = []
        sign_array_aux = []
        for j in range(0,days_ago-2):
            dif = data_raw[i][j+1]-data_raw[i][j]
            data_delta_raw_aux.append(dif)
        data_delta_raw.append(data_delta_raw_aux)

    # Matrix filled up with [-1,0+1] to detect the net growth
    sign_matrix = []
    for i in range(0,5):
        sign_array_aux = []
        for j in range(0, days_ago-3):
            dif = data_delta_raw[i][j+1] - data_delta_raw[i][j]
            if dif == 0:
                sign_array_aux.append(0)
            elif dif > 0:
                sign_array_aux.append(1)
            elif dif < 0:
                sign_array_aux.append(-1)
        sign_array_aux = sign_array_aux[::-1]
        sign_matrix.append(sign_array_aux)

    maxim =  max([sublist[-1] for sublist in data_raw])
    print(maxim)

    # transpose
    data = list(map(list, zip(*data_delta_raw)))
    sign_mat = list(map(list, zip(*sign_matrix)))

    columns = countries[0:5]
    yesterday = date.today() - timedelta(days=1)

    rows = [date.today()-timedelta(days=1),
            date.today()-timedelta(days=2),
            date.today()-timedelta(days=3),
            date.today()-timedelta(days=4),
            date.today()-timedelta(days=5),
            date.today()-timedelta(days=6),
            date.today()-timedelta(days=7)]

    #rows = [date.today(), date.today()-timedelta(days=1), date.today()-timedelta(days=2), date.today()-timedelta(days=3)]

    max_value = int(sum(max(data_delta_raw)))
    step = max_value*0.1
    step -= step % 1000 # round to the nearest 1k
    values = np.arange(0, int(sum(max(data_delta_raw))), step)
    value_increment = 100 # used to display yticks

    # Color vector
    colors = plt.cm.cool(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset
    y_offset = np.array([0.0] * (len(columns)))
    y_offset_tabla = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(0, n_rows):
        print("row", row)
        plt.bar(index, data[row], bar_width, bottom = y_offset, color=colors[row])
        y_offset =  data[row] + y_offset
        y_offset_tabla = data[row]
        cell_text.append(['%1.1f' % x for x in y_offset_tabla])
        print("data[row]", data[row])
        print("y_offset", y_offset)
        print(" ")

    # Revers colors and text labels to display the last value at the top
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    #sign_mat
    for i in range(0,6):
        for j in range(0, 5):
            val = sign_mat[i][j]
            i_table = i + 1
            if val == 0:
                the_table[(i_table, j)].get_text().set_color('black')
                #the_table[(i_table, j)].set_facecolor("#56b5fd")
            elif val >0:
                the_table[(i_table, j)].get_text().set_color('red')
            elif val < 0:
                the_table[(i_table, j)].get_text().set_color('green')




    # Adjust layout to make room for the table
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Daily infected")
    plt.yticks(values)
    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('TOP 5 Infected Countries')
    plt.subplots_adjust(bottom=0.27, top=0.94)
    plt.show()


else:
    if consulta in countries:
        index = [i for i, s in enumerate(countries) if consulta in s][0]

        ## DATOS GENERALES ##
        print("  --- DATOS "+str(countries[index])+" ---")
        print("Confirmados: "+str(confirmed[index][0]))
        print("Recuperados: "+str(recovered[index][0]))
        print("Muertes:     "+str(deaths[index][0]))
        print("Ratio:       "+str(mortality[index][0]))

        ## GRAFICA GENERAL ##
        plt.plot(day_confirmed[index][0], label = "Casos confirmados diarios", color="Blue")
        plt.plot(day_recovered[index][0], label = "Casos recuperados diarios", color="Green")
        plt.plot(day_deaths[index][2], label = "Muertes diarias", color = "Red")
        plt.title("Grafica general "+str(countries[index]))
        plt.legend()
        plt.show()

        ratio = np.divide(day_deaths[index][2], day_confirmed[index][0])
        ratio = np.multiply(ratio, 100)

        ## MORTALIDAD  ##
        plt.title("Evolucion de mortalidad "+str(countries[index]))
        plt.plot(ratio, label = "Ratio mortalidad")
        plt.legend()
        plt.show()


    else:
        print("No hay ningun país con ese nombre. Verifica mayúsculas. Todo en inglés")

