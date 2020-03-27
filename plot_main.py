# Packages
from urllib.request import urlopen
import json
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
import time
import math
import ast


path = '/home/pi/Documents/telegram/covid/'

consulta = 'Spain' #Comando que llega desde el bot
#consulta_comparativa = ['Spain', 'Italy', 'China']


############################################################################################################################################
##
## DATA INITIALIZATION
## Gathering all the data and storing it in a dictionary
##
############################################################################################################################################
def preprocess_data():
    #
    # Preprocess the data
    #
    general_data = []
    print("Starting data request...")
    amount_countries_request = 6
    countries = []
    response = urlopen("https://www.worldometers.info/coronavirus/#countries")
    page_source = str(response.read())

    ref_string_for_country = 'class="mt_a" href="country/'

    #number_of_countries = len(page_source.split(ref_string_for_country))

    for i in range(1, amount_countries_request):
        str1 = page_source.split(ref_string_for_country)[i][0:60]
        str2 = str1.split('/">')
        str3 = str2[1].split('<')
        countries.append(str3[0])


    # request info for each country
    for i in range(0, amount_countries_request-1):
        print("Analysing "+str(countries[i]+"..."))
        init = 1
        fin = 9
        if countries[i] == 'USA':
            countries[i] = 'US'
            init = 1
            fin = 6
        elif countries[i] == 'S. Korea':
            countries[i] = 'south-korea'

        response = urlopen("https://www.worldometers.info/coronavirus/country/"+str(countries[i]+"/"))
        page_source = str(response.read())
        str_ref = '<script type="text/javascript">'
        divided = page_source.split(str_ref)

        list_of_datatype = []

        name_country = []
        data_country = []
        for steps in range(init, fin):
            divided2 = divided[steps].split('</script>')[0]
            name = divided2.split('name:')

            if len(name)>2:
                name1 = (str(name[1].split(',')[0])).replace('\\', '')
                name2 = (str(name[2].split(',')[0])).replace('\\', '')
                name = [name1, name2]
            else:
                name = [(str(name[1].split(',')[0])).replace('\\', '')]

            if len(name) == 2 and  name[0] == name [1]:
                name = [name[0]]

            for n in range(0, len(name)):
                name_in = name[n]

                if name_in not in list_of_datatype:
                    name_country.append((name_in.replace("\'", ""))[1:len(name_in)-1])

                    data = (divided2.split('data:')[1]).split(" ")
                    lst_tosave = ast.literal_eval(data[1].replace("null", str(-1)).replace("nan", str(-1)))
                    data_country.append(lst_tosave)


        general_data.append([name_country, data_country])


    #
    # Creating an empty dict from list of lists
    # The current structure of data is:
    #       list = [[key1, key2, ...],[val1, val2, ...]]
    #
    data_dict = {}

    # List of countries is used as key list
    key_list = countries

    # each line contains a country (country1 -> [fields][data])
    i = 0

    for key in key_list:

        # Dictionary of a single country
        country_dict = {}

        # First column of data are the keys of a single country
        keyc_list = general_data[i][0]

        # Second column of data are the values
        value_list = general_data[i][1]

        # Iterating the elements in list
        #for j in range(0, len(value_list)):
        j = 0
        for keyc in keyc_list:
            country_dict[keyc] = value_list[j]
            j = j + 1

        # Creating a dict of dicts
        data_dict[key] = country_dict

        # iterate over all countries
        i = i + 1

    return data_dict





############################################################################################################################################
##
## Plot 1
## Casos diarios + tabla
##
############################################################################################################################################
def plot_evolution_days(countries, confirmed, recovered, deaths, day_confirmed, day_recovered, day_deaths, mortality):

    # gather the last 7 days of confirmed cases
    days_ago = 9
    data_raw =[day_confirmed[0][0][len(day_confirmed[0][0])-days_ago:len(day_confirmed[0][0])], # country 1
            day_confirmed[1][0][len(day_confirmed[1][0])-days_ago:len(day_confirmed[1][0])],
            day_confirmed[2][0][len(day_confirmed[2][0])-days_ago:len(day_confirmed[2][0])],
            day_confirmed[3][0][len(day_confirmed[3][0])-days_ago:len(day_confirmed[3][0])],
            day_confirmed[4][0][len(day_confirmed[4][0])-days_ago:len(day_confirmed[4][0])]]     # country n

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

    # transpose
    data = list(map(list, zip(*data_delta_raw)))
    sign_mat = list(map(list, zip(*sign_matrix)))

    columns = countries[0:5]
    yesterday = date.today() - timedelta(days=1)

    rows = [date.today()-timedelta(days=0),
            date.today()-timedelta(days=1),
            date.today()-timedelta(days=2),
            date.today()-timedelta(days=3),
            date.today()-timedelta(days=4),
            date.today()-timedelta(days=5),
            date.today()-timedelta(days=6)]

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
    fig = plt.figure()
    for row in range(0, n_rows):
        plt.bar(index, data[row], bar_width, bottom = y_offset, color=colors[row])
        y_offset =  data[row] + y_offset
        y_offset_tabla = data[row]
        cell_text.append(['%1.1f' % x for x in y_offset_tabla])


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
    #plt.show()
    fig.savefig(path+'figures/top5_totalInfected_dailyInfected.png')



############################################################################################################################################
##
## Plot 1B
## Muertes diarias + tabla
##
############################################################################################################################################
def plot_evolution2_days(countries, confirmed, recovered, deaths, day_confirmed, day_recovered, day_deaths, mortality):

    # gather the last 7 days of confirmed cases
    days_ago = 9
    data_raw =[day_deaths[0][0][len(day_deaths[0][0])-days_ago:len(day_deaths[0][0])], # country 1
            day_deaths[1][0][len(day_deaths[1][0])-days_ago:len(day_deaths[1][0])],
            day_deaths[2][0][len(day_deaths[2][0])-days_ago:len(day_deaths[2][0])],
            day_deaths[3][0][len(day_deaths[3][0])-days_ago:len(day_deaths[3][0])],
            day_deaths[4][0][len(day_deaths[4][0])-days_ago:len(day_deaths[4][0])]]     # country n

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

    # transpose
    data = list(map(list, zip(*data_delta_raw)))
    sign_mat = list(map(list, zip(*sign_matrix)))

    columns = countries[0:5]
    yesterday = date.today() - timedelta(days=1)

    rows = [date.today()-timedelta(days=0),
            date.today()-timedelta(days=1),
            date.today()-timedelta(days=2),
            date.today()-timedelta(days=3),
            date.today()-timedelta(days=4),
            date.today()-timedelta(days=5),
            date.today()-timedelta(days=6)]

    #rows = [date.today(), date.today()-timedelta(days=1), date.today()-timedelta(days=2), date.today()-timedelta(days=3)]

    max_value = int(sum(max(data_delta_raw)))
    step = max_value*0.1
    step -= step % 10 # round to the nearest 1k

    values = np.arange(0, int(sum(max(data_delta_raw))), step)
    value_increment = 100 # used to display yticks

    # Color vector
    colors = plt.cm.brg(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset
    y_offset = np.array([0.0] * (len(columns)))
    y_offset_tabla = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    fig = plt.figure()
    for row in range(0, n_rows):
        plt.bar(index, data[row], bar_width, bottom = y_offset, color=colors[row])
        y_offset =  data[row] + y_offset
        y_offset_tabla = data[row]
        cell_text.append(['%1.1f' % x for x in y_offset_tabla])

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
    plt.ylabel("Daily deaths")
    plt.yticks(values)
    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('TOP 5 Infected Countries')
    plt.subplots_adjust(bottom=0.27, top=0.94)
    #plt.show()
    fig.savefig(path+'figures/top5_totalInfected_dailyDeaths.png')



############################################################################################################################################
##
## Plot 2
## plot by country
##
############################################################################################################################################
def plot_by_country(country_consulted, countries, day_confirmed, day_recovered, day_deaths, confirmed, recovered, deaths, mortality):
        #country_consulted = pais a consultar la informacion

        if country_consulted in countries:
            index = [i for i, s in enumerate(countries) if consulta in s][0]

            ## DATOS GENERALES ##
            print("  --- DATOS "+str(countries[index])+" ---")
            print("Confirmados: "+str(confirmed[index][0]))
            print("Recuperados: "+str(recovered[index][0]))
            print("Muertes:     "+str(deaths[index][0]))
            print("Ratio:       "+str(mortality[index][0]))

            ## GRAFICA GENERAL ##
            fig = plt.figure()
            plt.plot(day_confirmed[index][0], label = "Casos confirmados diarios", color="Blue")
            plt.plot(day_recovered[index][0], label = "Casos recuperados diarios", color="Green")
            plt.plot(day_deaths[index][0], label = "Muertes diarias", color = "Red")
            plt.title("Grafica general "+str(countries[index]))
            plt.legend()
            #plt.show()
            fig.savefig(path+'figures/plot_'+country_consulted+'.png')

        else:
            print("No hay ningun pais con ese nombre. Verifica mayusculas. Todo en ingles")




#################################### EXECUTION ####################################
'''
#Data initialization
[countries, confirmed, recovered, deaths, day_confirmed, day_recovered, day_deaths, mortality] = req_data_initialization()

#Plot 1
plot_evolution_days(countries, confirmed, recovered, deaths, day_confirmed, day_recovered, day_deaths, mortality)

#Plot 1B
plot_evolution2_days(countries, confirmed, recovered, deaths, day_confirmed, day_recovered, day_deaths, mortality)

#Plot 2
plot_by_country('Spain', countries, day_confirmed, day_recovered, day_deaths, confirmed, recovered, deaths, mortality)
'''

req_data_initialization2()
