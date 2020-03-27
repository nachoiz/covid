#########################################################################
##
## 	badfunctions.py
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


#########################################################################
## req_data 
##
## Data extraction function is divided in 3 parts:
## data extraction -> dict transformation -> update data appendix
##
## input:   number_of_countries
##
## output:   data_dict   -> data dictionary
##           countries   -> list of countries
#########################################################################
def req_data(number_of_countries):
    general_data = []
    print("Starting data request...")
    countries = []
    response = urlopen("https://www.worldometers.info/coronavirus/#countries")
    page_source = str(response.read())

    ref_string_for_country = 'class="mt_a" href="country/'

    # First obtain a list w/ the name of all countries
    for i in range(1, number_of_countries):
        str1 = page_source.split(ref_string_for_country)[i][0:60]
        str2 = str1.split('/">')
        str3 = str2[1].split('<')
        countries.append(str3[0])

    # Request info for each country
    for i in range(0, number_of_countries-1):
        print("Analysing "+str(countries[i]+"..."))
        init = 1
        fin = 9
        # Special cases
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

    # Creating an empty dict from list of lists
    # The current structure of data is:
    #       list = [[key1, key2, ...],[val1, val2, ...]]
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


    #N = 100
    #y = list(data_dict['Spain']['Daily Deaths'])

    print(" ")
    print('Starting realtime request...')
    #Asking for realtime data
    response = urlopen("https://www.worldometers.info/coronavirus/#countries")
    page_source = str(response.read())

    for i in range(0, number_of_countries-1):
        print("Analysing "+str(countries[i]))
        #Reference for the source code seaching
        #href="/coronavirus/country/usa/">
        h_ref = str("href=\"/coronavirus/country/"+str(countries[i].lower())+"/\">")

        divided = (page_source.split(h_ref)[0]).split("</strong>")
        new_cases = int((divided[len(divided) - 3].split("<strong>")[1]).split("new cases")[0])
        new_deaths = int((divided[len(divided) - 2].split("<strong>")[1]).split("new deaths")[0])

        #Adding realtime data
        data_dict[countries[i]]['Currently Infected'].append(data_dict[countries[i]]['Currently Infected'][len(data_dict[countries[i]]['Currently Infected'])-1] + new_cases)
        data_dict[countries[i]]['Daily Cases'].append(new_cases)
        data_dict[countries[i]]['Cases'].append(data_dict[countries[i]]['Cases'][len(data_dict[countries[i]]['Cases'])-1] + new_cases)

        if countries[i] is not 'US':
            data_dict[countries[i]]['New Cases'].append(new_cases)

        data_dict[countries[i]]['Daily Deaths'].append(new_deaths)
        data_dict[countries[i]]['Deaths'].append(data_dict[countries[i]]['Deaths'][len(data_dict[countries[i]]['Deaths'])-1] + new_deaths)

    print("End of data gathering!")
    return [data_dict, countries]
    
    


#########################################################################
## plot_death_last_x_days 
##
## This plot represents deaths w.r.t. days for a group of countries.
##
## input:    data_dict         -> countries data
##           countries         -> list of countries
##           days              -> number of days in x axis
##           days_threshold    -> number of deaths to start counting
##           path              -> where to save the figure
##
## output:   death_last_X_threshold_X.png'
##         
#########################################################################
def plot_death_last_x_days(data_dict, countries, days, death_threshold, path):
    country_list = []
    country_death_list = []

    for country in countries:
        country_list = list(data_dict[country]['Deaths'])
        country_death_list.append(country_list)

    ind10_list = []
    list10 = []

    # set the threshold
    #death_threshold = 30
    for pais in country_death_list:
        ind10 = 0
        for elem in pais:
            if elem >= death_threshold:
                ind10 = pais.index(elem)
                list10.append(pais[ind10:])
                break
        ind10_list.append(ind10)

    # Plot
    lis_country = []
    lis_total = []
    for i in range(0, len(list10)):
        lis = np.clip(list10[i], days, 10000000)
        for j in range(0, len(lis)-1):
            if lis[j] != days:
                lis_country.append(lis[j])
        lis_total.append(lis_country)
        lis_country = []

    # Create figure
    fig = plt.figure()

    for i in range(0, len(lis_total)):
        plt.plot(lis_total[i], label=countries[i])

    plt.legend(loc='upper left')

    ax = fig.add_subplot(111)
    ax.tick_params(labeltop=False, labelright=True)
    ax.grid(True, linestyle='-.')

    plt.yscale('log')
    locs, labels = plt.yticks()                     # Get locations and labels
    print(locs)
    plt.yticks([locs[2], locs[3], locs[4]], [100, 1000, 10000])    # Set locations and labels
    
    plt.xlabel('Dias', fontsize=14)
    plt.ylabel('Muertes', fontsize=14)
    plt.title('Dias desde la muerte '+str(death_threshold))
    #plt.show()
    fig.savefig(path+'figures/death_last_'+str(days)+'_threshold_'+str(death_threshold)+'.png')
    print("*************************** plot_death_last_x_days FINISHED")
    
    
#########################################################################
## plot_heat_map 
##
## This plot represents daily deaths w.r.t. days for a group of countries.
##
## input:    data_dict         -> countries data
##           countries         -> list of countries
##           path              -> where to save the figure
##
## output:   deaths_daily_square.png'
##         
#########################################################################
def plot_heat_map(data_dict, countries, path):    
  # Nice to have 9 elements and 9 countries to build up a square matrix
  mat = np.zeros(shape=(len(countries),9))
  i = 0
  vector = []
  for country in countries:
      vector.append(data_dict[country]['Daily Deaths'][-9:])
      mat[i,:] = vector[i]
      i = i+1
  
  # Create figure
  fig, ax = plt.subplots()
  im = ax.imshow(mat)
  
  # We want to show all ticks...
  ax.set_xticks(np.arange(9))
  ax.set_yticks(np.arange(len(countries)))
  # ... and label them with the respective list entries
  
  dias = ["dia "+str(i) for i in range(10)]
  
  semana = []
  for i in range(0,9):
      dia = datetime.datetime.strftime(datetime.datetime.now() - timedelta(i), '%d-%m')
      semana.append(dia)
  
  semana = semana[::-1]
  
  ax.set_xticklabels(semana)
  ax.set_yticklabels(countries)
  
  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
  # Loop over data dimensions and create text annotations.
  for i in range(len(countries)):
      for j in range(9):
          text = ax.text(j, i, int(mat[i, j]),ha="center", va="center", color="w")
  
  ax.set_title("Muertos diarios")
  fig.tight_layout()
  #plt.show()
  fig.savefig(path+'figures/deaths_daily_square.png')
  print("*************************** plot_heat_map FINISHED")
  



#########################################################################
## plot_forecast 
##
## This plot represents daily deaths w.r.t. days for a group of countries.
##
## input:    data_dict         -> countries data
##           future_days       -> forecast days 
##           path              -> where to save the figure
##
## output:   plot_prediction.png'
##         
#########################################################################
def plot_forecast(datos, future_days, path):  
  ### Function ##########################################################
  # in-line function -> lambda t,a,b: a*numpy.exp(b*t)
  def func(t, a, b):
  	return a*np.exp(b*t)
      
  
  ### Current data ######################################################  
  country = datos
  for elem in country:
      if elem >= 10:
          ind = country.index(elem)
          break
  
  # Current day since record
  current_day_num = len(country)
  
  # Subscripts
  # 0 -> current variable
  # 1 -> 'future' variable
  
  days_ago = len(country)-ind    
  country_deaths = country[ind:]
  y0 = np.asarray(country_deaths)
  x0 = np.arange(y0.size)
  
  # Generate an x vector with the days
  dias0 = []
  for i in range(0,len(x0)):
      dia = datetime.datetime.strftime(datetime.datetime.now() - timedelta(i), '%d-%m')
      dias0.append(dia)
  dias0 = dias0[::-1]
  
  # Create plot figure
  fig = plt.figure()
  
  plt.legend(loc='upper left')
  ax = fig.add_subplot(111)
  
  # Yticks in both sides
  ax.tick_params(labeltop=False, labelright=True)
  
  # Grid on
  ax.grid(True, linestyle='-.')
  
  # Plot
  plt.plot(dias0,y0, 'bx')
  
  
  ### Future prevision ######################################################
  #future_days = 5
  
  # Generate an x vector with the future days
  dias1 = []
  for i in range(1,future_days+1):
      dia = datetime.datetime.strftime(datetime.datetime.now() + timedelta(i), '%d-%m')
      dias1.append(dia)
  
  # Standard deviation +/-
  std = 50. # muertos
  e = np.repeat(std, len(dias1))
  #plt.errorbar(x, y, yerr=e, fmt="none")
  
  # Extract the coefficients taking into account the historical data
  # popt -> best-fit parameters for 'a' and 'b'
  # pcov -> true variance and covariance of the parameters
  popt, pcov = opt.curve_fit(func,x0, y0)
  print("The fist parameters are:")
  print("a =", popt[0], "+/-", pcov[0,0]**0.5)
  print("b =", popt[1], "+/-", pcov[1,1]**0.5)
  
  #x = np.linspace(20,40)
  
  xfine = np.linspace(ind, ind+future_days, future_days)  # define values to plot the function for
  plt.plot(dias1, func(xfine, popt[0], popt[1]), 'r-')
  plt.errorbar(dias1, func(xfine, popt[0], popt[1]), yerr=e, uplims=True, lolims=True, fmt="none")
  
  # Turn of some xtick labels
  every_nth = 4
  for n, label in enumerate(ax.xaxis.get_ticklabels()):
      if n % every_nth != 0:
          label.set_visible(False)
  
  #plt.show()
  fig.savefig(path+'figures/forecast_'+str(future_days)+'.png')
  print("*************************** plot_forecast FINISHED")
  
  # Aproximar mejor la exponencial incluyendo weighted coeffs
  # https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
  
  # Breve explicacion de lo que hace 'curve_fit'
  # https://astrofrog.github.io/py4sci/_static/15.%20Fitting%20models%20to%20data.html