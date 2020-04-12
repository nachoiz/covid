#########################################################################
##
## 	badfunctions.py
##
## 	27/03/2020
##
#########################################################################

## MODULES
from urllib.request import urlopen, Request

from datetime import date, timedelta
import datetime
import time

import json
import ast
import re

# plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

# other
import math
import numpy as np
import scipy.optimize as opt

import difflib
import warnings
warnings.filterwarnings("ignore")

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
    url = "https://www.worldometers.info/coronavirus/#countries"
    req = Request(url, headers = {"User-Agent": "Mozilla/5.0"})
    #response = urlopen()
    response = urlopen(req)
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
        elif countries[i] == 'UK' or countries[i] == 'Turkey' or countries[i] == 'Switzerland' or countries[i] == 'Netherlands':
            init = 1
            fin = 6

        elif countries[i] == 'S. Korea':
            countries[i] = 'south-korea'

        url = "https://www.worldometers.info/coronavirus/country/"+str(countries[i]+"/")
        req = Request(url, headers = {"User-Agent": "Mozilla/5.0"})
        response = urlopen(req)
        page_source = str(response.read())

        str_ref = '<script type="text/javascript">'
        divided = page_source.split(str_ref)

        list_of_datatype = []
        name_country = []
        data_country = []
        # HAY QUE CALCULAR 'fin' ANTES!!!!!
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
            #print(len(name))
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


    #
    # Asking for realtime data
    #
    print(" ")
    print('Starting realtime request...')


    for i in range(0, number_of_countries-1):
    #for i in range(0, 1):
      url = "https://www.worldometers.info/coronavirus/country/{}".format(countries[i])
      req = Request(url, headers = {"User-Agent": "Mozilla/5.0"})
      response = urlopen(req)
      page_source = str(response.read())
      try:
        print("Analysing "+str(countries[i]))

        # Obtain html tree
        new_data = page_source.split('<div id="maincounter-wrap"')
        
        # Find cases inside html tree
        new_cases = new_data[1].split('<div class="maincounter-number">')
        new_cases = new_cases[1].split('/span>')
        new_cases = re.search('>(.*)<', new_cases[0])

        # Find deaths inside html tree
        new_deaths = new_data[2].split('<div class="maincounter-number">')
        new_deaths = new_deaths[1].split('/span>')
        new_deaths = re.search('>(.*)<', new_deaths[0])

        # from re match object to string
        new_cases = new_cases.group(1)
        new_deaths = new_deaths.group(1)

        # remove blank space and cast to int
        new_cases = int(new_cases.replace(' ','').replace(',',''))
        new_deaths = int(new_deaths.replace(' ','').replace(',',''))

        # Last cases/deaths
        last_cases = data_dict[countries[i]]['Cases'][-1]
        last_deaths = data_dict[countries[i]]['Deaths'][-1]
        
        #Adding realtime data
        # New cases/deaths are the current TOTAL cases/deaths
        delta_cases = new_cases - last_cases
        delta_deaths = new_deaths - last_deaths
        
        #data_dict[countries[i]]['_Cases'] = delta_cases
        #data_dict[countries[i]]['_Deaths'] = delta_deaths
        
        data_dict[countries[i]]['Cases'][-1] = last_cases + delta_cases
        data_dict[countries[i]]['Deaths'][-1] = last_deaths + delta_deaths
                
        try:
            data_dict[countries[i]]['Daily Cases'].append(delta_cases)
            data_dict[countries[i]]['Daily Deaths'].append(delta_deaths)
        except:
            print("Error adding realtime for "+countries[i])
        #print(data_dict[countries[i]]['Cases'])
        #print("")
      except:
        print("Error catching "+countries[i])
        data_dict[countries[i]]['status'] = 0
        continue
        
    
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

    plt.legend(loc='lower right')

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


def plot_cases_last_x_days(data_dict, countries, days, death_threshold, path):
    country_list = []
    country_death_list = []

    for country in countries:
        country_list = list(data_dict[country]['Cases'])
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

    plt.legend(loc='lower right')

    ax = fig.add_subplot(111)
    ax.tick_params(labeltop=False, labelright=True)
    ax.grid(True, linestyle='-.')

    plt.yscale('log')
    locs, labels = plt.yticks()                     # Get locations and labels
    print(locs)
    plt.yticks([locs[2], locs[3], locs[4], locs[5]], [100, 1000, 10000, int(1e5)])    # Set locations and labels

    plt.xlabel('Dias', fontsize=14)
    plt.ylabel('Casos', fontsize=14)
    plt.title('Dias desde el caso confirmado '+str(death_threshold))
    #plt.show()
    fig.savefig(path+'figures/case_last_'+str(days)+'_threshold_'+str(death_threshold)+'.png')
    print("*************************** plot_case_last_x_days FINISHED")

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
  #print("The fist parameters are:")
  #print("a =", popt[0], "+/-", pcov[0,0]**0.5)
  #print("b =", popt[1], "+/-", pcov[1,1]**0.5)

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



#########################################################################
## horizontal distribution
##
## This plot represents total deaths, recoveries and active cases
##
## input:    data_dict         -> countries data
##           countries         -> countries list
##           path              -> where to save the figure
##
## output:   plot_horizontal_bar.png
##
#########################################################################
def horizontal_distribution(data_dict, countries, path):
    #print(data_dict['US'])
    if len(countries) < 6:
        print("NO HAY SUFICIENTE INFORMACION PARA CREAR LA GRAFICA, HAY QUE PEDIR INFORMACION DE ALMENOS 7 PAISES")
    else:

        #Fixeamos y añadimos un nuevo key para US, que por defecto no tiene new recoveries
        data_dict['US']['New Recoveries'] = [0, data_dict['US']['Cases'][-1] - data_dict['US']['Deaths'][-1] - data_dict['US']['Currently Infected'][-1]]


        category_names = ['Muertes',
                          'Infectados actuales',
                          'Recuperados']

        results = {
            countries[0]: [data_dict[countries[0]]['Deaths'][-1], data_dict[countries[0]]['Currently Infected'][-1],  data_dict[countries[0]]['New Recoveries'][-1], 0, 0],
            countries[1]: [data_dict[countries[1]]['Deaths'][-1], data_dict[countries[1]]['Currently Infected'][-1],  data_dict[countries[1]]['New Recoveries'][-1], 0, 0],
            countries[2]: [data_dict[countries[2]]['Deaths'][-1], data_dict[countries[2]]['Currently Infected'][-1],  data_dict[countries[2]]['New Recoveries'][-1], 0, 0],
            countries[3]: [data_dict[countries[3]]['Deaths'][-1], data_dict[countries[3]]['Currently Infected'][-1],  data_dict[countries[3]]['New Recoveries'][-1], 0, 0],
            countries[4]: [data_dict[countries[4]]['Deaths'][-1], data_dict[countries[4]]['Currently Infected'][-1],  data_dict[countries[4]]['New Recoveries'][-1], 0, 0],
            countries[5]: [data_dict[countries[5]]['Deaths'][-1], data_dict[countries[5]]['Currently Infected'][-1],  data_dict[countries[5]]['New Recoveries'][-1], 0, 0],
        }

        def survey(results, category_names):
            """
            Parameters
            ----------
            results : dict
                A mapping from question labels to a list of answers per category.
                It is assumed all lists contain the same number of entries and that
                it matches the length of *category_names*.
            category_names : list of str
                The category labels.
            """
            labels = list(results.keys())
            data = np.array(list(results.values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, data.shape[1]))

            fig, ax = plt.subplots(figsize=(9.2, 5))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())
            column_colors =  ['Lightcoral', 'antiquewhite', 'yellowgreen']
            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=column_colors[i], align='center')
                xcenters = starts + widths / 2
                text_color = 'black'
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    #ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color) #Adding values to bar
                    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')

            return fig, ax

        survey(results, category_names)
        #plt.show()
        plt.savefig(path+'figures/horizontal_bar.png')
        print("*************************** plot_horizontal FINISHED")



#########################################################################
## obtain_message
##
## TCreates the message is going to be sent to the channel
##
## input:    data_dict
##           countries
##
## output:   message
##
#########################################################################
def obtain_message(data_dict, countries, path):
  #txtfile = open(path+"two_code_country.txt", 'r')
  #codeList = [line.split(' ') for line in txtfile.readlines()]
  #print(codeList)

  dcountry2code = {}
  country_code_list = []
  country_code2_list = []
  with open(path+"two_code_country.txt") as f:
    for line in f:
      (key, val) = line.split('/')
      country_code2_list.append(key)
      country_code_list.append(val)
      #dcountry2code[key] = val
    
    dcountry2code = {}

    i = 0
    for key in country_code_list:
        # Dictionary of a single country
        dcountry2code[key] = country_code2_list[i]
        i = i + 1
  '''   
  for k, v in dcountry2code.items():
    print(k, v)
  '''
  
  country_code_list = [x.replace('\n', '') for x in country_code_list]
  country_code_list = [x.replace('\t', '') for x in country_code_list]
  
  #print(country_code_list)
  OFFSET = 127462 - ord('A')

  def flag(code):
    code = code.upper()
    return chr(ord(code[0]) + OFFSET) + chr(ord(code[1]) + OFFSET)
    
  # obtain last spanish values from .txt
  last_info = []
  with open(path+"last_data.txt", 'r') as f:
    for elem in f:
      line = elem.split(',')
      f_name = line[0]
      f_cases = line[1].replace('x','').replace('\n','')
      f_death = line[2].replace('x','').replace('\n','')
      last_info.append([f_name, f_cases, f_death])
  #print(last_info)
  

      
     
  with open(path+"last_data.txt", "w") as text_file:
    for country in countries:
      text_file.write("{},{:x<6d},{:x<6d}\n".format(country,data_dict[country]['Cases'][-1],data_dict[country]['Deaths'][-1]))


  row_format = "{pais:<6s} | {casos:>8s} | {muertos:>7s}".format
  msg = "COVID INFO\nCountry    Cases    Deaths\n---------------------------\n"

  i = 1
  flag_code_list = []
  for country in countries:
    
    # Look for country last data
    cases_old = 0
    death_old = 0
    for i in range(0, len(countries)):
      if(country == last_info[i][0]):
        cases_old = last_info[i][1]
        death_old = last_info[i][2]
        
    # Search for the two-letter code of each country
    print(country)
    try:
      if len(country)>3:
        ind = country_code_list.index(country)
        flag_code = country_code2_list[ind]
      elif (country == "UK"):
        flag_code = "GB"
      else:
        flag_code = country
    except:
      flag_code = '-'
      print("NO code found")

    country_name = flag_code + " " + flag(flag_code)
    print(country_name)

    # Create message
    for key in data_dict[country]:
      if key == "Cases":
        cases = data_dict[country][key][-1]
        if (cases - int(cases_old) != 0):
          cases = str(cases) + "*"
        else:
          cases = str(cases) + " "
      elif key == "Deaths":
        deaths = data_dict[country][key][-1]
        if (deaths - int(death_old) != 0):
          deaths = str(deaths) + "*"
        else:
          deaths = str(deaths) + " "
    msg = msg + row_format(pais=country_name, casos=cases, muertos=deaths) + "\n"
    i = i+1

  message_markdown = "```" + msg + "```"
  return [msg, message_markdown]


#########################################################################
## 3D function - Requires PolyCollection
##              from mpl_toolkits.mplot3d import Axes3D
##
## Creates a 3D plot with accumulated cases
##
## input:    data_dict
##           countries
##
## output:   3d plot
##
#########################################################################
def global_contagios_3d(data_dict, countries, path):
    #print(data_dict['Spain']['Cases'])

    # Fixing random state for reproducibility
    def polygon_under_graph(xlist, ylist):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
        """
        return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make verts a list such that verts[i] is a list of (x, y) pairs defining
    # polygon i.
    verts = []

    # Set up the x sequence
    #xs = np.linspace(0, len(data_dict['Spain']['Cases'])-10)

    # The ith polygon will appear on the plane y = zs[i]
    zs = range(4)

    #print("PAISES: "+str(countries[3])+"  "+str(countries[2])+"  "+str(countries[1])+"  "+str(countries[0])+"  ")
    ys = [data_dict[countries[3]]['Cases'], data_dict[countries[2]]['Cases'], data_dict[countries[1]]['Cases'], data_dict[countries[0]]['Cases']]
    maxim_len = max(len(ys[0]), len(ys[1]), len(ys[2]), len(ys[3]))

    xs = np.arange(0, maxim_len)

    #print("ys", ys)
    ys_final = []
    for i in range(0, len(ys)):
        if len(ys[i]) < maxim_len:
            diff = maxim_len - len(ys[i])
            zer = [0]*diff
            ys_f = zer+ys[i]
            ys_final.append(list(ys_f))
            ys_f = []
        else:
            ys_final.append(list(ys[i]))

    for i in zs:
        verts.append(polygon_under_graph(xs, ys_final[i]))

    poly = PolyCollection(verts, facecolors=['r', 'g', 'b', 'y'], alpha=.6)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('Evolucion en dias')
    ax.set_ylabel(str(countries[3])+" "+str(countries[2])+" "+str(countries[1])+" "+str(countries[0]))
    ax.set_zlabel('Casos infectados')
    ax.set_xlim(0, maxim_len)
    ax.set_ylim(-1, 4)
    ax.set_zlim(0, max(ys_final[3]))
    ax.set_title("Casos acumulados [R,G,B,Y]     "+str(countries[3])+" "+str(countries[2])+" "+str(countries[1])+" "+str(countries[0]))
    #plt.show()
    plt.savefig(path+'figures/global_contagios_3d.png')
    print("*************************** global_contagios_3d FINISHED")



#########################################################################
## Stacked bar of deaths and cases
##
## Creates a double stacket plot of deaths and cases
##
## input:    data_dict
##           countries
##           number
##
## output:   stacket_plot_deaths
## output:   stacket_plot_cases
##
#########################################################################
def stacket_plot_deaths_and_cases(data_dict, countries, number, path):

    y = []
    labels = []
    for i in range(0, number-1):
        y.append(data_dict[countries[i]]['Deaths'])
        labels.append(countries[i])
    x = np.arange(0, len(y[0]))

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=labels)
    plt.xlabel('Dias')
    plt.ylabel('Muertes')
    plt.title('Evolucion muertes')
    ax.legend(loc='upper left')
    #plt.show()


    plt.savefig(path+'figures/stacked_deaths.png')
    print("*************************** stacked_deaths FINISHED")

    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window

    y = []
    labels = []
    for i in range(0, number-1):
        y.append(data_dict[countries[i]]['Cases'])
        labels.append(countries[i])
    x = np.arange(0, len(y[0]))

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=labels)
    plt.xlabel('Diass')
    plt.ylabel('Casos')
    plt.title('Evolucion casos')
    ax.legend(loc='upper left')
    #plt.show()


    plt.savefig(path+'figures/stacked_cases.png')
    print("*************************** stacked_cases FINISHED")

    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window


#########################################################################
## Ro contagious tax
##
## Creates an stacked plot with R0 of each country (cases, deaths)
##
## input:    data_dict
##           countries
##           number
##           Path
##
## output:   R0
##
#########################################################################
def evolution_R0(data_dict, countries, path):
    #plt.style.use('fivethirtyeight')
    #plt.style.context('Solarize_Light2')
    number=len(countries)-1

    ### Deaths ##
    #print(number)
    #print(countries)
    
    days_plotted = 15
    all_cases = []
    all_cases_sum = []
    maxv = 0
    minv = 2
    for j in range(0, number):
        cases = []
        for i in range(0, len(data_dict[countries[j]]['Deaths'])):
            cases1 = data_dict[countries[j]]['Deaths'][len(data_dict[countries[j]]['Deaths'])-1-i]
            cases2 = data_dict[countries[j]]['Deaths'][len(data_dict[countries[j]]['Deaths'])-2-i]

            if cases2 == 0:
                cases.append(0)
            else:
                cases.append(round(cases1/cases2, 3))

        cases.reverse()
        
        leng = len(data_dict[countries[number]]['Deaths'])
        if max(cases[leng-days_plotted-1:leng-1]) > maxv:
            maxv = max(cases[leng-days_plotted-1:leng-1])
        if min(cases[leng-days_plotted-1:leng-1]) < minv and min(cases[leng-days_plotted-1:leng-1]) != 0:
            minv = min(cases[leng-days_plotted-1:leng-1])
        all_cases.append(cases)

    cummulative = [0]*days_plotted
    for i in range(0, number):
        leng = len(data_dict[countries[number]]['Deaths'])
        second = data_dict[countries[number]]['Deaths'][leng-days_plotted-1:leng-1]
        cummulative = [x + y for x, y in zip(cummulative, second)]


    #Select how many days you want to see
    x = np.arange(0, days_plotted)
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(days_plotted)]
    date_list.reverse()

    fig, ax = plt.subplots()

    ax2 = ax.twinx()
    for i in range(0, len(all_cases)):
        ax2.plot(date_list, all_cases[i][len(all_cases[i])-days_plotted-1:len(all_cases[i])-1], label = countries[i])
        ax2.set_title("Tasa y acumulado de muertes")
        #print(countries[i])
        #print(all_cases[i][len(all_cases[i])-days_plotted-1:len(all_cases[i])-1])
        #print(" ")

    ax.fill_between(date_list, 0, cummulative, color='#BAD9FF', alpha=0.5)
    ax.set_ylabel("Muertes acumuladas")
    ax2.set_ylabel("Incremento")
    ax2.set_ylim(minv-0.1*minv, maxv+0.1*maxv)
    ax2.legend(loc='upper left')
    plt.gcf().autofmt_xdate()
    #plt.show()
    plt.savefig(path+'figures/ro_muertes.png')
    print("*************************** R0 Muertes FINISHED")

                             ### Cases ##

    days_plotted = 15
    all_cases = []
    all_cases_sum = []
    maxv = 0
    minv = 2
    for j in range(0, number):
        cases = []
        for i in range(0, len(data_dict[countries[j]]['Cases'])):
            cases1 = data_dict[countries[j]]['Cases'][len(data_dict[countries[j]]['Cases'])-1-i]
            cases2 = data_dict[countries[j]]['Cases'][len(data_dict[countries[j]]['Cases'])-2-i]

            if cases2 == 0:
                cases.append(0)
            else:
                cases.append(round(cases1/cases2, 3))

        cases.reverse()
        leng = len(data_dict[countries[number]]['Cases'])
        if max(cases[leng-days_plotted-1:leng-1]) > maxv:
            maxv = max(cases[leng-days_plotted-1:leng-1])
        if min(cases[leng-days_plotted-1:leng-1]) < minv and min(cases[leng-days_plotted-1:leng-1]) != 0:
            minv = min(cases[leng-days_plotted-1:leng-1])
        all_cases.append(cases)

    cummulative = [0]*days_plotted
    for i in range(0, number):
        leng = len(data_dict[countries[number]]['Cases'])
        second = data_dict[countries[number]]['Cases'][leng-days_plotted-1:leng-1]
        cummulative = [x + y for x, y in zip(cummulative, second)]


    #Select how many days you want to see
    x = np.arange(0, days_plotted)
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(days_plotted)]
    date_list.reverse()

    fig, ax = plt.subplots()

    ax2 = ax.twinx()
    for i in range(0, len(all_cases)):
        ax2.plot(date_list, all_cases[i][len(all_cases[i])-days_plotted-1:len(all_cases[i])-1], label = countries[i])
        ax2.set_title("Tasa y acumulado de casos")
        #print(countries[i])
        #print(all_cases[i][len(all_cases[i])-days_plotted-1:len(all_cases[i])-1])
        #print(" ")

    ax.fill_between(date_list, 0, cummulative, color='#BAD9FF', alpha=0.5)
    ax.set_ylabel("Casos acumulados")
    ax2.set_ylabel("Incremento")
    ax2.set_ylim(minv-0.1*minv, maxv+0.1*maxv)
    ax2.legend(loc='upper left')
    plt.gcf().autofmt_xdate()
    #plt.show()

    plt.savefig(path+'figures/ro_casos.png')
    print("*************************** R0 Casos FINISHED")



#### Execution space for testing ###

if False:
    print("Entering True")
    path = "Hello"
    number_of_countries = 11
    [data_dict, countries] = req_data(number_of_countries)

    for i in range(0, len(data_dict)):
        print(countries[i])
        print(data_dict[countries[i]]['status'])
    #stacket_plot_deaths_and_cases(data_dict, countries, 10, path)
    evolution_R0(data_dict, countries, 5, path)


