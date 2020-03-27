#########################################################################
##
## 	badfunctions.py
##
## 	27/03/2020
## 
#########################################################################

#########################################################################
## DATA EXTRACTION ######################################################
## req_data function is divided in 3 parts: 
## data extraction -> dict transformation -> update data appendix
##
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

    return [data_dict, countries]