from datetime import timedelta, datetime
import time
import requests
import os
import datetime
from datetime import timedelta, datetime
import urllib.request
import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
import numpy as np
import re

def read_files(fecha):

    StartTime = time.time()

    mydir = os.getcwd()

    fecha = datetime.strptime(fecha, "%d-%m-%Y").date()

    # #### creación archivos

    # ## Oferta Inicial

    # year = fecha.year
    # month = fecha.month
    # day = fecha.day

    # if fecha.month < 10:
    #     month = '0{}'.format(month)

    # if fecha.day < 10:
    #     day = '0{}'.format(day)

    # url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year,month,month,day)
    # response_oferta = urllib.request.urlopen(url_oferta)
    # data_oferta = response_oferta.read()

    # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/determinista/oferta.txt'), 'wb') as archivo:
    #     archivo.write(data_oferta)
    #     archivo.close()

    # ## MPO

    # url_MPO = 'http://www.xm.com.co/predespachoideal/{}-{}/iMAR{}{}_NAL.TXT'.format(year,month,month,day)
    # response_MPO = urllib.request.urlopen(url_MPO)
    # data_MPO = response_MPO.read()

    # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/determinista/MPO.txt'), 'wb') as archivo:
    #     archivo.write(data_MPO)
    #     archivo.close()

    # # AGC programado

    # url_AGC = 'http://www.xm.com.co/despachoprogramado/{}-{}/dAGC{}{}.TXT'.format(year,month,month,day)
    # response_AGC =  urllib.request.urlopen(url_AGC)
    # data_AGC = response_AGC.read()
    # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/determinista/AGC.txt'), 'wb') as archivo:
    #     archivo.write(data_AGC)
    #     archivo.close()

    # ## Pronostico Demanda

    # fechaD = fecha

    # while fechaD.weekday() != 0:
    #     fechaD = fechaD - timedelta(days=1)

    # yearD = fechaD.year
    # monthD = fechaD.month
    # dayD = fechaD.day

    # if monthD < 10:
    #     monthD = '0{}'.format(monthD)

    # if dayD < 10:
    #     dayD = '0{}'.format(dayD)

    # url_dem = 'http://www.xm.com.co/pronosticooficial/{}-{}/PRON_SIN{}{}.txt'.format(yearD,monthD,monthD,dayD)
    # response_dem = urllib.request.urlopen(url_dem)
    # data_dem = response_dem.read()

    # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/determinista/pronostico_dem.txt'), 'wb') as archivo:
    #     archivo.write(data_dem)
    #     archivo.close()

    #### Lectura de archivos

    agents_file = open('Casos_estudio/loc_size/ofe_dem/determinista/oferta.txt', encoding='utf8')
    agents_all_of_it = agents_file.read()
    agents_file.close()

    load_file = open('Casos_estudio/loc_size/ofe_dem/determinista/pronostico_dem.txt', encoding='utf8')
    load_all_of_it = load_file.read()
    load_file.close()

    MPO_file = open('Casos_estudio/loc_size/ofe_dem/determinista/MPO.txt', encoding='utf8')
    MPO_all_of_it = MPO_file.read()
    MPO_file.close()

    AGC_file = open('Casos_estudio/loc_size/ofe_dem/determinista/AGC.txt', encoding='utf8')
    AGC_all_of_it = AGC_file.read()
    AGC_file.close()

    ## costo de racionamiento

    # url_cro = 'http://www.upme.gov.co/CostosEnergia.asp'
    # page_cro = requests.get(url_cro)
    # print(page_cro)

    # soup_cro = BeautifulSoup(page_cro.content, 'lxml')

    # cro = {}
    # a = []

    # for link in soup_cro.find_all('td',{'width':'44%'}):
    #     cro[link.text] = 0

    # for link1 in soup_cro.find_all('td',{'align':'right'}):
    #     b = link1.text
    #     b = b.replace(',','')
    #     b = float(b.replace('.',''))/100
    #     a.append(b)

    # del cro['COSTO']

    # b = 0

    # for i in cro.keys():
    #     cro[i] = a[b]
    #     b += 1

    cro = {'CRO1': 1480.31,
        'CRO2': 2683.49,
        'CRO3': 4706.20,
        'CRO4': 9319.71}

    ReadingTime = time.time() - StartTime

    return agents_all_of_it, load_all_of_it, MPO_all_of_it, AGC_all_of_it, ReadingTime, fechaD, fecha, cro

def read_files_sc(fecha):

    agents_all = []
    load_all = []
    MPO_all = []
    AGC_all = []
    fecha_all = []
    fechaD_all = []

    for date in fecha:

        StartTime = time.time()

        mydir = os.getcwd()

        date = datetime.strptime(date, "%d-%m-%Y").date()

        # #### creación archivos

        # ## Oferta Inicial

        # year = date.year
        # month = date.month
        # day = date.day

        # if date.month < 10:
        #     month = '0{}'.format(month)

        # if date.day < 10:
        #     day = '0{}'.format(day)

        # url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year,month,month,day)
        # response_oferta = urllib.request.urlopen(url_oferta)
        # data_oferta = response_oferta.read()

        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/oferta_{}.txt'.format(date)), 'wb') as archivo:
        #     archivo.write(data_oferta)
        #     archivo.close()

        # ## MPO

        # url_MPO = 'http://www.xm.com.co/predespachoideal/{}-{}/iMAR{}{}_NAL.TXT'.format(year,month,month,day)
        # response_MPO = urllib.request.urlopen(url_MPO)
        # data_MPO = response_MPO.read()

        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/MPO_{}.txt'.format(date)), 'wb') as archivo:
        #     archivo.write(data_MPO)
        #     archivo.close()

        # # AGC programado

        # url_AGC = 'http://www.xm.com.co/despachoprogramado/{}-{}/dAGC{}{}.TXT'.format(year,month,month,day)
        # response_AGC =  urllib.request.urlopen(url_AGC)
        # data_AGC = response_AGC.read()
        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/AGC_{}.txt'.format(date)), 'wb') as archivo:
        #     archivo.write(data_AGC)
        #     archivo.close()

        # ## Pronostico Demanda

        # fechaD = date

        # while fechaD.weekday() != 0:
        #     fechaD = fechaD - timedelta(days=1)

        # yearD = fechaD.year
        # monthD = fechaD.month
        # dayD = fechaD.day

        # if monthD < 10:
        #     monthD = '0{}'.format(monthD)

        # if dayD < 10:
        #     dayD = '0{}'.format(dayD)

        # url_dem = 'http://www.xm.com.co/pronosticooficial/{}-{}/PRON_SIN{}{}.txt'.format(yearD,monthD,monthD,dayD)
        # response_dem = urllib.request.urlopen(url_dem)
        # data_dem = response_dem.read()

        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/pronostico_dem_{}.txt'.format(fechaD)), 'wb') as archivo:
        #     archivo.write(data_dem)
        #     archivo.close()

        #### Lectura de archivos

        agents_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/oferta_{}.txt'.format(date), encoding='utf8')
        agents_all_of_it = agents_file.read()
        agents_file.close()

        load_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/pronostico_dem_{}.txt'.format(fechaD), encoding='utf8')
        load_all_of_it = load_file.read()
        load_file.close()

        MPO_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/MPO_{}.txt'.format(date), encoding='utf8')
        MPO_all_of_it = MPO_file.read()
        MPO_file.close()

        AGC_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/AGC_{}.txt'.format(date), encoding='utf8')
        AGC_all_of_it = AGC_file.read()
        AGC_file.close()

        ## costo de racionamiento

        url_cro = 'http://www.upme.gov.co/CostosEnergia.asp'
        page_cro = requests.get(url_cro)

        soup_cro = BeautifulSoup(page_cro.content, 'lxml')

        cro = {}
        a = []

        for link in soup_cro.find_all('td',{'width':'44%'}):
            cro[link.text] = 0

        for link1 in soup_cro.find_all('td',{'align':'right'}):
            b = link1.text
            b = b.replace(',','')
            b = float(b.replace('.',''))/100
            a.append(b)

        del cro['COSTO']

        b = 0

        for i in cro.keys():
            cro[i] = a[b]
            b += 1

        agents_all.append(agents_all_of_it)
        load_all.append(load_all_of_it)
        MPO_all.append(MPO_all_of_it)
        AGC_all.append(MPO_all_of_it)
        fechaD_all.append(fechaD)
        fecha_all.append(date)

    ReadingTime = time.time() - StartTime

    return agents_all, load_all, MPO_all, AGC_all, ReadingTime, fechaD_all, fecha_all, cro

def organize_file_agents(agents_all_of_it, load_all_of_it, AGC_all_of_it, MPO_all_of_it, fechaD, fecha):

    StartTime = time.time()

    #### Organización archivos

    ## Oferta

    df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\n')])
    dic_OFEI = df_OFEI.to_dict('dict')

    none_val, agents_glb = list(dic_OFEI.items())[0]

    nul_val = []

    for key, value in agents_glb.items():
        if value == str(''):
            nul_val.append(key)

    for i in nul_val:
        del(agents_glb[i])

    #### Funciones para extraer diccionarios con cada componente de archivos globales

    ## Extracción strings

    def extract_str(element, agents_glb):
        idx = ', ' + str(element) + ', '
        dicc = {}
        for key, value in agents_glb.items():
            if value.find(idx) >= 0:
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')
        return dicc

    ## Extracción números

    def extract_num(element, agents_glb):
        idx = ', ' + str(element) + ', '
        dicc = {}
        for key, value in agents_glb.items():
            if value.find(idx) >= 0:
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]
        return dicc

    #### Extracción de componentes

    D = extract_num('D', agents_glb)
    P = extract_num('P', agents_glb)
    CONF = extract_num('CONF', agents_glb)
    C = extract_str('C', agents_glb)

    df_CONF = pd.DataFrame(CONF)
    df_CONF

    list_PAP = ['{} , PAP{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]
    list_PAPUSD = ['{} , PAPUSD{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]

    PAP = {}

    for idx in list_PAP:

        for key, value in agents_glb.items():

            if value.find(idx) >= 0:

                if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                    PAP[idx.split(' , ')[0]] = [int(value[value.find(idx)+len(idx)::].split(',')[1])]
    PAPUSD = {}

    for idx in list_PAPUSD:

        for key, value in agents_glb.items():

            if value.find(idx) >= 0:

                if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                    PAPUSD[idx.split(' , ')[0]] = [int(value[value.find(idx) + len(idx)::].split(',')[1])]

    MO = extract_num('MO', agents_glb)
    AGCP = extract_num('AGCP', agents_glb)
    AGCU = extract_num('AGCU', agents_glb)
    PRU = extract_num('PRU', agents_glb)
    CNA = extract_num('CNA', agents_glb)

    ## Pronostico Demanda

    days_pron = []

    for i in range(1,8):
        dates = str(fechaD + timedelta(days = i - 1))
        days_pron.append(dates)

    df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\n')])

    if fechaD.year >= 2020 and fechaD.month > 2:
        del df_PRON_DEM[0]
        df_PRON_DEM.columns -= 1
        for i in range(1, len(days_pron) + 1):
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)
    else:
        for i in range(1, len(days_pron) + 1):
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)

    del df_PRON_DEM['t']

    df_PRON_DEM = df_PRON_DEM.dropna()
    df_PRON_DEM.index += 1

    for i in range(1,8):
        dates = str(fechaD + timedelta(days = i - 1))
        df_PRON_DEM[str(dates)] = df_PRON_DEM[str(dates)].astype(float)

    df_PRON_DEM_fecha = df_PRON_DEM.loc[:,str(fechaD)]

    ## Despacho AGC programado

    df_AGC = pd.DataFrame([x.split(',') for x in AGC_all_of_it.split('\n')])

    ## MPO
    if fecha.year <= 2020 and fecha.month <= 11 and fecha.day <= 20:
        df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\n')]).dropna()
        df_MPO = df_MPO.drop([0], axis=1)
    else:
        df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\n')]).dropna()
        df_MPO = df_MPO.drop([0], axis=1)
        df_MPO.reset_index(inplace=True, drop=True)

    ## Demanda SIN

    actual_path = os.getcwd()
    file_path = os.path.join(actual_path, 'Casos_estudio/loc_size/dbDemandaReal/')

    if fecha.month <= 6:
        secon_idx = '1'
    else:
        secon_idx = '2'

    DemandaSIN_file = 'Demanda_Comercial_Por_Comercializador_{}SEME{}.xlsx'.format(fecha.year, secon_idx)

    df_demanda = pd.read_excel(os.path.join(file_path, DemandaSIN_file), sheet_name='Demanda_Comercial_Por_Comercial', header=2, index_col=0)

    df_demanda_fecha = df_demanda.loc[str(fecha)].replace(np.nan, 0)

    if fecha.year < 2020:
        df_demanda_fecha = df_demanda_fecha.drop(['Codigo Comercializador','Mercado','Version'], 1)
    else:
        df_demanda_fecha = df_demanda_fecha.drop(['Codigo Comercializador','Mercado','Versión'], 1)

    df_demanda_fecha.columns = [x+1 for x in range(24)]
    df_demanda_fecha = df_demanda_fecha.sum()

    df_demanda_fecha = df_demanda_fecha / 1000

    OrganizeTime = time.time() - StartTime

    return D, P, CONF, C, PAPUSD, PAP, MO, AGCP, AGCU, PRU, CNA, df_PRON_DEM_fecha, df_AGC, df_MPO, df_demanda_fecha, OrganizeTime

def organize_file_agents_sc(agents_all, load_all, AGC_all, MPO_all, fechaD, fecha):

    StartTime = time.time()

    D_all = []
    P_all = []
    CONF_all = []
    C_all = []
    PAPUSD_all = []
    PAP_all = []
    MO_all = []
    AGCP_all = []
    AGCU_all = []
    PRU_all = []
    CNA_all = []
    df_PRON_DEM_fecha_all = []
    df_AGC_all = []
    df_MPO_all = []
    df_demanda_fecha_all = []

    #### Organización archivos

    for agents_all_of_it in agents_all:

        ## Oferta

        df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\n')])
        dic_OFEI = df_OFEI.to_dict('dict')

        none_val, agents_glb = list(dic_OFEI.items())[0]

        nul_val = []

        for key, value in agents_glb.items():
            if value == str(''):
                nul_val.append(key)

        for i in nul_val:
            del(agents_glb[i])

        #### Funciones para extraer diccionarios con cada componente de archivos globales

        ## Extracción strings

        def extract_str(element, agents_glb):
            idx = ', ' + str(element) + ', '
            dicc = {}
            for key, value in agents_glb.items():
                if value.find(idx) >= 0:
                    dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')
            return dicc

        ## Extracción números

        def extract_num(element, agents_glb):
            idx = ', ' + str(element) + ', '
            dicc = {}
            for key, value in agents_glb.items():
                if value.find(idx) >= 0:
                    dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]
            return dicc

        #### Extracción de componentes

        D = extract_num('D', agents_glb)
        P = extract_num('P', agents_glb)
        CONF = extract_num('CONF', agents_glb)
        C = extract_str('C', agents_glb)

        df_CONF = pd.DataFrame(CONF)
        df_CONF

        list_PAP = ['{} , PAP{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]
        list_PAPUSD = ['{} , PAPUSD{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]

        PAP = {}

        for idx in list_PAP:

            for key, value in agents_glb.items():

                if value.find(idx) >= 0:

                    if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                        PAP[idx.split(' , ')[0]] = [int(value[value.find(idx)+len(idx)::].split(',')[1])]
        PAPUSD = {}

        for idx in list_PAPUSD:

            for key, value in agents_glb.items():

                if value.find(idx) >= 0:

                    if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                        PAPUSD[idx.split(' , ')[0]] = [int(value[value.find(idx) + len(idx)::].split(',')[1])]

        MO = extract_num('MO', agents_glb)
        AGCP = extract_num('AGCP', agents_glb)
        AGCU = extract_num('AGCU', agents_glb)
        PRU = extract_num('PRU', agents_glb)
        CNA = extract_num('CNA', agents_glb)

        D_all.append(D)
        P_all.append(P)
        CONF_all.append(CONF)
        C_all.append(C)
        PAP_all.append(PAP)
        PAPUSD_all.append(PAPUSD)
        MO_all.append(MO)
        AGCP_all.append(AGCP)
        AGCU_all.append(AGCU)
        PRU_all.append(PRU)
        CNA_all.append(CNA)

    ## Pronostico Demanda

    k = 0

    for load_all_of_it in load_all:

        days_pron = []

        for i in range(1,8):
            dates = str(fechaD[k] + timedelta(days = i - 1))
            days_pron.append(dates)

        df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\n')])

        if fechaD[k].year >= 2020 and fechaD[k].month > 2:
            del df_PRON_DEM[0]
            df_PRON_DEM.columns -= 1
            for i in range(1, len(days_pron) + 1):
                df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)
        else:
            for i in range(1, len(days_pron) + 1):
                df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)

        del df_PRON_DEM['t']

        df_PRON_DEM = df_PRON_DEM.dropna()
        df_PRON_DEM.index += 1

        for i in range(1,8):
            dates = str(fechaD[k] + timedelta(days = i - 1))
            df_PRON_DEM[str(dates)] = df_PRON_DEM[str(dates)].astype(float)

        df_PRON_DEM_fecha = df_PRON_DEM.loc[:,str(fechaD[k])]

        df_PRON_DEM_fecha_all.append(df_PRON_DEM_fecha)

        k += 1

    ## Despacho AGC programado

    for AGC_all_of_it in AGC_all:

        df_AGC = pd.DataFrame([x.split(',') for x in AGC_all_of_it.split('\n')])

        df_AGC_all.append(df_AGC)

    ## MPO

    k = 0

    for MPO_all_of_it in MPO_all:

        if fecha[k].year <= 2020 and fecha[k].month <= 11 and fecha[k].day <= 20:
            df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\n')]).dropna()
            df_MPO = df_MPO.drop([0], axis=1)
        else:
            df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\n')]).dropna()
            df_MPO = df_MPO.drop([0], axis=1)
            df_MPO.reset_index(inplace=True, drop=True)

        df_MPO_all.append(df_MPO)

        k += 1

    ## Demanda SIN

    for date in fecha:

        actual_path = os.getcwd()
        file_path = os.path.join(actual_path, 'Casos_estudio/loc_size/dbDemandaReal/')

        if date.month <= 6:
            secon_idx = '1'
        else:
            secon_idx = '2'

        DemandaSIN_file = 'Demanda_Comercial_Por_Comercializador_{}SEME{}.xlsx'.format(date.year, secon_idx)

        df_demanda = pd.read_excel(os.path.join(file_path, DemandaSIN_file), sheet_name='Demanda_Comercial_Por_Comercial', header=2, index_col=0)

        df_demanda_fecha = df_demanda.loc[str(date)].replace(np.nan, 0)

        if date.year < 2020:
            df_demanda_fecha = df_demanda_fecha.drop(['Codigo Comercializador','Mercado','Version'], 1)
        else:
            df_demanda_fecha = df_demanda_fecha.drop(['Codigo Comercializador','Mercado','Versión'], 1)

        df_demanda_fecha.columns = [x+1 for x in range(24)]
        df_demanda_fecha = df_demanda_fecha.sum()

        df_demanda_fecha = df_demanda_fecha / 1000

        df_demanda_fecha_all.append(df_demanda_fecha)

    OrganizeTime = time.time() - StartTime

    return D_all, P_all, CONF_all, C_all, PAPUSD_all, PAP_all, MO_all, AGCP_all, AGCU_all, PRU_all, CNA_all, df_PRON_DEM_fecha_all, df_AGC_all, df_MPO_all, df_demanda_fecha_all, OrganizeTime

def Desempate_ofertas(P):
    for p in P.keys():
        same_price = []
        for u in P.keys():
            if P[p][0] == P[u][0]:
                same_price.append(u)
        N = len(same_price)
        Delta_price = 0
        for n in range(N):
            oferente = random.choice(same_price)
            P[oferente][0] = P[oferente][0] + Delta_price
            Delta_price += 0.1
    return P

#### agrupación de información por unidades de generación

def data_process_unidades(D, P, PAP, MO, AGCU, PAPUSD, fecha):

    StartTime = time.time()

    G_Plants = []
    for u in D.keys():
        G_Plants.append(u)

    Generation_Units = []
    for u in D.keys():
        Generation_Units.append(u)

    # agrupación de unidades
    ALBAN = ['ANCHICAYA', 'BAJOANCHICAYA']
    GUATRON = ['GUADALUPE', 'TRONERAS']
    PAGUA = ['LAGUACA','PARAISO']
    TERMOVALLECC = ['TERMOVALLE1GAS', 'TERMOVALLE1VAPOR']
    TERMOSIERRACC = ['TERMOSIERRA1', 'TERMOSIERRA2', 'TERMOSIERRA3']
    TERMOCENTROCC = ['TERMOCENTRO1', 'TERMOCENTRO2', 'TERMOCENTRO3']
    TEBSABCC = ['TEBSA11', 'TEBSA12', 'TEBSA13', 'TEBSA14', 'TEBSA21', 'TEBSA22', 'TEBSA24']
    FLORESICC = ['FLORES1GAS', 'FLORES1VAPOR']
    FLORES4CC = ['FLORES2', 'FLORES3', 'FLORES4']
    TERMOEMCALICC = ['TERMOEMCALI1GAS', 'TERMOEMCALI1VAPOR']

    # unidades a ajustar por nombres similares a otras unidades
    settings_units = ['PRADO4', 'INGENIOSANCARLOS1', 'PORCEIIIMENOR', 'AUTOGARGOSSOGAMOSO', 'GUAVIOMENOR', 'URRAO', 'SANFRANCISCO(PUTUMAYO)']

    ############ Precios y disponibilidad por unidad ############

    for p in P.keys():
        if p == 'ALBAN':
            for c in ALBAN:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'GUATRON':
            for c in GUATRON:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'PAGUA':
            for c in PAGUA:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOVALLECC':
            for c in TERMOVALLECC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOSIERRACC':
            for c in TERMOSIERRACC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOCENTROCC':
            for c in TERMOCENTROCC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TEBSABCC':
            for c in TEBSABCC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'FLORESICC':
            for c in FLORESICC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'FLORES4CC':
            for c in FLORES4CC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOEMCALICC':
            for c in TERMOEMCALICC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        else:
            for u in Generation_Units:
                for s in settings_units:
                    if u in s:
                        index = Generation_Units.index(u)
                        G_Plants[index] = s
                    else:
                        if (p in u) and (u != 'URRAO'):
                            index = Generation_Units.index(u)
                            G_Plants[index] = p

    df_Unidades = pd.DataFrame()
    df_Unidades['P'] = Generation_Units
    Unidades = df_Unidades['P'].unique()

    #### Disponibilidad máxima por unidad

    Disponibilidad_unidad = {}
    Unidades = np.array(Generation_Units)

    for p in Generation_Units:
        Disponibilidad_unidad[p] = []
        for h in range(24):
            Disponibilidad_unidad[p].append(0)

    for p in Generation_Units:
        index_unit = np.where(Unidades == p)[0]
        for u in index_unit:
            Disponibilidad = D[Generation_Units[u]]
            for h in range(24):
                Disponibilidad_unidad[p][h] = Disponibilidad_unidad[p][h] + Disponibilidad[h]

    df_disponibilidad_unidad = pd.DataFrame()
    for p in Generation_Units:
        df_disponibilidad_unidad[p] = Disponibilidad_unidad[p]

    df_disponibilidad_unidad = df_disponibilidad_unidad.T
    df_disponibilidad_unidad.columns += 1

    #### Disponibilidad mínima por unidad

    time_disp = [i for i in range(24)]

    df_minop_unidad = pd.DataFrame(data=np.zeros((len(G_Plants),len(time_disp))), index=G_Plants, columns=time_disp)
    df_number_unit = df_minop_unidad.index.value_counts()

    for p in MO.keys():
        for u in G_Plants:
            if p == u:
                for t in range(24):
                    df_minop_unidad.loc[u,t] = round(MO[p][t] / df_number_unit.loc[u], 2)

    df_minop_unidad.index = Generation_Units

    for dm in df_disponibilidad_unidad.index:
        for t in range(24):
            if df_disponibilidad_unidad.loc[dm,t+1] == 0:
                df_minop_unidad.loc[dm,t] = 0

    df_minop_unidad.columns += 1

    #### Precios por unidad

    ## Precio de oferta por unidad

    df_oferta_unidad = pd.DataFrame(data=np.zeros(len(G_Plants)), index=G_Plants, columns=['Precio'])

    for p in P.keys():
        for u in G_Plants:
            if p == u:
                df_oferta_unidad.loc[u] = P[p][0]

    ## Precio de oferta de AGC por unidad

    df_oferta_unidad['P_AGC'] = 1000000

    for p in P.keys():
        for u in G_Plants:
            if p == u:
                df_oferta_unidad.loc[u,'P_AGC'] = P[p][0]


    ## Precio de oferta arranque y parada en dolares por unidad

    df_oferta_unidad['PAPUSD'] = 0

    for p in PAPUSD.keys():
        for u in G_Plants:
            if p == u:
                df_oferta_unidad.loc[u,'PAPUSD'] = PAPUSD[p][0]

    ## Precio de oferta arranque y parada en pesos colombianos por unidad

    df_oferta_unidad['PAP'] = 0

    for p in PAP.keys():
        for u in G_Plants:
            if p == u:
                df_oferta_unidad.loc[u,'PAP'] = PAP[p][0]

    df_oferta_unidad.index = Generation_Units

    G_Plants = list(set(G_Plants))

    ## Disponibilidad AGC por unidad

    df_AGCU = pd.DataFrame.from_dict(AGCU).T
    df_AGCU.columns += 1

    DataProcessTime = time.time() - StartTime

    return df_disponibilidad_unidad, df_minop_unidad, df_oferta_unidad, df_AGCU, DataProcessTime

#### agrupación de información por unidades de generación (escenarios)

def data_process_unidades_sc(D_all, P_all, PAP_all, MO_all, AGCU_all, PAPUSD_all, fecha):

    StartTime = time.time()

    df_disponibilidad_unidad_all = []
    df_minop_unidad_all = []
    df_oferta_unidad_all = []
    df_AGCU_all = []

    for i in range(len(D_all)):

        G_Plants = []
        for u in D_all[i].keys():
            G_Plants.append(u)

        Generation_Units = []
        for u in D_all[i].keys():
            Generation_Units.append(u)

        # agrupación de unidades
        ALBAN = ['ANCHICAYA', 'BAJOANCHICAYA']
        GUATRON = ['GUADALUPE', 'TRONERAS']
        PAGUA = ['LAGUACA','PARAISO']
        TERMOVALLECC = ['TERMOVALLE1GAS', 'TERMOVALLE1VAPOR']
        TERMOSIERRACC = ['TERMOSIERRA1', 'TERMOSIERRA2', 'TERMOSIERRA3']
        TERMOCENTROCC = ['TERMOCENTRO1', 'TERMOCENTRO2', 'TERMOCENTRO3']
        TEBSABCC = ['TEBSA11', 'TEBSA12', 'TEBSA13', 'TEBSA14', 'TEBSA21', 'TEBSA22', 'TEBSA24']
        FLORESICC = ['FLORES1GAS', 'FLORES1VAPOR']
        FLORES4CC = ['FLORES2', 'FLORES3', 'FLORES4']
        TERMOEMCALICC = ['TERMOEMCALI1GAS', 'TERMOEMCALI1VAPOR']

        # unidades a ajustar por nombres similares a otras unidades
        settings_units = ['PRADO4', 'INGENIOSANCARLOS1', 'PORCEIIIMENOR', 'AUTOGARGOSSOGAMOSO', 'GUAVIOMENOR', 'URRAO', 'SANFRANCISCO(PUTUMAYO)']

        ############ Precios y disponibilidad por unidad ############

        for p in P_all[i].keys():
            if p == 'ALBAN':
                for c in ALBAN:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'GUATRON':
                for c in GUATRON:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'PAGUA':
                for c in PAGUA:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOVALLECC':
                for c in TERMOVALLECC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOSIERRACC':
                for c in TERMOSIERRACC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOCENTROCC':
                for c in TERMOCENTROCC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TEBSABCC':
                for c in TEBSABCC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'FLORESICC':
                for c in FLORESICC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'FLORES4CC':
                for c in FLORES4CC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOEMCALICC':
                for c in TERMOEMCALICC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            else:
                for u in Generation_Units:
                    for s in settings_units:
                        if u in s:
                            index = Generation_Units.index(u)
                            G_Plants[index] = s
                        else:
                            if (p in u) and (u != 'URRAO'):
                                index = Generation_Units.index(u)
                                G_Plants[index] = p

        df_Unidades = pd.DataFrame()
        df_Unidades['P'] = Generation_Units
        Unidades = df_Unidades['P'].unique()

        #### Disponibilidad máxima por unidad

        Disponibilidad_unidad = {}
        Unidades = np.array(Generation_Units)

        for p in Generation_Units:
            Disponibilidad_unidad[p] = []
            for h in range(24):
                Disponibilidad_unidad[p].append(0)

        for p in Generation_Units:
            index_unit = np.where(Unidades == p)[0]
            for u in index_unit:
                Disponibilidad = D_all[i][Generation_Units[u]]
                for h in range(24):
                    Disponibilidad_unidad[p][h] = Disponibilidad_unidad[p][h] + Disponibilidad[h]

        df_disponibilidad_unidad = pd.DataFrame()
        for p in Generation_Units:
            df_disponibilidad_unidad[p] = Disponibilidad_unidad[p]

        df_disponibilidad_unidad = df_disponibilidad_unidad.T
        df_disponibilidad_unidad.columns += 1

        #### Disponibilidad mínima por unidad

        time_disp = [x for x in range(24)]

        df_minop_unidad = pd.DataFrame(data=np.zeros((len(G_Plants),len(time_disp))), index=G_Plants, columns=time_disp)
        df_number_unit = df_minop_unidad.index.value_counts()

        for p in MO_all[i].keys():
            for u in G_Plants:
                if p == u:
                    for t in range(24):
                        df_minop_unidad.loc[u,t] = round(MO_all[i][p][t] / df_number_unit.loc[u], 2)

        df_minop_unidad.index = Generation_Units

        for dm in df_disponibilidad_unidad.index:
            for t in range(24):
                if df_disponibilidad_unidad.loc[dm,t+1] == 0:
                    df_minop_unidad.loc[dm,t] = 0

        #### Precios por unidad

        ## Precio de oferta por unidad

        df_oferta_unidad = pd.DataFrame(data=np.zeros(len(G_Plants)), index=G_Plants, columns=['Precio'])

        for p in P_all[i].keys():
            for u in G_Plants:
                if p == u:
                    df_oferta_unidad.loc[u] = P_all[i][p][0]

        ## Precio de oferta de AGC por unidad

        df_oferta_unidad['P_AGC'] = 1000000

        for p in P_all[i].keys():
            for u in G_Plants:
                if p == u:
                    df_oferta_unidad.loc[u,'P_AGC'] = P_all[i][p][0]

        ## Precio de oferta arranque y parada en dolares por unidad

        df_oferta_unidad['PAPUSD'] = 0

        for p in PAPUSD_all[i].keys():
            for u in G_Plants:
                if p == u:
                    df_oferta_unidad.loc[u,'PAPUSD'] = PAPUSD_all[i][p][0]

        ## Precio de oferta arranque y parada en pesos colombianos por unidad

        df_oferta_unidad['PAP'] = 0

        for p in PAP_all[i].keys():
            for u in G_Plants:
                if p == u:
                    df_oferta_unidad.loc[u,'PAP'] = PAP_all[i][p][0]

        df_oferta_unidad.index = Generation_Units

        G_Plants = list(set(G_Plants))

        #### Disponibilidad AGC por unidad

        df_AGCU = pd.DataFrame.from_dict(AGCU_all[i]).T

        df_AGCU.columns += 1

        #### Indices dobles
        ## Disponibilidad máxima por unidad
        df_disponibilidad_unidad['sce'] = i+1
        df_disponibilidad_unidad['unidades'] = df_disponibilidad_unidad.index
        df_disponibilidad_unidad = df_disponibilidad_unidad.set_index(['unidades','sce'])

        ## Disponibilidad mínima por unidad
        df_minop_unidad['sce'] = i+1
        df_minop_unidad['unidades'] = df_minop_unidad.index
        df_minop_unidad = df_minop_unidad.set_index(['unidades','sce'])
        df_minop_unidad.columns += 1

        ## Precios por unidad
        df_oferta_unidad['sce'] = i+1
        df_oferta_unidad['unidades'] = df_oferta_unidad.index
        df_oferta_unidad = df_oferta_unidad.set_index(['unidades','sce'])

        ## Disponibilidad AGC por unidad
        df_AGCU['sce'] = i+1
        df_AGCU['unidades'] = df_AGCU.index
        df_AGCU = df_AGCU.set_index(['unidades','sce'])

        #### Listas con info de todos los escenarios

        df_disponibilidad_unidad_all.append(df_disponibilidad_unidad)
        df_minop_unidad_all.append(df_minop_unidad)
        df_oferta_unidad_all.append(df_oferta_unidad)
        df_AGCU_all.append(df_AGCU)

    DataProcessTime = time.time() - StartTime

    return df_disponibilidad_unidad_all, df_minop_unidad_all, df_oferta_unidad_all, df_AGCU_all, DataProcessTime

#### agrupación de información por datos de recursos real e ideal de XM

def data_process_RI(db_files, MO, PAP, PAPUSD, fecha):

    StartTime = time.time()

    ## Disponibilidad máxima

    fecha_pd = pd.to_datetime(fecha)
    quarter = str(fecha_pd.to_period('Q'))[-1]

    fecha = datetime.strptime(fecha, "%d-%m-%Y").date()

    if quarter == '1' or quarter == '2':
        real_idx = '1'
    else:
        real_idx = '2'

    if fecha.year >= 2016 and fecha.year <= 2017:

        DispComercial = 'dbDispComercial/Disponibilidad_Comercial_(kW)_{}_{}.xlsx'.format(fecha.year,real_idx)
        sheet_name = 'Disponibilidad_Comercial_(kW)_S'

    else:

        DispComercial = 'dbDispComercial/Disponibilidad_Comercial_(kW)_{}.xlsx'.format(fecha.year)
        sheet_name = 'Disponibilidad_Comercial_(kW)'

    df_DispComercial = pd.read_excel(os.path.join(db_files, DispComercial), sheet_name=sheet_name, header=0, index_col=0)

    df_DispComercial_fecha = df_DispComercial.loc[str(fecha)].replace(np.nan, 0)

    df_DispComercial_fecha.index = df_DispComercial_fecha.loc[:,'Recurso']

    if fecha.year < 2020:
        df_DispComercial_fecha = df_DispComercial_fecha.drop(['Recurso','Código Agente','Version'], axis=1)
    else:
        df_DispComercial_fecha = df_DispComercial_fecha.drop(['Recurso','Codigo Recurso','Código Agente','Version'], axis=1)

    df_DispComercial_fecha.columns = [x+1 for x in range(24)]

    df_DispComercial_fecha = df_DispComercial_fecha / 1e3

    ## Precios de oferta, oferta AGC, arranque/parada en pesos y dolares

    if fecha.year <= 2017:
        sheet_name = 'Precio_Oferta_($kWh).rdl'
    else:
        sheet_name = 'Precio_Oferta_(Valor_kWh)'

    PrecioOferta_file = 'dbPrecioOferta/Precio_Oferta_($kWh)_{}.xlsx'.format(fecha.year)

    df_oferta = pd.read_excel(os.path.join(db_files, PrecioOferta_file), sheet_name=sheet_name, header=0, index_col=0)

    df_oferta_fecha = df_oferta.loc[str(fecha)].replace(np.nan, 0)

    df_oferta_fecha.index = df_oferta_fecha.loc[:,'Recurso']
    df_oferta_fecha = df_oferta_fecha.drop(['Recurso','Código Agente'], axis=1)

    df_oferta_fecha_complete = pd.DataFrame(index=df_DispComercial_fecha.index, columns=['Precio','P_AGC','PAPUSD','PAP'])

    df_oferta_fecha_complete.loc[:,'Precio'] = 0
    df_oferta_fecha_complete.loc[:,'P_AGC'] = 1e6
    df_oferta_fecha_complete.loc[:,'PAPUSD'] = 0
    df_oferta_fecha_complete.loc[:,'PAP'] = 0

    for i in df_oferta_fecha_complete.index:
        for j in df_oferta_fecha.index:
            if i == j:
                df_oferta_fecha_complete.loc[i,'Precio'] = df_oferta_fecha.loc[j,'Precio de Oferta Ideal $/kWh'] * 1e3
                df_oferta_fecha_complete.loc[i,'P_AGC'] = df_oferta_fecha.loc[j,'Precio de Oferta Ideal $/kWh'] * 1e3

    for i in df_oferta_fecha_complete.index:
        for j in PAP.keys():
            if i.replace(' ','') == j:
                df_oferta_fecha_complete.loc[i,'PAPUSD'] = PAPUSD[i.replace(' ','')][0]
                df_oferta_fecha_complete.loc[i,'PAP'] = PAP[i.replace(' ','')][0]

    ## minimos operativos

    df_minop = pd.DataFrame(index=df_DispComercial_fecha.index, columns=[x+1 for x in range(24)]).fillna(0.)

    for mi in MO.keys():
        for i in df_minop.index:
            if i.replace(' ','') == mi:
                for t in df_minop.columns:
                    df_minop.loc[i,t] = MO[i.replace(' ','')][t-1]

    DataProcessTime = time.time() - StartTime

    return df_DispComercial_fecha, df_minop, df_oferta_fecha_complete, DataProcessTime

    #### agrupación de información por plantas de generación

#### agrupación de información por datos de recursos real e ideal de XM (escenarios)

def data_process_RI_sc(db_files, MO_all, PAP_all, PAPUSD_all, fecha):

    StartTime = time.time()

    df_DispComercial_fecha_all = []
    df_minop_all = []
    df_oferta_fecha_complete_all = []

    for k in range(len(MO_all)):

        ## Disponibilidad máxima

        fecha_ = fecha[k].strftime('%d-%m-%Y')
        fecha_pd = pd.to_datetime(fecha_)
        quarter = str(fecha_pd.to_period('Q'))[-1]

        fecha_ = datetime.strptime(fecha_, "%d-%m-%Y").date()

        if quarter == '1' or quarter == '2':
            real_idx = '1'
        else:
            real_idx = '2'

        if fecha_.year >= 2016 and fecha_.year <= 2017:

            DispComercial = 'dbDispComercial/Disponibilidad_Comercial_(kW)_{}_{}.xlsx'.format(fecha_.year,real_idx)
            sheet_name = 'Disponibilidad_Comercial_(kW)_S'

        else:

            DispComercial = 'dbDispComercial/Disponibilidad_Comercial_(kW)_{}.xlsx'.format(fecha_.year)
            sheet_name = 'Disponibilidad_Comercial_(kW)'

        df_DispComercial = pd.read_excel(os.path.join(db_files, DispComercial), sheet_name=sheet_name, header=0, index_col=0)

        df_DispComercial_fecha = df_DispComercial.loc[str(fecha_)].replace(np.nan, 0)

        df_DispComercial_fecha.index = df_DispComercial_fecha.loc[:,'Recurso']

        if fecha_.year < 2020:
            df_DispComercial_fecha = df_DispComercial_fecha.drop(['Recurso','Código Agente','Version'], axis=1)
        else:
            df_DispComercial_fecha = df_DispComercial_fecha.drop(['Recurso','Codigo Recurso','Código Agente','Version'], axis=1)

        df_DispComercial_fecha.columns = [x+1 for x in range(24)]

        df_DispComercial_fecha = df_DispComercial_fecha / 1e3

        ## Precios de oferta, oferta AGC, arranque/parada en pesos y dolares

        if fecha_.year <= 2017:
            sheet_name = 'Precio_Oferta_($kWh).rdl'
        else:
            sheet_name = 'Precio_Oferta_(Valor_kWh)'

        PrecioOferta_file = 'dbPrecioOferta/Precio_Oferta_($kWh)_{}.xlsx'.format(fecha_.year)

        df_oferta = pd.read_excel(os.path.join(db_files, PrecioOferta_file), sheet_name=sheet_name, header=0, index_col=0)

        df_oferta_fecha = df_oferta.loc[str(fecha_)].replace(np.nan, 0)

        df_oferta_fecha.index = df_oferta_fecha.loc[:,'Recurso']
        df_oferta_fecha = df_oferta_fecha.drop(['Recurso','Código Agente'], axis=1)

        df_oferta_fecha_complete = pd.DataFrame(index=df_DispComercial_fecha.index, columns=['Precio','P_AGC','PAPUSD','PAP'])

        df_oferta_fecha_complete.loc[:,'Precio'] = 0
        df_oferta_fecha_complete.loc[:,'P_AGC'] = 1e6
        df_oferta_fecha_complete.loc[:,'PAPUSD'] = 0
        df_oferta_fecha_complete.loc[:,'PAP'] = 0

        for i in df_oferta_fecha_complete.index:
            for j in df_oferta_fecha.index:
                if i == j:
                    df_oferta_fecha_complete.loc[i,'Precio'] = df_oferta_fecha.loc[j,'Precio de Oferta Ideal $/kWh'] * 1e3
                    df_oferta_fecha_complete.loc[i,'P_AGC'] = df_oferta_fecha.loc[j,'Precio de Oferta Ideal $/kWh'] * 1e3

        for i in df_oferta_fecha_complete.index:
            for j in PAP_all[k].keys():
                if i.replace(' ','') == j:
                    df_oferta_fecha_complete.loc[i,'PAPUSD'] = PAPUSD_all[k][i.replace(' ','')][0]
                    df_oferta_fecha_complete.loc[i,'PAP'] = PAP_all[k][i.replace(' ','')][0]

        ## minimos operativos

        df_minop = pd.DataFrame(index=df_DispComercial_fecha.index, columns=[x+1 for x in range(24)]).fillna(0.)

        for mi in MO_all[k].keys():
            for i in df_minop.index:
                if i.replace(' ','') == mi:
                    for t in df_minop.columns:
                        df_minop.loc[i,t] = MO_all[k][i.replace(' ','')][t-1]

        #### Indice doble

        ## Disponibilidad máxima

        df_DispComercial_fecha['sce'] = k+1
        df_DispComercial_fecha['plantas'] = df_DispComercial_fecha.index
        df_DispComercial_fecha = df_DispComercial_fecha.set_index(['plantas','sce'])

        ## Mínimos operativos

        df_minop['sce'] = k+1
        df_minop['plantas'] = df_minop.index
        df_minop = df_minop.set_index(['plantas','sce'])

        ## Precios de oferta, oferta AGC, arranque/parada en pesos y dolares

        df_oferta_fecha_complete['sce'] = k+1
        df_oferta_fecha_complete['plantas'] = df_oferta_fecha_complete.index
        df_oferta_fecha_complete = df_oferta_fecha_complete.set_index(['plantas','sce'])

        #### Listas con info de todos los escenarios

        df_DispComercial_fecha_all.append(df_DispComercial_fecha)
        df_minop_all.append(df_minop)
        df_oferta_fecha_complete_all.append(df_oferta_fecha_complete)

    DataProcessTime = time.time() - StartTime

    return df_DispComercial_fecha_all, df_minop_all, df_oferta_fecha_complete_all, DataProcessTime

    #### agrupación de información por plantas de generación

#### agrupación de información por plantas de generación

def data_process_planta(D, P, PAP, MO, AGCP, PAPUSD, fecha):

    StartTime = time.time()

    G_Plants = []
    for u in D.keys():
        G_Plants.append(u)

    Generation_Units = []
    for u in D.keys():
        Generation_Units.append(u)

    ALBAN = ['ANCHICAYA', 'BAJOANCHICAYA']
    GUATRON = ['GUADALUPE', 'TRONERAS']
    PAGUA = ['LAGUACA','PARAISO']
    TERMOVALLECC = ['TERMOVALLE1GAS', 'TERMOVALLE1VAPOR']
    TERMOSIERRACC = ['TERMOSIERRA1', 'TERMOSIERRA2', 'TERMOSIERRA3']
    TERMOCENTROCC = ['TERMOCENTRO1', 'TERMOCENTRO2', 'TERMOCENTRO3']
    TEBSABCC = ['TEBSA11', 'TEBSA12', 'TEBSA13', 'TEBSA14', 'TEBSA21', 'TEBSA22', 'TEBSA24']
    FLORESICC = ['FLORES1GAS', 'FLORES1VAPOR']
    FLORES4CC = ['FLORES2', 'FLORES3', 'FLORES4']
    TERMOEMCALICC = ['TERMOEMCALI1GAS', 'TERMOEMCALI1VAPOR']

    # ajustes de unidades
    settings_units = ['PRADO4', 'INGENIOSANCARLOS1', 'PORCEIIIMENOR', 'AUTOGARGOSSOGAMOSO', 'GUAVIOMENOR', 'URRAO', 'SANFRANCISCO(PUTUMAYO)']

    ############ Precios y disponibilidad por planta ############

    for p in P.keys():
        if p == 'ALBAN':
            for c in ALBAN:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'GUATRON':
            for c in GUATRON:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'PAGUA':
            for c in PAGUA:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOVALLECC':
            for c in TERMOVALLECC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOSIERRACC':
            for c in TERMOSIERRACC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOCENTROCC':
            for c in TERMOCENTROCC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TEBSABCC':
            for c in TEBSABCC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'FLORESICC':
            for c in FLORESICC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'FLORES4CC':
            for c in FLORES4CC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        elif p == 'TERMOEMCALICC':
            for c in TERMOEMCALICC:
                for u in Generation_Units:
                    if c in u:
                        index = Generation_Units.index(u)
                        G_Plants[index] = p
        else:
            for u in Generation_Units:
                for s in settings_units:
                    if u in s:
                        index = Generation_Units.index(u)
                        G_Plants[index] = s
                    else:
                        if (p in u) and (u != 'URRAO'):
                            index = Generation_Units.index(u)
                            G_Plants[index] = p

    df_Plantas = pd.DataFrame()
    df_Plantas['P'] = G_Plants
    Planta = df_Plantas['P'].unique()

    Generation_plants = []
    for p in Planta:
        Generation_plants.append(p)

    #### Disponibilidad máxima por planta

    Disponibilidad_planta = {}
    Unidades = np.array(G_Plants)

    for p in Generation_plants:
        Disponibilidad_planta[p] = []
        for h in range(24):
            Disponibilidad_planta[p].append(0)

    for p in Generation_plants:
        index_plant = np.where(Unidades == p)[0]
        for u in index_plant:
            Disponibilidad = D[Generation_Units[u]]
            for h in range(24):
                Disponibilidad_planta[p][h] = Disponibilidad_planta[p][h] + Disponibilidad[h]

    df_disponibilidad_planta = pd.DataFrame()
    for p in Generation_plants:
        if p == 'INGENIOSANCARLOS1':
            df_disponibilidad_planta[p] = D[p]
        else:
            df_disponibilidad_planta[p] = Disponibilidad_planta[p]

    df_disponibilidad_planta = df_disponibilidad_planta.T

    df_disponibilidad_planta.columns += 1

    #### Disponibilidad mínima por planta

    Min_operativos = {}

    for p in Generation_plants:
        Min_operativos[p] = []
        for h in range(24):
            Min_operativos[p].append(0)

    for p in MO.keys():
        for u in Generation_plants:
            if p == u:
                Min_operativos[u] = MO[p]

    df_minop_planta = pd.DataFrame()
    for p in Generation_plants:
        df_minop_planta[p] = Min_operativos[p]

    df_minop_planta = df_minop_planta.T

    for dm in df_disponibilidad_planta.index:
        for t in range(24):
            if df_disponibilidad_planta.loc[dm,t+1] == 0:
                df_minop_planta.loc[dm,t] = 0

    df_minop_planta.columns += 1

    #### Precios por planta

    ## Precio de oferta por planta

    Generation_Plants_Price = {}
    Index_generation_Plants = []

    Index_generation_Plants.append('Precio')

    for p in Generation_plants:
        Generation_Plants_Price[p] = [0]

    for p in P.keys():
        for u in Generation_plants:
            if p == u:
                Generation_Plants_Price[u][0] = P[p][0]

    ## Precio de oferta de AGC por planta

    Index_generation_Plants.append('P_AGC')

    for p in Generation_plants:
        Generation_Plants_Price[p].append(1000000)

    for p in P.keys():
        for u in Generation_plants:
            if p == u:
                Generation_Plants_Price[u][1] = P[p][0]

    ## Precio de oferta arranque y parada en dolares por planta

    Index_generation_Plants.append('PAPUSD')

    for p in Generation_plants:
        Generation_Plants_Price[p].append(0)

    for p in PAPUSD.keys():
        for u in Generation_plants:
            if p == u:
                Generation_Plants_Price[u][2] = PAPUSD[p][0]

    ## Precio de oferta arranque y parada en pesos colombianos por planta

    Index_generation_Plants.append('PAP')

    for p in Generation_plants:
        Generation_Plants_Price[p].append(0)

    for p in PAP.keys():
        for u in Generation_plants:
            if p == u:
                Generation_Plants_Price[u][3] = PAP[p][0]

    df_oferta_planta = pd.DataFrame.from_dict(Index_generation_Plants)

    for p in Generation_plants:
        df_oferta_planta[p] = Generation_Plants_Price[p]

    df_oferta_planta = df_oferta_planta.T
    df_oferta_planta.columns = df_oferta_planta.loc[0,:]
    df_oferta_planta = df_oferta_planta.drop([0], axis=0)

    G_Plants = list(set(G_Plants))

    ## Disponibilidad de AGC por planta

    df_AGCP = pd.DataFrame.from_dict(AGCP).T
    df_AGCP.columns += 1

    DataProcessTime = time.time() - StartTime

    return df_disponibilidad_planta, df_minop_planta, df_oferta_planta, df_AGCP, DataProcessTime

#### agrupación de información por plantas de generación (escenarios)

def data_process_planta_sc(D_all, P_all, PAP_all, MO_all, AGCP_all, PAPUSD_all, fecha):

    StartTime = time.time()

    df_disponibilidad_planta_all = []
    df_minop_planta_all = []
    df_oferta_planta_all = []
    df_AGCP_all = []

    for i in range(len(D_all)):

        G_Plants = []
        for u in D_all[i].keys():
            G_Plants.append(u)

        Generation_Units = []
        for u in D_all[i].keys():
            Generation_Units.append(u)

        ALBAN = ['ANCHICAYA', 'BAJOANCHICAYA']
        GUATRON = ['GUADALUPE', 'TRONERAS']
        PAGUA = ['LAGUACA','PARAISO']
        TERMOVALLECC = ['TERMOVALLE1GAS', 'TERMOVALLE1VAPOR']
        TERMOSIERRACC = ['TERMOSIERRA1', 'TERMOSIERRA2', 'TERMOSIERRA3']
        TERMOCENTROCC = ['TERMOCENTRO1', 'TERMOCENTRO2', 'TERMOCENTRO3']
        TEBSABCC = ['TEBSA11', 'TEBSA12', 'TEBSA13', 'TEBSA14', 'TEBSA21', 'TEBSA22', 'TEBSA24']
        FLORESICC = ['FLORES1GAS', 'FLORES1VAPOR']
        FLORES4CC = ['FLORES2', 'FLORES3', 'FLORES4']
        TERMOEMCALICC = ['TERMOEMCALI1GAS', 'TERMOEMCALI1VAPOR']

        # ajustes de unidades
        settings_units = ['PRADO4', 'INGENIOSANCARLOS1', 'PORCEIIIMENOR', 'AUTOGARGOSSOGAMOSO', 'GUAVIOMENOR', 'URRAO', 'SANFRANCISCO(PUTUMAYO)']

        ############ Precios y disponibilidad por planta ############

        for p in P_all[i].keys():
            if p == 'ALBAN':
                for c in ALBAN:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'GUATRON':
                for c in GUATRON:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'PAGUA':
                for c in PAGUA:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOVALLECC':
                for c in TERMOVALLECC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOSIERRACC':
                for c in TERMOSIERRACC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOCENTROCC':
                for c in TERMOCENTROCC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TEBSABCC':
                for c in TEBSABCC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'FLORESICC':
                for c in FLORESICC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'FLORES4CC':
                for c in FLORES4CC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            elif p == 'TERMOEMCALICC':
                for c in TERMOEMCALICC:
                    for u in Generation_Units:
                        if c in u:
                            index = Generation_Units.index(u)
                            G_Plants[index] = p
            else:
                for u in Generation_Units:
                    for s in settings_units:
                        if u in s:
                            index = Generation_Units.index(u)
                            G_Plants[index] = s
                        else:
                            if (p in u) and (u != 'URRAO'):
                                index = Generation_Units.index(u)
                                G_Plants[index] = p

        df_Plantas = pd.DataFrame()
        df_Plantas['P'] = G_Plants
        Planta = df_Plantas['P'].unique()

        Generation_plants = []
        for p in Planta:
            Generation_plants.append(p)

        #### Disponibilidad máxima por planta

        Disponibilidad_planta = {}
        Unidades = np.array(G_Plants)

        for p in Generation_plants:
            Disponibilidad_planta[p] = []
            for h in range(24):
                Disponibilidad_planta[p].append(0)

        for p in Generation_plants:
            index_plant = np.where(Unidades == p)[0]
            for u in index_plant:
                Disponibilidad = D_all[i][Generation_Units[u]]
                for h in range(24):
                    Disponibilidad_planta[p][h] = Disponibilidad_planta[p][h] + Disponibilidad[h]

        df_disponibilidad_planta = pd.DataFrame()
        for p in Generation_plants:
            if p == 'INGENIOSANCARLOS1':
                df_disponibilidad_planta[p] = D_all[i][p]
            else:
                df_disponibilidad_planta[p] = Disponibilidad_planta[p]

        df_disponibilidad_planta = df_disponibilidad_planta.T

        df_disponibilidad_planta.columns += 1

        #### Disponibilidad mínima por planta

        Min_operativos = {}

        for p in Generation_plants:
            Min_operativos[p] = []
            for h in range(24):
                Min_operativos[p].append(0)

        for p in MO_all[i].keys():
            for u in Generation_plants:
                if p == u:
                    Min_operativos[u] = MO_all[i][p]

        df_minop_planta = pd.DataFrame()
        for p in Generation_plants:
            df_minop_planta[p] = Min_operativos[p]

        df_minop_planta = df_minop_planta.T

        for dm in df_disponibilidad_planta.index:
            for t in range(24):
                if df_disponibilidad_planta.loc[dm,t+1] == 0:
                    df_minop_planta.loc[dm,t] = 0

        df_minop_planta.columns += 1

        #### Precios por planta

        ## Precio de oferta por planta

        Generation_Plants_Price = {}
        Index_generation_Plants = []

        Index_generation_Plants.append('Precio')

        for p in Generation_plants:
            Generation_Plants_Price[p] = [0]

        for p in P_all[i].keys():
            for u in Generation_plants:
                if p == u:
                    Generation_Plants_Price[u][0] = P_all[i][p][0]

        ## Precio de oferta de AGC por planta

        Index_generation_Plants.append('P_AGC')

        for p in Generation_plants:
            Generation_Plants_Price[p].append(1000000)

        for p in P_all[i].keys():
            for u in Generation_plants:
                if p == u:
                    Generation_Plants_Price[u][1] = P_all[i][p][0]

        ## Precio de oferta arranque y parada en dolares por planta

        Index_generation_Plants.append('PAPUSD')

        for p in Generation_plants:
            Generation_Plants_Price[p].append(0)

        for p in PAPUSD_all[i].keys():
            for u in Generation_plants:
                if p == u:
                    Generation_Plants_Price[u][2] = PAPUSD_all[i][p][0]

        ## Precio de oferta arranque y parada en pesos colombianos por planta

        Index_generation_Plants.append('PAP')

        for p in Generation_plants:
            Generation_Plants_Price[p].append(0)

        for p in PAP_all[i].keys():
            for u in Generation_plants:
                if p == u:
                    Generation_Plants_Price[u][3] = PAP_all[i][p][0]

        df_oferta_planta = pd.DataFrame.from_dict(Index_generation_Plants)

        for p in Generation_plants:
            df_oferta_planta[p] = Generation_Plants_Price[p]

        df_oferta_planta = df_oferta_planta.T
        df_oferta_planta.columns = df_oferta_planta.loc[0,:]
        df_oferta_planta = df_oferta_planta.drop([0], axis=0)

        G_Plants = list(set(G_Plants))

        ## Disponibilidad de AGC por planta

        df_AGCP = pd.DataFrame.from_dict(AGCP_all[i]).T
        df_AGCP.columns += 1

        #### Indices dobles
        ## Disponibilidad máxima por planta
        df_disponibilidad_planta['sce'] = i+1
        df_disponibilidad_planta['plantas'] = df_disponibilidad_planta.index
        df_disponibilidad_planta = df_disponibilidad_planta.set_index(['plantas','sce'])

        ## Disponibilidad mínima por planta
        df_minop_planta['sce'] = i+1
        df_minop_planta['plantas'] = df_minop_planta.index
        df_minop_planta = df_minop_planta.set_index(['plantas','sce'])

        ## Precios por planta
        df_oferta_planta['sce'] = i+1
        df_oferta_planta['plantas'] = df_oferta_planta.index
        df_oferta_planta = df_oferta_planta.set_index(['plantas','sce'])

        ## Disponibilidad de AGC por planta
        df_AGCP['sce'] = i+1
        df_AGCP['plantas'] = df_AGCP.index
        df_AGCP = df_AGCP.set_index(['plantas','sce'])

        #### Listas con info de todos los escenarios
        df_disponibilidad_planta_all.append(df_disponibilidad_planta)
        df_minop_planta_all.append(df_minop_planta)
        df_oferta_planta_all.append(df_oferta_planta)
        df_AGCP_all.append(df_AGCP)

    DataProcessTime = time.time() - StartTime

    return df_disponibilidad_planta_all, df_minop_planta_all, df_oferta_planta_all, df_AGCP_all, DataProcessTime

## Precios de arranque y parada

def read_files_PAP(fecha):

    agents_all = []
    AGC_all = []

    for date in fecha:

        StartTime = time.time()

        mydir = os.getcwd()

        date = datetime.strptime(date, "%d-%m-%Y").date()

        # #### creación archivos

        # ## Oferta Inicial

        # year = date.year
        # month = date.month
        # day = date.day

        # if date.month < 10:
        #     month = '0{}'.format(month)

        # if date.day < 10:
        #     day = '0{}'.format(day)

        # url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year,month,month,day)
        # response_oferta = urllib.request.urlopen(url_oferta)
        # data_oferta = response_oferta.read()

        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/oferta_{}.txt'.format(date)), 'wb') as archivo:
        #     archivo.write(data_oferta)
        #     archivo.close()

        # # AGC programado

        # url_AGC = 'http://www.xm.com.co/despachoprogramado/{}-{}/dAGC{}{}.TXT'.format(year,month,month,day)
        # response_AGC =  urllib.request.urlopen(url_AGC)
        # data_AGC = response_AGC.read()

        # with open(os.path.join(mydir, 'Casos_estudio/loc_size/ofe_dem/escenarios/AGC_{}.txt'.format(date)), 'wb') as archivo:
        #     archivo.write(data_AGC)
        #     archivo.close()

        #### Lectura de archivos

        agents_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/oferta_{}.txt'.format(date), encoding='utf8')
        agents_all_of_it = agents_file.read()
        agents_file.close()

        AGC_file = open('Casos_estudio/loc_size/ofe_dem/escenarios/AGC_{}.txt'.format(date), encoding='utf8')
        AGC_all_of_it = AGC_file.read()
        AGC_file.close()

        agents_all.append(agents_all_of_it)
        AGC_all.append(AGC_all_of_it)

    ReadingTime = time.time() - StartTime

    return agents_all, AGC_all

## 

def organize_file_agents_PAP_MO(agents_all, AGC_all):

    StartTime = time.time()

    PAPUSD_all = []
    PAP_all = []
    MO_all = []

    #### Organización archivos

    for agents_all_of_it in agents_all:

        ## Oferta

        df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\n')])
        dic_OFEI = df_OFEI.to_dict('dict')

        none_val, agents_glb = list(dic_OFEI.items())[0]

        nul_val = []

        for key, value in agents_glb.items():
            if value == str(''):
                nul_val.append(key)

        for i in nul_val:
            del(agents_glb[i])

        #### Funciones para extraer diccionarios con cada componente de archivos globales

        ## Extracción strings

        def extract_str(element, agents_glb):
            idx = ', ' + str(element) + ', '
            dicc = {}
            for key, value in agents_glb.items():
                if value.find(idx) >= 0:
                    dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')
            return dicc

        ## Extracción números

        def extract_num(element, agents_glb):
            idx = ', ' + str(element) + ', '
            dicc = {}
            for key, value in agents_glb.items():
                if value.find(idx) >= 0:
                    dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]
            return dicc

        #### Extracción de componentes

        CONF = extract_num('CONF', agents_glb)
        MO = extract_num('MO', agents_glb)

        df_CONF = pd.DataFrame(CONF)

        list_PAP = ['{} , PAP{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]
        list_PAPUSD = ['{} , PAPUSD{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]

        PAP = {}

        for idx in list_PAP:

            for key, value in agents_glb.items():

                if value.find(idx) >= 0:

                    if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                        PAP[idx.split(' , ')[0]] = [int(value[value.find(idx)+len(idx)::].split(',')[1])]
        PAPUSD = {}

        for idx in list_PAPUSD:

            for key, value in agents_glb.items():

                if value.find(idx) >= 0:

                    if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                        PAPUSD[idx.split(' , ')[0]] = [int(value[value.find(idx) + len(idx)::].split(',')[1])]

        PAP_all.append(PAP)
        PAPUSD_all.append(PAPUSD)
        MO_all.append(MO)

    ## Holgura

    for AGC_all_of_it in AGC_all:

        df_AGC = pd.DataFrame([x.split(',') for x in AGC_all_of_it.split('\n')])
        df_AGC = df_AGC.set_index([0])
        df_AGC = df_AGC.loc['"Total"',:].astype(float)

    OrganizeTime = time.time() - StartTime

    return PAPUSD_all, PAP_all, MO_all, df_AGC, OrganizeTime
