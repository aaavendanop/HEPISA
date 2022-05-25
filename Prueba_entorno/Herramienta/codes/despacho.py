from pyomo.environ import *
import numpy as np
import pandas as pd
import math
import os
import time
import datetime
from datetime import timedelta, datetime
import urllib.request

import matplotlib.pyplot as plt

def pyomo_df(element):

    data = value(element)
    df_data = pd.DataFrame(data, index=['1','2'], columns=['1'])
    df_data = df_data.drop(['2'], axis=0)
    df_data = df_data.rename(columns={'1':fecha[0]})
    return df_data

def pyomo1_df(element):

    data = {i: value(v) for i, v in element.items()}
    keys = data.items()
    idx = pd.MultiIndex.from_tuples(keys)
    df_data = pd.DataFrame(data, index=[0]).stack().loc[0]
    return df_data

def pyomo2_df(element):

    if len(element) != 0:
        data = {(i, j): value(v) for (i, j), v in element.items()}
        keys, values = zip(*data.items())
        idx = pd.MultiIndex.from_tuples(keys)
        df_data = pd.DataFrame(data, index=[0], ).stack().loc[0]
        df_data = df_data.rename(columns={1:fecha[0]})
    else:
        df_data = 0
    return df_data

def descarga_archivos(fecha):

    mydir = os.getcwd()
    aux = mydir.split('/')
    last_element = aux[-1]
    aux.remove(last_element)
    before_path = '/'.join(aux)

    day1 = int(fecha.day)
    month1 = int(fecha.month)
    year1 = int(fecha.year)

    if month1 < 10:
        month1 = '0{}'.format(month1)

    if day1 < 10:
        day1 = '0{}'.format(day1)

    url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year1,month1,month1,day1)
    response_oferta = urllib.request.urlopen(url_oferta)
    data_oferta = response_oferta.read()

    with open(os.path.join(before_path , 'Casos de estudio/archivos_despacho/scrap_files/oferta.txt'), 'wb') as archivo:
        archivo.write(data_oferta)
        archivo.close()

    ## Pronostico Demanda

    fecha_dem = str(year1) + '-' + str(month1) + '-' + str(day1)
    fecha_dem = datetime.strptime(fecha_dem, '%Y-%m-%d').date()

    while fecha_dem.weekday() != 0:
        fecha_dem = fecha_dem - timedelta(days=1)

    year_dem = fecha_dem.year
    month_dem = fecha_dem.month
    day_dem = fecha_dem.day

    if month_dem < 10:
        month_dem = '0{}'.format(month_dem)

    if day_dem < 10:
        day_dem = '0{}'.format(day_dem)

    url_dem = 'http://www.xm.com.co/pronosticooficial/{}-{}/PRON_SIN{}{}.txt'.format(year_dem,month_dem,month_dem,day_dem)
    response_dem = urllib.request.urlopen(url_dem)
    data_dem = response_dem.read()

    with open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/pronostico_dem.txt'), 'wb') as archivo:
        archivo.write(data_dem)
        archivo.close()

    ## MPO

    url_MPO = 'http://www.xm.com.co/predespachoideal/{}-{}/iMAR{}{}_NAL.TXT'.format(year1,month1,month1,day1)
    response_MPO = urllib.request.urlopen(url_MPO)
    data_MPO = response_MPO.read()

    with open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/MPO.txt'), 'wb') as archivo:
        archivo.write(data_MPO)
        archivo.close()

    #### Lectura de archivos

    agents_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/oferta.txt'), encoding='utf8')
    agents_all_of_it = agents_file.read()
    agents_file.close()

    load_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/pronostico_dem.txt'), encoding='utf8')
    load_all_of_it = load_file.read()
    load_file.close()

    MPO_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/MPO.txt'), encoding='utf8')
    MPO_all_of_it = MPO_file.read()
    MPO_file.close()

    return agents_all_of_it, load_all_of_it, MPO_all_of_it


def opt_despacho(fecha, file_ofer, file_load, file_MPO, txt_Pot_max, txt_Ene, txt_eff_ch, txt_eff_dc, txt_eff_SoC, txt_SoC_min, txt_tdp, txt_td, combo):

    ################################################# Load files ################################################################

    StartTime = time.time()

    year = str(fecha.year)
    month = str(fecha.month)
    day = str(fecha.day)

    load_all_of_it = file_load
    agents_all_of_it = file_ofer
    MPO_all_of_it = file_MPO

    df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\n')])
    dic_OFEI = df_OFEI.to_dict('dict')

    none_val, agents_glb = list(dic_OFEI.items())[0]

    if int(month) == 12:
        year = int(year) - 1

    date_forecast_load = year + '-' + month + '-' + day
    date_forecast_load = datetime.strptime(date_forecast_load, '%Y-%m-%d').date()

    ##

    days_pron = []

    for i in range(1,8):
        tomorrow = str(date_forecast_load + timedelta(days=i-1))
        days_pron.append(tomorrow)

    df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\n')])

    if int(year) >= 2020 and int(month) > 2:
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
        tomorrow = str(date_forecast_load + timedelta(days=i-1))
        df_PRON_DEM[str(tomorrow)] = df_PRON_DEM[str(tomorrow)].astype(float)

    nul_val = []

    for key, value in agents_glb.items():
        if value == str(''):
            nul_val.append(key)

    for i in nul_val:
        del(agents_glb[i])

    def extract_str(element, agents_glb):
        idx = ', ' + str(element) + ', '
        dicc = {}
        for key, value in agents_glb.items():
            if value.find(idx) >= 0:
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')
        return dicc

    def extract_num(element, agents_glb):
        idx = ', ' + str(element) + ', '
        dicc = {}
        for key, value in agents_glb.items():
            if value.find(idx) >= 0:
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]
        return dicc

    D = extract_num('D', agents_glb)
    P = extract_num('P', agents_glb)
    CONF = extract_num('CONF', agents_glb)
    C = extract_str('C', agents_glb)
    PAPUSD = extract_num('PAPUSD1', agents_glb)
    PAP = extract_num('PAP1', agents_glb)
    MO = extract_num('MO', agents_glb)
    AGCP = extract_num('AGCP', agents_glb)
    AGCU = extract_num('AGCU', agents_glb)
    PRU = extract_num('PRU', agents_glb)
    CNA = extract_num('CNA', agents_glb)

    ##

    df_agente_precio = pd.DataFrame(columns=('Planta','Precio'))

    for key, value in D.items():
        for key1, value1 in P.items():
            if key[:] == key1[:] or key[:-1] == key1[:]:
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]

    df_agente_precio = df_agente_precio.drop(df_agente_precio[df_agente_precio['Planta']=='GECELCA32'].index)

    for key, value in D.items():
        for key1, value1 in P.items():
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):
                df_agente_precio.loc[len(df_agente_precio)+2] = key, value1[0]

    df_agente_precio.index = df_agente_precio['Planta']
    df_agente_precio = df_agente_precio.drop(['Planta'],axis=1)

    ##
    df_disponibilidad_maxima = pd.DataFrame.from_dict(D,orient='index').transpose()
    df_disponibilidad_maxima.index += 1
    df_disponibilidad_maxima = df_disponibilidad_maxima.transpose()
    ##

    df_disponibilidad_minima = pd.DataFrame.from_dict(MO,orient='index')

    for key, value in D.items():
        for key1, value1 in MO.items():
            if key[:-1] == key1[:]:
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):
                df_disponibilidad_minima.loc[key] = MO[key1]

    df_disponibilidad_minima = df_disponibilidad_minima.drop(['ALBAN'])
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['GUATRON'])
    # df_disponibilidad_minima = df_disponibilidad_minima.drop(['PAGUA'])

    df_disponibilidad_minima = df_disponibilidad_minima.transpose()
    df_disponibilidad_minima.index += 1
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()

    ##
    df_PAP = pd.DataFrame(columns=('Planta','PAP'))

    for key, value in D.items():
        for key1, value1 in PAP.items():
            if key[:] == key1[:] or key[:-1] == key1[:]:
                df_PAP.loc[len(df_PAP)] = key, value1[0]

    df_PAP = df_PAP.drop(df_PAP[df_PAP['Planta']=='GECELCA32'].index)

    for key, value in D.items():
        for key1, value1 in PAP.items():
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):
                df_PAP.loc[len(df_PAP)+2] = key, value1[0]

    df_PAP.index = df_PAP['Planta']
    df_PAP = df_PAP.drop(['Planta'],axis=1)

    ##
    df_con_max = df_agente_precio.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan,int(0))
    del df_con_max['union']

    df_disponibilidad_maxima = df_con_max.drop(['Precio'], axis=1)

    df_precio = df_con_max
    for i in range(1,25):
        del df_precio[i]

    ##

    df_con_min = df_PAP.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan,int(0))
    for i in range(1,25):
        del df_con_min[i]
    del df_con_min['union']
    df_con_min = df_con_min.rename(columns={0:'PAP'})
    df_con_all_min = df_con_min.merge(df_disponibilidad_minima, how='outer', indicator='union',left_index=True, right_index=True).replace(np.nan,int(0))
    del df_con_all_min['union']

    df_disponibilidad_minima = df_con_all_min.drop(['PAP'], axis=1)

    df_PAP = df_con_all_min
    for i in range(1,25):
        del df_PAP[i]

    df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\n')])
    df_MPO = df_MPO.dropna()
    df_MPO = df_MPO.drop([0], axis=1)

    ReadingTime = time.time() - StartTime

    ################################################ Sets Definitions ################################################
    StartTime = time.time()
    model = ConcreteModel()
    ##
    dim = df_MPO.shape

    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)              ## scheduling periods
    # model.td = RangeSet(5,7)                                                        ## discharge periods
    model.tdp = RangeSet(16,18)                                                       ## previous discharge periods
    model.i = Set(initialize=df_disponibilidad_maxima.index.tolist(), ordered=True) ## Units
    model.s = RangeSet(1)                                                         ## Batteries in the system
    model.umbral = RangeSet(1,4)                                                    ## Umbrales costo de escasez

    ########################################## Parameters definitions ####################################

    ###
    a = {}
    aa = {}
    k = 0
    kk = 0

    for i in range(1,dim[0]+1):
        for j in range(1,dim[1]+1):
            k += 1
            aa[k] = int(df_MPO.iloc[i-1,j-1])

    def MPO_init(model):
        return aa
    model.MPO = Param(model.t, initialize=MPO_init)

    for i in range(1,dim[0]+1):
        for j in range(1,dim[1]+1):
            kk += 1
            max_number = max(aa.items())[1]
            a[kk] = (int(df_MPO.iloc[i-1,j-1]) ** 2)/max_number

    def PC_init(model):
        return a
    model.PC = Param(model.t, initialize=PC_init)

    model.PC_max = Param(model.s, model.t, initialize=int(txt_Pot_max))                ## Power capacity max charging
    model.PD_max = Param(model.s, model.t, initialize=int(txt_Pot_max))                ## Power capacity max discharging
    model.Eficiencia_descarga = Param(model.s, initialize=float(txt_eff_ch))           ## Discharge efficiency
    model.Eficiencia_carga = Param(model.s, initialize=float(txt_eff_dc))              ## Charge efficiency
    model.Eficiencia_SoC = Param(model.s, initialize=float(txt_eff_SoC))               ## Storage efficiency
    model.E_max = Param(model.s, initialize=float(txt_Ene))                            ## Energy storage max limit
    model.SoC_min = Param(model.s, initialize=float(txt_SoC_min)*float(txt_Ene)) ## Minimum state of charge
    # model.SoC_max = Param(model.s, initialize=float(txt_SoC_max.get())*float(txt_Ene.get())) ## Maximum state of charge
    model.PCdes = Param(model.s, model.t, initialize=0)        ## Charging power capacity on discharge periods
    model.PDdes = Param(model.s, model.t, initialize=30)        ## Discharging power capacity on discharge periods

    td_i = txt_td
    td_f = txt_td
    tdp_i = txt_tdp
    tdp_f = txt_tdp

    CRO_est = {1: 1480.31,
               2: 2683.49,
               3: 4706.20,
               4: 9319.71}

    def CRO_init(model):
        return CRO_est
    model.CRO_est = Param(model.umbral, initialize=CRO_init)    ## Costo incremental de racionamiento de energía

    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################
    model.status = Var(model.i, model.t, within=Binary, initialize=0)          ## Commitment of unit i at time t
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)     ## Power dispatch of unit i at time t
    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)            ## Binary Status of battery charge
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)            ## Binary Status of battery discharge
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)  ## Power in battery charge
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)  ## Power in battery discharge
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                  ## Energy of battery
    model.V_DoC = Var(model.s, model.t, domain=NonNegativeReals)                  ## Discharge state
    model.V_Rac = Var(model.t, bounds=(0,2400))  ## Energy of rationing

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################
    StartTime = time.time()

    ## Objective function

    def cost_rule(model):
        return sum(model.P[i,t] * df_precio.loc[i,'Precio'] for i in model.i for t in model.t) + \
                sum(df_PAP.loc[i,'PAP'] * model.status[i,t] for i in model.i for t in model.t) + \
                sum((CRO_est[1] * 1000) * model.V_DoC[s,t] for s in model.s for t in model.t if t >= tdp_i and t <= tdp_f) + \
                sum((CRO_est[1] * 1000) * model.V_Rac[t] for t in model.t) + \
                sum(model.PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################
    ## Power limits

    def P_lim_max_rule(model,i,t):
        return model.P[i,t] <= df_disponibilidad_maxima.loc[i,t] * model.status[i,t]
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,t):
        return model.P[i,t] >= df_disponibilidad_minima.loc[i,t] * model.status[i,t]
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)

    ## Power Balance

    def power_balance_rule(model,t,s):
        return sum(model.P[i,t] for i in model.i) + model.V_Rac[t] + \
                model.V_PD[s,t] == df_PRON_DEM.loc[t,str(fecha)] + model.V_PC[s,t]
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)

    ##### Batteries

    ## Causalidad de la carga/descarga

    def sim_rule(model,s,t):
        return model.B_PC[s,t] + model.B_PD[s,t] <= 1
    model.sim = Constraint(model.s, model.t, rule=sim_rule)

    def power_c_max_rule(model,s,t):
        return model.V_PC[s,t] <= model.PC_max[s,t] * model.B_PC[s,t]
    model.power_c_max = Constraint(model.s, model.t, rule=power_c_max_rule)

    def power_d_max_rule(model,s,t):
        return model.V_PD[s,t] <= model.PD_max[s,t] * model.B_PD[s,t]
    model.power_d_max = Constraint(model.s, model.t, rule=power_d_max_rule)

    ## Balance almacenamiento

    def energy_rule(model,s,t):
        if t == 1:
            return model.V_SoC[s,t] == model.SoC_min[s] + (model.Eficiencia_carga[s] * model.V_PC[s,t] - (model.V_PD[s,t]) * (1/model.Eficiencia_descarga[s]))
        else:
            return model.V_SoC[s,t] == model.V_SoC[s,t-1] * (1 - model.Eficiencia_SoC[s]) + (model.Eficiencia_carga[s] * model.V_PC[s,t] - model.V_PD[s,t] * (1/model.Eficiencia_descarga[s]))
    model.energy = Constraint(model.s, model.t, rule=energy_rule)

    ## Balance de Estado de Carga

    def energy_balance_rule(model,s,t):
        return model.V_DoC[s,t] == model.E_max[s] - model.V_SoC[s,t]
    model.energy_balance = Constraint(model.s, model.t, rule=energy_balance_rule)

    ## Capacidad mínima y máxima de almacenamiento

    def energy_min_limit_rule(model,s,t):
        return model.V_SoC[s,t] >= model.SoC_min[s]
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)

    def energy_max_limit_rule(model,s,t):
        return model.V_SoC[s,t] <= model.E_max[s]
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)

    ## Carga y descarga requerida

    def power_required_dh_rule(model,s,t):
        if t >= td_i and t <= td_f:
            return model.V_PD[s,t] == model.PDdes[s,t]
        else:
            return Constraint.Skip
    model.power_required_dh = Constraint(model.s, model.t, rule=power_required_dh_rule)

    # def power_required_ch_rule(model,s,t):
    #     if t >= td_i and t <= td_f:
    #         return model.V_PC[s,t] >= model.PCdes[s,t] * model.B_PC[s,t]
    #     else:
    #         return Constraint.Skip
    # model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)

    # Configuracion:

    solver_selected = combo

    if solver_selected== 'CPLEX':
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    df_SOC = pyomo1_df(model.V_PD)

    return df_SOC, tiempo

def opt_despacho_2(fecha, file_ofer, file_load, txt_Pot_max, txt_Ene, txt_eff_ch, txt_eff_dc, txt_eff_SoC, txt_SoC_min, txt_SoC_MT, txt_tdp, txt_td, combo):

    ##################################### Load file ####################################

    StartTime = time.time()

    type_sim = 1 #predespacho ideal

    day1 = int(fecha.day)
    month1 = int(fecha.month)
    year1 = int(fecha.year)

    if month1 < 10:
        month1 = '0{}'.format(month1)

    if day1 < 10:
        day1 = '0{}'.format(day1)

    load_all_of_it = file_load
    agents_all_of_it = file_ofer

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

    ## Pronostico Demanda

    days_pron = []

    date_forecast_load = str(year1) + '-' + str(month1) + '-' + str(day1)
    date_forecast_load = datetime.strptime(date_forecast_load, '%Y-%m-%d').date()

    for i in range(1,8):
        dates = str(date_forecast_load + timedelta(days = i - 1))
        days_pron.append(dates)

    df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\n')])

    if year1 >= 2020 and int(month1) > 2:
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
        dates = str(date_forecast_load + timedelta(days = i - 1))
        df_PRON_DEM[str(dates)] = df_PRON_DEM[str(dates)].astype(float)

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
    PAPUSD = extract_num('PAPUSD1', agents_glb)
    PAP = extract_num('PAP1', agents_glb)
    MO = extract_num('MO', agents_glb)
    AGCP = extract_num('AGCP', agents_glb)
    AGCU = extract_num('AGCU', agents_glb)
    PRU = extract_num('PRU', agents_glb)
    CNA = extract_num('CNA', agents_glb)

    #### Desempate de ofertas

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

    ####

    df_agente_precio = pd.DataFrame(columns=('Planta','Precio'))

    for key, value in D.items():
        for key1, value1 in P.items():
            if key[:] == key1[:] or key[:-1] == key1[:]:
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]

    df_agente_precio = df_agente_precio.drop(df_agente_precio[df_agente_precio['Planta'] == 'GECELCA32'].index)

    for key, value in D.items():
        for key1, value1 in P.items():
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):
                df_agente_precio.loc[len(df_agente_precio) + 2] = key, value1[0]

    df_agente_precio.index = df_agente_precio['Planta']
    df_agente_precio = df_agente_precio.drop(['Planta'], axis=1)

    ##

    df_disponibilidad_maxima = pd.DataFrame.from_dict(D, orient='index').transpose()
    df_disponibilidad_maxima.index += 1
    df_disponibilidad_maxima = df_disponibilidad_maxima.transpose()

    ##

    df_disponibilidad_minima = pd.DataFrame.from_dict(MO, orient='index')

    for key, value in D.items():
        for key1, value1 in MO.items():
            if key[:-1] == key1[:]:
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):
                df_disponibilidad_minima.loc[key] = MO[key1]
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):
                df_disponibilidad_minima.loc[key] = MO[key1]

    df_disponibilidad_minima = df_disponibilidad_minima.drop(['ALBAN'])
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['GUATRON'])
    # df_disponibilidad_minima = df_disponibilidad_minima.drop(['PAGUA'])

    df_disponibilidad_minima = df_disponibilidad_minima.transpose()
    df_disponibilidad_minima.index += 1
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()

    if type_sim == 1:
        df_disponibilidad_minima = df_disponibilidad_minima.clip(upper=0)
    else:
        df_disponibilidad_minima = df_disponibilidad_minima

    ##

    df_PAP = pd.DataFrame(columns=('Planta','PAP'))

    for key, value in D.items():
        for key1, value1 in PAP.items():
            if key[:] == key1[:] or key[:-1] == key1[:]:
                df_PAP.loc[len(df_PAP)] = key, value1[0]

    df_PAP = df_PAP.drop(df_PAP[df_PAP['Planta'] == 'GECELCA32'].index)

    for key, value in D.items():
        for key1, value1 in PAP.items():
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):
                df_PAP.loc[len(df_PAP) + 2] = key, value1[0]

    df_PAP.index = df_PAP['Planta']
    df_PAP = df_PAP.drop(['Planta'], axis=1)

    ##

    df_con_max = df_agente_precio.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))
    del df_con_max['union']

    df_disponibilidad_maxima = df_con_max.drop(['Precio'], axis=1)

    df_precio = df_con_max

    for i in range(1,25):
        del df_precio[i]

    ##

    df_con_min = df_PAP.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))

    for i in range(1,25):
        del df_con_min[i]

    del df_con_min['union']

    df_con_min = df_con_min.rename(columns={0:'PAP'})
    df_con_all_min = df_con_min.merge(df_disponibilidad_minima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))
    del df_con_all_min['union']

    df_disponibilidad_minima = df_con_all_min.drop(['PAP'], axis=1)

    df_PAP = df_con_all_min

    for i in range(1,25):
        del df_PAP[i]

    ## Demanda SIN

    mydir = os.getcwd()
    files_path = os.path.join(mydir, 'Casos de estudio/archivos_despacho')

    if year1 == 2019 and int(month1) <= 6 :
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME1_2019.xlsx'),
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)
    elif year1 == 2019 and int(month1) >= 7:
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME2_2019.xlsx'),
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)
    elif year1 == 2020 and int(month1) <= 6:
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME1_2020.xlsx'),
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)
    elif year1 == 2020 and int(month1) >= 7:
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME2_2020.xlsx'),
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)
    else:
        print('No existen archivos de demanda para esa fecha')

    df_demanda = df_demanda.drop([0], axis=0)
    df_demanda = df_demanda.fillna(0)
    a = df_demanda.loc[1,:]
    for i in range(28):
        df_demanda = df_demanda.rename(columns={'Unnamed: {}'.format(i):a[i]})
    df_demanda = df_demanda.drop(['Codigo Comercializador','Mercado','Version'], 1)
    df_demanda = df_demanda.drop([1], axis=0)
    df_demanda = df_demanda.reset_index(drop=True)

    df_demanda_fecha = pd.DataFrame()

    fecha_SIN = str(year1) + '-' + str(month1) + '-' + str(day1)

    for i in range(len(df_demanda)):
        if df_demanda.loc[i,'Fecha'] == fecha_SIN:
            df_demanda_fecha = df_demanda_fecha.append(df_demanda.loc[i,:], ignore_index=True)

    df_demanda_fecha = df_demanda_fecha.drop(['Fecha'], 1)
    df_demanda_fecha = df_demanda_fecha.rename(columns={'0':0})
    df_demanda_fecha = df_demanda_fecha.reindex(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])

    df_demanda_fecha = df_demanda_fecha.sum(axis=0)
    df_demanda_fecha.index += 1

    for i in range(len(df_demanda_fecha)):
        df_demanda_fecha.loc[i+1] = round(df_demanda_fecha.loc[i+1] / 1000, 2)

    ReadingTime = time.time() - StartTime

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    model = ConcreteModel()
    ##

    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)              ## scheduling periods
    model.tr = Set(initialize=[txt_td])                                                        ## discharge periods
    model.tdp = Set(initialize=[txt_tdp])                                                     ## previous discharge periods
    model.i = Set(initialize=df_disponibilidad_maxima.index.tolist(), ordered=True) ## Units
    model.s = RangeSet(1)                                                           ## Batteries in the system
    ########################################## Parameters definitions ####################################

    model.PC_max = Param(model.s, initialize=int(txt_Pot_max))           ## Power capacity max charging
    model.PD_max = Param(model.s, initialize=int(txt_Pot_max))           ## Power capacity max discharging
    model.Eficiencia_descarga = Param(model.s, initialize=float(txt_eff_dc))      ## Discharge efficiency
    model.Eficiencia_carga = Param(model.s, initialize=float(txt_eff_ch))         ## Charge efficiency
    model.Eficiencia_SoC = Param(model.s, initialize=float(txt_eff_SoC))       ## Storage efficiency
    model.E_max = Param(initialize=int(txt_Ene))                     ## Energy storage max limit
    SoC_min = float(txt_SoC_min)                                                  ## Minimum state of charge
    SoC_max = 1                                                     ## Maximum state of charge
    SoC_MT = float(txt_SoC_MT)                                                   ## Minimum technical state of charge
    model.PCreq = Param(initialize=0)            ## Charging power capacity on discharge periods
    model.PDreq = Param(initialize=30)            ## Discharging power capacity on discharge periods
    K_e = 1                                                         ## Scalling factor [$/MWh]

    td_i = int(txt_td)
    td_f = int(txt_td)
    tpd_i = int(txt_tdp)
    tpd_f = int(txt_tdp)

    ## PC definitions

    PC = round((K_e * df_PRON_DEM.loc[:,str(fecha)]) / max(df_PRON_DEM.loc[:,str(fecha)]),4)

    ## costo de racionamiento

    cro = {'CRO1': 1480.31,
            'CRO2': 2683.49,
            'CRO3': 4706.20,
            'CRO4': 9319.71}

    ## Estado de conexión del SAEB al SIN

    model.ECS = Param(model.s, model.t, initialize=1, mutable=True)

    # model.ECS[1,7] = 0

    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################
    model.status = Var(model.i, model.t, within=Binary, initialize=0)           ## Commitment of unit i at time t
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)      ## Power dispatch of unit i at time t
    model.V_Rac = Var(model.t, bounds=(0, 2400))   ## Energy of rationing
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)               ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)               ## Shut-Down cost of unit i
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)               ## Start-Up status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)               ## Shut-Down status of unit i

    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery charge
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery discharge
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery charge
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery discharge
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                ## Energy of battery
    model.V_SoD = Var(model.s, model.t, domain=NonNegativeReals)                ## Discharge state
    model.V_SoC_E = Var(model.s, model.t, domain=NonNegativeReals)              ## State of Charge with storage efficiency

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################
    StartTime = time.time()

    ## Objective function

    def cost_rule(model):
        return sum(model.P[i,t] * df_precio.loc[i,'Precio'] for i in model.i for t in model.t) + \
                sum(model.costSD[i,t] + model.costSU[i,t] for i in model.i for t in model.t) + \
                sum((cro['CRO1'] * 1000) * model.V_SoD[s,t] * model.E_max for s in model.s for t in model.t if t == tpd_i) + \
                sum((cro['CRO1'] * 1000) * model.V_Rac[t] for t in model.t) + \
                sum(PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################

    #### Dispath constraints

    ## Power limits

    def P_lim_max_rule(model,i,t):
        return model.P[i,t] <= df_disponibilidad_maxima.loc[i,t] * model.status[i,t]
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,t):
        return model.P[i,t] >= df_disponibilidad_minima.loc[i,t] * model.status[i,t]
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)

    ## PAP cost

    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == df_PAP.loc[i,'PAP'] * model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,t):
        return model.costSD[i,t] == df_PAP.loc[i,'PAP'] * model.SD[i,t]
    model.CostSDfn = Constraint(model.i, model.t, rule=CostSDfn_init)

    ## Integer Constraint

    def bin_cons1_rule(model,i,t):
        if t == model.t.first():
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] #- model.onoff_to[i,t]
        else:
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons2_rule(model,i,t):
        return model.SU[i,t] + model.SD[i,t] <= 1
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)

    ## Power Balance

    def power_balance_rule(model,t,s):
        return sum(model.P[i,t] for i in model.i) + model.V_Rac[t] + \
                model.V_PD[s,t] * model.ECS[s,t] == df_demanda_fecha[t] + model.V_PC[s,t] * model.ECS[s,t]
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)

    ##### Batteries

    ## Balance almacenamiento

    def energy_rule(model,s,t):
        if t == 1:
            return model.V_SoC[s,t] == SoC_min + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] * (1 / model.E_max) - \
                    model.V_PD[s,t] * (1 / model.Eficiencia_descarga[s]) * (1 / model.E_max))
        else:
            return model.V_SoC[s,t] == model.V_SoC_E[s,t-1] + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] * (1 / model.E_max) - \
                    model.V_PD[s,t] * (1 / model.Eficiencia_descarga[s]) * (1 / model.E_max))
    model.energy = Constraint(model.s, model.t, rule=energy_rule)

    ## Afectación del estado de carga por eficiencia de almacenamiento

    def efe_storage_1_rule(model,s,t):
        return -(model.B_PC[s,t] + model.B_PD[s,t]) + model.V_SoC[s,t] * (1 - model.Eficiencia_SoC[s]) <= model.V_SoC_E[s,t]
    model.efe_storage_1 = Constraint(model.s, model.t, rule=efe_storage_1_rule)

    def efe_storage_2_rule(model,s,t):
        return model.V_SoC[s,t] * (1 - model.Eficiencia_SoC[s]) + (model.B_PC[s,t] + model.B_PD[s,t]) >= model.V_SoC_E[s,t]
    model.efe_storage_2 = Constraint(model.s, model.t, rule=efe_storage_2_rule)

    def efe_storage_3_rule(model,s,t):
        return -(1 - model.B_PC[s,t]) + model.V_SoC[s,t] <= model.V_SoC_E[s,t]
    model.efe_storage_3 = Constraint(model.s, model.t, rule=efe_storage_3_rule)

    def efe_storage_4_rule(model,s,t):
        return model.V_SoC[s,t] + (1 - model.B_PC[s,t]) >= model.V_SoC_E[s,t]
    model.efe_storage_4 = Constraint(model.s, model.t, rule=efe_storage_4_rule)

    def efe_storage_5_rule(model,s,t):
        return -(1 - model.B_PD[s,t]) + model.V_SoC[s,t] <= model.V_SoC_E[s,t]
    model.efe_storage_5 = Constraint(model.s, model.t, rule=efe_storage_5_rule)

    def efe_storage_6_rule(model,s,t):
        return model.V_SoC[s,t] + (1 - model.B_PD[s,t]) >= model.V_SoC_E[s,t]
    model.efe_storage_6 = Constraint(model.s, model.t, rule=efe_storage_6_rule)

    ## Balance de Estado de Carga

    def energy_balance_rule(model,s,t):
        return model.V_SoD[s,t] == 1 - model.V_SoC[s,t]
    model.energy_balance = Constraint(model.s, model.t, rule=energy_balance_rule)

    ## Capacidad mínima y máxima de almacenamiento

    def energy_min_limit_rule(model,s,t):
        return model.V_SoC[s,t] >= SoC_min
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)

    def energy_max_limit_rule(model,s,t):
        return model.V_SoC[s,t] <= SoC_max
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)

    ## mínimo técnico del sistema de almacenamiento

    def energy_min_tec_rule(model,s,t):
        return model.V_SoC[s,t] >= SoC_MT
    model.energy_min_tec = Constraint(model.s, model.t, rule=energy_min_tec_rule)

    ## Causalidad de la carga/descarga

    def sim_rule(model,s,t):
        if model.ECS[s,t].value == 1:
            return model.B_PC[s,t] + model.B_PD[s,t] <= 1
        else:
            return Constraint.Skip
    model.sim = Constraint(model.s, model.t, rule=sim_rule)

    def power_c_max_rule(model,s,t):
        if model.ECS[s,t].value == 1:
            return model.V_PC[s,t] <= model.PC_max[s] * model.B_PC[s,t]
        else:
            return Constraint.Skip
    model.power_c_max = Constraint(model.s, model.t, rule=power_c_max_rule)

    def power_d_max_rule(model,s,t):
        if model.ECS[s,t].value == 1:
            return model.V_PD[s,t] <= model.PD_max[s] * model.B_PD[s,t]
        else:
            return Constraint.Skip
    model.power_d_max = Constraint(model.s, model.t, rule=power_d_max_rule)

    ## Carga y descarga requerida

    def power_required_dc_rule(model,s,t):
        if t == td_i and model.ECS[s,t].value == 1:
            return model.V_PD[s,t] == model.PDreq
        else:
            return model.V_PD[s,t] == 0
    model.power_required_dc = Constraint(model.s, model.t, rule=power_required_dc_rule)

    def power_required_ch_rule(model,s,t):
        if t == td_i and model.ECS[s,t].value == 1:
            return model.V_PC[s,t] >= model.PCreq
        else:
            return Constraint.Skip
    model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)

    # Configuracion:

    solver_selected = combo

    if solver_selected== 'CPLEX':
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    df_SOC = pyomo1_df(model.V_PD)

    return df_SOC, tiempo
