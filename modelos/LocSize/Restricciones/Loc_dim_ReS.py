# coding: utf-8

## Libraries
from pyomo.environ import *
import pandas as pd
import numpy as np
import math
import requests
import datetime
from datetime import datetime, timedelta
import os
from funciones.save_files import *
from funciones.read_download_files import *
import errno
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## Despacho incluyendo red limitando número de BESS (DPRLNSC: despacho progamado minimizando restricciones incluyendo pérdidas limitando número de BESS con enfoque estocástico)
def DPRLNSC(file_system, db_file, df_DEM, df_disponibilidad_maxima, df_disponibilidad_minima, d_type, df_ofe, fecha, df_PI, df_MPO,
            txt_eff, txt_SOC_min, txt_SOC_ini, txt_time_sim, txt_re_inv, txt_C_Pot, txt_C_Bat, txt_autoD, num_BESS, s_study, sensi, combo):

    StartTime = time.time()

    try:
        os.mkdir('Casos_estudio/loc_size/h5_files')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if os.path.exists('Casos_estudio/loc_size/h5_files/file_name.txt'):
        pass
    else:
        with open('Casos_estudio/loc_size/h5_files/file_name.txt', 'w') as f:
            f.write(' ')

    actual_path = os.getcwd()

    file_name_path = open('Casos_estudio/loc_size/h5_files/file_name.txt')
    file_name_content = file_name_path.read()
    file_name_path.close()

    if file_system.name == 'Colombia_115.xlsx':
        db_system_column = 'bus_over_110'
        h5_name = 'colombia_over_110'
    elif file_system.name == 'Colombia_115_UPME2019.xlsx':
        db_system_column = 'bus_over_110_UPME2019'
        h5_name = 'colombia_over_110_UPME2019'
    elif file_system.name == 'Colombia_220.xlsx':
        db_system_column = 'bus_over_200'
        h5_name = 'colombia_over_220'
    elif file_system.name == 'Colombia_220_ISA2021.xlsx':
        db_system_column = 'bus_over_200_UPME2019'
        h5_name = 'colombia_over_220_ISA'
    elif file_system.name == 'Bogota.xlsx':
        db_system_column = 'bus_BOG'
        h5_name = 'Bogota'
    elif file_system.name == 'Colombia_15_Mod_2018.xlsx':
        db_system_column = 'bus_15N'
        h5_name = 'colombia_15N_2018'
    elif file_system.name == 'Colombia_15_Mod_2030.xlsx':
        db_system_column = 'bus_15N'
        h5_name = 'colombia_15N_2030'
    else:
        print('Sistema equivalente no encontrado!!')

    print(file_system.name)

    if file_name_content != file_system.name:

        df_System_data = pd.read_excel(file_system, sheet_name='System_data', header=0, index_col=0)
        df_SM_Unit = pd.read_excel(file_system, sheet_name='SM_Unit', header=0, index_col=0)

        if file_system.name == 'Bogota.xlsx':
            df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)
            df_Bus_pl = pd.read_excel(file_system, sheet_name='Bus_pl', header=0, index_col=0)
            df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)
            df_Branch_pl = pd.read_excel(file_system, sheet_name='Branch_pl', header=0, index_col=0)

        else:

            df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)
            df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)

        df_gen_db_un = pd.read_excel(db_file, sheet_name='gen_un', header=0, index_col=0)
        df_gen_db_pl = pd.read_excel(db_file, sheet_name='gen_pl', header=0, index_col=0)
        df_gen_db_RealIdeal = pd.read_excel(db_file, sheet_name='gen_ide_real', header=0, index_col=0)
        df_gen_db_un.index = df_gen_db_un.loc[:,'name']
        df_gen_db_un.drop(['name'], axis=1, inplace=True)
        df_gen_db_pl.index = df_gen_db_pl.loc[:,'name']
        df_gen_db_pl.drop(['name'], axis=1, inplace=True)
        df_gen_db_RealIdeal.index = df_gen_db_RealIdeal.loc[:,'name']
        df_gen_db_RealIdeal.drop(['name'], axis=1, inplace=True)

        ## gen incidence matrix

        if file_system.name == 'Bogota.xlsx':

            df_SM_map_un = pd.DataFrame(index=df_gen_db_un.index, columns=df_Bus.index.tolist())
            df_SM_map_pl = pd.DataFrame(index=df_gen_db_pl.index, columns=df_Bus_pl.index.tolist())
            df_SM_map_RealIdeal = pd.DataFrame(index=df_gen_db_RealIdeal.index, columns=df_Bus_pl.index.tolist())

        else:

            df_SM_map_un = pd.DataFrame(index=df_gen_db_un.index, columns=df_Bus.index.tolist())
            df_SM_map_pl = pd.DataFrame(index=df_gen_db_pl.index, columns=df_Bus.index.tolist())
            df_SM_map_RealIdeal = pd.DataFrame(index=df_gen_db_RealIdeal.index, columns=df_Bus.index.tolist())

        for i in df_SM_map_un.index:
            for j in df_SM_map_un.columns:
                if df_gen_db_un.loc[i,db_system_column] == j:
                    df_SM_map_un.loc[i,j] = 1
                else:
                    df_SM_map_un.loc[i,j] = 0

        for i in df_SM_map_pl.index:
            for j in df_SM_map_pl.columns:
                if df_gen_db_pl.loc[i,db_system_column] == j:
                    df_SM_map_pl.loc[i,j] = 1
                else:
                    df_SM_map_pl.loc[i,j] = 0

        for i in df_SM_map_RealIdeal.index:
            for j in df_SM_map_RealIdeal.columns:
                if df_gen_db_RealIdeal.loc[i,db_system_column] == j:
                    df_SM_map_RealIdeal.loc[i,j] = 1
                else:
                    df_SM_map_RealIdeal.loc[i,j] = 0

        ## line incidence matrix

        if file_system.name == 'Bogota.xlsx':

            df_line_map_un = pd.DataFrame(index=df_Branch.index.tolist(), columns=df_Bus.index.tolist()).fillna(0)
            df_line_map_pl = pd.DataFrame(index=df_Branch_pl.index.tolist(), columns=df_Bus_pl.index.tolist()).fillna(0)

            # unidades

            for i in df_line_map_un.index:
                for j in df_line_map_un.columns:
                    if j == df_Branch.loc[i,'from']:
                        df_line_map_un.loc[i,j] = 1
                    if j == df_Branch.loc[i,'to']:
                        df_line_map_un.loc[i,j] = -1

            # plantas

            for i in df_line_map_pl.index:
                for j in df_line_map_pl.columns:
                    if df_Branch_pl.loc[i,'from'] == j:
                        df_line_map_pl.loc[i,j] = 1
                    if df_Branch_pl.loc[i,'to'] == j:
                        df_line_map_pl.loc[i,j] = -1

        else:

            df_line_map = pd.DataFrame(index=df_Branch.index.tolist(), columns=df_Bus.index.tolist()).fillna(0)

            for i in df_line_map.index:
                for j in df_line_map.columns:
                    if df_Branch.loc[i,'from'] == j:
                        df_line_map.loc[i,j] = 1
                    if df_Branch.loc[i,'to'] == j:
                        df_line_map.loc[i,j] = -1

        ## save files

        with pd.HDFStore('Casos_estudio/loc_size/h5_files/{}.h5'.format(h5_name)) as store:
            store['df_System_data'] = df_System_data
            store['df_SM_Unit'] = df_SM_Unit
            store['df_SM_map_un'] = df_SM_map_un
            store['df_SM_map_pl'] = df_SM_map_pl
            store['df_SM_map_RealIdeal'] = df_SM_map_RealIdeal

            if file_system.name == 'Bogota.xlsx':

                store['df_Branch'] = df_Branch
                store['df_Branch_pl'] = df_Branch_pl
                store['df_Bus'] = df_Bus
                store['df_Bus_pl'] = df_Bus_pl
                store['df_line_map_un'] = df_line_map_un
                store['df_line_map_pl'] = df_line_map_pl

            else:

                store['df_Branch'] = df_Branch
                store['df_Bus'] = df_Bus
                store['df_line_map'] = df_line_map

            store['df_gen_db_un'] = df_gen_db_un
            store['df_gen_db_pl'] = df_gen_db_pl
            store['df_gen_db_RealIdeal'] = df_gen_db_RealIdeal

        with open('Casos_estudio/loc_size/h5_files/file_name.txt', 'w') as archivo:
            archivo.write(file_system.name)

    else:

        with pd.HDFStore('Casos_estudio/loc_size/h5_files/{}.h5'.format(h5_name)) as store:
            df_System_data = store['df_System_data']
            df_SM_Unit = store['df_SM_Unit']
            df_SM_map_un = store['df_SM_map_un']
            df_SM_map_pl = store['df_SM_map_pl']
            df_SM_map_RealIdeal = store['df_SM_map_RealIdeal']

            if file_system.name == 'Bogota.xlsx':

                df_Branch = store['df_Branch']
                df_Branch_pl = store['df_Branch_pl']
                df_Bus = store['df_Bus']
                df_Bus_pl = store['df_Bus_pl']
                df_line_map_un = store['df_line_map_un']
                df_line_map_pl = store['df_line_map_pl']

            else:

                df_Branch = store['df_Branch']
                df_line_map = store['df_line_map']
                df_Bus = store['df_Bus']

            df_gen_db_un = store['df_gen_db_un']
            df_gen_db_pl = store['df_gen_db_pl']
            df_gen_db_RealIdeal = store['df_gen_db_RealIdeal']

    if d_type == 'Unidades':

        df_SM_map = df_SM_map_un
        df_gen_db = df_gen_db_un

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch
            df_Bus = df_Bus
            df_line_map = df_line_map_un

    elif d_type == 'Acorde a disponibilidad Real':

        df_SM_map = df_SM_map_RealIdeal
        df_gen_db = df_gen_db_RealIdeal

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch_pl
            df_Bus = df_Bus_pl
            df_line_map = df_line_map_pl

    else:

        df_SM_map = df_SM_map_pl
        df_gen_db = df_gen_db_pl

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch_pl
            df_Bus = df_Bus_pl
            df_line_map = df_line_map_pl

    #### Load

    ## Load factor by bus

    if file_system.name == 'Bogota.xlsx' and (d_type == 'Plantas' or d_type == 'Acorde a disponibilidad Real'):

        df_load_factor = pd.read_excel(file_system, sheet_name='load_factor_pl', header=0, index_col=1)

    else:

        df_load_factor = pd.read_excel(file_system, sheet_name='load_factor', header=0, index_col=1)

    ## bus, sce --> tuple
    busSce = []
    for bu in df_Bus.index:
        for sce in range(len(df_DEM)):
            busSce.append((bu,sce+1))

    ## bus, sce MultiIndex
    idxL = pd.MultiIndex.from_tuples(busSce, names=['buses','sce'])

    ## Load by bus parameter scenarios

    if file_system.name == 'Colombia_115.xlsx':

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)
        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        if t in [1,2,3,4,5,6,22,23,24]:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'min_factor'] * df_DEM[sc].loc[t]
                        else:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    elif file_system.name == 'Colombia_115_UPME2019.xlsx':

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)

        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        if t in [1,2,3,4,5,6,22,23,24]:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'min_factor'] * df_DEM[sc].loc[t]
                        else:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    else:

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)

        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    ReadingTime = time.time() - StartTime

    #### Generators

    ## Generators index
    gen_idx = []
    for i in range(len(df_disponibilidad_maxima)):
        aux = df_disponibilidad_maxima[i].index.get_level_values(0).tolist()
        gen_idx = gen_idx + aux

    gen_idx = list(set(gen_idx))

    ## gen, sce --> tuples
    genSce = []
    for gen in gen_idx:
        for sce in range(len(df_disponibilidad_maxima)):
            genSce.append((gen,sce+1))

    ## gen, sce MultiIndex
    idx = pd.MultiIndex.from_tuples(genSce, names=['plantas','sce'])

    #### Generators and offers complete dataset for all scenarios

    ## max disp
    df_maxDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_maxima)):
        for gen, sce in df_maxDispC.index:
            for co in df_maxDispC.columns:
                if (gen, sce) in df_disponibilidad_maxima[i].index:
                    df_maxDispC.loc[(gen,sce),co] = df_disponibilidad_maxima[i].loc[(gen,sce),co]

    ## min disp
    df_minDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_minima)):
        for gen, sce in df_minDispC.index:
            for co in df_minDispC.columns:
                if (gen, sce) in df_disponibilidad_minima[i].index:
                    df_minDispC.loc[(gen,sce),co] = df_disponibilidad_minima[i].loc[(gen,sce),co]

    ## ofertas
    df_ofeC = pd.DataFrame(index=idx, columns=['Precio','PAPUSD','PAP']).fillna(0)

    for i in range(len(df_ofe)):
        for gen, sce in df_ofeC.index:
            for co in df_ofeC.columns:
                if (gen, sce) in df_ofe[i].index:
                    df_ofeC.loc[(gen,sce),co] = df_ofe[i].loc[(gen,sce),co]

    #### MPO
    df_MPO_C = pd.DataFrame()

    for i in range(len(df_MPO)):
        df_MPO_C[i+1] = df_MPO[i].iloc[0]

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    model = ConcreteModel()

    N_horas = int(txt_time_sim)                                                                    ## Total time steps

    model.t = RangeSet(1, N_horas)                                                  ## scheduling periods
    model.i = Set(initialize=gen_idx, ordered=True) ## Units
    model.b = Set(initialize=df_Bus.index.tolist(), ordered=True)                   ## Bus system set
    model.l = Set(initialize=df_Branch.index.tolist(), ordered=True)                ## Lines set
    model.s = RangeSet(len(df_disponibilidad_maxima))                               ## Scenario number
    model.L = RangeSet(3)

    ########################################## Parameters definitions ####################################

    # Sbase [MVA]
    S_base = df_System_data.loc['S_base'][0]
    model.MVA_base = Param(initialize=S_base)

    # Valor TRM para el día bajo estudio

    url_trm = 'https://www.datos.gov.co/resource/mcec-87by.json'
    data_trm = requests.get(url_trm).json()

    trm_date = fecha[0]
    print(trm_date)
    trm_year, trm_month, trm_day = trm_date.year, trm_date.month, trm_date.day

    if trm_month < 10:
        trm_month = '0{}'.format(trm_month)

    if trm_day < 10:
        trm_day = '0{}'.format(trm_day)

    trm_date_format = '{}-{}-{}T00:00:00.000'.format(trm_year, trm_month, trm_day)

    date_trm_to = []

    for i in range(len(data_trm)):
        date_trm_to.append(data_trm[i]['vigenciadesde'])

    trm_value = None

    for j in date_trm_to:
        if j == trm_date_format:
            index = date_trm_to.index(j)
            trm_value = float(data_trm[index]['valor'])

    if trm_value == None:

        while trm_date.weekday() != 5:
            trm_date = trm_date - timedelta(days=1)

        trm_year, trm_month, trm_day = trm_date.year, trm_date.month, trm_date.day

        if trm_month < 10:
            trm_month = '0{}'.format(trm_month)

        if trm_day < 10:
            trm_day = '0{}'.format(trm_day)

        trm_date_format = '{}-{}-{}T00:00:00.000'.format(trm_year, trm_month, trm_day)

        for j in date_trm_to:
            if j == trm_date_format:

                index = date_trm_to.index(j)
                trm_value = float(data_trm[index]['valor'])

    ### BESS parameters

    SOC_min = 1 - float(txt_SOC_min)                                                               ## Minimum State of Charge of BESS
    SOC_ini = float(txt_SOC_ini)                                                               ## Initial State of Charge of BESS
    SOC_max = 1.0
    BESS_number = num_BESS
    Big_number = 1e20

    model.n_dc = Param(initialize=float(txt_eff))                                         ## Discharging efficiency
    model.n_ch = Param(initialize=float(txt_eff))                                         ## Charging efficiency
    model.n_SoC = Param(initialize=float(txt_autoD))                                   ## Self-discharging

    #### BESS parameters
    Costo_potencia = round(int(txt_C_Pot) * trm_value / (365 * int(txt_re_inv) * 24),2)
    Costo_energia = round(int(txt_C_Bat) * trm_value / (365 * int(txt_re_inv) * 24),2)
    model.Costo_potencia = Param(initialize=Costo_potencia * N_horas)         ## Costo del inversor de potencia de la batería
    model.Costo_energia = Param(initialize=Costo_energia * N_horas)           ## Costo de los modulos de baterías

    # #### line parameters

    ## Susceptance of each line
    def susceptance_init(model,l):
        return 1 / df_Branch.loc[l,'X']
    model.susceptance = Param(model.l, rule=susceptance_init)

    ## Conductance of each line
    def conductance_init(model,l):
        return df_Branch.loc[l,'R'] /(df_Branch.loc[l,'X'] ** 2 + df_Branch.loc[l,'R'] ** 2)
    model.conductance = Param(model.l, initialize=conductance_init)

    ##angular difference slope
    delta_theta = (20 * math.pi / 180)

    def alpha_init(model,L):
        return delta_theta * (2 * L - 1)
    model.alpha = Param(model.L, initialize=alpha_init)

    #################################### Scenarios Parameters ##############################

    prob = {}

    for s in model.s:
        prob[s] = 1 / len(model.s)
    model.prob = Param(model.s, initialize=prob)

    #################################### Define Variables ##############################

    ## Dispatch variables
    model.status = Var(model.i, model.t, within=Binary, initialize=0)                       ## Commitment of unit i at time t
    model.P = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)         ## Power dispatch of unit i at time t
    model.costSU = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)    ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)    ## Shut-Down cost of unit i
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)                       ## Startup status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)                       ## Shutdown status of unit i
    model.V_Rac = Var(model.b, model.t, bounds=(0,2400), initialize=0)                      ## Energy of rationing

    ## Power Flow variables
    model.theta = Var(model.b, model.s, model.t, bounds=(-math.pi,math.pi))                 ## Voltage angle
    model.pf = Var(model.l, model.s, model.t, within=Reals)                                 ## Power flow line l at time t scenario u
    model.pf_complete = Var(model.l, model.s, model.t, within=Reals, initialize=0)

    if file_system.name == 'Bogota.xlsx' and (d_type == 'Pl' or d_type == 'Acorde a disponibilidad Real'):

        Slack_bus = df_System_data.loc['Slack_bus_pl'][0]

    else:

        Slack_bus = df_System_data.loc['Slack_bus'][0]

    for s in model.s:
        for t in model.t:
            model.theta[Slack_bus,s,t].fix(0)

    ## Losses
    model.theta_sr_pos = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.theta_sr_neg = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.line_losses = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.theta_aux = Var(model.l, model.s, model.t, model.L, domain=NonNegativeReals, initialize=0)

    ## BESS variables
    model.BESS_inst = Var(model.b, within=Binary)                     ## Status of battery charge
    model.BESS_status = Var(model.b, model.t, within=Binary)                     ## Status of battery discharge
    model.Pot_Ba_ch = Var(model.b, model.s, model.t, domain=NonNegativeReals)            # Power in battery charge [kW]
    model.Pot_Ba_dc = Var(model.b, model.s, model.t, domain=NonNegativeReals)            # Power in battery discharge [kW]
    model.e_b = Var(model.b, model.s, model.t, domain=NonNegativeReals)                  # Energy of battery [kWh]
    model.P_BESS = Var(model.b, domain = NonNegativeReals, bounds=(0,1e6), initialize=0) ## Power Size (New)
    model.E_BESS = Var(model.b, domain = NonNegativeReals, bounds=(0,1e6), initialize=0) ## Energy Size (New)

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################

    StartTime = time.time()

    ## Objective function definition

    def cost_rule(model):
        return sum(model.prob[s] * sum(df_ofeC.loc[(i,s),'Precio'] * model.P[i,s,t] for i in model.i for t in model.t) + \
            sum(model.costSD[i,s,t] + model.costSU[i,s,t] for i in model.i for t in model.t) + \
            sum((model.P[i,s,t] - df_PI.loc[(i,s),t]) * int(df_MPO_C.loc[t,s]) for i in model.i for t in model.t) for s in model.s) + \
            sum(model.P_BESS[b] * model.Costo_potencia + model.E_BESS[b] * model.Costo_energia for b in model.b)
    model.cost_comp = Objective(rule=cost_rule, sense=minimize)

    def cost_rule(model):
        return sum(model.prob[s] * sum(df_ofeC.loc[(i,s),'Precio'] * model.P[i,s,t] for i in model.i for t in model.t) + \
            sum((model.P[i,s,t] - df_PI.loc[(i,s),t]) * int(df_MPO_C.loc[t,s]) for i in model.i for t in model.t) for s in model.s) + \
            sum(model.P_BESS[b] * model.Costo_potencia + model.E_BESS[b] * model.Costo_energia for b in model.b)
    model.cost_simp = Objective(rule=cost_rule, sense=minimize)

    ##### Constraints

    ## Power limits

    def P_lim_max_comp_rule(model,i,s,t):
        return model.P[i,s,t] <= df_maxDispC.loc[(i,s),t] * model.status[i,t]
    model.P_lim_max_comp = Constraint(model.i, model.s, model.t, rule=P_lim_max_comp_rule)

    def P_lim_max_simp_rule(model,i,s,t):
        return model.P[i,s,t] <= df_maxDispC.loc[(i,s),t]
    model.P_lim_max_simp = Constraint(model.i, model.s, model.t, rule=P_lim_max_simp_rule)

    def P_lim_min_rule(model,i,s,t):
        return model.P[i,s,t] >= df_minDispC.loc[(i,s),t]
    model.P_lim_min = Constraint(model.i, model.s, model.t, rule=P_lim_min_rule)

    ## PAP cost & Integer Constraint

    def bin_cons1_rule(model,i,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            if t == model.t.first():
                return model.SU[i,t] - model.SD[i,t] == model.status[i,t]# - model.onoff_t0[i]
            else:
                return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
        else:
            return Constraint.Skip
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons3_rule(model,i,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.SU[i,t] + model.SD[i,t] <= 1
        else:
            return Constraint.Skip
    model.bin_cons3 = Constraint(model.i, model.t, rule=bin_cons3_rule)

    def CostSUfn_init(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.costSU[i,s,t] == df_ofeC.loc[(i,s),'PAP'] * model.SU[i,t]
        else:
            return Constraint.Skip
    model.CostSUfn = Constraint(model.i, model.s, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.costSD[i,s,t] == df_ofeC.loc[(i,s),'PAP'] * model.SD[i,t]
        else:
            return Constraint.Skip
    model.CostSDfn = Constraint(model.i, model.s, model.t, rule=CostSDfn_init)

    ## Ramp Constraints

    def ramp_up_fn_rule(model,i,s,t):
        if t > 1:
            if df_gen_db.loc[i,'centralmente'] == 'SI':
                if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
                    return model.P[i,s,t] - model.P[i,s,t-1] <= df_gen_db.loc[i,'Ramp_Up'] * model.status[i,t-1] + \
                            df_minDispC.loc[(i,s),t] * (model.status[i,t] - model.status[i,t-1]) + df_maxDispC.loc[(i,s),t] * (1 - model.status[i,t])
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ramp_up_fn = Constraint(model.i, model.s, model.t, rule=ramp_up_fn_rule)

    def ramp_dw_fn_rule(model,i,s,t):
        if t > 1:
            if df_gen_db.loc[i,'centralmente'] == 'SI':
                if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
                    return model.P[i,s,t-1] - model.P[i,s,t] <= df_gen_db.loc[i,'Ramp_Down'] * model.status[i,t] + \
                            df_minDispC.loc[(i,s),t] * (model.status[i,t-1] - model.status[i,t]) + df_maxDispC.loc[(i,s),t] * (1 - model.status[i,t-1])
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ramp_dw_fn = Constraint(model.i, model.s, model.t, rule=ramp_dw_fn_rule)

    ## Power Balance

    def power_balance_rule(model,b,s,t):
        return sum(model.P[i,s,t] for i in model.i if df_SM_map.loc[i,b]) + model.Pot_Ba_dc[b,s,t] + \
            sum(model.pf[l,s,t] for l in model.l if b == df_Branch.loc[l,'to']) == df_load_cal.loc[(b,s),t] + model.Pot_Ba_ch[b,s,t] + \
            sum((model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t]) for l in model.l if b == df_Branch.loc[l,'from'])
    model.power_balance = Constraint(model.b, model.s, model.t, rule=power_balance_rule)

    #### Angle definition

    ##
    def theta_sr_dec_rule(model,l,s,t):
        return sum(model.theta[b,s,t] * df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0) == model.theta_sr_pos[l,s,t] - model.theta_sr_neg[l,s,t]
    model.theta_sr_dec = Constraint(model.l, model.s, model.t, rule=theta_sr_dec_rule)

    ##
    def abs_definition_rule(model,l,s,t):
        return sum(model.theta_aux[l,s,t,L] for L in model.L) == model.theta_sr_pos[l,s,t] + model.theta_sr_neg[l,s,t]
    model.abs_definition = Constraint(model.l, model.s, model.t, rule=abs_definition_rule)

    ##
    def max_angular_difference_rule(model,l,s,t,L):
        return model.theta_aux[l,s,t,L] <= delta_theta
    model.max_angular_difference = Constraint(model.l, model.s, model.t, model.L, rule=max_angular_difference_rule)

    ####DC transmission network security constraint

    def line_flow_rule(model,l,s,t):
        return model.pf[l,s,t] == model.MVA_base * model.susceptance[l] * sum(model.theta[b,s,t] * df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)
    model.line_flow = Constraint(model.l, model.s, model.t, rule=line_flow_rule)

    ##
    def line_min_rule(model,l,s,t):
        return - model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t] <= df_Branch.loc[l,"Flowlimit"]
    model.line_min = Constraint(model.l, model.s, model.t, rule=line_min_rule)

    def line_max_rule(model,l,s,t):
        return model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t] <= df_Branch.loc[l,"Flowlimit"]
    model.line_max = Constraint(model.l, model.s, model.t, rule=line_max_rule)

    ##
    def line_min_1_rule(model,l,s,t):
        return model.pf[l,s,t] >= - df_Branch.loc[l,'Flowlimit']
    model.line_min_1 = Constraint(model.l, model.s, model.t, rule=line_min_1_rule)

    def line_max_1_rule(model,l,s,t):
        return model.pf[l,s,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_max_1 = Constraint(model.l, model.s, model.t, rule=line_max_1_rule)

    #### Losses

    ##
    def losses_rule(model,l,s,t):
        return model.line_losses[l,s,t] == model.MVA_base * model.conductance[l] * sum(model.alpha[L] * model.theta_aux[l,s,t,L] for L in model.L)
    model.losses = Constraint(model.l, model.s, model.t, rule=losses_rule)

    ##
    def losses_max_rule(model,l,s,t):
        return model.line_losses[l,s,t] <= df_Branch.loc[l,'Flowlimit']
    model.losses_max = Constraint(model.l, model.s, model.t, rule=losses_max_rule)

    ### BESS Constraints

    ## power charging Constraints

    def power_c_max_rule(model,b,s,t):
        return model.Pot_Ba_ch[b,s,t] <= Big_number * model.BESS_status[b,t]
    model.power_c_max = Constraint(model.b, model.s, model.t, rule=power_c_max_rule)

    def power_c_max_2_rule(model,b,s,t):
        return model.Pot_Ba_ch[b,s,t] <= model.P_BESS[b]
    model.power_c_max_2 = Constraint(model.b, model.s, model.t, rule=power_c_max_2_rule)

    ## power dischraging Constraints

    def power_d_max_rule(model,b,s,t):
        return model.Pot_Ba_dc[b,s,t] <= Big_number * (1 - model.BESS_status[b,t]) - Big_number * (1 - model.BESS_inst[b])
    model.power_d_max = Constraint(model.b, model.s, model.t, rule=power_d_max_rule)

    def power_d_max_2_rule(model,b,s,t):
        return model.Pot_Ba_dc[b,s,t] <= model.P_BESS[b]
    model.power_d_max_2 = Constraint(model.b, model.s, model.t, rule=power_d_max_2_rule)

    ##

    def power_bat_rule(model,b,t):
        return model.BESS_status[b,t] <= model.BESS_inst[b]
    model.power_bat = Constraint(model.b, model.t, rule=power_bat_rule)

    ## relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,b,s,t):
        if t == model.t.first():
            return model.e_b[b,s,t] == model.E_BESS[b] * SOC_ini + model.n_ch * model.Pot_Ba_ch[b,s,t] - model.Pot_Ba_dc[b,s,t] / model.n_dc
        else:
            return model.e_b[b,s,t] == model.e_b[b,s,t-1] * (1 - model.n_SoC) + model.n_ch * model.Pot_Ba_ch[b,s,t] - \
                                        model.Pot_Ba_dc[b,s,t] / model.n_dc
    model.energy = Constraint(model.b, model.s, model.t, rule=energy_rule)

    ## Energy limits

    def energy_limit_rule(model,b,s,t):
        return model.e_b[b,s,t] <= model.E_BESS[b]
    model.energy_limit = Constraint(model.b, model.s, model.t, rule=energy_limit_rule)

    ## Energy min limit

    def energy_limit_min_rule(model,b,s,t):
        return model.e_b[b,s,t] >= model.E_BESS[b] * SOC_min
    model.energy_limit_min = Constraint(model.b, model.s, model.t, rule=energy_limit_min_rule)

    ## Energy max limit

    def energy_limit_max_rule(model,b,s,t):
        return model.e_b[b,s,t] <= model.E_BESS[b] * SOC_max
    model.energy_limit_max = Constraint(model.b, model.s, model.t, rule=energy_limit_max_rule)

    ## Energy binary variable behavior

    def energy_beh_rule(model,b):
        return model.E_BESS[b] <= model.BESS_inst[b] * Big_number
    model.energy_beh = Constraint(model.b, rule=energy_beh_rule)

    ## BESS number limit

    def bess_number_rule(model):
        return sum(model.BESS_inst[b] for b in model.b) <= BESS_number
    model.bess_number = Constraint(rule=bess_number_rule)

    ## complete power flow

    def complete_pf_rule(model,l,s,t):
        return model.pf_complete[l,s,t] == model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t]
    model.complete_pf = Constraint(model.l, model.s, model.t, rule=complete_pf_rule)

    #### Model Construction

    if s_study == 'No':
        model.bess_number.deactivate()

    if type(sensi) == str:
        ## Función Objetivo
        model.cost_simp.deactivate()
        ## Restricciones
        model.P_lim_max_simp.deactivate()
    else:
        ## Función Objetivo
        model.cost_comp.deactivate()
        ## Restricciones
        # Limites en generación
        model.P_lim_max_comp.deactivate()
        # Operación y costos arranque/parada de unidades térmicas
        model.bin_cons1.deactivate()
        model.bin_cons3.deactivate()
        model.CostSUfn.deactivate()
        model.CostSDfn.deactivate()
        # Rampas
        model.ramp_up_fn.deactivate()
        model.ramp_dw_fn.deactivate()
        # Pérdidas
        model.abs_definition.deactivate()
        model.max_angular_difference.deactivate()
        model.line_min.deactivate()
        model.line_max.deactivate()
        model.losses.deactivate()
        model.losses_max.deactivate()

        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in sensi:
            model.ramp_up_fn.deactivate()
            model.ramp_dw_fn.deactivate()

            ## Función Objetivo
            model.P_lim_max_comp.activate()
            model.P_lim_max_simp.deactivate()

            # Operación y costos arranque/parada de unidades térmicas
            model.bin_cons1.activate()
            model.bin_cons3.activate()
            model.CostSUfn.activate()
            model.CostSDfn.activate()

        if 'Pérdidas' in sensi:
            model.abs_definition.activate()
            model.max_angular_difference.activate()
            model.line_min.activate()
            model.line_max.activate()
            model.losses.activate()
            model.losses_max.activate()
            model.complete_pf.activate()

    ###Solution#####

    # Configuracion:

    solver_selected = combo

    if solver_selected == "CPLEX":
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    print('Reading DATA time:',round(ReadingTime,3), '[s]')
    print('Modeling time:',round(ModelingTime,3), '[s]')
    print('Solving time:',round(SolvingTime,3), '[s]')
    print('Total time:', tiempo)

    #################################################################################
    ####################### Creación de Archivo Excel ###############################
    #################################################################################

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc_res.xlsx'

    Output_data = {}

    for v in model.component_objects(Var):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df_mod(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 1:
            df = pyomo1_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 0:
            Output_data[str(v)] = pyomo_df(v)

    for v in model.component_objects(Objective):
        Output_data[str(v)] = pyomo_df(v)

    if type(sensi) == str:
        cost_name = 'cost_simp'
    else:
        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in sensi:
            cost_name = 'cost_comp'
        else:
            cost_name = 'cost_simp'

    writer = pd.ExcelWriter(os.path.join(mydir, name_file), engine = 'xlsxwriter')

    for idx in Output_data.keys():
        Output_data[idx].to_excel(writer, sheet_name=idx, index=True)
    writer.save()
    # writer.close()
    ##########################################################################

    return Output_data['P_BESS'], Output_data['E_BESS'], name_file, tiempo, df_Branch, df_Bus, Output_data[cost_name], Output_data

def DPRLNSC_H(file_system, db_file, df_DEM, df_disponibilidad_maxima, df_disponibilidad_minima, d_type, df_ofe, fecha, df_PI, df_MPO,
            txt_eff, txt_SOC_min, txt_SOC_ini, txt_time_sim, txt_re_inv, txt_C_Pot, txt_C_Bat, txt_autoD, num_BESS, s_study, sensi, combo):

    StartTime = time.time()

    try:
        os.mkdir('Casos_estudio/loc_size/h5_files')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if os.path.exists('Casos_estudio/loc_size/h5_files/file_name.txt'):
        pass
    else:
        with open('Casos_estudio/loc_size/h5_files/file_name.txt', 'w') as f:
            f.write(' ')

    actual_path = os.getcwd()

    file_name_path = open('Casos_estudio/loc_size/h5_files/file_name.txt')
    file_name_content = file_name_path.read()
    file_name_path.close()

    if file_system.name == 'Colombia_115.xlsx':
        db_system_column = 'bus_over_110'
        h5_name = 'colombia_over_110'
    elif file_system.name == 'Colombia_115_UPME2019.xlsx':
        db_system_column = 'bus_over_110_UPME2019'
        h5_name = 'colombia_over_110_UPME2019'
    elif file_system.name == 'Colombia_220.xlsx':
        db_system_column = 'bus_over_200'
        h5_name = 'colombia_over_220'
    elif file_system.name == 'Colombia_220_ISA2021.xlsx':
        db_system_column = 'bus_over_200_UPME2019'
        h5_name = 'colombia_over_220_ISA'
    elif file_system.name == 'Bogota.xlsx':
        db_system_column = 'bus_BOG'
        h5_name = 'Bogota'
    elif file_system.name == 'Colombia_15_Mod_2018.xlsx':
        db_system_column = 'bus_15N'
        h5_name = 'colombia_15N_2018'
    elif file_system.name == 'Colombia_15_Mod_2030.xlsx':
        db_system_column = 'bus_15N'
        h5_name = 'colombia_15N_2030'
    else:
        print('Sistema equivalente no encontrado!!')

    if file_name_content != file_system.name:

        df_System_data = pd.read_excel(file_system, sheet_name='System_data', header=0, index_col=0)
        df_SM_Unit = pd.read_excel(file_system, sheet_name='SM_Unit', header=0, index_col=0)

        if file_system.name == 'Bogota.xlsx':
            df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)
            df_Bus_pl = pd.read_excel(file_system, sheet_name='Bus_pl', header=0, index_col=0)
            df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)
            df_Branch_pl = pd.read_excel(file_system, sheet_name='Branch_pl', header=0, index_col=0)

        else:

            df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)
            df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)

        df_gen_db_un = pd.read_excel(db_file, sheet_name='gen_un', header=0, index_col=0)
        df_gen_db_pl = pd.read_excel(db_file, sheet_name='gen_pl', header=0, index_col=0)
        df_gen_db_RealIdeal = pd.read_excel(db_file, sheet_name='gen_ide_real', header=0, index_col=0)
        df_gen_db_un.index = df_gen_db_un.loc[:,'name']
        df_gen_db_un.drop(['name'], axis=1, inplace=True)
        df_gen_db_pl.index = df_gen_db_pl.loc[:,'name']
        df_gen_db_pl.drop(['name'], axis=1, inplace=True)
        df_gen_db_RealIdeal.index = df_gen_db_RealIdeal.loc[:,'name']
        df_gen_db_RealIdeal.drop(['name'], axis=1, inplace=True)

        ## gen incidence matrix

        if file_system.name == 'Bogota.xlsx':

            df_SM_map_un = pd.DataFrame(index=df_gen_db_un.index, columns=df_Bus.index.tolist())
            df_SM_map_pl = pd.DataFrame(index=df_gen_db_pl.index, columns=df_Bus_pl.index.tolist())
            df_SM_map_RealIdeal = pd.DataFrame(index=df_gen_db_RealIdeal.index, columns=df_Bus_pl.index.tolist())

        else:

            df_SM_map_un = pd.DataFrame(index=df_gen_db_un.index, columns=df_Bus.index.tolist())
            df_SM_map_pl = pd.DataFrame(index=df_gen_db_pl.index, columns=df_Bus.index.tolist())
            df_SM_map_RealIdeal = pd.DataFrame(index=df_gen_db_RealIdeal.index, columns=df_Bus.index.tolist())

        for i in df_SM_map_un.index:
            for j in df_SM_map_un.columns:
                if df_gen_db_un.loc[i,db_system_column] == j:
                    df_SM_map_un.loc[i,j] = 1
                else:
                    df_SM_map_un.loc[i,j] = 0

        for i in df_SM_map_pl.index:
            for j in df_SM_map_pl.columns:
                if df_gen_db_pl.loc[i,db_system_column] == j:
                    df_SM_map_pl.loc[i,j] = 1
                else:
                    df_SM_map_pl.loc[i,j] = 0

        for i in df_SM_map_RealIdeal.index:
            for j in df_SM_map_RealIdeal.columns:
                if df_gen_db_RealIdeal.loc[i,db_system_column] == j:
                    df_SM_map_RealIdeal.loc[i,j] = 1
                else:
                    df_SM_map_RealIdeal.loc[i,j] = 0

        ## line incidence matrix

        if file_system.name == 'Bogota.xlsx':

            df_line_map_un = pd.DataFrame(index=df_Branch.index.tolist(), columns=df_Bus.index.tolist()).fillna(0)
            df_line_map_pl = pd.DataFrame(index=df_Branch_pl.index.tolist(), columns=df_Bus_pl.index.tolist()).fillna(0)

            # unidades

            for i in df_line_map_un.index:
                for j in df_line_map_un.columns:
                    if j == df_Branch.loc[i,'from']:
                        df_line_map_un.loc[i,j] = 1
                    if j == df_Branch.loc[i,'to']:
                        df_line_map_un.loc[i,j] = -1

            # plantas

            for i in df_line_map_pl.index:
                for j in df_line_map_pl.columns:
                    if df_Branch_pl.loc[i,'from'] == j:
                        df_line_map_pl.loc[i,j] = 1
                    if df_Branch_pl.loc[i,'to'] == j:
                        df_line_map_pl.loc[i,j] = -1

        else:

            df_line_map = pd.DataFrame(index=df_Branch.index.tolist(), columns=df_Bus.index.tolist()).fillna(0)

            for i in df_line_map.index:
                for j in df_line_map.columns:
                    if df_Branch.loc[i,'from'] == j:
                        df_line_map.loc[i,j] = 1
                    if df_Branch.loc[i,'to'] == j:
                        df_line_map.loc[i,j] = -1

        ## save files

        with pd.HDFStore('Casos_estudio/loc_size/h5_files/{}.h5'.format(h5_name)) as store:
            store['df_System_data'] = df_System_data
            store['df_SM_Unit'] = df_SM_Unit
            store['df_SM_map_un'] = df_SM_map_un
            store['df_SM_map_pl'] = df_SM_map_pl
            store['df_SM_map_RealIdeal'] = df_SM_map_RealIdeal

            if file_system.name == 'Bogota.xlsx':

                store['df_Branch'] = df_Branch
                store['df_Branch_pl'] = df_Branch_pl
                store['df_Bus'] = df_Bus
                store['df_Bus_pl'] = df_Bus_pl
                store['df_line_map_un'] = df_line_map_un
                store['df_line_map_pl'] = df_line_map_pl

            else:

                store['df_Branch'] = df_Branch
                store['df_Bus'] = df_Bus
                store['df_line_map'] = df_line_map

            store['df_gen_db_un'] = df_gen_db_un
            store['df_gen_db_pl'] = df_gen_db_pl
            store['df_gen_db_RealIdeal'] = df_gen_db_RealIdeal

        with open('Casos_estudio/loc_size/h5_files/file_name.txt', 'w') as archivo:
            archivo.write(file_system.name)

    else:

        with pd.HDFStore('Casos_estudio/loc_size/h5_files/{}.h5'.format(h5_name)) as store:
            df_System_data = store['df_System_data']
            df_SM_Unit = store['df_SM_Unit']
            df_SM_map_un = store['df_SM_map_un']
            df_SM_map_pl = store['df_SM_map_pl']
            df_SM_map_RealIdeal = store['df_SM_map_RealIdeal']

            if file_system.name == 'Bogota.xlsx':

                df_Branch = store['df_Branch']
                df_Branch_pl = store['df_Branch_pl']
                df_Bus = store['df_Bus']
                df_Bus_pl = store['df_Bus_pl']
                df_line_map_un = store['df_line_map_un']
                df_line_map_pl = store['df_line_map_pl']

            else:

                df_Branch = store['df_Branch']
                df_line_map = store['df_line_map']
                df_Bus = store['df_Bus']

            df_gen_db_un = store['df_gen_db_un']
            df_gen_db_pl = store['df_gen_db_pl']
            df_gen_db_RealIdeal = store['df_gen_db_RealIdeal']

    if d_type == 'Unidades':

        df_SM_map = df_SM_map_un
        df_gen_db = df_gen_db_un

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch
            df_Bus = df_Bus
            df_line_map = df_line_map_un

    elif d_type == 'Acorde a disponibilidad Real':

        df_SM_map = df_SM_map_RealIdeal
        df_gen_db = df_gen_db_RealIdeal

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch_pl
            df_Bus = df_Bus_pl
            df_line_map = df_line_map_pl

    else:

        df_SM_map = df_SM_map_pl
        df_gen_db = df_gen_db_pl

        if file_system.name == 'Bogota.xlsx':

            df_Branch = df_Branch_pl
            df_Bus = df_Bus_pl
            df_line_map = df_line_map_pl

    #### Load

    ## Load factor by bus

    if file_system.name == 'Bogota.xlsx' and (d_type == 'Plantas' or d_type == 'Acorde a disponibilidad Real'):

        df_load_factor = pd.read_excel(file_system, sheet_name='load_factor_pl', header=0, index_col=1)

    else:

        df_load_factor = pd.read_excel(file_system, sheet_name='load_factor', header=0, index_col=1)

    ## bus, sce --> tuple
    busSce = []
    for bu in df_Bus.index:
        for sce in range(len(df_DEM)):
            busSce.append((bu,sce+1))

    ## bus, sce MultiIndex
    idxL = pd.MultiIndex.from_tuples(busSce, names=['buses','sce'])

    ## Load by bus parameter scenarios

    if file_system.name == 'Colombia_115.xlsx':

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)
        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        if t in [1,2,3,4,5,6,22,23,24]:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'min_factor'] * df_DEM[sc].loc[t]
                        else:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    elif file_system.name == 'Colombia_115_UPME2019.xlsx':

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)

        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        if t in [1,2,3,4,5,6,22,23,24]:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'min_factor'] * df_DEM[sc].loc[t]
                        else:
                            df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    else:

        df_load_cal = pd.DataFrame(index=idxL, columns=[x+1 for x in range(24)]).fillna(0)

        for bu in df_Bus.index:
            for sc in range(len(df_DEM)):
                for t in df_load_cal.columns:
                    if bu in df_load_factor.index:
                        df_load_cal.loc[(bu,sc+1),t] = df_load_factor.loc[bu,'max_factor'] * df_DEM[sc].loc[t]

    ReadingTime = time.time() - StartTime

    #### Generators

    ## Generators index
    gen_idx = []
    for i in range(len(df_disponibilidad_maxima)):
        aux = df_disponibilidad_maxima[i].index.get_level_values(0).tolist()
        gen_idx = gen_idx + aux

    gen_idx = list(set(gen_idx))

    ## gen, sce --> tuples
    genSce = []
    for gen in gen_idx:
        for sce in range(len(df_disponibilidad_maxima)):
            genSce.append((gen,sce+1))

    ## gen, sce MultiIndex
    idx = pd.MultiIndex.from_tuples(genSce, names=['plantas','sce'])

    #### Generators and offers complete dataset for all scenarios

    ## max disp
    df_maxDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_maxima)):
        for gen, sce in df_maxDispC.index:
            for co in df_maxDispC.columns:
                if (gen, sce) in df_disponibilidad_maxima[i].index:
                    df_maxDispC.loc[(gen,sce),co] = df_disponibilidad_maxima[i].loc[(gen,sce),co]

    ## min disp
    df_minDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_minima)):
        for gen, sce in df_minDispC.index:
            for co in df_minDispC.columns:
                if (gen, sce) in df_disponibilidad_minima[i].index:
                    df_minDispC.loc[(gen,sce),co] = df_disponibilidad_minima[i].loc[(gen,sce),co]

    ## ofertas
    df_ofeC = pd.DataFrame(index=idx, columns=['Precio','PAPUSD','PAP']).fillna(0)

    for i in range(len(df_ofe)):
        for gen, sce in df_ofeC.index:
            for co in df_ofeC.columns:
                if (gen, sce) in df_ofe[i].index:
                    df_ofeC.loc[(gen,sce),co] = df_ofe[i].loc[(gen,sce),co]

    #### MPO
    df_MPO_C = pd.DataFrame()

    for i in range(len(df_MPO)):
        df_MPO_C[i+1] = df_MPO[i].iloc[0]

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    model = ConcreteModel()

    N_horas = int(txt_time_sim)                                                                    ## Total time steps

    model.t = RangeSet(1, N_horas)                                                  ## scheduling periods
    model.i = Set(initialize=gen_idx, ordered=True) ## Units
    model.b = Set(initialize=df_Bus.index.tolist(), ordered=True)                   ## Bus system set
    model.l = Set(initialize=df_Branch.index.tolist(), ordered=True)                ## Lines set
    model.s = RangeSet(len(df_disponibilidad_maxima))                               ## Scenario number
    model.L = RangeSet(3)

    ########################################## Parameters definitions ####################################

    # Sbase [MVA]
    S_base = df_System_data.loc['S_base'][0]
    model.MVA_base = Param(initialize=S_base)

    ## system parameters
    model.delta_RP = Param(initialize=0.03)
    model.reserva_max = Param(initialize=100)
    model.Holgura = Param(initialize=300)
    model.delta_RRF = Param(initialize=0.00833)
    model.delta_RSF = Param(initialize=0.5)

    # Valor TRM para el día bajo estudio

    url_trm = 'https://www.datos.gov.co/resource/mcec-87by.json'
    data_trm = requests.get(url_trm).json()

    trm_date = fecha[0]
    print(trm_date)
    trm_year, trm_month, trm_day = trm_date.year, trm_date.month, trm_date.day

    if trm_month < 10:
        trm_month = '0{}'.format(trm_month)

    if trm_day < 10:
        trm_day = '0{}'.format(trm_day)

    trm_date_format = '{}-{}-{}T00:00:00.000'.format(trm_year, trm_month, trm_day)

    date_trm_to = []

    for i in range(len(data_trm)):
        date_trm_to.append(data_trm[i]['vigenciadesde'])

    trm_value = None

    for j in date_trm_to:
        if j == trm_date_format:
            index = date_trm_to.index(j)
            trm_value = float(data_trm[index]['valor'])

    if trm_value == None:

        while trm_date.weekday() != 5:
            trm_date = trm_date - timedelta(days=1)

        trm_year, trm_month, trm_day = trm_date.year, trm_date.month, trm_date.day

        if trm_month < 10:
            trm_month = '0{}'.format(trm_month)

        if trm_day < 10:
            trm_day = '0{}'.format(trm_day)

        trm_date_format = '{}-{}-{}T00:00:00.000'.format(trm_year, trm_month, trm_day)

        for j in date_trm_to:
            if j == trm_date_format:

                index = date_trm_to.index(j)
                trm_value = float(data_trm[index]['valor'])

    ### BESS parameters

    SOC_min = 1 - float(txt_SOC_min)                                                               ## Minimum State of Charge of BESS
    SOC_ini = float(txt_SOC_ini)                                                               ## Initial State of Charge of BESS
    SOC_max = 1.0
    BESS_number = num_BESS
    Big_number = 1e20

    model.n_dc = Param(initialize=round(pow(txt_eff,0.5),4))                                         ## Discharging efficiency
    model.n_ch = Param(initialize=round(pow(txt_eff,0.5),4))                                         ## Charging efficiency
    model.n_SoC = Param(initialize=float(txt_autoD))                                   ## Self-discharging

    #### BESS parameters
    Costo_potencia = round(int(txt_C_Pot) * trm_value / (365 * int(txt_re_inv) * 24),2)
    Costo_energia = round(int(txt_C_Bat) * trm_value / (365 * int(txt_re_inv) * 24),2)
    model.Costo_potencia = Param(initialize=Costo_potencia * N_horas)         ## Costo del inversor de potencia de la batería
    model.Costo_energia = Param(initialize=Costo_energia * N_horas)           ## Costo de los modulos de baterías

    # #### line parameters

    ## Susceptance of each line
    def susceptance_init(model,l):
        return 1 / df_Branch.loc[l,'X']
    model.susceptance = Param(model.l, rule=susceptance_init)

    ## Conductance of each line
    def conductance_init(model,l):
        return df_Branch.loc[l,'R'] /(df_Branch.loc[l,'X'] ** 2 + df_Branch.loc[l,'R'] ** 2)
    model.conductance = Param(model.l, initialize=conductance_init)

    ##angular difference slope
    delta_theta = (20 * math.pi / 180)

    def alpha_init(model,L):
        return delta_theta * (2 * L - 1)
    model.alpha = Param(model.L, initialize=alpha_init)

    #################################### Scenarios Parameters ##############################

    prob = {}

    for s in model.s:
        prob[s] = 1 / len(model.s)
    model.prob = Param(model.s, initialize=prob)

    #################################### Define Variables ##############################

    ## Dispatch variables
    model.status = Var(model.i, model.t, within=Binary, initialize=0)                       ## Commitment of unit i at time t
    model.P = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)         ## Power dispatch of unit i at time t
    model.costSU = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)    ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)    ## Shut-Down cost of unit i
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)                       ## Startup status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)                       ## Shutdown status of unit i
    model.V_Rac = Var(model.b, model.t, bounds=(0,2400), initialize=0)                      ## Energy of rationing

    ## Power Flow variables
    model.theta = Var(model.b, model.s, model.t, bounds=(-math.pi,math.pi))                 ## Voltage angle
    model.pf = Var(model.l, model.s, model.t, within=Reals)                                 ## Power flow line l at time t scenario u
    model.pf_complete = Var(model.l, model.s, model.t, within=Reals, initialize=0)

    if file_system.name == 'Bogota.xlsx' and (d_type == 'Pl' or d_type == 'Acorde a disponibilidad Real'):

        Slack_bus = df_System_data.loc['Slack_bus_pl'][0]

    else:

        Slack_bus = df_System_data.loc['Slack_bus'][0]

    for s in model.s:
        for t in model.t:
            model.theta[Slack_bus,s,t].fix(0)

    ## Losses
    model.theta_sr_pos = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.theta_sr_neg = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.line_losses = Var(model.l, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.theta_aux = Var(model.l, model.s, model.t, model.L, domain=NonNegativeReals, initialize=0)

    ## reservas
    model.RPUP = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.RPDN = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.RSUP = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)
    model.RSDN = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0)

    ## BESS variables
    model.BESS_inst = Var(model.b, within=Binary)                     ## Status of battery charge
    model.BESS_status = Var(model.b, model.t, within=Binary)                     ## Status of battery discharge
    model.Pot_Ba_ch = Var(model.b, model.s, model.t, domain=NonNegativeReals)            # Power in battery charge [kW]
    model.Pot_Ba_dc = Var(model.b, model.s, model.t, domain=NonNegativeReals)            # Power in battery discharge [kW]
    model.e_b = Var(model.b, model.s, model.t, domain=NonNegativeReals)                  # Energy of battery [kWh]
    model.P_BESS = Var(model.b, domain = NonNegativeReals, initialize=0) ## Power Size (New)
    model.E_BESS = Var(model.b, domain = NonNegativeReals, initialize=0) ## Energy Size (New)

    ## BESS reserve variables

    model.RRUP_ch =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RRUP_dc =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RRUP =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RRDN_ch =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RRDN_dc =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RRDN =  Var(model.b, model.s, model.t, domain=NonNegativeReals)

    model.RSUP_ch =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RSUP_dc =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RSUP_n =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RSDN_ch =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RSDN_dc =  Var(model.b, model.s, model.t, domain=NonNegativeReals)
    model.RSDN_n =  Var(model.b, model.s, model.t, domain=NonNegativeReals)

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################

    StartTime = time.time()

    ## Objective function definition

    def cost_rule(model):
        return sum(model.prob[s] * sum(df_ofeC.loc[(i,s),'Precio'] * model.P[i,s,t] for i in model.i for t in model.t) + \
            sum(model.costSD[i,s,t] + model.costSU[i,s,t] for i in model.i for t in model.t) + \
            sum((model.P[i,s,t] - df_PI.loc[(i,s),t]) * int(df_MPO_C.loc[t,s]) for i in model.i for t in model.t) for s in model.s) + \
            sum(model.P_BESS[b] * model.Costo_potencia + model.E_BESS[b] * model.Costo_energia for b in model.b)
    model.cost_comp = Objective(rule=cost_rule, sense=minimize)

    ##### Constraints

    ## Power limits

    def P_lim_max_comp_rule(model,i,s,t):
        return model.P[i,s,t] + model.RPUP[i,s,t] + model.RSUP[i,s,t] <= df_maxDispC.loc[(i,s),t] * model.status[i,t]
    model.P_lim_max_comp = Constraint(model.i, model.s, model.t, rule=P_lim_max_comp_rule)

    def P_lim_min_rule(model,i,s,t):
        return model.P[i,s,t] - (model.RPDN[i,s,t] + model.RSDN[i,s,t]) >= df_minDispC.loc[(i,s),t] * model.status[i,t]
    model.P_lim_min = Constraint(model.i, model.s, model.t, rule=P_lim_min_rule)

    ## PAP cost & Integer Constraint

    def bin_cons1_rule(model,i,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            if t == model.t.first():
                return model.SU[i,t] - model.SD[i,t] == model.status[i,t]# - model.onoff_t0[i]
            else:
                return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
        else:
            return Constraint.Skip
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons3_rule(model,i,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.SU[i,t] + model.SD[i,t] <= 1
        else:
            return Constraint.Skip
    model.bin_cons3 = Constraint(model.i, model.t, rule=bin_cons3_rule)

    def CostSUfn_init(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.costSU[i,s,t] == df_ofeC.loc[(i,s),'PAP'] * model.SU[i,t]
        else:
            return Constraint.Skip
    model.CostSUfn = Constraint(model.i, model.s, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
            return model.costSD[i,s,t] == df_ofeC.loc[(i,s),'PAP'] * model.SD[i,t]
        else:
            return Constraint.Skip
    model.CostSDfn = Constraint(model.i, model.s, model.t, rule=CostSDfn_init)

    ## Ramp Constraints

    def ramp_up_fn_rule(model,i,s,t):
        if t > 1:
            if df_gen_db.loc[i,'centralmente'] == 'SI':
                if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
                    return model.P[i,s,t] - model.P[i,s,t-1] <= df_gen_db.loc[i,'Ramp_Up'] * model.status[i,t-1] + \
                            df_minDispC.loc[(i,s),t] * (model.status[i,t] - model.status[i,t-1]) + df_maxDispC.loc[(i,s),t] * (1 - model.status[i,t])
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ramp_up_fn = Constraint(model.i, model.s, model.t, rule=ramp_up_fn_rule)

    def ramp_dw_fn_rule(model,i,s,t):
        if t > 1:
            if df_gen_db.loc[i,'centralmente'] == 'SI':
                if df_gen_db.loc[i,'Gen_type'] == 'TERMICA':
                    return model.P[i,s,t-1] - model.P[i,s,t] <= df_gen_db.loc[i,'Ramp_Down'] * model.status[i,t] + \
                            df_minDispC.loc[(i,s),t] * (model.status[i,t-1] - model.status[i,t]) + df_maxDispC.loc[(i,s),t] * (1 - model.status[i,t-1])
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ramp_dw_fn = Constraint(model.i, model.s, model.t, rule=ramp_dw_fn_rule)

    ## ---
    def reserva_primaria_minima_rule(model,i,s,t):
        return model.RPUP[i,s,t] >= model.P[i,s,t] * model.delta_RP
    model.reserva_primaria_minima = Constraint(model.i, model.s, model.t, rule=reserva_primaria_minima_rule)

    def reserva_secundaria_maxima_rule(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'HIDRAULICA':
            return model.RSUP[i,s,t] <= model.reserva_max * model.status[i,t]
        else:
            return Constraint.Skip
    model.reserva_secundaria_maxima = Constraint(model.i, model.s, model.t, rule=reserva_secundaria_maxima_rule)

    def reserva_primeria_simetrica_rule(model,i,s,t):
        return model.RPUP[i,s,t] == model.RPDN[i,s,t]
    model.reserva_primeria_simetrica = Constraint(model.i, model.s, model.t, rule=reserva_primeria_simetrica_rule)

    def reserva_secundaria_simetrica_rule(model,i,s,t):
        if df_gen_db.loc[i,'Gen_type'] == 'HIDRAULICA':
            return model.RSUP[i,s,t] == model.RSDN[i,s,t]
        else:
            return Constraint.Skip
    model.reserva_secundaria_simetrica = Constraint(model.i, model.s, model.t, rule=reserva_secundaria_simetrica_rule)

    # ## Reserva
    # def reserva_up_power_rule(model,b,s,t):
    #     return sum((model.status[i,t] * df_maxDispC.loc[(i,s),t] - model.P[i,s,t]) for i in model.i if df_SM_map.loc[i,b]) + (model.P_BESS[b] - model.Pot_Ba_dc[b,s,t] + model.Pot_Ba_ch[b,s,t]) >= Holgura[s-1][t]
    # model.reserva_up_power = Constraint(model.b, model.s, model.t, rule=reserva_up_power_rule)

    # def reserva_up_energy_rule(model,b,s,t):
    #     return model.e_b[b,s,t] - (model.P_BESS[b] - model.Pot_Ba_dc[b,s,t]) * 0.5 >= model.E_BESS[b] * SOC_min
    # model.reserva_up_energy = Constraint(model.b, model.s, model.t, rule=reserva_up_energy_rule)

    ## Power Balance

    def power_balance_rule(model,b,s,t):
        return sum(model.P[i,s,t] for i in model.i if df_SM_map.loc[i,b]) + model.Pot_Ba_dc[b,s,t] + \
            sum(model.pf[l,s,t] for l in model.l if b == df_Branch.loc[l,'to']) == df_load_cal.loc[(b,s),t] + model.Pot_Ba_ch[b,s,t] + \
            sum((model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t]) for l in model.l if b == df_Branch.loc[l,'from'])
    model.power_balance = Constraint(model.b, model.s, model.t, rule=power_balance_rule)

    ## reserve balance

    def reserva_balance_rule(model,s,t):
        return sum(model.RSUP[i,s,t] for i in model.i) + sum(model.RSUP_n[b,s,t] for b in model.b) == model.Holgura
    model.reserva_balance = Constraint(model.s, model.t, rule=reserva_balance_rule)

    #### Angle definition

    ##
    def theta_sr_dec_rule(model,l,s,t):
        return sum(model.theta[b,s,t] * df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0) == model.theta_sr_pos[l,s,t] - model.theta_sr_neg[l,s,t]
    model.theta_sr_dec = Constraint(model.l, model.s, model.t, rule=theta_sr_dec_rule)

    ##
    def abs_definition_rule(model,l,s,t):
        return sum(model.theta_aux[l,s,t,L] for L in model.L) == model.theta_sr_pos[l,s,t] + model.theta_sr_neg[l,s,t]
    model.abs_definition = Constraint(model.l, model.s, model.t, rule=abs_definition_rule)

    ##
    def max_angular_difference_rule(model,l,s,t,L):
        return model.theta_aux[l,s,t,L] <= delta_theta
    model.max_angular_difference = Constraint(model.l, model.s, model.t, model.L, rule=max_angular_difference_rule)

    ####DC transmission network security constraint

    def line_flow_rule(model,l,s,t):
        return model.pf[l,s,t] == model.MVA_base * model.susceptance[l] * sum(model.theta[b,s,t] * df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)
    model.line_flow = Constraint(model.l, model.s, model.t, rule=line_flow_rule)

    ##
    def line_min_rule(model,l,s,t):
        return - model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t] <= df_Branch.loc[l,"Flowlimit"]
    model.line_min = Constraint(model.l, model.s, model.t, rule=line_min_rule)

    def line_max_rule(model,l,s,t):
        return model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t] <= df_Branch.loc[l,"Flowlimit"]
    model.line_max = Constraint(model.l, model.s, model.t, rule=line_max_rule)

    ##
    def line_min_1_rule(model,l,s,t):
        return model.pf[l,s,t] >= - df_Branch.loc[l,'Flowlimit']
    model.line_min_1 = Constraint(model.l, model.s, model.t, rule=line_min_1_rule)

    def line_max_1_rule(model,l,s,t):
        return model.pf[l,s,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_max_1 = Constraint(model.l, model.s, model.t, rule=line_max_1_rule)

    #### Losses

    ##
    def losses_rule(model,l,s,t):
        return model.line_losses[l,s,t] == model.MVA_base * model.conductance[l] * sum(model.alpha[L] * model.theta_aux[l,s,t,L] for L in model.L)
    model.losses = Constraint(model.l, model.s, model.t, rule=losses_rule)

    ##
    def losses_max_rule(model,l,s,t):
        return model.line_losses[l,s,t] <= df_Branch.loc[l,'Flowlimit']
    model.losses_max = Constraint(model.l, model.s, model.t, rule=losses_max_rule)

    ### BESS Constraints

    ## power charging Constraints

    def power_c_max_rule(model,b,s,t):
        return model.Pot_Ba_ch[b,s,t] <= Big_number * model.BESS_status[b,t]
    model.power_c_max = Constraint(model.b, model.s, model.t, rule=power_c_max_rule)

    def power_c_max_2_rule(model,b,s,t):
        return model.Pot_Ba_ch[b,s,t] <= model.P_BESS[b]
    model.power_c_max_2 = Constraint(model.b, model.s, model.t, rule=power_c_max_2_rule)

    ## power dischraging Constraints

    def power_d_max_rule(model,b,s,t):
        return model.Pot_Ba_dc[b,s,t] <= Big_number * (1 - model.BESS_status[b,t]) - Big_number * (1 - model.BESS_inst[b])
    model.power_d_max = Constraint(model.b, model.s, model.t, rule=power_d_max_rule)

    def power_d_max_2_rule(model,b,s,t):
        return model.Pot_Ba_dc[b,s,t] <= model.P_BESS[b]
    model.power_d_max_2 = Constraint(model.b, model.s, model.t, rule=power_d_max_2_rule)

    ##

    def power_bat_rule(model,b,t):
        return model.BESS_status[b,t] <= model.BESS_inst[b]
    model.power_bat = Constraint(model.b, model.t, rule=power_bat_rule)

    ## BESS RRF reserve limits

    def RRF_UP_dc_rule(model,b,s,t):
        return model.RRUP_dc[b,s,t] <= model.P_BESS[b] - model.Pot_Ba_dc[b,s,t]
    model.RRF_UP_dc = Constraint(model.b, model.s, model.t, rule = RRF_UP_dc_rule)

    def RRF_UP_ch_rule(model,b,s,t):
        return model.RRUP_ch[b,s,t] <= model.Pot_Ba_ch[b,s,t]
    model.RRF_UP_ch = Constraint(model.b, model.s, model.t, rule = RRF_UP_ch_rule)

    def RRF_UP_rule(model,b,s,t):
        return model.RRUP[b,s,t] == model.RRUP_ch[b,s,t] + model.RRUP_dc[b,s,t]
    model.RRF_UP = Constraint(model.b, model.s, model.t, rule = RRF_UP_rule)

    def RRF_DN_dc_rule(model,b,s,t):
        return model.RRDN_dc[b,s,t] <= model.Pot_Ba_dc[b,s,t]
    model.RRF_DN_dc = Constraint(model.b, model.s, model.t, rule = RRF_DN_dc_rule)

    def RRF_DN_ch_rule(model,b,s,t):
        return model.RRDN_ch[b,s,t] <= model.P_BESS[b] - model.Pot_Ba_ch[b,s,t]
    model.RRF_DN_ch = Constraint(model.b, model.s, model.t, rule = RRF_DN_ch_rule)

    def RRF_DN_rule(model,b,s,t):
        return model.RRDN[b,s,t] == model.RRDN_ch[b,s,t] + model.RRDN_dc[b,s,t]
    model.RRF_DN = Constraint(model.b, model.s, model.t, rule = RRF_DN_rule)

    def RRF_Simetric_rule(model,b,s,t):
        return model.RRDN[b,s,t] == model.RRUP[b,s,t]
    model.RRF_Simetric = Constraint(model.b, model.s, model.t, rule = RRF_Simetric_rule)

    ## BESS RSF reserve limits

    def RSF_UP_dc_rule(model,b,s,t):
        return model.RSUP_dc[b,s,t] <= model.P_BESS[b] - model.Pot_Ba_dc[b,s,t]
    model.RSF_UP_dc = Constraint(model.b, model.s, model.t, rule = RSF_UP_dc_rule)

    def RSF_UP_ch_rule(model,b,s,t):
        return model.RSUP_ch[b,s,t] <= model.Pot_Ba_ch[b,s,t]
    model.RSF_UP_ch = Constraint(model.b, model.s, model.t, rule = RSF_UP_ch_rule)

    def RSF_UP_rule(model,b,s,t):
        return model.RSUP_n[b,s,t] == model.RSUP_ch[b,s,t] + model.RSUP_dc[b,s,t]
    model.RSF_UP = Constraint(model.b, model.s, model.t, rule = RSF_UP_rule)

    def RSF_DN_dc_rule(model,b,s,t):
        return model.RSDN_dc[b,s,t] <= model.Pot_Ba_dc[b,s,t]
    model.RSF_DN_dc = Constraint(model.b, model.s, model.t, rule = RSF_DN_dc_rule)

    def RSF_DN_ch_rule(model,b,s,t):
        return model.RSDN_ch[b,s,t] <= model.P_BESS[b] - model.Pot_Ba_ch[b,s,t]
    model.RSF_DN_ch = Constraint(model.b, model.s, model.t, rule = RSF_DN_ch_rule)

    def RSF_DN_rule(model,b,s,t):
        return model.RSDN_n[b,s,t] == model.RSDN_ch[b,s,t] + model.RSDN_dc[b,s,t]
    model.RSF_DN = Constraint(model.b, model.s, model.t, rule = RSF_DN_rule)

    def RSF_Simetric_rule(model,b,s,t):
        return model.RSDN_n[b,s,t] == model.RSUP_n[b,s,t]
    model.RSF_Simetric = Constraint(model.b, model.s, model.t, rule = RSF_Simetric_rule)


    ## relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,b,s,t):
        if t == model.t.first():
            return model.e_b[b,s,t] == model.E_BESS[b] * SOC_ini + model.n_ch * model.Pot_Ba_ch[b,s,t] - model.Pot_Ba_dc[b,s,t] / model.n_dc
        else:
            return model.e_b[b,s,t] == model.e_b[b,s,t-1] * (1 - model.n_SoC) + model.n_ch * model.Pot_Ba_ch[b,s,t] - \
                                        model.Pot_Ba_dc[b,s,t] / model.n_dc
    model.energy = Constraint(model.b, model.s, model.t, rule=energy_rule)

    ## Energy limits

    def energy_limit_rule(model,b,s,t):
        return model.e_b[b,s,t] + model.RRDN_ch[b,s,t]*model.delta_RRF + model.RSDN_ch[b,s,t]*model.delta_RSF <= model.E_BESS[b]
    model.energy_limit = Constraint(model.b, model.s, model.t, rule=energy_limit_rule)

    ## Energy min limit

    def energy_limit_min_rule(model,b,s,t):
        return model.e_b[b,s,t] - model.RRUP_dc[b,s,t]*model.delta_RRF - model.RSUP_dc[b,s,t]*model.delta_RSF>= model.E_BESS[b] * SOC_min
    model.energy_limit_min = Constraint(model.b, model.s, model.t, rule=energy_limit_min_rule)

    ## Energy max limit

    def energy_limit_max_rule(model,b,s,t):
        return model.e_b[b,s,t] <= model.E_BESS[b] * SOC_max
    model.energy_limit_max = Constraint(model.b, model.s, model.t, rule=energy_limit_max_rule)

    ## Energy binary variable behavior

    def energy_beh_rule(model,b):
        return model.E_BESS[b] <= model.BESS_inst[b] * Big_number
    model.energy_beh = Constraint(model.b, rule=energy_beh_rule)

    ## BESS number limit

    def bess_number_rule(model):
        return sum(model.BESS_inst[b] for b in model.b) <= BESS_number
    model.bess_number = Constraint(rule=bess_number_rule)

    ## complete power flow

    def complete_pf_rule(model,l,s,t):
        return model.pf_complete[l,s,t] == model.pf[l,s,t] + 0.5 * model.line_losses[l,s,t]
    model.complete_pf = Constraint(model.l, model.s, model.t, rule=complete_pf_rule)

    #### Model Construction

    if s_study == 'No':
        # model.bess_number.deactivate()
        pass

    if type(sensi) == str:
        pass
    else:
        ## Función Objetivo
        model.cost_comp.deactivate()
        ## Restricciones
        # Limites en generación
        model.P_lim_max_comp.deactivate()
        # Operación y costos arranque/parada de unidades térmicas
        model.bin_cons1.deactivate()
        model.bin_cons3.deactivate()
        model.CostSUfn.deactivate()
        model.CostSDfn.deactivate()
        # Rampas
        model.ramp_up_fn.deactivate()
        model.ramp_dw_fn.deactivate()
        # Pérdidas
        model.abs_definition.deactivate()
        model.max_angular_difference.deactivate()
        model.line_min.deactivate()
        model.line_max.deactivate()
        model.losses.deactivate()
        model.losses_max.deactivate()

        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in sensi:
            model.ramp_up_fn.deactivate()
            model.ramp_dw_fn.deactivate()

            ## Función Objetivo
            model.P_lim_max_comp.activate()
            model.P_lim_max_simp.deactivate()

            # Operación y costos arranque/parada de unidades térmicas
            model.bin_cons1.activate()
            model.bin_cons3.activate()
            model.CostSUfn.activate()
            model.CostSDfn.activate()

        if 'Pérdidas' in sensi:
            model.abs_definition.activate()
            model.max_angular_difference.activate()
            model.line_min.activate()
            model.line_max.activate()
            model.losses.activate()
            model.losses_max.activate()
            model.complete_pf.activate()

    ###Solution#####

    # Configuracion:

    solver_selected = combo

    if solver_selected == "CPLEX":
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    print('Reading DATA time:',round(ReadingTime,3), '[s]')
    print('Modeling time:',round(ModelingTime,3), '[s]')
    print('Solving time:',round(SolvingTime,3), '[s]')
    print('Total time:', tiempo)

    #################################################################################
    ####################### Creación de Archivo Excel ###############################
    #################################################################################

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc_res.xlsx'

    Output_data = {}

    for v in model.component_objects(Var):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df_mod(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 1:
            df = pyomo1_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 0:
            Output_data[str(v)] = pyomo_df(v)

    for v in model.component_objects(Objective):
        Output_data[str(v)] = pyomo_df(v)

    if type(sensi) == str:
        cost_name = 'cost_simp'
    else:
        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in sensi:
            cost_name = 'cost_comp'
        else:
            cost_name = 'cost_simp'

    writer = pd.ExcelWriter(os.path.join(mydir, name_file), engine = 'xlsxwriter')

    for idx in Output_data.keys():
        Output_data[idx].to_excel(writer, sheet_name=idx, index=True)
    writer.save()
    # writer.close()
    ##########################################################################

    return Output_data['P_BESS'], Output_data['E_BESS'], name_file, tiempo, df_Branch, df_Bus, Output_data[cost_name], Output_data

# ## Despacho ideal escenarios (DISC: despacho ideal con enfoque estocástico)
def DISC(df_demSIN, df_disponibilidad_maxima, df_disponibilidad_minima, df_ofe, cro, combo):

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    ## Generators index
    gen_idx = []
    for i in range(len(df_disponibilidad_maxima)):
        aux = df_disponibilidad_maxima[i].index.get_level_values(0).tolist()
        gen_idx = gen_idx + aux

    gen_idx = list(set(gen_idx))

    ## gen, sce --> tuples
    genSce = []
    for gen in gen_idx:
        for sce in range(len(df_disponibilidad_maxima)):
            genSce.append((gen,sce+1))

    ## gen, sce MultiIndex
    idx = pd.MultiIndex.from_tuples(genSce, names=['plantas','sce'])

    #### Generators complete dataset for all scenarios

    ## max disp
    df_maxDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_maxima)):
        for gen, sce in df_maxDispC.index:
            for co in df_maxDispC.columns:
                if (gen, sce) in df_disponibilidad_maxima[i].index:
                    df_maxDispC.loc[(gen,sce),co] = df_disponibilidad_maxima[i].loc[(gen,sce),co]

    ## min disp
    df_minDispC = pd.DataFrame(index=idx, columns=[x+1 for x in range(24)]).fillna(0)

    for i in range(len(df_disponibilidad_minima)):
        for gen, sce in df_minDispC.index:
            for co in df_minDispC.columns:
                if (gen, sce) in df_disponibilidad_minima[i].index:
                    df_minDispC.loc[(gen,sce),co] = df_disponibilidad_minima[i].loc[(gen,sce),co]

    ## ofertas
    df_ofeC = pd.DataFrame(index=idx, columns=['Precio','PAPUSD','PAP']).fillna(0)

    for i in range(len(df_ofe)):
        for gen, sce in df_ofeC.index:
            for co in df_ofeC.columns:
                if (gen, sce) in df_ofe[i].index:
                    df_ofeC.loc[(gen,sce),co] = df_ofe[i].loc[(gen,sce),co]

    ## demanda
    df_demSce = pd.DataFrame()

    for i in range(len(df_demSIN)):
        df_demSce[i+1] = df_demSIN[i]

    model = ConcreteModel()

    model.t = Set(initialize=df_demSce.index.tolist(), ordered=True)            ## scheduling periods
    model.i = Set(initialize=gen_idx, ordered=True)                             ## Units
    model.s = RangeSet(len(df_disponibilidad_maxima))                           ## scenarios
    ########################################## Parameters definitions ####################################

    # def CRO_init(model):
    #     return cro
    model.CRO_est = Param(initialize=cro)    ## Costo incremental de racionamiento de energía

    #################################### Scenarios Parameters ##############################

    prob = {}

    for s in model.s:
        prob[s] = 1 / len(model.s)
    model.prob = Param(model.s, initialize=prob)

    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################
    model.status = Var(model.i, model.t, within=Binary, initialize=0)               ## Commitment of unit i at time t
    model.P = Var(model.i, model.s, model.t, domain=NonNegativeReals, initialize=0) ## Power dispatch of unit i at time t
    model.V_Rac = Var(model.s, model.t, bounds=(0,2400))                            ## Energy of rationing
    model.costSU = Var(model.i, model.s, model.t, domain=NonNegativeReals)          ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.s, model.t, domain=NonNegativeReals)          ## Shut-Down cost of unit i

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################
    StartTime = time.time()

    ## Objective function

    def cost_rule(model):
        return sum(model.prob[s] * sum(model.P[i,s,t] * df_ofeC.loc[(i,s),'Precio'] for i in model.i for t in model.t) + \
                sum(model.costSD[i,s,t] + model.costSU[i,s,t] for i in model.i for t in model.t) + \
                sum((model.CRO_est) * model.V_Rac[s,t] for s in model.s for t in model.t) for s in model.s)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################

    #### Dispath constraints

    ## Power limits

    def P_lim_max_rule(model,i,s,t):
        if (i,s) in df_maxDispC.index:
            return model.P[i,s,t] <= df_maxDispC.loc[(i,s),t] * model.status[i,t]
        else:
            return model.P[i,s,t] <= 0
    model.P_lim_max = Constraint(model.i, model.s, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,s,t):
        if (i,s) in df_minDispC.index:
            return model.P[i,s,t] >= df_minDispC.loc[(i,s),t]
        else:
            return model.P[i,s,t] >= 0
    model.P_lim_min = Constraint(model.i, model.s, model.t, rule=P_lim_min_rule)

    ## PAP cost

    def CostSUfn_init(model,i,s,t):
        if t == model.t.first():
            return model.costSU[i,s,t] >= df_ofeC.loc[(i,s),'PAP'] * model.status[i,t]
        else:
            return model.costSU[i,s,t] >= df_ofeC.loc[(i,s),'PAP'] * (model.status[i,t] - model.status[i,t-1])
    model.CostSUfn = Constraint(model.i, model.s, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,s,t):
        if t == model.t.first():
            return model.costSD[i,s,t] >= df_ofeC.loc[(i,s),'PAP'] * model.status[i,t]
        else:
            return model.costSD[i,s,t] >= df_ofeC.loc[(i,s),'PAP'] * (model.status[i,t-1] - model.status[i,t])
    model.CostSDfn = Constraint(model.i, model.s, model.t, rule=CostSDfn_init)

    ## Power Balance

    def power_balance_rule(model,s,t):
        return sum(model.P[i,s,t] for i in model.i) + model.V_Rac[s,t] == df_demSce.loc[t,s]
    model.power_balance = Constraint(model.s, model.t, rule=power_balance_rule)

    ###Solution#####

    # Configuracion:

    solver_selected = combo

    if solver_selected == "CPLEX":
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    print('Modeling time:',round(ModelingTime,3), '[s]')
    print('Solving time:',round(SolvingTime,3), '[s]')
    print('Total time:', tiempo)

    #################################################################################
    ####################### Creación de Archivo Excel ###############################
    #################################################################################

    if len(model.P) == 0:
        df_gen_i = pd.DataFrame()
    else:
        df_gen_i = pyomo3_df(model.P)
        df_gen_i = df_gen_i.T

    if len(model.cost) == 0:
        df_cost = pd.DataFrame()
    else:
        df_cost = pyomo_df(model.cost)

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc_rest(ide).xlsx'

    path = os.path.join(mydir, name_file)

    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')

    df_gen_i.to_excel(writer, sheet_name='gen', index=True)
    df_cost.to_excel(writer, sheet_name='cost', index=True)
    writer.save()
    # writer.close()
    ##########################################################################

    return df_gen_i, tiempo, name_file

def graph_results_res(Output_data, sensi):

    time = Output_data['P'].columns

    title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'14'}

    st.markdown('### Graficación de Resultados:')

    st.write('#### Generación')

    ## Generación

    fig = go.Figure()
    for s in range(len(Output_data['P'].index.levels[1])):
        for g in Output_data['P'].index.levels[0]:
            if sum(Output_data['P'].loc[(g,s+1),:]) > 0:
                fig.add_trace(go.Scatter(x=time, y=Output_data['P'].loc[(g,s+1),:], name=g, line_shape='hv'))

        if len(Output_data['P'].index.levels[1]) == 1:
            fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                title='Generación',
                xaxis_title='Hora',
                yaxis_title='[MW]')
        else:
            fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                            title='Generación en el escenario {}'.format(s+1),
                            xaxis_title='Hora',
                            yaxis_title='[MW]')

        fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig)

    ## power flow

    st.markdown('#### Flujos de potencia')

    fig = go.Figure()

    if 'Pérdidas' in sensi:
        real_pf = Output_data['pf_complete']
    else:
        real_pf = Output_data['pf']

    fig = go.Figure()
    for s in range(len(real_pf.index.levels[1])):
        for l in real_pf.index.levels[0]:
            fig.add_trace(go.Scatter(x=time, y=real_pf.loc[(l,s+1),:], name=l, line_shape='hv'))

        if len(Output_data['P'].index.levels[1]) == 1:
            fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                title='Flujos de potencia',
                xaxis_title='Hora',
                yaxis_title='[MW]')
        else:
            fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                            title='Flujos de potencia en el escenario {}'.format(s+1),
                            xaxis_title='Hora',
                            yaxis_title='[MW]')

        fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig)

    ## SAE

    st.markdown('#### Operación SAE')

    fig = go.Figure()

    if sum(Output_data['Pot_Ba_ch'].loc[(b,s),t] for b,s in Output_data['Pot_Ba_ch'].index for t in Output_data['Pot_Ba_ch'].columns) <= 1e-5:
        st.info('¡Ningún Sistema de Almacenamieto de Energía fue Instalado!')
    else:
        for s in range(len(Output_data['Pot_Ba_ch'].index.levels[1])):
            for b in Output_data['Pot_Ba_ch'].index.levels[0]:
                if sum(Output_data['Pot_Ba_ch'].loc[(b,s+1),:]) > 0:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=time, y=Output_data['e_b'].loc[(b,s+1),:], name='Energía [MWh]', line_shape='linear'), secondary_y=False)
                    fig.add_trace(go.Scatter(x=time, y=Output_data['Pot_Ba_ch'].loc[(b,s+1),2:24], name='Carga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)
                    fig.add_trace(go.Scatter(x=time, y=Output_data['Pot_Ba_dc'].loc[(b,s+1),2:24], name='Descarga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)

                    if len(Output_data['P'].index.levels[1]) == 1:
                        fig.update_layout(legend=dict(y=1, traceorder='reversed', font_size=8),
                            title='Operación SAE ubicado en el nodo {}'.format(b),
                            autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
                    else:
                        fig.update_layout(legend=dict(y=1, traceorder='reversed', font_size=8),
                                        title='Operación SAE ubicado en el nodo {} en el escenario {}'.format(b,s+1),
                                        autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')

                    fig.update_xaxes(title_text='Hora', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
                    fig.update_yaxes(title_text='Energía [MWh]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=False)
                    fig.update_yaxes(title_text='Potencia [MW]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=True)
                    st.plotly_chart(fig)