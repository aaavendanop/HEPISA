# coding: utf-8
from pyomo.environ import *
import pandas as pd
import os
import time
from datetime import timedelta

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funciones.read_download_files import *
from funciones.save_files import *

# import urllib.request
# import math
# import datetime
# import numpy as np

def opt_despacho(df_disp, df_minop, df_ofe, df_MPO, df_PRON_DEM, df_SAEInfo, SAE_num, num_tdp_i, num_tdp_f, num_td_i,
                num_td_f, combo):

    ################################################# Load files ################################################################

    StartTime = time.time()

    ReadingTime = time.time() - StartTime

    ################################################ Sets Definitions ################################################
    StartTime = time.time()
    model = ConcreteModel()
    ##

    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)      ## scheduling periods
    model.i = Set(initialize=df_disp.index.tolist(), ordered=True)          ## Units
    model.s = RangeSet(SAE_num)                                             ## Batteries in the system
    model.umbral = RangeSet(1,4)                                            ## Umbrales costo de escasez

    ########################################## Parameters definitions ####################################

    ###

    def MPO_init(model,t):
        return int(df_MPO.loc[0,t])
    model.MPO = Param(model.t, initialize=MPO_init)

    def PC_init(model,t):
        return round(model.MPO[t] ** 2 / max(model.MPO[t] for t in model.t),2)
    model.PC = Param(model.t, initialize=PC_init)

    def PC_max_init(model,s):
        return df_SAEInfo.loc['Potencia máxima de carga [MW]','SAE {}'.format(s)]
    model.PC_max = Param(model.s, initialize=PC_max_init)                           ## Power capacity max charging

    def PD_max_init(model,s):
        return df_SAEInfo.loc['Potencia máxima de descarga [MW]','SAE {}'.format(s)]
    model.PD_max = Param(model.s, initialize=PD_max_init)                           ## Power capacity max discharging

    def PC_des_init(model,s):
        return df_SAEInfo.loc['Valor de carga requerido [MW]','SAE {}'.format(s)]
    model.PCdes = Param(model.s, initialize=PC_des_init)                            ## Charging power capacity on discharge periods

    def PD_des_init(model,s):
        return df_SAEInfo.loc['Valor de descarga requerido [MW]','SAE {}'.format(s)]
    model.PDdes = Param(model.s, initialize=PD_des_init)                            ## Discharging power capacity on discharge periods

    def E_max_init(model,s):
        return df_SAEInfo.loc['Energía [MWh]','SAE {}'.format(s)]
    model.E_max = Param(model.s, initialize=E_max_init)                             ## Energy storage max limit

    def effDescarga_init(model,s):
        return round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]','SAE {}'.format(s)],0.5),4)
    model.Eficiencia_descarga = Param(model.s, initialize=effDescarga_init)         ## Discharge efficiency

    def effCarga_init(model,s):
        return round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]','SAE {}'.format(s)],0.5),4)
    model.Eficiencia_carga = Param(model.s, initialize=effCarga_init)               ## Charge efficiency

    def effSoC_init(model,s):
        return df_SAEInfo.loc['Autodescarga [%/h]','SAE {}'.format(s)]
    model.Eficiencia_SoC = Param(model.s, initialize=effSoC_init)                   ## Storage efficiency

    def socMIN_init(model,s):
        return df_SAEInfo.loc['Estado de carga mínimo [pu]','SAE {}'.format(s)]
    model.SoC_min = Param(model.s, initialize=socMIN_init)                          ## Minimum state of charge

    def socMAX_init(model,s):
        return df_SAEInfo.loc['Estado de carga máximo [pu]','SAE {}'.format(s)]
    model.SoC_max = Param(model.s, initialize=socMAX_init)                          ## Maximum state of charge

    def loadSIN_init(model,t):
        return df_PRON_DEM.loc[t]
    model.loadSIN = Param(model.t, initialize=loadSIN_init)                         ## Load

    def pOfe_init(model,i):
        return df_ofe.loc[i,'Precio']
    model.pOfe = Param(model.i, initialize=pOfe_init)                               ## Price offer

    def pArrPa_init(model,i):
        return df_ofe.loc[i,'PAP']
    model.pArrPa = Param(model.i, initialize=pArrPa_init)                           ## Start-Up and Shut-Down cost

    def maxDisp_init(model,i,t):
        return df_disp.loc[i,t]
    model.maxDisp = Param(model.i, model.t, initialize=maxDisp_init)                ## Maximum availability

    def minDisp_init(model,i,t):
        return df_minop.loc[i,t]
    model.minDisp = Param(model.i, model.t, initialize=minDisp_init)                ## Minimum availability

    model.td_i = Param(initialize=(num_td_i+1))
    model.td_f = Param(initialize=(num_td_f+1))
    model.tdp_i = Param(initialize=(num_tdp_i+1))
    model.tdp_f = Param(initialize=(num_tdp_f+1))

    CRO_est = {1: 1480.31,
                2: 2683.49,
                3: 4706.20,
                4: 9319.71}

    def CRO_init(model):
        return CRO_est
    model.CRO_est = Param(model.umbral, initialize=CRO_init)    ## Costo incremental de racionamiento de energía

    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################
    ## Generators
    model.status = Var(model.i, model.t, within=Binary, initialize=0)           ## Commitment of unit i at time t
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)               ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)               ## Shut-Down cost of unit i
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)               ## Start-Up status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)               ## Shut-Down status of unit i
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)      ## Power dispatch of unit i at time t
    ## BESS
    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery charge
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery discharge
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery charge
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery discharge
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                ## Energy of battery
    model.V_DoC = Var(model.s, model.t, domain=NonNegativeReals)                ## Discharge state
    ## System
    model.V_Rac = Var(model.t, bounds=(0,2400))                                 ## Energy of rationing

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################
    StartTime = time.time()

    ## Objective function

    def cost_rule(model):
        return sum(model.P[i,t] * model.pOfe[i] for i in model.i for t in model.t) + \
                sum(model.costSD[i,t] + model.costSU[i,t] for i in model.i for t in model.t) + \
                sum((CRO_est[1] * 1000) * model.V_DoC[s,t] for s in model.s for t in model.t if t >= model.tdp_i and t <= model.tdp_f) + \
                sum((CRO_est[1] * 1000) * model.V_Rac[t] for t in model.t) + \
                sum(model.PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################
    ## Power limits

    def P_lim_max_rule(model,i,t):
        return model.P[i,t] <= model.maxDisp[i,t] * model.status[i,t]
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,t):
        return model.P[i,t] >= model.minDisp[i,t] * model.status[i,t]
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)

    ## PAP cost

    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == model.pArrPa[i] * model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,t):
        return model.costSD[i,t] == model.pArrPa[i] * model.SD[i,t]
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
        return sum(model.P[i,t] for i in model.i) + model.V_Rac[t] + model.V_PD[s,t] == model.loadSIN[t] + model.V_PC[s,t]
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)

    ##### Batteries

    ## Causalidad de la carga/descarga

    def sim_rule(model,s,t):
        return model.B_PC[s,t] + model.B_PD[s,t] <= 1
    model.sim = Constraint(model.s, model.t, rule=sim_rule)

    def power_c_max_rule(model,s,t):
        return model.V_PC[s,t] <= model.PC_max[s] * model.B_PC[s,t]
    model.power_c_max = Constraint(model.s, model.t, rule=power_c_max_rule)

    def power_d_max_rule(model,s,t):
        return model.V_PD[s,t] <= model.PD_max[s] * model.B_PD[s,t]
    model.power_d_max = Constraint(model.s, model.t, rule=power_d_max_rule)

    ## Balance almacenamiento

    def energy_rule(model,s,t):
        if t == 1:
            return model.V_SoC[s,t] == model.SoC_min[s] * model.E_max[s] + model.Eficiencia_carga[s] * model.V_PC[s,t] - \
                    model.V_PD[s,t] * (1/model.Eficiencia_descarga[s])
        else:
            return model.V_SoC[s,t] == model.V_SoC[s,t-1] * (1 - model.Eficiencia_SoC[s]) + (model.Eficiencia_carga[s] * model.V_PC[s,t] - model.V_PD[s,t] * (1/model.Eficiencia_descarga[s]))
    model.energy = Constraint(model.s, model.t, rule=energy_rule)

    ## Balance de Estado de Carga

    def energy_balance_rule(model,s,t):
        return model.V_DoC[s,t] == model.E_max[s] - model.V_SoC[s,t]
    model.energy_balance = Constraint(model.s, model.t, rule=energy_balance_rule)

    ## Capacidad mínima y máxima de almacenamiento

    def energy_min_limit_rule(model,s,t):
        return model.V_SoC[s,t] >= model.SoC_min[s] * model.E_max[s]
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)

    def energy_max_limit_rule(model,s,t):
        return model.V_SoC[s,t] <= model.E_max[s]
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)

    ## Carga y descarga requerida

    def power_required_dh_rule(model,s,t):
        if t >= model.td_i and t <= model.td_f:
            return model.V_PD[s,t] == model.PDdes[s]
        else:
            return Constraint.Skip
    model.power_required_dh = Constraint(model.s, model.t, rule=power_required_dh_rule)

    def power_required_ch_rule(model,s,t):
        if t >= model.td_i and t <= model.td_f:
            return model.V_PC[s,t] >= model.PCdes[s]
        else:
            return Constraint.Skip
    model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)

    # Configuracion:

    solver_selected = combo

    if solver_selected== "CPLEX":
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    #################################################################################
    ####################### Creación de Archivo Excel ###############################
    #################################################################################

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_opeXM_SAE.xlsx'

    Output_data = {}

    for v in model.component_objects(Var):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 1:
            df = pyomo1_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 0:
            Output_data[str(v)] = pyomo_df(v)

    for v in model.component_objects(Param):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df(v)
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

    writer = pd.ExcelWriter(os.path.join(mydir, name_file), engine = 'xlsxwriter')

    for idx in Output_data.keys():
        Output_data[idx].to_excel(writer, sheet_name=idx, index=True)
    writer.save()
    # writer.close()
    ##########################################################################

    df_SOC = pyomo1_df(model.V_PD)

    return Output_data, tiempo

def opt_despacho_2(df_disp, df_minop, df_ofe, df_MPO, df_PRON_DEM, df_SAEInfo, SAE_num, num_tdp_i, num_tdp_f, num_td_i,
                num_td_f, combo):

    ##################################### Load file ####################################

    StartTime = time.time()

    ReadingTime = time.time() - StartTime

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    model = ConcreteModel()
    ##

    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)              ## scheduling periods
    model.i = Set(initialize=df_disp.index.tolist(), ordered=True) ## Units
    model.s = RangeSet(1)                                                           ## Batteries in the system
    ########################################## Parameters definitions ####################################

    ##

    def MPO_init(model,t):
        return int(df_MPO.loc[0,t])
    model.MPO = Param(model.t, initialize=MPO_init)

    def PC_max_init(model,s):
        return df_SAEInfo.loc['Potencia máxima de carga [MW]','SAE {}'.format(s)]
    model.PC_max = Param(model.s, initialize=PC_max_init)                           ## Power capacity max charging

    def PD_max_init(model,s):
        return df_SAEInfo.loc['Potencia máxima de descarga [MW]','SAE {}'.format(s)]
    model.PD_max = Param(model.s, initialize=PD_max_init)                           ## Power capacity max discharging

    def PC_req_init(model,s):
        return df_SAEInfo.loc['Valor de carga requerido [MW]','SAE {}'.format(s)]
    model.PCreq = Param(model.s, initialize=PC_req_init)                            ## Charging power capacity on discharge periods

    def PD_req_init(model,s):
        return df_SAEInfo.loc['Valor de descarga requerido [MW]','SAE {}'.format(s)]
    model.PDreq = Param(model.s, initialize=PD_req_init)                            ## Discharging power capacity on discharge periods

    def E_max_init(model,s):
        return df_SAEInfo.loc['Energía [MWh]','SAE {}'.format(s)]
    model.E_max = Param(model.s, initialize=E_max_init)                             ## Energy storage max limit

    def effDescarga_init(model,s):
        return round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]','SAE {}'.format(s)],0.5),4)
    model.Eficiencia_descarga = Param(model.s, initialize=effDescarga_init)         ## Discharge efficiency

    def effCarga_init(model,s):
        return round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]','SAE {}'.format(s)],0.5),4)
    model.Eficiencia_carga = Param(model.s, initialize=effCarga_init)               ## Charge efficiency

    def effSoC_init(model,s):
        return df_SAEInfo.loc['Autodescarga [%/h]','SAE {}'.format(s)]
    model.Eficiencia_SoC = Param(model.s, initialize=effSoC_init)                   ## Storage efficiency

    def socMIN_init(model,s):
        return df_SAEInfo.loc['Estado de carga mínimo [pu]','SAE {}'.format(s)]
    model.SoC_min = Param(model.s, initialize=socMIN_init)                          ## Minimum state of charge

    def socMAX_init(model,s):
        return df_SAEInfo.loc['Estado de carga máximo [pu]','SAE {}'.format(s)]
    model.SoC_max = Param(model.s, initialize=socMAX_init)                          ## Maximum state of charge

    def socMT_init(model,s):
        return df_SAEInfo.loc['Estado de carga mínimo técnico [pu]','SAE {}'.format(s)]
    model.SoC_MT = Param(model.s, initialize=socMT_init)                            ## Technical minimum state of charge

    model.K_e = Param(initialize=1)                                                 ## Scalling factor [$/MWh]

    def loadSIN_init(model,t):
        return df_PRON_DEM.loc[t]
    model.loadSIN = Param(model.t, initialize=loadSIN_init)                         ## Load

    def pOfe_init(model,i):
        return df_ofe.loc[i,'Precio']
    model.pOfe = Param(model.i, initialize=pOfe_init)                               ## Price offer

    def pArrPa_init(model,i):
        return df_ofe.loc[i,'PAP']
    model.pArrPa = Param(model.i, initialize=pArrPa_init)                           ## Start-Up and Shut-Down cost

    def maxDisp_init(model,i,t):
        return df_disp.loc[i,t]
    model.maxDisp = Param(model.i, model.t, initialize=maxDisp_init)                ## Maximum availability

    def minDisp_init(model,i,t):
        return df_minop.loc[i,t]
    model.minDisp = Param(model.i, model.t, initialize=minDisp_init)                ## Minimum availability

    model.td_i = Param(initialize=(num_td_i+1))
    model.td_f = Param(initialize=(num_td_f+1))
    model.tdp_i = Param(initialize=(num_tdp_i+1))
    model.tdp_f = Param(initialize=(num_tdp_f+1))

    ## PC definitions

    def PC_init(model,t):
        return round((model.K_e * df_PRON_DEM.loc[t]) / max(df_PRON_DEM.loc[t] for t in model.t),2)
    model.PC = Param(model.t, initialize=PC_init)

    ## costo de racionamiento

    cro = {'CRO1': 1480.31,
            'CRO2': 2683.49,
            'CRO3': 4706.20,
            'CRO4': 9319.71}

    ## Estado de conexión del SAEB al SIN

    model.ECS = Param(model.s, model.t, initialize=1, mutable=True)

    # model.ECS[1,7] = 0

    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################
    ## Generators
    model.status = Var(model.i, model.t, within=Binary, initialize=0)           ## Commitment of unit i at time t
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)      ## Power dispatch of unit i at time t
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)               ## Start-Up cost of uit i
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)               ## Shut-Down cost of unit i
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)               ## Start-Up status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)               ## Shut-Down status of unit i
    ## BESS
    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery charge
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery discharge
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery charge
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery discharge
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                ## Energy of battery
    model.V_SoD = Var(model.s, model.t, domain=NonNegativeReals)                ## Discharge state
    model.V_SoC_E = Var(model.s, model.t, domain=NonNegativeReals)              ## State of Charge with storage efficiency
    ## System
    model.V_Rac = Var(model.t, bounds=(0, 2400))                                ## Energy of rationing

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################
    StartTime = time.time()

    ## Objective function

    def cost_rule(model):
        return sum(model.P[i,t] * model.pOfe[i] for i in model.i for t in model.t) + \
                sum(model.costSD[i,t] + model.costSU[i,t] for i in model.i for t in model.t) + \
                sum((cro['CRO1'] * 1000) * model.V_SoD[s,t] * model.E_max[s] for s in model.s for t in model.t if t >= model.tdp_i and t <= model.tdp_f) + \
                sum((cro['CRO1'] * 1000) * model.V_Rac[t] for t in model.t) + \
                sum(model.PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################

    #### Dispath constraints

    ## Power limits

    def P_lim_max_rule(model,i,t):
        return model.P[i,t] <= model.maxDisp[i,t] * model.status[i,t]
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,t):
        return model.P[i,t] >= model.minDisp[i,t] * model.status[i,t]
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)

    ## PAP cost

    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == model.pArrPa[i] * model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,t):
        return model.costSD[i,t] == model.pArrPa[i] * model.SD[i,t]
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
                model.V_PD[s,t] * model.ECS[s,t] == model.loadSIN[t] + model.V_PC[s,t] * model.ECS[s,t]
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)

    ##### Batteries

    ## Balance almacenamiento

    def energy_rule(model,s,t):
        if t == 1:
            return model.V_SoC[s,t] == model.SoC_min[s] + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] / model.E_max[s] - \
                    model.V_PD[s,t] / (model.E_max[s] * model.Eficiencia_descarga[s]))
        else:
            return model.V_SoC[s,t] == model.V_SoC_E[s,t-1] + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] / model.E_max[s]  - \
                    model.V_PD[s,t] / (model.E_max[s] * model.Eficiencia_descarga[s]))
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
        return model.V_SoC[s,t] >= model.SoC_min[s]
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)

    def energy_max_limit_rule(model,s,t):
        return model.V_SoC[s,t] <= model.SoC_max[s]
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)

    ## mínimo técnico del sistema de almacenamiento

    def energy_min_tec_rule(model,s,t):
        return model.V_SoC[s,t] >= model.SoC_MT[s]
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
        if (t >= model.td_i) and (t <= model.td_f) and (model.ECS[s,t].value == 1):
            return model.V_PD[s,t] == model.PDreq[s]
        else:
            return model.V_PD[s,t] == 0
    model.power_required_dc = Constraint(model.s, model.t, rule=power_required_dc_rule)

    def power_required_ch_rule(model,s,t):
        if (t >= model.td_i) and (t <= model.td_f) and (model.ECS[s,t].value == 1):
            return model.V_PC[s,t] >= model.PCreq[s]
        else:
            return Constraint.Skip
    model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)

    # Configuracion:

    solver_selected = combo

    if solver_selected== "CPLEX":
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    #################################################################################
    ####################### Creación de Archivo Excel ###############################
    #################################################################################

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_opeXM_SAE.xlsx'

    Output_data = {}

    for v in model.component_objects(Var):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 1:
            df = pyomo1_df(v)
            df = df.T
            Output_data[str(v)] = df
        elif sets == 0:
            Output_data[str(v)] = pyomo_df(v)

    for v in model.component_objects(Param):
        sets = v.dim()
        if sets == 3:
            df = pyomo3_df(v)
            df = df.T
            Output_data[str(v)] = df
        if sets == 2:
            df = pyomo2_df(v)
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

    writer = pd.ExcelWriter(os.path.join(mydir, name_file), engine = 'xlsxwriter')

    for idx in Output_data.keys():
        Output_data[idx].to_excel(writer, sheet_name=idx, index=True)
    writer.save()
    # writer.close()

    return Output_data, tiempo

def graph_results_opeXM(Output_data):

    time = Output_data['P'].columns
    time = list(map(lambda x: x - 1, time))

    title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'14'}

    st.markdown('### Graficación de Resultados:')

    st.write('#### Generación')

    ## Generación

    fig = go.Figure()
    for g in Output_data['P'].index:
        if sum(Output_data['P'].loc[g,1:24]) > 0:
            fig.add_trace(go.Scatter(x=time, y=Output_data['P'].loc[g,1:24], name=g, line_shape='hv'))

    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
            title='Generación',
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

    V_SoC = Output_data['V_SoC'].loc[1,:]
    V_SoC[0] = 0
    V_SoC = V_SoC.sort_index()

    for n in Output_data['V_PC'].index:
        if sum(Output_data['V_PC'].loc[n,1:24]) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=time, y=V_SoC.loc[0:23], name='Energía [MWh]', line_shape='linear'), secondary_y=False)
            fig.add_trace(go.Scatter(x=time, y=Output_data['V_PC'].loc[n,:], name='Carga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)
            fig.add_trace(go.Scatter(x=time, y=Output_data['V_PD'].loc[n,:], name='Descarga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)
            fig.add_vrect(x0=Output_data['tdp_i'].values[0][0]-1, x1=Output_data['tdp_f'].values[0][0], line_width=1,
                fillcolor="green", opacity=0.3, annotation_text="TDP", annotation_position="outside top left")
            fig.add_vrect(x0=Output_data['td_i'].values[0][0]-1, x1=Output_data['td_f'].values[0][0], line_width=1,
                fillcolor="red", opacity=0.3, annotation_text="TD", annotation_position="outside top right")

            fig.update_layout(legend=dict(y=1, traceorder='reversed', font_size=8),
                    title='Operación SAE número {}'.format(n),
                    autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(title_text='Hora', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
            fig.update_yaxes(title_text='Energía [MWh]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=False)
            fig.update_yaxes(title_text='Potencia [MW]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=True)
            st.plotly_chart(fig)