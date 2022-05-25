import pandas as pd
from pandas import ExcelWriter
from pyomo.environ import *
import math
import time
from datetime import datetime, timedelta, date
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from funciones.save_files import *
import streamlit as st

# para poder corer GLPK desde una API
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
###

def opt_dim(file_system, txt_eff, txt_SOC_min, txt_SOC_ini, txt_time_sim, txt_re_inv, txt_C_Pot, txt_C_Bat, txt_autoD, combo, info_):

    StartTime = time.time()
    file_name_path = open('Casos_estudio/loc_size/h5_files/file_name.txt')
    file_name_content = file_name_path.read()
    file_name_path.close()

    if file_name_content != file_system.name:

        df_System_data = pd.read_excel(file_system, sheet_name='System_data', header=0, index_col=0)
        df_SM_Unit = pd.read_excel(file_system, sheet_name='SM_Unit', header=0, index_col=0)
        df_SM_map = pd.read_excel(file_system, sheet_name='SM_map', header=0, index_col=0)
        df_Renewable = pd.read_excel(file_system, sheet_name='Renewable', header=0, index_col=0)
        df_Reservoir = pd.read_excel(file_system, sheet_name='Reservoir', header=0, index_col=1).drop(['Unnamed: 0'], axis=1)
        df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)
        df_line_map = pd.read_excel(file_system, sheet_name='Branch_map', header=0, index_col=0)
        df_load = pd.read_excel(file_system, sheet_name='load', header=0, index_col=0)
        df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)

        with pd.HDFStore('Casos_estudio/loc_size/h5_files/store.h5') as store:
            store['df_System_data'] = df_System_data
            store['df_SM_Unit'] = df_SM_Unit
            store['df_SM_map'] = df_SM_map
            store['df_Renewable'] = df_Renewable
            store['df_Reservoir'] = df_Reservoir
            store['df_Branch'] = df_Branch
            store['df_line_map'] = df_line_map
            store['df_load'] = df_load
            store['df_Bus'] = df_Bus

        with open('Casos_estudio/loc_size/h5_files/file_name.txt', 'w') as archivo:
            archivo.write(file_system.name)

    else:
        with pd.HDFStore('Casos_estudio/loc_size/h5_files/store.h5') as store:
            df_System_data = store['df_System_data']
            df_SM_Unit = store['df_SM_Unit']
            df_SM_map = store['df_SM_map']
            df_Renewable = store['df_Renewable']
            df_Reservoir = store['df_Reservoir']
            df_Branch = store['df_Branch']
            df_line_map = store['df_line_map']
            df_load = store['df_load']
            df_Bus = store['df_Bus']

    #Total time steps
    N_horas = int(txt_time_sim)
    ReadingTime = time.time() - StartTime

    #### gen classification by type

    ## Hydro
    C_hydro = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'HYDRO'].index)
    C_hydro = C_hydro.tolist()
    C_hydro.sort()
    hydro_idx = df_SM_Unit.index[C_hydro]

    ## Fossil fuels
    C_acpm = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'ACPM'].index)
    C_acpm = C_acpm.tolist()
    C_diesel = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'DIESEL'].index)
    C_diesel = C_diesel.tolist()
    C_coal = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'COAL'].index)
    C_coal = C_coal.tolist()
    C_combustoleo = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'COMBUSTOLEO'].index)
    C_combustoleo = C_combustoleo.tolist()
    C_oil = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'OIL'].index)
    C_oil = C_oil.tolist()
    C_gas = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'GAS'].index)
    C_gas = C_gas.tolist()
    C_thermal = C_gas + C_acpm + C_diesel + C_coal + C_combustoleo + C_oil
    C_thermal.sort()
    thermal_idx = df_SM_Unit.index[C_thermal]

    ## Wind
    C_wind = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'WIND'].index)
    C_wind = C_wind.tolist()
    C_wind.sort()
    wind_idx = df_SM_Unit.index[C_wind]

    ## Solar
    C_solar = df_SM_Unit.index.get_indexer_for(df_SM_Unit[df_SM_Unit.Fuel_Type == 'SOLAR'].index)
    C_solar = C_solar.tolist()
    C_solar.sort()
    solar_idx = df_SM_Unit.index[C_solar]

    ## Reservoir
    embalse_Rio = {'AGREGADO BOGOTA':['BOGOTA N.R.'], 'ALTOANCHICAYA':['ALTOANCHICAYA','DIGUA'], 'AMANI':['MIEL I','DESV. GUARINO','DESV. MANSO'],
            'BETANIA':['BETANIA CP','MAGDALENA BETANIA'], 'CALIMA1':['CALIMA'], 'CHUZA':['CHUZA'], 'EL QUIMBO':['EL QUIMBO'],
            'ESMERALDA':['BATA'], 'GUAVIO':['GUAVIO'], 'MIRAFLORES':['TENCHE'], 'MUNA':['BOGOTA N.R.'], 'PENOL':['NARE'], 'PLAYAS':['GUATAPE'],
            'PORCE II':['PORCE2 CP','PORCE II'], 'PORCE III':['PORCE III'], 'PRADO':['PRADO'], 'PUNCHINA':['SAN CARLOS'], 'RIOGRANDE2':['GRANDE'],
            'SALVAJINA':['CAUCA SALVAJINA'], 'SAN LORENZO':['A. SAN LORENZO'], 'TOPOCORO':['SOGAMOSO'],
            'TRONERAS':['CONCEPCION','DESV. EEPPM (NEC,PAJ,DOL)','GUADALUPE'], 'URRA1':['SINU URRA']}

    ################################################ Sets Definitions ################################################

    model = ConcreteModel()

    ## Sets
    model.t = RangeSet(1,N_horas)                                       ## Periodos de tiempo
    model.tt = SetOf(model.t)                                           ## Periodos de tiempo para relaciones intertemporales
    model.i = Set(initialize=thermal_idx, ordered=True)                 ## Unidades térmicas de generación
    model.w = Set(initialize=wind_idx, ordered=True)                    ## Unidades eólicas de generación
    model.b = Set(initialize=df_Bus.index.tolist(), ordered=True)       ## Número de nodos del sistema
    model.s = Set(initialize=solar_idx, ordered=True)                   ## Unidades solares de generación
    model.j = Set(initialize=hydro_idx, ordered=True)                   ## Unidades hidráulicas de generación
    model.l = Set(initialize=df_Branch.index.tolist(), ordered=True)    ## Número de líneas del sistema
    model.r = Set(initialize=df_Reservoir.index.tolist(), ordered=True) ## Número de embalses del sistema (sí aplica)

    ## Linearization sets
    L = 3
    model.L = RangeSet(1,L)                                         ##
    model.k = Set(initialize=['k1','k2','k3'], ordered=True)          ## Cantidad de segmentos linealizados

    ########################################## Parameters definitions ####################################

    #### System parameters
    Big_number = 1e20                                               ##
    model.MVA_base = Param(initialize=100)                       ## Base del sistema

    #### BESS parameters
    Costo_potencia = round(int(txt_C_Pot)/(365*int(txt_re_inv)*24),2)
    Costo_energia = round(int(txt_C_Bat)/(365*int(txt_re_inv)*24),2)
    model.Costo_potencia = Param(initialize=Costo_potencia*N_horas)         ## Costo del inversor de potencia de la batería
    model.Costo_energia = Param(initialize=Costo_energia*N_horas)           ## Costo de los modulos de baterías
    model.Eficiencia_descarga = Param(initialize=round(pow(txt_eff,0.5),2))              ## Eficiencia global del BESS
    model.Eficiencia_carga = Param(initialize=round(pow(txt_eff,0.5),2))                 ##
    model.Eff_self = Param(initialize=float(txt_autoD))
    SOC_min = 1 - float(txt_SOC_min)                                                   ##
    SOC_ini = float(txt_SOC_ini)                                                  ##

    #### gen parameters
    ## slope segment k generator i (thermal)
    def slope_init(model,i,k):
        if model.k.ord(k) == 1:
            return (df_SM_Unit.loc[i,"a"]*(df_SM_Unit.loc[i,"k1"]+df_SM_Unit.loc[i,"Pmin"])+df_SM_Unit.loc[i,"b"])
        else:
            return (df_SM_Unit.loc[i,"a"]*(df_SM_Unit.loc[i,k]+df_SM_Unit.loc[i,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[i,"b"])
    model.slope = Param(model.i, model.k, initialize=slope_init)

    ## Minimum production cost of unit i at Pmin (thermal)
    def fg_min_init(model,i):
        return (df_SM_Unit.loc[i,"a"]*df_SM_Unit.loc[i,"Pmin"]*df_SM_Unit.loc[i,"Pmin"]+df_SM_Unit.loc[i,"b"]*df_SM_Unit.loc[i,"Pmin"]+\
                                df_SM_Unit.loc[i,"c"])
    model.fg_min = Param(model.i, initialize=fg_min_init)

    ## on-off status at t=0 (thermal)
    def onoff_t0_init(model,i):
        if df_SM_Unit.loc[i,"IniT_ON"] > 0:
            a = 1
        else:
            a = 0
        return a
    model.onoff_t0 = Param(model.i, initialize=onoff_t0_init)

    ## Used for minimum up time constraints (thermal)
    def L_up_min_init(model, i):
        return min(len(model.t), (df_SM_Unit.loc[i,"Min_ON"]-df_SM_Unit.loc[i,"IniT_ON"])*model.onoff_t0[i])
    model.L_up_min = Param(model.i, rule=L_up_min_init)

    ## Used for minimum down time constraints (thermal)
    def L_down_min_init(model,i):
        return min(len(model.t), (df_SM_Unit.loc[i,"Min_OFF"]-df_SM_Unit.loc[i,"IniT_off"])*(1-model.onoff_t0[i]))
    model.L_down_min = Param(model.i, rule=L_down_min_init)

    ## slope segment k generator i (hydro)
    def slope_Hydro_init(model,j,k):
        if model.k.ord(k) == 1:
            return (df_SM_Unit.loc[j,"a"]*(df_SM_Unit.loc[j,"k1"]+df_SM_Unit.loc[j,"Pmin"])+df_SM_Unit.loc[j,"b"])
        else:
            return (df_SM_Unit.loc[j,"a"]*(df_SM_Unit.loc[j,k]+df_SM_Unit.loc[j,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[j,"b"])
    model.slope_j = Param(model.j, model.k, initialize=slope_Hydro_init)

    ## Minimum production cost of unit i at Pmin (hydro)
    def fg_min_j_init(model,j):
        return (df_SM_Unit.loc[j,"a"]*df_SM_Unit.loc[j,"Pmin"]*df_SM_Unit.loc[j,"Pmin"]+df_SM_Unit.loc[j,"b"]*df_SM_Unit.loc[j,"Pmin"]+\
                                df_SM_Unit.loc[j,"c"])
    model.fg_min_j = Param(model.j, initialize=fg_min_j_init)

    ## maximum waterflow in turbine
    def Q_max_init(model,j):
        return df_SM_Unit.loc[j,'Qmax']
    model.Q_max = Param(model.j, initialize=Q_max_init)

    ## minimum waterflow in turbine
    def Q_min_init(model,j):
        return df_SM_Unit.loc[j,'Qmin']
    model.Q_min = Param(model.j, initialize=Q_min_init)

    ## maximum water volume in the reservoir
    def Vol_max_init(model,r):
        return df_Reservoir.loc[r,'Vmax']
    model.Vol_max = Param(model.r, initialize=Vol_max_init)

    ## minimum water volume in the reservoir
    def Vol_min_init(model,r):
        return df_Reservoir.loc[r,'Vmin']
    model.Vol_min = Param(model.r, initialize=Vol_min_init)

    ## forecast water in the reservoir

    fecha = date(2019, 7, 1)
    fecha_ant = fecha
    fecha_ant -= timedelta(days=1)

    actual_path = os.getcwd()

    db_files = os.path.join(actual_path, 'Casos_estudio/loc_size')

    df_aportes = pd.read_hdf(os.path.join(db_files, 'dbAportes/Aportes_Diario_{}.h5').format(fecha.year),'Aportes_{}'.format(fecha.year))

    df_aportes_f = df_aportes.drop(df_aportes[df_aportes['Fecha']!=str(fecha)].index)

    df_aporte_embalse = pd.DataFrame(index=embalse_Rio.keys(), columns=['Aporte m3/s'])

    for key, value in embalse_Rio.items():
        aporte = 0
        for i in df_aportes_f.index:
            if df_aportes_f.loc[i,'Nombre Río'] in value:
                aporte = aporte + df_aportes_f.loc[i,'Aportes Caudal m3/s']
        df_aporte_embalse.loc[key,'Aporte m3/s'] = aporte

    def I_init(model,r,t):
        return round(df_aporte_embalse.loc[r,'Aporte m3/s'] / 24, 4)
    model.I = Param(model.r, model.t, initialize=I_init)

    ## maximum spillage in the reservoir
    def spill_max_init(model,r):
        return df_Reservoir.loc[r,'Smax']
    model.spill_max = Param(model.r, initialize=spill_max_init)

    ## conversion factor
    def cov_factor_init(model,j):
        return df_SM_Unit.loc[j,'Factor_conversion']
    model.cov_factor = Param(model.j, initialize=cov_factor_init)

    ##

    df_reservas = pd.read_hdf(os.path.join(db_files, 'dbReservas/Reservas_Diario_{}.h5').format(fecha_ant.year),'Reservas_{}'.format(fecha_ant.year))

    df_reservas_f = df_reservas.drop(df_reservas[df_reservas['Fecha']!=str(fecha_ant)].index)

    df_reserva_embalse = pd.DataFrame(index=embalse_Rio.keys(), columns=['Reserva Mm3'])

    for key, value in embalse_Rio.items():
        reserva = 0
        for i in df_reservas_f.index:
            if df_reservas_f.loc[i,'Nombre Embalse'] in key:
                reserva = reserva + df_reservas_f.loc[i,'Volumen Útil Diario Mm3']
        df_reserva_embalse.loc[key,'Reserva Mm3'] = reserva

    #### line parameters
    ## angular difference slope
    delta_theta = (20 * math.pi / 180)

    def alpha_init(model,L):
        return delta_theta * (2 * L - 1)
    model.alpha = Param(model.L, initialize=alpha_init)

    ## Conductance of each line
    def conductance_init(model,l):
        return df_Branch.loc[l,'R'] / (df_Branch.loc[l,'R']**2 + df_Branch.loc[l,'X']**2)
    model.conductance = Param(model.l, rule=conductance_init)

    ## Susceptance of each line
    def susceptance_init(model,l):
        return 1 / df_Branch.loc[l,'X']
    model.susceptance = Param(model.l, rule=susceptance_init)

    ####################################################### VARIABLES ######################################################

    #### Thermal gen variables
    model.status = Var(model.i, model.t, within=Binary, initialize=0)                   ## Commitment of unit i at time t
    model.P = Var(model.i, model.t, domain=NonNegativeReals)                            ## Power dispatch of unit i at time t
    model.P_seg = Var(model.i, model.t, model.k, domain=NonNegativeReals, initialize=0)               ## Power dispatch segment k of i at time t
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)                       ## Startup status of unit i
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)                       ## Shutdown status of unit i
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)                       ## Startup cost of unit i
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)                       ## Shutdown cost of unit i
    model.pcost = Var(model.t, domain=NonNegativeReals, initialize=0)                                 ## Period operation cost
    model.costgen = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)                      ## Operation cost by generator

    #### Hydro gen variables
    model.status_j = Var(model.j, model.t, within=Binary, initialize=0)
    model.P_j = Var(model.j, model.t, domain=NonNegativeReals, initialize=0)                            ## Power dispatch of unit i at time t
    model.P_seg_j = Var(model.j, model.t, model.k, domain=NonNegativeReals, initialize=0)               ## Power dispatch segment k of i at time t
    model.pcost_j = Var(model.t, domain=NonNegativeReals, initialize=0)
    model.costgen_j = Var(model.j, model.t, domain=NonNegativeReals, initialize=0)

    model.Q = Var(model.j, model.t, domain=NonNegativeReals, initialize=0)

    def Vol0_init(model,r,t):
        return df_reserva_embalse.loc[r,'Reserva Mm3']
    model.Vol = Var(model.r, model.t, domain=NonNegativeReals, initialize=Vol0_init)
    model.spill = Var(model.r, model.t, domain=NonNegativeReals, initialize=0)

    #### Renewable variables
    model.Pw = Var(model.w, model.t, domain=NonNegativeReals)  ## Power dispatch of wind turbine w at time t
    model.Ps = Var(model.s, model.t, domain=NonNegativeReals)

    #### DC power flow variables
    model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))                      ## Voltage angle
    model.pf = Var(model.l, model.t, within=Reals)                                                    ## Power flow line l at time t scenario u
    model.theta_sr = Var(model.l, model.t, domain=Reals, initialize=0)
    model.pf_complete = Var(model.l, model.t, initialize=0)

    Slack_bus = df_System_data.loc['Slack_bus'][0]

    for t in model.t:
        model.theta[Slack_bus,t].fix(0)

    #### Power losses variables
    model.theta_sr_pos = Var(model.l, model.t, domain = NonNegativeReals, initialize=0)
    model.theta_sr_neg = Var(model.l, model.t, domain = NonNegativeReals, initialize=0)
    model.line_losses = Var(model.l, model.t, domain = NonNegativeReals, initialize=0)
    model.theta_aux = Var(model.l, model.t, model.L, domain = NonNegativeReals, initialize=0)

    #### BESS variables
    model.Carga_b = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery charge
    model.Descarga_b = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery discharge
    model.Pot_Ba_ch = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery charge
    model.Pot_Ba_dc = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery discharge
    model.e_b = Var(model.b, model.t, domain = NonNegativeReals, initialize=0)               ## Energy of battery
    model.C_Potencia = Var(model.b, domain = NonNegativeReals, initialize=0) ##
    model.E_max = Var(model.b, domain = NonNegativeReals, initialize=0) ##

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################

    StartTime = time.time()
    #### Costs functions
    ## Thermal
    def P_sum_rule(model,i,t):
        return model.P[i,t] == model.status[i,t]*df_SM_Unit.loc[i,"Pmin"] + sum(model.P_seg[i,t,k] for k in model.k)
    model.P_sum = Constraint(model.i, model.t, rule=P_sum_rule)

    def costgen_rule(model, i, t):
        return model.costgen[i,t] == model.status[i,t]*model.fg_min[i] + sum(model.P_seg[i,t,k]*model.slope[i,k] for k in model.k)
    model.costgen_fn = Constraint(model.i, model.t, rule=costgen_rule)

    def pcost_rule(model, t):
        return model.pcost[t] == sum(model.costgen[i,t] + model.costSU[i,t] + model.costSD[i,t] for i in model.i)
    model.costfnperiod = Constraint(model.t, rule=pcost_rule)

    ## Hydro
    def P_Hydro_sum_rule(model,j,t):
        return model.P_j[j,t] == model.status_j[j,t] * df_SM_Unit.loc[j,"Pmin"] + sum(model.P_seg_j[j,t,k] for k in model.k)
    model.P_Hydro_sum = Constraint(model.j, model.t, rule=P_Hydro_sum_rule)

    def costgen_Hydro_rule(model, j, t):
        return model.costgen_j[j,t] == model.status_j[j,t] * model.fg_min_j[j] + sum(model.P_seg_j[j,t,k]*model.slope_j[j,k] for k in model.k)
    model.costgen_Hydro_fn = Constraint(model.j, model.t, rule=costgen_Hydro_rule)

    def pcost_Hydro_rule(model, t):
        return model.pcost_j[t] == sum(model.costgen_j[j,t] for j in model.j)
    model.costfnperiod_Hydro = Constraint(model.t, rule=pcost_Hydro_rule)

    #### Objective function
    def cost_comp_rule(model):
        return sum(model.pcost[t] for t in model.t) + sum(model.pcost_j[t] for t in model.t) + sum(model.C_Potencia[b]*model.Costo_potencia + model.E_max[b]*model.Costo_energia for b in model.b)
    model.cost_comp = Objective(rule=cost_comp_rule, sense=minimize)

    def cost_rule_simp(model):
        return sum(df_SM_Unit.loc[i,'b'] * model.P[i,t] for i in model.i for t in model.t) + \
            sum(df_SM_Unit.loc[j,'b'] * model.P_j[j,t] for j in model.j for t in model.t) + \
            sum(model.C_Potencia[b]*model.Costo_potencia + model.E_max[b]*model.Costo_energia for b in model.b)
    model.cost_simp = Objective(rule=cost_rule_simp, sense=minimize)

    ###################################################### CONSTRAINTS ######################################################

    ## Startup/down cost
    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == df_SM_Unit.loc[i,"CSU"]*model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,t):
        return model.costSD[i,t] == df_SM_Unit.loc[i,"CSU"]*model.SD[i,t]
    model.CostSDfn = Constraint(model.i, model.t, rule=CostSDfn_init)

    #### power limits of generations
    ## Thermal
    def P_lim_min_comp_rule(model,i,t):
        return model.P[i,t] >= df_SM_Unit.loc[i,"Pmin"]*model.status[i,t]
    model.P_min_lim_comp = Constraint(model.i, model.t, rule=P_lim_min_comp_rule)

    def P_lim_max_comp_rule(model,i,t):
        return model.P[i,t] <= df_SM_Unit.loc[i,"Pmax"]*model.status[i,t]
    model.P_lim_max_comp = Constraint(model.i, model.t, rule=P_lim_max_comp_rule)

    def P_seg_lim_max_comp_rule(model,i,t,k):
        return model.P_seg[i,t,k] <= (df_SM_Unit.loc[i,"k2"]-df_SM_Unit.loc[i,"k1"])*model.status[i,t]
    model.P_seg_lim_max_comp = Constraint(model.i,model.t, model.k, rule=P_seg_lim_max_comp_rule)

    def P_lim_min_simp_rule(model,i,t):
        return model.P[i,t] >= 0
    model.P_min_lim_simp = Constraint(model.i, model.t, rule=P_lim_min_simp_rule)

    def P_lim_max_simp_rule(model,i,t):
        return model.P[i,t] <= df_SM_Unit.loc[i,"Pmax"]
    model.P_max_lim_simp = Constraint(model.i, model.t, rule=P_lim_max_simp_rule)

    ## Hydro
    def P_Hydro_lim_min_comp_rule(model,j,t):
        return model.P_j[j,t] >= df_SM_Unit.loc[j,"Pmin"] * model.status_j[j,t]
    model.P_Hydro_lim_min_comp = Constraint(model.j, model.t, rule=P_Hydro_lim_min_comp_rule)

    def P_Hydro_lim_max_comp_rule(model,j,t):
        return model.P_j[j,t] <= df_SM_Unit.loc[j,"Pmax"] * model.status_j[j,t]
    model.P_Hydro_lim_max_comp = Constraint(model.j, model.t, rule=P_Hydro_lim_max_comp_rule)

    def P_Hydro_seg_lim_max_comp_rule(model,j,t,k):
        return model.P_seg_j[j,t,k] <= (df_SM_Unit.loc[j,"k2"]-df_SM_Unit.loc[j,"k1"]) * model.status_j[j,t]
    model.P_Hydro_seg_lim_max_comp = Constraint(model.j, model.t, model.k, rule=P_Hydro_seg_lim_max_comp_rule)

    def P_Hydro_lim_min_simp_rule(model,j,t):
        return model.P_j[j,t] >= 0
    model.P_Hydro_lim_min_simp = Constraint(model.j, model.t, rule=P_Hydro_lim_min_simp_rule)

    def P_Hydro_lim_max_simp_rule(model,j,t):
        return model.P_j[j,t] <= df_SM_Unit.loc[j,"Pmax"]
    model.P_Hydro_lim_max_simp = Constraint(model.j, model.t, rule=P_Hydro_lim_max_simp_rule)

    ## Wind
    def maxPw_rule(model, w, t):
        return model.Pw[w,t] <= df_Renewable.loc[t,w]
    model.maxPw = Constraint(model.w, model.t, rule=maxPw_rule)

    ## Solar
    def maxPs_rule(model, s, t):
        return model.Ps[s,t] <= df_Renewable.loc[t,s]
    model.maxPs = Constraint(model.s, model.t, rule=maxPs_rule)

    #### Integer Constraint

    def bin_cons1_rule(model,i,t):
        if t == model.t.first():
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]
        else:
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons3_rule(model,i,t):
        return model.SU[i,t] + model.SD[i,t] <= 1
    model.bin_cons3 = Constraint(model.i, model.t, rule=bin_cons3_rule)

    #### Min up_dn time

    def min_up_dn_time_1_rule(model,i,t):
        if model.L_up_min[i] + model.L_down_min[i] > 0 and t < model.L_up_min[i] + model.L_down_min[i]:
            return model.status[i,t] == model.onoff_t0[i]
        else:
            return Constraint.Skip
    model.min_up_dn_time_1 = Constraint(model.i, model.t, rule=min_up_dn_time_1_rule)

    def min_up_dn_time_2_rule(model,i,t):
        return sum(model.SU[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,"Min_ON"]+1 and tt <= t) <= model.status[i,t]
    model.min_up_dn_time_2 = Constraint(model.i, model.t, rule=min_up_dn_time_2_rule)

    def min_up_dn_time_3_rule(model,i,t):
        return sum(model.SD[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,"Min_OFF"]+1 and tt <= t) <= 1-model.status[i,t]
    model.min_up_dn_time_3 = Constraint(model.i, model.t, rule=min_up_dn_time_3_rule)

    #### Ramp constraints

    def ramp_up_fn_rule(model,i,t):
        if t > 1:
            return model.P[i,t] - model.P[i,t-1] <= df_SM_Unit.loc[i,"Ramp_Up"]*model.status[i,t-1] + df_SM_Unit.loc[i,"Pmin"]*(model.status[i,t]-model.status[i,t-1]) + df_SM_Unit.loc[i,"Pmax"]*(1-model.status[i,t])
        else:
            return Constraint.Skip
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)

    def ramp_dw_fn_rule(model,i,t):
        if t > 1:
            return model.P[i,t-1] - model.P[i,t] <= df_SM_Unit.loc[i,"Ramp_Down"]*model.status[i,t] + df_SM_Unit.loc[i,"Pmin"]*(model.status[i,t-1]-model.status[i,t]) + df_SM_Unit.loc[i,"Pmax"]*(1-model.status[i,t-1])
        else:
            return Constraint.Skip
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)

    #### Reservoir constraints

    ## Water flow in turbine
    def water_flow_min_rule(model,j,t):
        return model.Q[j,t] >= model.Q_min[j]
    model.water_flow_min = Constraint(model.j, model.t, rule=water_flow_min_rule)

    def water_flow_max_rule(model,j,t):
        return model.Q[j,t] <= model.Q_max[j]
    model.water_flow_max = Constraint(model.j, model.t, rule=water_flow_max_rule)

    ## water volumen in the reservoir
    def water_volume_min_rule(model,r,t):
        return model.Vol[r,t] >= model.Vol_min[r]
    model.water_volume_min = Constraint(model.r, model.t, rule=water_volume_min_rule)

    def water_volume_max_rule(model,r,t):
        return model.Vol[r,t] <= model.Vol_max[r]
    model.water_volume_max = Constraint(model.r, model.t, rule=water_volume_max_rule)

    ## Spillage in the reservoir
    def spillage_max_rule(model,r,t):
        return model.spill[r,t] <= model.spill_max[r]
    model.spillage_max = Constraint(model.r, model.t, rule=spillage_max_rule)

    ## water conservation
    def water_conserv_rule(model,r,t):
        if t == 1:
            return Constraint.Skip
        else:
            return model.Vol[r,t] == model.Vol[r,t-1] + 3600 * (model.I[r,t] - sum(model.Q[j,t] for j in model.j if r in df_SM_Unit.loc[j,'Reservoir_name/River'].split(';')) - model.spill[r,t]) / 1e6
    model.water_conserv = Constraint(model.r, model.t, rule=water_conserv_rule)

    ## Power conversion
    def power_conversion_rule(model,j,t):
        return model.P_j[j,t] == model.cov_factor[j] * model.Q[j,t]
    model.power_conversion = Constraint(model.j, model.t, rule=power_conversion_rule)

    #### Angle definition

    def theta_sr_dec_rule(model,l,t):
        return sum(model.theta[b,t] * df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0) == model.theta_sr_pos[l,t] - model.theta_sr_neg[l,t]
    model.theta_sr_dec = Constraint(model.l, model.t, rule=theta_sr_dec_rule)

    ##

    def abs_definition_rule(model,l,t):
        return sum(model.theta_aux[l,t,L] for L in model.L) == model.theta_sr_neg[l,t] + model.theta_sr_pos[l,t]
    model.abs_definition = Constraint(model.l, model.t, rule=abs_definition_rule)

    ##

    def max_angular_difference_rule(model,l,t,L):
        return model.theta_aux[l,t,L] <= delta_theta
    model.max_angular_difference = Constraint(model.l, model.t, model.L, rule=max_angular_difference_rule)

    ## DC transmission network security constraint

    def line_flow_rule(model,l,t):
        return model.pf[l,t] == model.MVA_base * (model.susceptance[l]) * (model.theta_sr_pos[l,t] - model.theta_sr_neg[l,t])
    model.line_flow = Constraint(model.l, model.t, rule=line_flow_rule)

    ## transmission lines limits

    def line_min_rule(model,l,t):
        return - model.pf[l,t] + 0.5 * model.line_losses[l,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_min = Constraint(model.l, model.t, rule=line_min_rule)

    def line_max_rule(model,l,t):
        return model.pf[l,t] + 0.5 * model.line_losses[l,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_max = Constraint(model.l, model.t, rule=line_max_rule)

    ##

    def line_min_1_rule(model,l,t):
        return model.pf[l,t] >= - df_Branch.loc[l,'Flowlimit']
    model.line_min_1 = Constraint(model.l, model.t, rule=line_min_1_rule)

    def line_max_1_rule(model,l,t):
        return model.pf[l,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_max_1 = Constraint(model.l, model.t, rule=line_max_1_rule)

    # ##

    # def line_convex_1_rule(model,l,t):
    #     return model.pf[l,t] / (model.MVA_base * model.susceptance[l]) + (model.theta_sr_pos[l,t] - model.theta_sr_neg[l,t]) == 0
    # model.line_convex_1 = Constraint(model.l, model.t, rule=line_convex_1_rule)

    ## Losses

    ##

    def losses_rule(model,l,t):
        return model.line_losses[l,t] == model.MVA_base * model.conductance[l] * sum(model.alpha[L] * model.theta_aux[l,t,L] for L in model.L)
    model.losses = Constraint(model.l, model.t, rule=losses_rule)

    ##

    def losses_max_rule(model,l,t):
        return model.line_losses[l,t] <= df_Branch.loc[l,'Flowlimit']
    model.losses_max = Constraint(model.l, model.t, rule=losses_max_rule)

    # # ##

    # def losses_convex_1_rule(model,l,t):
    #     return - model.line_losses[l,t] / (model.MVA_base * model.conductance[l]) + sum(model.alpha[L] * model.theta_aux[l,t,L] for L in model.L) == 0
    # model.losses_convex_1 = Constraint(model.l, model.t, rule=losses_convex_1_rule)

    ## Power Balance

    def power_balance_rule(model,b,t):
        return sum(model.P[i,t] for i in model.i if df_SM_map.loc[i,b]) + \
                sum(model.P_j[j,t] for j in model.j if df_SM_map.loc[j,b]) + \
                sum(model.Ps[s,t] for s in model.s if df_SM_map.loc[s,b]) + \
                sum(model.Pw[w,t] for w in model.w if df_SM_map.loc[w,b]) + \
                sum(model.pf[l,t] for l in model.l if b == df_Branch.loc[l,'to']) + \
                model.Pot_Ba_dc[b,t] - model.Pot_Ba_ch[b,t] == df_load.loc[t,b] + sum((model.pf[l,t] + 0.5 * model.line_losses[l,t]) for l in model.l if b == df_Branch.loc[l,'from'])
    model.power_balance = Constraint(model.b, model.t, rule=power_balance_rule)

    ############################################################################################################################

    ## power charging Constraints

    def power_c_max_rule(model,b,t):
        return model.Pot_Ba_ch[b,t] <= Big_number * model.Carga_b[b,t]
    model.power_c_max = Constraint(model.b, model.t, rule=power_c_max_rule)

    def power_c_max_2_rule(model,b,t):
        return model.Pot_Ba_ch[b,t] <= model.C_Potencia[b] #* model.area_param[b]
    model.power_c_max_2 = Constraint(model.b, model.t, rule=power_c_max_2_rule)

    # power dischraging Constraints

    def power_d_max_rule(model,b,t):
        return model.Pot_Ba_dc[b,t] <= Big_number * model.Descarga_b[b,t]
    model.power_d_max = Constraint(model.b, model.t, rule=power_d_max_rule)

    def power_d_max_2_rule(model,b,t):
        return model.Pot_Ba_dc[b,t] <= model.C_Potencia[b] #* model.area_param[b]
    model.power_d_max_2 = Constraint(model.b, model.t, rule=power_d_max_2_rule)

    ## Simultaneous charging and discharging Constraint

    def sim_rule(model,b,t):
        return model.Carga_b[b,t] + model.Descarga_b[b,t] <= 1
    model.sim = Constraint(model.b, model.t, rule=sim_rule)

    ## relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,b,t):
        if t == 1:
            return model.e_b[b,t] == model.E_max[b] * SOC_ini + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga
        else:
            return model.e_b[b,t] == model.e_b[b,t-1] * (1 - model.Eff_self) + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga   
    model.energy = Constraint(model.b, model.t, rule=energy_rule)

    def energy_limit_rule(model,b,t):
        return model.e_b[b,t] <= model.E_max[b]
    model.energy_limit = Constraint(model.b, model.t, rule=energy_limit_rule)

    def energy_limit_min_rule(model,b,t):
        return model.e_b[b,t] >= model.E_max[b] * SOC_min
    model.energy_limit_min = Constraint(model.b, model.t, rule=energy_limit_min_rule)

    ## complete power flow

    def complete_pf_rule(model,l,t):
        return model.pf_complete[l,t] == model.pf[l,t] + 0.5 * model.line_losses[l,t]
    model.complete_pf = Constraint(model.l, model.t, rule=complete_pf_rule)

    #### Model construction

    if type(info_) == str:

        if info_ == 'Simplificado':
            ## Objective function
            model.cost_comp.deactivate()

            ## Costraints

            # Elementos función objetivo
            model.P_sum.deactivate()
            model.costgen_fn.deactivate()
            model.costfnperiod.deactivate()
            model.P_Hydro_sum.deactivate()
            model.costgen_Hydro_fn.deactivate()
            model.costfnperiod_Hydro.deactivate()
            # Costos arranque/parada
            model.CostSUfn.deactivate()
            model.CostSDfn.deactivate()
            # Límites de Generación
            model.P_min_lim_comp.deactivate()
            model.P_lim_max_comp.deactivate()
            model.P_seg_lim_max_comp.deactivate()
            model.P_Hydro_lim_min_comp.deactivate()
            model.P_Hydro_lim_max_comp.deactivate()
            model.P_Hydro_seg_lim_max_comp.deactivate()
            # Operación de unidades térmicas
            model.bin_cons1.deactivate()
            model.bin_cons3.deactivate()
            # Tiempos mínimos de encendido/apagado
            model.min_up_dn_time_1.deactivate()
            model.min_up_dn_time_2.deactivate()
            model.min_up_dn_time_3.deactivate()
            # Rampas Térmicos
            model.ramp_up_fn.deactivate()
            model.ramp_dw_fn.deactivate()
            # pérdidas
            model.abs_definition.deactivate()
            model.max_angular_difference.deactivate()
            model.line_min.deactivate()
            model.line_max.deactivate()
            model.losses.deactivate()
            model.losses_max.deactivate()
            # flujo de potencia
            model.complete_pf.deactivate()
            # reservoirs
            model.water_flow_min.deactivate()
            model.water_flow_max.deactivate()
            model.water_volume_min.deactivate()
            model.water_volume_max.deactivate()
            model.spillage_max.deactivate()
            model.water_conserv.deactivate()
            model.power_conversion.deactivate()

        else:
            ## Objective function
            model.cost_simp.deactivate()

            ## Constraints

            # Límites de Generación
            model.P_min_lim_simp.deactivate()
            model.P_max_lim_simp.deactivate()
            model.P_Hydro_lim_min_simp.deactivate()
            model.P_Hydro_lim_max_simp.deactivate()

    else:
        ## Objective function
        model.cost_comp.deactivate()

        ## Costraints

        # Elementos función objetivo
        model.P_sum.deactivate()
        model.costgen_fn.deactivate()
        model.costfnperiod.deactivate()
        model.P_Hydro_sum.deactivate()
        model.costgen_Hydro_fn.deactivate()
        model.costfnperiod_Hydro.deactivate()
        # Costos arranque/parada
        model.CostSUfn.deactivate()
        model.CostSDfn.deactivate()
        # Límites de Generación
        model.P_min_lim_comp.deactivate()
        model.P_lim_max_comp.deactivate()
        model.P_seg_lim_max_comp.deactivate()
        model.P_Hydro_lim_min_comp.deactivate()
        model.P_Hydro_lim_max_comp.deactivate()
        model.P_Hydro_seg_lim_max_comp.deactivate()
        # Operación de unidades térmicas
        model.bin_cons1.deactivate()
        model.bin_cons3.deactivate()
        # Tiempos mínimos de encendido/apagado
        model.min_up_dn_time_1.deactivate()
        model.min_up_dn_time_2.deactivate()
        model.min_up_dn_time_3.deactivate()
        # Rampas Térmicos (OK)
        model.ramp_up_fn.deactivate()
        model.ramp_dw_fn.deactivate()
        # pérdidas
        model.abs_definition.deactivate()
        model.max_angular_difference.deactivate()
        model.line_min.deactivate()
        model.line_max.deactivate()
        model.losses.deactivate()
        model.losses_max.deactivate()
        # flujo de potencia
        model.complete_pf.deactivate()
        # reservoirs
        model.water_flow_min.deactivate()
        model.water_flow_max.deactivate()
        model.water_volume_min.deactivate()
        model.water_volume_max.deactivate()
        model.spillage_max.deactivate()
        model.water_conserv.deactivate()
        model.power_conversion.deactivate()

        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in info_:
            model.ramp_up_fn.activate()
            model.ramp_dw_fn.activate()
            model.cost_comp.activate()
            model.cost_simp.deactivate()

            # Límites de Generación
            model.P_min_lim_comp.activate()
            model.P_lim_max_comp.activate()
            model.P_seg_lim_max_comp.activate()
            model.P_Hydro_lim_min_comp.activate()
            model.P_Hydro_lim_max_comp.activate()
            model.P_Hydro_seg_lim_max_comp.activate()

            model.P_min_lim_simp.deactivate()
            model.P_max_lim_simp.deactivate()
            model.P_Hydro_lim_min_simp.deactivate()
            model.P_Hydro_lim_max_simp.deactivate()
            # Elementos función objetivo
            model.P_sum.activate()
            model.costgen_fn.activate()
            model.costfnperiod.activate()
            model.P_Hydro_sum.activate()
            model.costgen_Hydro_fn.activate()
            model.costfnperiod_Hydro.activate()
            # Operación de unidades térmicas
            model.bin_cons1.activate()
            model.bin_cons3.activate()
            # Costos arranque/parada
            model.CostSUfn.activate()
            model.CostSDfn.activate()

            # Tiempos mínimos de encendido/apagado
            model.min_up_dn_time_1.activate()
            model.min_up_dn_time_2.activate()
            model.min_up_dn_time_3.activate()

        if 'Embalses' in info_:
            model.water_flow_min.activate()
            model.water_flow_max.activate()
            model.water_volume_min.activate()
            model.water_volume_max.activate()
            model.spillage_max.activate()
            model.water_conserv.activate()
            model.power_conversion.activate()

        if 'Pérdidas' in info_:
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
        os.environ['NEOS_EMAIL'] = 'hepisaGEB@gmail.com'
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)
    model.C_Potencia.display()
    model.E_max.display()
    print('Reading DATA time:',round(ReadingTime,3), '[s]')
    print('Modeling time:',round(ModelingTime,3), '[s]')
    print('Solving time:',round(SolvingTime,3), '[s]')
    print('Total time:', tiempo)

    # #################################################################################
    # ####################### Creación de Archivo Excel ###############################
    # #################################################################################

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc_ope.xlsx'

    Output_data = {}

    for v in model.component_objects(Var):
        sets = v.dim()
        if sets == 3:
            Output_data[str(v)] = pyomo3_df(v)
        if sets == 2:
            Output_data[str(v)] = pyomo2_df_mod(v)
        elif sets == 1:
            Output_data[str(v)] = pyomo1_df(v)
        elif sets == 0:
            Output_data[str(v)] = pyomo_df(v)

    for v in model.component_objects(Objective):
        Output_data[str(v)] = pyomo_df(v)

    if type(info_) == str:
        if info_ == 'Simplificado':
            cost_name = 'cost_simp'
        else:
            cost_name = 'cost_comp'
    else:
        if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in info_:
            cost_name = 'cost_comp'
        else:
            cost_name = 'cost_simp'

    with pd.ExcelWriter(os.path.join(mydir, name_file)) as writer:
        for idx in Output_data.keys():
            Output_data[idx].to_excel(writer, sheet_name= idx)
        writer.save()
        # writer.close()

    return Output_data['C_Potencia'], Output_data['E_max'], name_file, tiempo, df_Branch, df_Bus, Output_data[cost_name], Output_data

def graph_results_ope(Output_data, info_):

    if type(info_) == str:

        df_pf = Output_data['pf']
        df_gen_i = Output_data['P']
        df_gen_j = Output_data['P_j']
        df_gen_w = Output_data['Pw']
        df_gen_so = Output_data['Ps']
        df_Pot_Ba_ch = Output_data['Pot_Ba_ch']
        df_Pot_Ba_dc = Output_data['Pot_Ba_dc']
        df_e_b = Output_data['e_b']

        if info_ == 'Simplificado':
            pass
        else:
            df_losses = Output_data['line_losses']
            df_complete_pf = Output_data['pf_complete']

            Losses = {}
            completePf = {}

            for i in df_losses.columns:
                Losses[i] = df_losses[i].values.tolist()

            for i in df_complete_pf.columns:
                completePf[i] = df_complete_pf[i].values.tolist()
    else:
        df_pf = Output_data['pf']
        df_gen_i = Output_data['P']
        df_gen_j = Output_data['P_j']
        df_gen_w = Output_data['Pw']
        df_gen_so = Output_data['Ps']
        df_Pot_Ba_ch = Output_data['Pot_Ba_ch']
        df_Pot_Ba_dc = Output_data['Pot_Ba_dc']
        df_e_b = Output_data['e_b']

        if 'Pérdidas' in info_:
            df_losses = Output_data['line_losses']
            df_complete_pf = Output_data['pf_complete']

            Losses = {}
            completePf = {}

            for i in df_losses.columns:
                Losses[i] = df_losses[i].values.tolist()

            for i in df_complete_pf.columns:
                completePf[i] = df_complete_pf[i].values.tolist()

    time = df_pf.index

    pf = {}
    genThermal = {}
    genHydro = {}
    genWind = {}
    genSolar = {}
    PotCh = {}
    PotDc = {}
    Energy = {}

    for i in df_gen_i.columns:
        genThermal[i] = df_gen_i[i].values.tolist()

    for i in df_gen_j.columns:
        genHydro[i] = df_gen_j[i].values.tolist()

    for i in df_gen_w.columns:
        genWind[i] = df_gen_w[i].values.tolist()

    for i in df_gen_so.columns:
        genSolar[i] = df_gen_so[i].values.tolist()

    for i in df_pf.columns:
        pf[i] = df_pf[i].values.tolist()

    for i in df_Pot_Ba_ch.columns:
        PotCh[i] = df_Pot_Ba_ch[i].values.tolist()

    for i in df_Pot_Ba_dc.columns:
        PotDc[i] = df_Pot_Ba_dc[i].values.tolist()

    for i in df_e_b.columns:
        Energy[i] = df_e_b[i].values.tolist()


    title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'14'}

    st.markdown('### Graficación de Resultados:')

    st.write('#### Generación')

    ## Thermal

    fig = go.Figure()
    for i in genThermal.keys():
        if sum(genThermal[i]) > 0:
            fig.add_trace(go.Scatter(x=time, y=genThermal[i], name=str(i), line_shape='hv'))

    fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                    title='Generación térmica',
                    xaxis_title='Hora',
                    yaxis_title='[MW]')
    st.plotly_chart(fig)

    ## Hydro

    fig = go.Figure()
    for i in genHydro.keys():
        if sum(genHydro[i]) > 0:
            fig.add_trace(go.Scatter(x=time, y=genHydro[i], name=str(i), line_shape='hv'))

    fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                    title='Generación Hidráulica',
                    xaxis_title='Hora',
                    yaxis_title='[MW]')
    st.plotly_chart(fig)

    ## Wind

    if len(df_gen_w) == 0:
        print('')
    else:

        fig = go.Figure()
        for i in genWind.keys():
            if sum(genWind[i]) > 0:
                fig.add_trace(go.Scatter(x=time, y=genWind[i], name=str(i), line_shape='hv'))

        fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                        title='Generación Eólica',
                        xaxis_title='Hora',
                        yaxis_title='[MW]')
        st.plotly_chart(fig)

    ## Solar

    if len(df_gen_so) == 0:
        print('')
    else:

        fig = go.Figure()
        for i in genSolar.keys():
            if sum(genSolar[i]) > 0:
                fig.add_trace(go.Scatter(x=time, y=genSolar[i], name=str(i), line_shape='hv'))

        fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                        title='Generación Solar',
                        xaxis_title='Hora',
                        yaxis_title='[MW]')
        st.plotly_chart(fig)

    ## power flow

    st.markdown('#### Flujos de potencia')

    fig = go.Figure()

    if type(info_) == str:
        if info_ == 'Simplificado':
            real_pf = pf
        else:
            real_pf = completePf
    else:
        if 'Pérdidas' in info_:
            real_pf = completePf
        else:
            real_pf = pf

    for i in real_pf.keys():

        fig.add_trace(go.Scatter(x=time, y=pf[i], name=str(i), line_shape='hv'))

    fig.update_layout(autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),
                    title='Flujos de potencia',
                    xaxis_title='Hora',
                    yaxis_title='[MW]')
    st.plotly_chart(fig)

    ## SAE

    st.markdown('#### Operación SAE')

    fig = go.Figure()

    if sum(PotCh[i][j] for i in PotCh.keys() for j in range(1,len(time))) <= 0:
        st.info('¡Ningún Sistema de Almacenamieto de Energía fue Instalado!')
    else:
        for i in PotCh.keys():
            if sum(PotCh[i]) > 0:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=time, y=Energy[i][0:24], name='Energía [MWh]', line_shape='linear'), secondary_y=False)
                fig.add_trace(go.Scatter(x=time, y=PotCh[i][1:25], name='Carga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)
                fig.add_trace(go.Scatter(x=time, y=PotDc[i][1:25], name='Descarga [MW]', line=dict(dash='dot'), line_shape='hv'), secondary_y=True)
                fig.update_layout(title_text='Operación SAE ubicado en el nodo {}'.format(i), legend=dict(y=1, traceorder='reversed', font_size=8),
                                autosize=True, width=700, height=500, plot_bgcolor='rgba(0,0,0,0)')
                fig.update_xaxes(title_text='Hora', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
                fig.update_yaxes(title_text='Energía [MWh]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=False)
                fig.update_yaxes(title_text='Potencia [MW]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True, secondary_y=True)
                st.plotly_chart(fig)