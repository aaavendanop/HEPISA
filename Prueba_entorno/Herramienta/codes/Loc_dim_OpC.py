import pandas as pd
from pandas import ExcelWriter
from pyomo.environ import *
from pyomo import environ as pym
from pyomo import kernel as pmo
import math
import time
from datetime import datetime, timedelta
import os
import numpy as np

# para poder corer GLPK desde una API
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
###

def opt_dim(file_system, txt_eff, txt_SOC_min, txt_SOC_ini, txt_time_sim, txt_re_inv, txt_C_Pot, txt_C_Bat, txt_ope_fact, txt_con_fact,
            txt_con_cost, combo):

    StartTime = time.time()

    df_System_data = pd.read_excel(file_system, sheet_name='System_data', header=0, index_col=0)
    df_SM_Unit = pd.read_excel(file_system, sheet_name='SM_Unit', header=0, index_col=0)
    df_SM_map = pd.read_excel(file_system, sheet_name='SM_map', header=0, index_col=0)
    df_Renewable = pd.read_excel(file_system, sheet_name='Renewable', header=0, index_col=0)
    df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)
    df_line_map = pd.read_excel(file_system, sheet_name='line_map', header=0, index_col=0)
    df_load = pd.read_excel(file_system, sheet_name='load', header=0, index_col=0)
    df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)

    #Total time steps
    N_horas = int(txt_time_sim)
    ReadingTime = time.time() - StartTime

    ###### Index definitions ######

    C_hydro = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'HYDRO'].index))
    C_hydro = C_hydro.tolist()
    C_acpm = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'ACPM'].index))
    C_acpm = C_acpm.tolist()
    C_diesel = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'DIESEL'].index))
    C_diesel = C_diesel.tolist()
    C_coal = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'COAL'].index))
    C_coal = C_coal.tolist()
    C_combustoleo = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'COMBUSTOLEO'].index))
    C_combustoleo = C_combustoleo.tolist()
    C_gas = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'GAS'].index))
    C_gas = C_gas.tolist()
    C_wind = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'WIND'].index))
    C_wind = C_wind.tolist()
    C_solar = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'SOLAR'].index))
    C_solar = C_solar.tolist()
    C_thermal = C_acpm + C_diesel + C_coal + C_combustoleo + C_gas

    thermal_idx = df_SM_Unit.index[C_thermal]
    hydro_idx = df_SM_Unit.index[C_hydro]
    solar_idx = df_SM_Unit.index[C_solar]
    wind_idx = df_SM_Unit.index[C_wind]

    ################################################ Sets Definitions ################################################
    StartTime = time.time()

    model = ConcreteModel()

    model.t = RangeSet(1,N_horas)                                               ## Time Set
    model.tt = SetOf(model.t)                                              ## Time Set relations intertemporal
    model.i = Set(initialize=thermal_idx, ordered=True)                    ## Thermal generators set
    model.w = Set(initialize=wind_idx, ordered=True)                       ## Wind generators set
    model.so = Set(initialize=solar_idx, ordered=True)                     ## Solar generators set
    model.j = Set(initialize=hydro_idx, ordered=True)                      ## Hydro generators set
    model.b = Set(initialize=df_Bus.index.tolist(), ordered=True)          ## Bus system set
    model.l = Set(initialize=df_Branch.index.tolist(), ordered=True)       ## Lines Set
    model.k = Set(initialize=['k1','k2','k3'], ordered=True)               ## Index number of linearized segments

    ########################################## Parameters definitions ####################################

    model.delta_theta = Param(initialize=15*math.pi/180)
    L = 3
    def alpha_init(model):
        return sum(model.delta_theta * (2*l - 1) / L for l in range(1,L))
    model.alpha = Param(initialize=alpha_init)


    # Piecewise Linear Approximation of Quadratic Cost Curve (Upper Approximation)

    ## slope segment k generator i
    def slope_init(model,i,k):
        if model.k.ord(k) == 1:
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,'k1']+df_SM_Unit.loc[i,'Pmin'])+df_SM_Unit.loc[i,'b'])
        else:
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,k]+df_SM_Unit.loc[i,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[i,'b'])
    model.slope = Param(model.i, model.k, initialize=slope_init)

    ## Minimum production cost of unit i at Pmin
    def fg_min_init(model,i):
        return (df_SM_Unit.loc[i,'a']*df_SM_Unit.loc[i,'Pmin']*df_SM_Unit.loc[i,'Pmin']+df_SM_Unit.loc[i,'b']*df_SM_Unit.loc[i,'Pmin'] + df_SM_Unit.loc[i,'c'])
    model.fg_min = Param(model.i, initialize=fg_min_init)

    #### Hydro

    def slope_Hydro_init(model,j,k):
        if model.k.ord(k) == 1:
            return (df_SM_Unit.loc[j,'a']*(df_SM_Unit.loc[j,'k1']+df_SM_Unit.loc[j,'Pmin'])+df_SM_Unit.loc[j,'b'])
        else:
            return (df_SM_Unit.loc[j,'a']*(df_SM_Unit.loc[j,k]+df_SM_Unit.loc[j,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[j,'b'])
    model.slope_j = Param(model.j, model.k, initialize=slope_Hydro_init)

    def fg_min_j_init(model,j):
        return (df_SM_Unit.loc[j,'a']*df_SM_Unit.loc[j,'Pmin']*df_SM_Unit.loc[j,'Pmin']+df_SM_Unit.loc[j,'b']*df_SM_Unit.loc[j,'Pmin'] + df_SM_Unit.loc[j,'c'])
    model.fg_min_j = Param(model.j, initialize=fg_min_j_init)

    #Parameters for minimum up time constraints

    def onoff_t0_init(model,i):
        if df_SM_Unit.loc[i,'IniT_ON'] > 0:
            a = 1
        else:
            a = 0
        return a
    model.onoff_t0 = Param(model.i, initialize=onoff_t0_init)

    def L_up_min_init(model,i):
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_ON']-df_SM_Unit.loc[i,'IniT_ON'])*model.onoff_t0[i])
    model.L_up_min = Param(model.i, initialize=L_up_min_init)

    def L_down_min_init(model,i):
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_OFF']-df_SM_Unit.loc[i,'IniT_off'])*(1-model.onoff_t0[i]))
    model.L_down_min = Param(model.i, initialize=L_down_min_init)

    ## Conductance of each line

    def Conductance_init(model,l):
        return (df_Branch.loc[l,'R'])/(df_Branch.loc[l,'R']*df_Branch.loc[l,'R'] + df_Branch.loc[l,'X']*df_Branch.loc[l,'X'])
    model.Conductance = Param(model.l, rule=Conductance_init)

    # Sbase [MVA]
    S_base = df_System_data.loc['S_base'][0]
    model.MVA_base = Param(initialize=S_base)

    re_inv = int(txt_re_inv)
    discount = 0
    Costo_potencia = int(txt_C_Pot)
    Costo_energia = int(txt_C_Bat)
    Costo_potencia_eqv = round(Costo_potencia*(1-discount/100)/(365*re_inv),2)
    Costo_energia_eqv = round(Costo_energia*(1-discount/100)/(365*re_inv),2)

    ####Parameters BESS
    Big_number = 1e20
    model.Costo_potencia = Param(initialize=Costo_potencia_eqv) ## Costo del inversor de potencia de la batería
    model.Costo_energia = Param(initialize=Costo_energia_eqv)  ## Costo de los modulos de baterías
    Eficiencia = float(txt_eff)        ## Eficiencia global del BESS
    SOC_min = 1-float(txt_SOC_min)       ## Minimum State of Charge of BESS
    SOC_ini = float(txt_SOC_ini)       ## Initial State of Charge of BESS

    model.Eficiencia_descarga = Param(initialize=Eficiencia)      ## Eficiencia global del BESS
    model.Eficiencia_carga = Param(initialize=Eficiencia)

    #################################### Define Variables ##############################

    model.status = Var(model.i, model.t, within=Binary, initialize=0)
    model.status_j = Var(model.j, model.t, within=Binary, initialize=0)
    model.P_i = Var(model.i, model.t, domain=NonNegativeReals)
    model.P_j = Var(model.j, model.t, domain=NonNegativeReals)
    model.P_seg_i = Var(model.i, model.t, model.k, domain=NonNegativeReals)
    model.P_seg_j = Var(model.j, model.t, model.k, domain=NonNegativeReals)

    model.pcost = Var(model.t, domain=NonNegativeReals)
    model.pcost_j = Var(model.t, domain=NonNegativeReals)
    model.costgen = Var(model.i, model.t, domain=NonNegativeReals)
    model.costgen_j = Var(model.j, model.t, domain=NonNegativeReals)

    model.Pw = Var(model.w, model.t, domain=NonNegativeReals)
    model.Ps = Var(model.so, model.t, domain=NonNegativeReals)
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)
    ## model.cost = Var(doc='total operation cost')

    model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))
    model.pf = Var(model.l, model.t)

    model.theta_sr = Var(model.l, model.t)
    model.theta_sr_pos = Var(model.l, model.t)
    model.theta_sr_neg = Var(model.l, model.t)
    model.theta_sr_abs = Var(model.l, model.t)
    model.line_losses = Var(model.l, model.t, domain=Reals, bounds=(-1e6,1e6), initialize=0)

    model.abs_var = Var(model.l, model.t, domain=Binary, initialize=0)
    model.line_losses_pos = Var(model.l, model.t, initialize=0)
    model.line_losses_neg = Var(model.l, model.t, domain=NonNegativeReals, initialize=0)
    model.line_losses_abs = Var(model.l, model.t, domain=NonNegativeReals, initialize=0)

    # positive and negative pf (new)

    model.pf_pos = Var(model.l, model.t, domain=NonNegativeReals)
    model.pf_neg = Var(model.l, model.t, domain=NonNegativeReals)

    ############ BESS#######
    model.u_ch = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery charge
    model.u_dc = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery discharge
    model.Pot_Ba_ch = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery charge
    model.Pot_Ba_dc = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery discharge
    model.e_b = Var(model.b, model.t, domain=NonNegativeReals, initialize=0)            ## Energy of battery
    model.C_Pot = Var(model.b, domain = NonNegativeReals, bounds=(0,1e6), initialize=0) ## Power Size
    model.E_max = Var(model.b, domain = NonNegativeReals, bounds=(0,1e12), initialize=0) ## Energy Size

    ###slack

    Slack_bus = df_System_data.loc['Slack_bus'][0]

    for t in model.t:
        model.theta[Slack_bus,t].fix(0)

    ModelingTime = time.time() - StartTime

    ###################################################### MODEL ######################################################

    StartTime = time.time()

    ## Objective function definition

    ## Thermal

    def P_sum_rule(model,i,t):
        return model.P_i[i,t] == model.status[i,t]*df_SM_Unit.loc[i,'Pmin'] + sum(model.P_seg_i[i,t,k] for k in model.k)
    model.P_sum = Constraint(model.i, model.t, rule=P_sum_rule)

    def costgen_rule(model, i, t):
        return model.costgen[i,t] == model.status[i,t]*model.fg_min[i] + sum(model.P_seg_i[i,t,k]*model.slope[i,k] for k in model.k)
    model.costgen_fn = Constraint(model.i, model.t, rule=costgen_rule)

    def pcost_rule(model, t):
        return model.pcost[t] == sum(model.costgen[i,t] + model.costSU[i,t] + model.costSD[i,t]  for i in model.i)
    model.costfnperiod = Constraint(model.t, rule=pcost_rule)

    ## Hydro

    def P_Hydro_sum_rule(model,j,t):
        return model.P_j[j,t] == model.status_j[j,t] * df_SM_Unit.loc[j,'Pmin'] + sum(model.P_seg_j[j,t,k] for k in model.k)
    model.P_Hydro_sum = Constraint(model.j, model.t, rule=P_Hydro_sum_rule)

    def costgen_Hydro_rule(model, j, t):
        return model.costgen_j[j,t] == model.status_j[j,t] * model.fg_min_j[j] + sum(model.P_seg_j[j,t,k]*model.slope_j[j,k] for k in model.k)
    model.costgen_Hydro_fn = Constraint(model.j, model.t, rule=costgen_Hydro_rule)

    def pcost_Hydro_rule(model, t):
        return model.pcost_j[t] == sum(model.costgen_j[j,t] for j in model.j)
    model.costfnperiod_Hydro = Constraint(model.t, rule=pcost_Hydro_rule)

    ## Objective function

    def cost_rule(model):
        return int(txt_ope_fact) * (sum(model.pcost[t] for t in model.t) + sum(model.pcost_j[t] for t in model.t)) + \
            sum(model.C_Pot[b] * model.Costo_potencia + model.E_max[b] * model.Costo_energia for b in model.b) + \
            int(txt_con_fact) * (sum(model.pf_pos[l,t] + model.pf_neg[l,t] + 0.5 * model.line_losses_pos[l,t] +
                                0.5 * model.line_losses_neg[l,t] - df_Branch.loc[l,'Flowlimit'] for l in model.l for t in model.t)) * float(txt_con_cost)
    model.cost = Objective(rule=cost_rule, sense=minimize)

    ##### Constraints

    ## Startup/down cost

    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == df_SM_Unit.loc[i,'CSU']*model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def CostSDfn_init(model,i,t):
        return model.costSD[i,t] == df_SM_Unit.loc[i,'CSU']*model.SD[i,t]
    model.CostSDfn = Constraint(model.i, model.t, rule=CostSDfn_init)

    ## Power limits of generators

    # Thermal

    def P_lim_min_rule(model,i,t):
        return model.P_i[i,t] >= df_SM_Unit.loc[i,'Pmin']*model.status[i,t]
    model.P_min_lim = Constraint(model.i, model.t, rule=P_lim_min_rule)

    def P_lim_max_rule(model,i,t):
        return model.P_i[i,t] <= df_SM_Unit.loc[i,'Pmax']*model.status[i,t]
    model.P_max_lim = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_seg_base_max_rule(model,i,t,k):
        return model.P_seg_i[i,t,k]<=(df_SM_Unit.loc[i,'k2']-df_SM_Unit.loc[i,'k1'])*model.status[i,t];
    model.P_seg_base_max = Constraint(model.i,model.t,model.k, rule=P_seg_base_max_rule)

    # Hydro

    def P_Hydro_lim_min_rule(model,j,t):
        return model.P_j[j,t] >= df_SM_Unit.loc[j,'Pmin'] * model.status_j[j,t]
    model.P_Hydro_lim_min = Constraint(model.j, model.t, rule=P_Hydro_lim_min_rule)

    def P_Hydro_lim_max_rule(model,j,t):
        return model.P_j[j,t] <= df_SM_Unit.loc[j,'Pmax'] * model.status_j[j,t]
    model.P_Hydro_lim_max = Constraint(model.j, model.t, rule=P_Hydro_lim_max_rule)

    def P_Hydro_seg_lim_max_rule(model,j,t,k):
        return model.P_seg_j[j,t,k] <= (df_SM_Unit.loc[j,'k2']-df_SM_Unit.loc[j,'k1']) * model.status_j[j,t]
    model.P_Hydro_seg_lim_max = Constraint(model.j, model.t, model.k, rule=P_Hydro_seg_lim_max_rule)

    # Wind

    def Wind_lim_max_rule(model,w,t):
        return model.Pw[w,t] <= df_Renewable.loc[t,w]
    model.Wind_max_lim = Constraint(model.w, model.t, rule=Wind_lim_max_rule)

    # Solar

    def maxPs_rule(model, so, t):
        return model.Ps[so,t] <= df_Renewable.loc[t,so]
    model.maxPs = Constraint(model.so, model.t, rule=maxPs_rule)

    ## Integer Constraint

    def bin_cons1_rule(model,i,t):
        if t == model.t.first():
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]
        else:
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons2_rule(model,i,t):
        return model.SU[i,t] + model.SD[i,t] <= 1
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)

    # ## Min up_dn time

    # def min_up_dn_time_1_rule(model,i,t):
    #     if model.L_up_min[i] + model.L_down_min[i] > 0 and t < model.L_up_min[i] + model.L_down_min[i]:
    #         return model.status[i,t] == model.onoff_t0[i]
    #     else:
    #         return Constraint.Skip
    # model.min_up_dn_time_1 = Constraint(model.i, model.t, rule=min_up_dn_time_1_rule)

    # def min_up_dn_time_2_rule(model,i,t):
    #     return sum(model.SU[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_ON']+1 and tt <= t) <= model.status[i,t]
    # model.min_up_dn_time_2 = Constraint(model.i, model.t, rule=min_up_dn_time_2_rule)

    # def min_up_dn_time_3_rule(model,i,t):
    #     return sum(model.SD[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_OFF']+1 and tt <= t) <= 1-model.status[i,t]
    # model.min_up_dn_time_3 = Constraint(model.i, model.t, rule=min_up_dn_time_3_rule)

    ## ramp constraints

    def ramp_up_fn_rule(model,i,t):
        if t > 1:
            return model.P_i[i,t] - model.P_i[i,t-1] <= df_SM_Unit.loc[i,'Ramp_Up']*model.status[i,t-1] + df_SM_Unit.loc[i,'Pmin']*(model.status[i,t]-model.status[i,t-1]) + df_SM_Unit.loc[i,'Pmax']*(1-model.status[i,t])
        else:
            return Constraint.Skip
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)

    def ramp_dw_fn_rule(model,i,t):
        if t > 1:
            return model.P_i[i,t-1] - model.P_i[i,t] <= df_SM_Unit.loc[i,'Ramp_Down']*model.status[i,t] + df_SM_Unit.loc[i,'Pmin']*(model.status[i,t-1]-model.status[i,t]) + df_SM_Unit.loc[i,'Pmax']*(1-model.status[i,t-1])
        else:
            return Constraint.Skip
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)

    ## Angle definition

    def theta_sr_fn_rule(model,l,t):
        return model.theta_sr[l,t] == sum(model.theta[b,t]*df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)
    model.theta_sr_fn = Constraint(model.l, model.t, rule=theta_sr_fn_rule)

    def abs_definition_1_rule(model,l,t):
        return model.theta_sr[l,t] <= Big_number * model.abs_var[l,t]
    model.abs_definition_1 = Constraint(model.l, model.t, rule=abs_definition_1_rule)

    def abs_definition_2_rule(model,l,t):
        return model.theta_sr[l,t] >= -(1 - model.abs_var[l,t]) * 20
    model.ans_definition_2 = Constraint(model.l, model.t, rule=abs_definition_2_rule)

    def theta_positive_1_rule(model,l,t):
        return model.theta_sr_pos[l,t] <= Big_number * (model.abs_var[l,t])
    model.theta_positive_1 = Constraint(model.l, model.t, rule=theta_positive_1_rule)

    def theta_positive_2_rule(model,l,t):
        return model.theta_sr_pos[l,t] <= - model.theta_sr[l,t]
    model.theta_positive_2 = Constraint(model.l, model.t, rule=theta_positive_2_rule)

    def theta_negative_1_rule(model,l,t):
        return model.theta_sr_neg[l,t] <= Big_number * (1 - model.abs_var[l,t])
    model.theta_negative_1 = Constraint(model.l, model.t, rule=theta_negative_1_rule)

    def theta_negative_2_rule(model,l,t):
        return model.theta_sr_neg[l,t] <= model.theta_sr[l,t]
    model.theta_negative_2 = Constraint(model.l, model.t, rule=theta_negative_2_rule)

    def q_rule(model,l,t):
        return model.theta_sr_neg[l,t] <= 0
    model.q = Constraint(model.l, model.t, rule=q_rule)

    def qq_rule(model,l,t):
        return model.theta_sr_pos[l,t] <= 0
    model.qq = Constraint(model.l, model.t, rule=qq_rule)

    def e_rule(model,l,t):
        return model.theta_sr[l,t] == - model.theta_sr_pos[l,t] - - model.theta_sr_neg[l,t]
    model.e = Constraint(model.l, model.t, rule=e_rule)

    def total_rule(model,l,t):
        return model.theta_sr_abs[l,t] == - model.theta_sr_pos[l,t] + - model.theta_sr_neg[l,t]
    model.total = Constraint(model.l, model.t, rule=total_rule)

    ####DC transmission network security constraint

    def line_flow_rule(model, l, t):
        return model.pf[l,t] == model.MVA_base*(1/df_Branch.loc[l,'X'])*(- model.theta_sr_pos[l,t] - - model.theta_sr_neg[l,t])
    model.line_flow = Constraint(model.l, model.t, rule=line_flow_rule)

    def line_min_rule(model, l, t):
        return model.pf[l,t] + 0.5 * model.line_losses[l,t] >= - df_Branch.loc[l,'Flowlimit']
    model.line_min = Constraint(model.l, model.t, rule=line_min_rule)

    def line_max_rule(model, l, t):
        return model.pf[l,t] + 0.5 * model.line_losses[l,t] <= df_Branch.loc[l,'Flowlimit']
    model.line_max = Constraint(model.l, model.t, rule=line_max_rule)

    # positive and negative constraint powerflow (new)

    def line_flow_abs_rule(model,l,t):
        return model.pf[l,t] == model.pf_pos[l,t] - model.pf_neg[l,t]
    model.line_flow_abs = Constraint(model.l, model.t, rule=line_flow_abs_rule)

    # Losses

    def losses_rule(model,l,t):
        return model.line_losses[l,t] == model.MVA_base * model.alpha * model.Conductance[l] * (model.theta_sr[l,t])
    model.losses = Constraint(model.l, model.t, rule=losses_rule)

    def losses_abs_rule(model, l, t):
        return model.line_losses[l,t] == model.line_losses_pos[l,t] - model.line_losses_neg[l,t]
    model.losses_abs = Constraint(model.l, model.t, rule=losses_abs_rule)

    #--------------------------------DC PF------------------------------------------

    #contribución al balance de los BESS

    def power_balance_rule(model,t,b):
        return sum(model.P_i[i,t] for i in model.i if df_SM_map.loc[i,b]) +sum(model.P_j[j,t] for j in model.j if df_SM_map.loc[j,b]) +\
            sum(model.Ps[so,t] for so in model.so if df_SM_map.loc[so,b]) + sum(model.Pw[w,t] for w in model.w if df_SM_map.loc[w,b]) -\
            sum((model.pf[l,t] + 0.5 * model.line_losses[l,t])*df_line_map.loc[l,b] for l in model.l if df_line_map.loc[l,b] != 0) +\
            model.Pot_Ba_dc[b,t] - model.Pot_Ba_ch[b,t] == df_load.loc[t,b]
    model.power_balance = Constraint(model.t, model.b, rule=power_balance_rule)

    #--------------------------------BESS------------------------------------------

    ############################################################################################################################

    ## power charging Constraints

    def power_c_max_rule(model,b,t):
        return model.Pot_Ba_ch[b,t] <= Big_number*model.u_ch[b,t]
    model.power_c_max = Constraint(model.b, model.t, rule=power_c_max_rule)

    def power_c_max_2_rule(model,b,t):
        return model.Pot_Ba_ch[b,t] <= model.C_Pot[b]
    model.power_c_max_2 = Constraint(model.b, model.t, rule=power_c_max_2_rule)

    ## power dischraging Constraints

    def power_d_max_rule(model,b,t):
        return model.Pot_Ba_dc[b,t] <= Big_number * model.u_dc[b,t]
    model.power_d_max = Constraint(model.b, model.t, rule=power_d_max_rule)

    def power_d_max_2_rule(model,b,t):
        return model.Pot_Ba_dc[b,t] <= model.C_Pot[b]
    model.power_d_max_2 = Constraint(model.b, model.t, rule=power_d_max_2_rule)

    ## Simultaneous charging and discharging Constraint

    def sim_rule(model,b,t):
        return model.u_ch[b,t] + model.u_dc[b,t] <= 1
    model.sim = Constraint(model.b, model.t, rule=sim_rule)

    ## relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,b,t):
        if t == 1:
            return model.e_b[b,t] == model.E_max[b] * SOC_ini + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga
        else:
            return model.e_b[b,t] == model.e_b[b,t-1] + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga
    model.energy = Constraint(model.b, model.t, rule=energy_rule)

    ### Energy limits

    def energy_limit_rule(model,b,t):
        return model.e_b[b,t] <= model.E_max[b]
    model.energy_limit = Constraint(model.b, model.t, rule=energy_limit_rule)

    def energy_limit_min_rule(model,b,t):
        return model.e_b[b,t] >= model.E_max[b] * SOC_min
    model.energy_limit_min = Constraint(model.b, model.t, rule=energy_limit_min_rule)

    ###Solution#####

    def pyomo_postprocess(options=None, instance=None, results=None):
        model.C_Pot.display()
        model.E_max.display()
        model.pf.display()
        model.P_i.display()
        model.P_j.display()
    # Configuracion:

    solver_selected = combo

    if solver_selected == 'CPLEX':
#        if __name__ == '__main__':
        # This emulates what the pyomo command-line tools does
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
        results.write()
        print('\nDisplaying Solution\n' + '-'*60)
        pyomo_postprocess(None, model, results)
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        results.write()
        print('\nDisplaying Solution\n' + '-'*60)
        pyomo_postprocess(None, model, results)

    SolvingTime = time.time() - StartTime

    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)

    tiempo = timedelta(seconds=TotalTime)

    print('Reading DATA time:',round(ReadingTime,3), '[s]')
    print('Modeling time:',round(ModelingTime,3), '[s]')
    print('Solving time:',round(SolvingTime,3), '[s]')
    print('Total time:', tiempo)

    #################################################################################
    #######################Creación de Archivo Excel#################################
    #################################################################################

    V_pf = np.ones((len(model.l),len(model.t)))
    V_gen_i = np.ones((len(model.i),len(model.t)))
    V_gen_j = np.ones((len(model.j),len(model.t)))
    V_gen_w = np.ones((len(model.w),len(model.t)))
    V_gen_so = np.ones((len(model.so),len(model.t)))
    V_losses = np.ones((len(model.l),len(model.t)))
    V_Pot_Ba_ch = np.ones((len(model.b),len(model.t)))
    V_Pot_Ba_dc = np.ones((len(model.b),len(model.t)))
    V_e_b = np.ones((len(model.b),len(model.t)))
    V_cost = model.cost.value()
    V_E_size = np.ones(len(model.E_max))
    V_P_size = np.ones(len(model.C_Pot))

# Se multiplica por 0.5 por formulación de pérdidas
    for l in model.l:
        for t in model.t:
            V_pf[model.l.ord(l)-1, t-1] = model.pf[l,t].value + 0.5 * model.line_losses[l,t].value

    for i in model.i:
        for t in model.t:
            V_gen_i[model.i.ord(i)-1, t-1] = model.P_i[i,t].value

    for j in model.j:
        for t in model.t:
            V_gen_j[model.j.ord(j)-1, t-1] = model.P_j[j,t].value

    for w in model.w:
        for t in model.t:
            V_gen_w[model.w.ord(w)-1, t-1] = model.Pw[w,t].value

    for so in model.so:
        for t in model.t:
            V_gen_so[model.so.ord(so)-1, t-1] = model.Ps[so,t].value

    for l in model.l:
        for t in model.t:
            V_losses[model.l.ord(l)-1, t-1] = 0.5 * model.line_losses[l,t].value

    for b in model.b:
        for t in model.t:
            V_Pot_Ba_ch[model.b.ord(b)-1, t-1] = model.Pot_Ba_ch[b,t].value

    for b in model.b:
        for t in model.t:
            V_Pot_Ba_dc[model.b.ord(b)-1, t-1] = model.Pot_Ba_dc[b,t].value

    for b in model.b:
        for t in model.t:
            V_e_b[model.b.ord(b)-1, t-1] = model.e_b[b,t].value

    for b in model.b:
        V_E_size[model.b.ord(b)-1] = model.E_max[b].value

    for b in model.b:
        V_P_size[model.b.ord(b)-1] = model.C_Pot[b].value


    df_pf = pd.DataFrame(V_pf)
    df_gen_i = pd.DataFrame(V_gen_i)
    df_gen_j = pd.DataFrame(V_gen_j)
    df_gen_w = pd.DataFrame(V_gen_w)
    df_gen_so = pd.DataFrame(V_gen_so)
    df_losses = pd.DataFrame(V_losses)
    df_Pot_Ba_ch = pd.DataFrame(V_Pot_Ba_ch)
    df_Pot_Ba_dc = pd.DataFrame(V_Pot_Ba_dc)
    df_e_b = pd.DataFrame(V_e_b)
    df_E_size= pd.DataFrame(V_E_size)
    df_P_size  = pd.DataFrame(V_P_size)
    df_cost = pd.DataFrame(V_cost, index=['1','2'], columns=['Cost'])
    df_cost  = df_cost.drop(['2'], axis=0)

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc.xlsx'

    path = os.path.join(mydir, name_file)

    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')

    df_pf.to_excel(writer, sheet_name='pf', index=True)
    df_gen_i.to_excel(writer, sheet_name='gen_Thermal', index=True)
    df_gen_j.to_excel(writer, sheet_name='gen_Hydro', index=True)
    df_gen_w.to_excel(writer, sheet_name='gen_Wind', index=True)
    df_gen_so.to_excel(writer, sheet_name='gen_Solar', index=True)
    df_losses.to_excel(writer, sheet_name='Losses', index=True)
    df_Pot_Ba_ch.to_excel(writer, sheet_name='BESS_Ch_Power', index=True)
    df_Pot_Ba_dc.to_excel(writer, sheet_name='BESS_Dc_Power', index=True)
    df_e_b.to_excel(writer, sheet_name='BESS_Energy', index=True)
    df_E_size.to_excel(writer, sheet_name='Energy_size', index=True)
    df_P_size.to_excel(writer, sheet_name='Power_size', index=True)
    df_cost.to_excel(writer, sheet_name='cost', index=True)
    writer.save()
    writer.close()
    ##########################################################################

    return model.C_Pot, model.E_max, name_file, tiempo
