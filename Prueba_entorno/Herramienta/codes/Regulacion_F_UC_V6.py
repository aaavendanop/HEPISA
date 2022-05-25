# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:20:08 2020

UNIT COMMITMENT WITH ESS AND FREQUENCY CONSTRAINS

@author: Andres Felipe Peñaranda Bayona
"""

# In[Librerias]

from pyomo.environ import *
from pyomo import environ as pym
from pyomo import kernel as pmo
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import numpy as np
import math

# In[Study case]

def UC_ESS_with_Freq(system_data,Simulation_hours,opt_option):


    df_System_data = pd.read_excel(system_data, sheet_name='System_data', header=0, index_col=0)
    df_bus = pd.read_excel(system_data, sheet_name='Bus', header=0, index_col=0)
    df_SM_Unit = pd.read_excel(system_data, sheet_name='SM_Unit', header=0, index_col=0)
    df_SM_map = pd.read_excel(system_data, sheet_name='SM_map', header=0, index_col=0)
    df_Renewable = pd.read_excel(system_data, sheet_name='Renewable', header=0, index_col=0)
    df_branch = pd.read_excel(system_data, sheet_name='Branch', header=0, index_col=0)
    df_line_map = pd.read_excel(system_data, sheet_name='line_map', header=0, index_col=0)
    df_load = pd.read_excel(system_data, sheet_name='load', header=0, index_col=0)
    df_Reserve = pd.read_excel(system_data, sheet_name='Reserve', header=0, index_col=0)
    df_ESS_Unit = pd.read_excel(system_data, sheet_name='ESS_Unit', header=0, index_col=0)
    df_ESS_map = pd.read_excel(system_data, sheet_name='ESS_map', header=0, index_col=0)
    df_E_price = pd.read_excel(system_data, sheet_name='ESS_Energy_price', header=0, index_col=0)
    df_C_DR = pd.read_excel(system_data, sheet_name ='C_DR_load', header=0, index_col=0)
    df_P_C_DR = pd.read_excel(system_data, sheet_name ='PDR', header=0, index_col=0)


    # In[Model]

    ## Simulation type

    model = ConcreteModel()

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # SETS
    #-------------------------------------------------------------------------------------------------------------------------------------------



    model.t = RangeSet(1,Simulation_hours)                                        # Horizon simulation time
    model.tt = SetOf(model.t)

    I = []
    for i in df_SM_Unit.index.tolist():
        if df_SM_Unit.loc[i,'Fuel_Type'] != 'SOLAR' and df_SM_Unit.loc[i,'Fuel_Type'] != 'WIND':
            I.append(i)
    model.i = Set(initialize=I)                                                   # Generation units set

    model.b = Set(initialize=df_bus.index.tolist(), ordered=True)                 # Buses of system set
    model.k = Set(initialize=['k1', 'k2', 'k3'],ordered=True)                     # Segments of generation cost for generation units

    W = []
    for i in df_SM_Unit.index.tolist():
        if df_SM_Unit.loc[i,'Fuel_Type'] == 'WIND':
            W.append(i)
    model.w = Set(initialize=W)                                                   # Wind generation units set

    S = []
    for i in df_SM_Unit.index.tolist():
        if df_SM_Unit.loc[i,'Fuel_Type'] == 'SOLAR':
            S.append(i)
    model.s = Set(initialize=S)                                                   # Pv generation units set

    model.l = Set(initialize=df_line_map.index.tolist(),ordered=True)             # lines of system set
    model.n = Set(initialize=df_ESS_map.index.tolist(), ordered=True)             # Energy storage sistem units set

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETERS
    #-------------------------------------------------------------------------------------------------------------------------------------------

    model.MVA_base = Param(initialize=df_System_data.loc['S_base','Data'])        # Power base of the system

    model.Nominal_Frequency = Param(initialize=df_System_data.loc['N_freq','Data'])  # Nominal frequency of the system [Hz]
    model.Max_Frequency_desviation = Param(initialize=df_System_data.loc['Max_D_freq','Data'])  # Maximum frequency desviation [Hz/s]
    model.Minimum_Frequency = Param(initialize=df_System_data.loc['Min_freq','Data'])    # Minimum frequency limit [Hz]
    model.DB_Frequency = Param(initialize=df_System_data.loc['DB_freq','Data'])   # Dead Band governors frequency [Hz]

    model.Delta_t1 = Param(initialize=df_System_data.loc['Delta_1_RF','Data'])    # Time delta of post-contingency inertia response [h]
    model.Delta_t2 = Param(initialize=df_System_data.loc['Delta_2_RF','Data'])    # Time delta of primary frequency response [h]
    model.Delta_t3 = Param(initialize=df_System_data.loc['Delta_3_RF','Data'])    # Time delta of second frequency response [h]


    def Power_max(model,i):
        return df_SM_Unit.loc[i,'Pmax']
    model.PG_max = Param(model.i, initialize = Power_max)                         # Maximum power capacity of each generator [MW]

    def Power_min(model,i):
        return df_SM_Unit.loc[i,'Pmin']
    model.PG_min = Param(model.i, initialize = Power_min)                         # Minimum power capacity of each generator [MW]

    def Power_seg_max(model,i,k):
        return df_SM_Unit.loc[i,'k2']-df_SM_Unit.loc[i,'k1']
    model.P_seg_max = Param(model.i, model.k, initialize = Power_seg_max)         # Maximum power capacity of each segment k of each generator [MW]

    def Cost_seg(model,i,k):
        if model.k.ord(k) == 1:
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,'k1']+df_SM_Unit.loc[i,'Pmin'])+df_SM_Unit.loc[i,'b'])
        else:
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,k]+df_SM_Unit.loc[i,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[i,'b'])
    model.C_seg = Param(model.i, model.k, initialize = Cost_seg)                  # Cost of power of each segment k of each generator [$/MW]

    def fg_min_init(model,i):
        return ((df_SM_Unit.loc[i,'a']*df_SM_Unit.loc[i,'Pmin']*df_SM_Unit.loc[i,'Pmin'])+(df_SM_Unit.loc[i,'b']*df_SM_Unit.loc[i,'Pmin'])+(df_SM_Unit.loc[i,'c']))
    model.FG_min= Param(model.i, initialize=fg_min_init)                          # Price of generate minimum power for each generator [$]

    def onoff_t0_init(model,i):
        if df_SM_Unit.loc[i,'IniT_ON'] > 0:
            a = 1
        else:
            a = 0
        return a
    model.onoff_t0 = Param(model.i, initialize=onoff_t0_init)                     # Initial ON/OFF signal [Binary]

    def L_up_min_init(model,i):
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_ON']-df_SM_Unit.loc[i,'IniT_ON'])*model.onoff_t0[i])
    model.L_up_min = Param(model.i, initialize=L_up_min_init)                     # Minimum time for each generator when it is ON

    def L_down_min_init(model,i):
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_OFF']-df_SM_Unit.loc[i,'IniT_off'])*(1-model.onoff_t0[i]))
    model.L_down_min = Param(model.i, initialize=L_down_min_init)                 # Minimum time for each generator when it is OFF

    def Reserve_max_rule(model,i):
        return df_SM_Unit.loc[i,'R']*df_SM_Unit.loc[i,'Pmax']
    model.Reserve_Max = Param(model.i, initialize=Reserve_max_rule)               # Maximum power of reserve for each generator [MW]

    def CR_rule(model,i):
        return df_SM_Unit.loc[i,'b']
    model.C_Reserve = Param(model.i, initialize=CR_rule)                          # Cost of Primary reserve [$/MW]

    def Ramp_power(model,i):
        return df_SM_Unit.loc[i,'Ramp_Up']
    model.Ramp = Param(model.i, initialize=Ramp_power)                            # Ramp power for each generator [MW/s]


    def Power_wind(model,w,t):
        return df_Renewable.loc[t,w]
    model.Pw_max = Param(model.w, model.t, initialize = Power_wind)               # Power dispatch of wind turbine w at time t

    def Power_solar(model,s,t):
        return df_Renewable.loc[t,s]
    model.Ppv_max = Param(model.s, model.t, initialize = Power_solar)             # Power dispatch of pv systems s at time t


    def C_P_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'C_Potencia']
    model.C_Pot = Param(model.n, initialize=C_P_Ba_rule)                          # Power Size [MW]

    def C_E_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'C_Energia']
    model.E_max = Param(model.n, initialize=C_E_Ba_rule)                          # Energy Size [MWh]

    def C_nch_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'n_ch_eff']
    model.n_ch = Param(model.n, initialize=C_nch_Ba_rule)                         # Charge efficency of ESS [p.u.]

    def C_ndc_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'n_dc_eff']
    model.n_dc = Param(model.n, initialize=C_ndc_Ba_rule)                         # Discharge efficency of ESS [p.u.]

    def C_sdc_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'Self_discharge']
    model.s_dc = Param(model.n, initialize=C_sdc_Ba_rule)                         # Self-Discharge efficency of ESS [p.u./h]

    def C_SOCmin_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'SOC_min']
    model.SOC_min = Param(model.n, initialize=C_SOCmin_Ba_rule)                   # Minimum State of Charge of BESS

    def C_SOCini_Ba_rule(model,n):
        return df_ESS_Unit.loc[n,'SOC_ini']
    model.SOC_ini = Param(model.n, initialize=C_SOCini_Ba_rule)                   # Initial State of Charge of BESS

    def C_D_Ba_rule(model,t):
        return df_E_price.loc[t,'CB_MWh']
    model.C_D_Ba = Param(model.t, initialize=C_D_Ba_rule)                         # Cost of Energy Storage System Degradation [$/MWh]

    def Delta_energy_rule(model,n):
        return df_ESS_Unit.loc[n,'C_Potencia']*(model.Delta_t1 + model.Delta_t2 + 0.5*model.Delta_t3)
    model.Delta_energy = Param(model.n, initialize=Delta_energy_rule)             # Delta energy for contingency supply [MWh]


    def Bus_load(model,b,t):
        return df_load.loc[t,b]
    model.P_load = Param(model.b, model.t, initialize = Bus_load)                 # Load of each bus in the test system [MW]


    def CDR_rule(model,b,t):
        return df_C_DR.loc[t,b]
    model.C_DR = Param(model.b, model.t, initialize=CDR_rule)                     # Cost of power reduction of customer [$/MW]

    def DR_Max_rule(model,b,t):
        return df_load.loc[t,b]*(1-df_P_C_DR.loc[t,b])
    model.Pot_DR_Max = Param(model.b, model.t, initialize=DR_Max_rule)            # Maximum power reduction for each consumer


    def Power_imbalance_value(model,t):
        return df_Reserve.loc[t,'R1']
    model.Power_imbalance = Param(model.t,initialize=Power_imbalance_value)       # Power imbalance at contingency [MW]

    def Inertia_rule(model,i):
        return df_SM_Unit.loc[i,'H']
    model.Inertia = Param(model.i, initialize = Inertia_rule)                     # Inertia value of each generator [s]
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # VARIABLES
    #-------------------------------------------------------------------------------------------------------------------------------------------

    model.P = Var(model.i, model.t, domain=NonNegativeReals)                      # Power dispatch of unit i at time t [MW]
    model.status = Var(model.i, model.t, within=Binary)                           # Status [ON/OFF] for each generator [Binary]
    model.P_seg = Var(model.i, model.t, model.k, domain=NonNegativeReals)         # Power dispatch segment k of i at time t [MW]
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)                 # Start UP cost for each generator [$]
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)                 # Star UP signal [Binary]
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)                 # Star DOWN signal [Binary]
    model.Reserve = Var(model.i, model.t, domain= NonNegativeReals)               # Power resever for primary reserve [MW]

    model.Pw = Var(model.w, model.t, domain=NonNegativeReals)                     # Power dispatch of unit w at time t [MW]
    model.Ppv = Var(model.s, model.t, domain=NonNegativeReals)                    # Power dispatch of unit p at time t [MW]

    model.Pot_DR = Var(model.b, model.t, domain = NonNegativeReals)               # Power reduction of customer [MW]

    model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))                # Voltage angle [rad]
    model.pf = Var(model.l, model.t)                                              # Power flow through the line l [MW]
    model.P_bus = Var(model.b, model.t)                                           # Power bus balance [MW]
    model.PG_bus = Var(model.b, model.t)                                          # Power of SM in bus b [MW]
    model.PW_bus = Var(model.b, model.t)                                          # Power of wind turbine in bus b [MW]
    model.PPV_bus = Var(model.b, model.t)                                         # Power of Pv systems in bus b [MW]
    model.PDC_bus = Var(model.b, model.t)                                         # Charge power of ESS in bus b [MW]
    model.PCH_bus = Var(model.b, model.t)                                         # Discharge power of ESS in bus b [MW]

    for t in model.t:
        model.theta[df_System_data.loc['Slack_bus','Data'],t].fix(0)              # Slack angle [rad]

    model.Pot_Ba_ch = Var(model.n, model.t, initialize=0)                         # Power in battery charge [MW]
    model.Pot_Ba_dc = Var(model.n, model.t, initialize=0)                         # Power in battery discharge [MW]
    model.u_ch = Var(model.n, model.t, within=Binary, initialize=0)               # Status of battery charge {Binary}
    model.u_dc = Var(model.n, model.t, within=Binary, initialize=0)               # Status of battery discharge [Binary]
    model.e_b = Var(model.n, model.t, domain=NonNegativeReals, initialize=0)      # Energy of battery [MWh]

    model.PC_Power_imbalance = Var(model.t, bounds=(0,1e6))                       # Power imbalance post contingency [MW]

    model.Total_inertia = Var(model.t, bounds=(0,1e6))                            # Total inertia of the system at time t

    model.A_ch = Var(model.i, model.n, model.t, domain=NonNegativeReals)          # Charge Auxiliary 
    model.A_dc = Var(model.i, model.n, model.t, domain=NonNegativeReals)          # Discharge Auxiliary variables

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # OBJETIVE FUNCTION
    #-------------------------------------------------------------------------------------------------------------------------------------------

    def Cost_rule(model):
        return sum(sum((model.costSU[i,t])+(model.FG_min[i]*model.status[i,t])+(sum(model.P_seg[i,t,k]*model.C_seg[i,k] for k in model.k)) + (model.C_Reserve[i]*model.Reserve[i,t]) for i in model.i) for t in model.t) + sum(sum(model.C_D_Ba[t]*(model.Pot_Ba_ch[n,t]) for n in model.n) for t in model.t) + sum(sum((model.C_DR[b,t]*model.Pot_DR[b,t]) for b in model.b) for t in model.t)
    model.Objetivo = Objective(rule = Cost_rule, sense=minimize)

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # CONSTRAINS
    #-------------------------------------------------------------------------------------------------------------------------------------------

    ## Power generation SM limits constraints

    def P_seg_base_max_rule1(model,i,t,k):
        return model.P_seg[i,t,k]>=0
    model.P_seg_lim1 = Constraint(model.i, model.t, model.k, rule=P_seg_base_max_rule1)

    def P_seg_base_max_rule2(model,i,t,k):
        return model.P_seg[i,t,k]<=model.P_seg_max[i,k]*model.status[i,t];
    model.P_seg_lim2 = Constraint(model.i, model.t, model.k, rule=P_seg_base_max_rule2)

    def P_sum_rule(model,i,t):
        return model.P[i,t] == model.status[i,t]*model.PG_min[i] + sum(model.P_seg[i,t,k] for k in model.k)
    model.P_sum = Constraint(model.i, model.t, rule=P_sum_rule)

    def P_lim_max_rule(model,i,t):
        return model.P[i,t] + model.Reserve[i,t] <= model.PG_max[i]*model.status[i,t]
    model.P_max_lim = Constraint(model.i, model.t, rule=P_lim_max_rule)

    def P_lim_min_rule(model,i,t):
        return model.P[i,t] >= model.PG_min[i]*model.status[i,t]
    model.P_min_lim = Constraint(model.i, model.t, rule=P_lim_min_rule)

    def bin_cons1_rule(model,i,t):
        if t == model.t.first():
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]
        else:
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)

    def bin_cons2_rule(model,i,t):
        return model.SU[i,t] + model.SD[i,t] <= 1
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)

    def CostSUfn_init(model,i,t):
        return model.costSU[i,t] == df_SM_Unit.loc[i,'CSU']*model.SU[i,t]
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)

    def ramp_up_fn_rule(model,i,t):
        if t > 1:
            return model.P[i,t] - model.P[i,t-1] <= df_SM_Unit.loc[i,'Ramp_Up']
        else:
            return Constraint.Skip
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)

    def ramp_dw_fn_rule(model,i,t):
        if t > 1:
            return model.P[i,t] - model.P[i,t-1] >= -df_SM_Unit.loc[i,'Ramp_Down']
        else:
            return Constraint.Skip
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)

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

    ## Renewable constraints

    ## Power generation WIND limits constraints

    def P_lim_max_rule_w(model,w,t):
        return model.Pw[w,t] <= model.Pw_max[w,t]
    model.P_max_lim_w = Constraint(model.w, model.t, rule=P_lim_max_rule_w)


    ## Power generation SOLAR limits constraints

    def P_lim_max_rule_s(model,s,t):
        return model.Ppv[s,t] <= model.Ppv_max[s,t]
    model.P_max_lim_s = Constraint(model.s, model.t, rule=P_lim_max_rule_s)


    ## Demand Response constraints

    def Demand_response_rule(model,b,t):
        return model.Pot_DR[b,t] <= model.Pot_DR_Max[b,t]
    model.Demand_response = Constraint(model.b, model.t, rule=Demand_response_rule)


    ## ESS constraints

    def power_c_max_rule(model,n,t):
        return model.Pot_Ba_ch[n,t] <= model.C_Pot[n]*model.u_ch[n,t]
    model.power_c_max = Constraint(model.n, model.t, rule=power_c_max_rule)

    def power_c_max_2_rule(model,n,t):
        return model.Pot_Ba_ch[n,t] >= 0
    model.power_c_max_2 = Constraint(model.n, model.t, rule=power_c_max_2_rule)

    def power_d_max_rule(model,n,t):
        return model.Pot_Ba_dc[n,t] <= model.C_Pot[n]*model.u_dc[n,t]
    model.power_d_max = Constraint(model.n, model.t, rule=power_d_max_rule)

    def power_d_max_2_rule(model,n,t):
        return model.Pot_Ba_dc[n,t] >= 0
    model.power_d_max_2 = Constraint(model.n, model.t, rule=power_d_max_2_rule)

    def sim_rule(model,n,t):
        return model.u_ch[n,t] + model.u_dc[n,t] <= 1
    model.sim = Constraint(model.n, model.t, rule=sim_rule)

    # relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,n,t):
        if t == 1:
            return model.e_b[n,t] == (model.E_max[n]*model.SOC_ini[n]) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])
        else:
            return model.e_b[n,t] == (model.e_b[n,t-1]*(1-model.s_dc[n])) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])
    model.energy = Constraint(model.n, model.t, rule=energy_rule)

    # Energy limits

    def energy_limit_rule(model,n,t):
        return model.e_b[n,t] <= model.E_max[n]
    model.energy_limit = Constraint(model.n, model.t, rule=energy_limit_rule)

    def energy_limit_min_rule(model,n,t):
        return model.e_b[n,t] >= model.E_max[n]*model.SOC_min[n]
    model.energy_limit_min = Constraint(model.n, model.t, rule=energy_limit_min_rule)

    def energy_final_value_rule(model,n,t):
        return model.e_b[n,model.t.first()] == model.e_b[n,model.t.last()]
    model.energy_final_value = Constraint(model.n, model.t, rule=energy_final_value_rule)


    ## Power balance constrains

    def line_flow_rule(model, t, l):
        return model.pf[l,t] == model.MVA_base*(1/df_branch.loc[l,'X'])*sum(model.theta[b,t]*df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)
    model.line_flow = Constraint(model.t, model.l, rule=line_flow_rule)

    def line_min_rule(model, t, l):
        return model.pf[l,t] >= - df_branch.loc[l,'Flowlimit']
    model.line_min = Constraint(model.t, model.l, rule=line_min_rule)

    def line_max_rule(model, t, l):
        return model.pf[l,t] <= df_branch.loc[l,'Flowlimit']
    model.line_max = Constraint(model.t, model.l, rule=line_max_rule)

    def Power_bus_rule(model, t, b):
        return model.P_bus[b,t] == sum(model.pf[l,t]*df_line_map.loc[l,b] for l in model.l if df_line_map.loc[l,b] != 0)
    model.Power_bus = Constraint(model.t, model.b, rule=Power_bus_rule)

    def PowerG_bus_rule(model, t, b):
        return model.PG_bus[b,t] == sum(model.P[i,t] for i in model.i if df_SM_map.loc[i,b])
    model.PowerG_bus = Constraint(model.t, model.b, rule=PowerG_bus_rule)

    def PowerW_bus_rule(model, t, b):
        return model.PW_bus[b,t] == sum(model.Pw[w,t] for w in model.w if df_SM_map.loc[w,b])
    model.PowerW_bus = Constraint(model.t, model.b, rule=PowerW_bus_rule)

    def PowerPv_bus_rule(model, t, b):
        return model.PPV_bus[b,t] == sum(model.Ppv[s,t] for s in model.s if df_SM_map.loc[s,b])
    model.PowerPv_bus = Constraint(model.t, model.b, rule=PowerPv_bus_rule)

    def PowerDC_bus_rule(model, t, b):
        return model.PDC_bus[b,t] == sum(model.Pot_Ba_dc[n,t] for n in model.n if df_ESS_map.loc[n,b])
    model.PowerDC_bus = Constraint(model.t, model.b, rule=PowerDC_bus_rule)

    def PowerCH_bus_rule(model, t, b):
        return model.PCH_bus[b,t] == sum(model.Pot_Ba_ch[n,t] for n in model.n if df_ESS_map.loc[n,b])
    model.PowerCH_bus = Constraint(model.t, model.b, rule=PowerCH_bus_rule)

    def power_balance_rule(model,t):
        return sum(model.P[i,t] for i in model.i) + sum(model.Pw[w,t] for w in model.w) + sum(model.Ppv[s,t] for s in model.s) + sum(model.Pot_Ba_dc[n,t] for n in model.n) == sum(model.P_load[b,t]- model.Pot_DR[b,t] for b in model.b) + sum(model.Pot_Ba_ch[n,t] for n in model.n)
    model.power_balance = Constraint(model.t, rule=power_balance_rule)

    def power_balance_rule2(model, t, b):
        return model.PG_bus[b,t] + model.PW_bus[b,t] + model.PPV_bus[b,t] + model.PDC_bus[b,t] - model.PCH_bus[b,t] - model.P_load[b,t] - model.Pot_DR[b,t] == model.P_bus[b,t]
    model.power_balance2 = Constraint(model.t, model.b, rule=power_balance_rule2)


    ## Primary Reserve limits

    def Primary_reserve_rule1(model,i,t):
        return model.Reserve[i,t]<= model.Reserve_Max[i]
    model.Primary_reserve1 = Constraint(model.i, model.t, rule = Primary_reserve_rule1)

    def Primary_reserve_rule2(model,i,t):
        return  model.Reserve[i,t]>=0
    model.Primary_reserve2 = Constraint(model.i, model.t, rule = Primary_reserve_rule2)

    def Primary_reserve_rule3(model,t):
        return sum(model.Reserve[i,t] for i in model.i) >= model.PC_Power_imbalance[t]
    model.Primary_reserve3 = Constraint(model.t, rule = Primary_reserve_rule3)


    ## Post Contingency Frequency Dynamics

    # System power imbalance

    def Power_imbalance_rule1(model,t):
        return model.PC_Power_imbalance[t] == model.Power_imbalance[t] - sum(model.C_Pot[n] - model.Pot_Ba_dc[n,t] + model.Pot_Ba_ch[n,t] for n in model.n)
    model.Power_Imbalance_1 = Constraint(model.t, rule = Power_imbalance_rule1)

    # Energy of ESS in contingency

    def contingency_energy_rule1(model,n,t):
        return (model.e_b[n,t]*(1-model.s_dc[n])) - model.Delta_energy[n] >= model.E_max[n]*model.SOC_min[n]
    model.contingency_energy_1 = Constraint(model.n, model.t, rule=contingency_energy_rule1)

    def contingency_energy_rule2(model,n,t):
        return (model.e_b[n,t]*(1-model.s_dc[n])) - model.Delta_energy[n] <= model.E_max[n]
    model.contingency_energy_2 = Constraint(model.n, model.t, rule=contingency_energy_rule2)

    # # Total inertia value

    def Total_inertia_rule(model,t):
        return model.Total_inertia[t] == sum(model.Inertia[i]*model.PG_max[i]*model.status[i,t] for i in model.i)/model.Nominal_Frequency
    model.Total_Inertia_value = Constraint(model.t, rule = Total_inertia_rule)

    # # Frequency desviation limit

    def Frequency_desviation_rule(model,t):
        return model.Total_inertia[t] >= model.PC_Power_imbalance[t]/(2*model.Max_Frequency_desviation)
    Frequency_limits1 = Constraint(model.t, rule = Frequency_desviation_rule)

    def Frequency_desviation_rule2(model,i,t):
        return (model.Reserve[i,t]*model.Power_imbalance[t]) - sum(model.Reserve[i,t]*model.C_Pot[n] for n in model.n) + sum(model.A_dc[i,n,t] for n in model.n) - sum(model.A_ch[i,n,t] for n in model.n) <= 2*model.Ramp[i]*model.Total_inertia[t]*(model.Nominal_Frequency - model.Minimum_Frequency - model.DB_Frequency)
    model.Frequency_limits2 = Constraint(model.i, model.t, rule = Frequency_desviation_rule2)

    ## Reformulation - Linearization Technique (RLT)

    def Reformulation1_rule(model,i,n,t):
        return model.A_ch[i,n,t] - (model.Reserve_Max[i]*model.Pot_Ba_ch[n,t]) - (model.C_Pot[n]*model.Reserve[i,t]) + (model.C_Pot[n]*model.Reserve_Max[i]) >= 0
    model.Reformulation1 = Constraint(model.i, model.n, model.t, rule = Reformulation1_rule)

    def Reformulation2_rule(model,i,n,t):
        return - model.A_ch[i,n,t] + (model.C_Pot[n]*model.Reserve[i,t])>= 0
    model.Reformulation2 = Constraint(model.i, model.n, model.t, rule = Reformulation2_rule)

    def Reformulation3_rule(model,i,n,t):
        return - model.A_ch[i,n,t] + (model.Reserve_Max[i]*model.Pot_Ba_ch[n,t])>= 0
    model.Reformulation3 = Constraint(model.i, model.n, model.t, rule = Reformulation3_rule)

    def Reformulation4_rule(model,i,n,t):
        return model.A_dc[i,n,t] - (model.Reserve_Max[i]*model.Pot_Ba_dc[n,t]) - (model.C_Pot[n]*model.Reserve[i,t]) + (model.C_Pot[n]*model.Reserve_Max[i]) >= 0
    model.Reformulation4 = Constraint(model.i, model.n, model.t, rule = Reformulation4_rule)

    def Reformulation5_rule(model,i,n,t):
        return - model.A_dc[i,n,t] + (model.C_Pot[n]*model.Reserve[i,t])>= 0
    model.Reformulation5 = Constraint(model.i, model.n, model.t, rule = Reformulation5_rule)

    def Reformulation6_rule(model,i,n,t):
        return - model.A_dc[i,n,t] + (model.Reserve_Max[i]*model.Pot_Ba_dc[n,t])>= 0
    model.Reformulation6 = Constraint(model.i, model.n, model.t, rule = Reformulation6_rule)

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # SOLVER CONFIGURATION
    #-------------------------------------------------------------------------------------------------------------------------------------------

    def pyomo_postprocess(options=None, instance=None, results=None):
        model.Objetivo.display()


    if opt_option == 1:
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

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # DATA
    #-------------------------------------------------------------------------------------------------------------------------------------------

    ############################################ DATA ####################################################################################

    P = {}
    Reserve = {}
    Pw = {}
    Ppv = {}
    P_Ba_dc = {}
    P_Ba_ch = {}
    P_Ba = {}
    Energy_Ba = {}
    P_load = {}
    RoCoF_Ini = []
    RoCoF_Ba = []
    RoCoF_lim = []
    Delta_f_Ini = []
    Delta_f_Ba = []
    Delta_f_Max = []
    f_nadir_Ini = []
    f_nadir_Ba = []
    f_min = []
    Time = []
    Reserve_obj = []
    Reserve_BESS = {}
    Generacion = {}

    C_nadir = sum(model.Ramp[i]for i in model.i)

    Data1 = []
    Data2 = []
    Data3 = []
    Data4 = []
    Data5 = []
    Data6 = []
    Data7 = []

    for t in model.t:
        RoCoF_Ini.append((model.Power_imbalance[t])/(2*model.Total_inertia[t].value))
        RoCoF_Ba.append((model.PC_Power_imbalance[t].value)/(2*model.Total_inertia[t].value))
        RoCoF_lim.append(model.Max_Frequency_desviation.value)
        f_nadir_Ini.append(model.Nominal_Frequency.value - model.DB_Frequency.value - (((model.Power_imbalance[t])**2)/(2*C_nadir*model.Total_inertia[t].value)))
        f_nadir_Ba.append(model.Nominal_Frequency.value - model.DB_Frequency.value - (((model.PC_Power_imbalance[t].value)**2)/(2*C_nadir*model.Total_inertia[t].value)))
        f_min.append(model.Minimum_Frequency.value)
        Delta_f_Ini.append(model.Nominal_Frequency.value - f_nadir_Ini[t-1])
        Delta_f_Ba.append(model.Nominal_Frequency.value - f_nadir_Ba[t-1])
        Delta_f_Max.append(model.Nominal_Frequency.value - model.Minimum_Frequency.value)
        Reserve_obj.append(model.Power_imbalance[t])
        Time.append(t)

        P_H = 0
        P_Gas = 0
        P_Coal = 0
        P_Oil = 0
        P_Wind = 0
        P_Solar = 0
        P_BESS = 0


        for i in model.i:
            if df_SM_Unit.loc[i,'Fuel_Type'] == 'HYDRO':
                P_H = model.P[i,t].value + P_H
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'GAS':
                P_Gas = model.P[i,t].value + P_Gas
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'COAL':
                P_Coal = model.P[i,t].value + P_Coal
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'ACPM' or df_SM_Unit.loc[i,'Fuel_Type'] == 'DIESEL' or df_SM_Unit.loc[i,'Fuel_Type'] == 'COMBUSTOLEO':
                P_Oil = model.P[i,t].value + P_Oil

        Data1.append(P_H)
        Data2.append(P_Gas)
        Data3.append(P_Coal)
        Data4.append(P_Oil)

        for w in model.w:
            P_Wind = df_Renewable.loc[t,w] + P_Wind

        Data5.append(P_Wind)

        for s in model.s:
            P_Solar = df_Renewable.loc[t,s] + P_Solar

        Data6.append(P_Solar)

        for n in model.n:
            P_BESS = model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value + P_BESS

        Data7.append(P_BESS)

    Generacion['Hydro'] = Data1
    Generacion['Natural gas'] = Data2
    Generacion['Coal'] = Data3
    Generacion['Oil'] = Data4
    Generacion['Wind'] = Data5
    Generacion['Solar'] = Data6
    Generacion['BESS'] = Data7


    for i in model.i:
        Data1 = []
        Data2 = []
        for t in model.t:
            Data1.append(model.P[i,t].value)
            Data2.append(model.Reserve[i,t].value)
        P[i] = Data1
        Reserve[i] = Data2

    for w in model.w:
        Data1 = []
        for t in model.t:
            Data1.append(df_Renewable.loc[t,w])
        Pw[w] = Data1

    for s in model.s:
        Data1 = []
        for t in model.t:
            Data1.append(df_Renewable.loc[t,s])
        Ppv[s] = Data1

    for n in model.n:
        Data1 = []
        Data2 = []
        Data3 = []
        Data4 = []
        Data5 = []
        for t in model.t:
            Data1.append(model.Pot_Ba_dc[n,t].value)
            Data2.append(model.e_b[n,t].value)
            Data3.append(model.Pot_Ba_ch[n,t].value)
            Data4.append(model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value)
            Data5.append(model.C_Pot[n] - model.Pot_Ba_dc[n,t].value + model.Pot_Ba_ch[n,t].value)
        P_Ba_dc[n] = Data1
        Energy_Ba[n] = Data2
        P_Ba_ch[n] = Data3
        P_Ba[n] = Data4
        Reserve_BESS[n] = Data5


    for b in model.b:
        Data1 = []
        for t in model.t:
            Data1.append(model.P_load[b,t])
        P_load[b] = Data1

    ############################################ PLOTS ###################################################################################

    ## Plot configuration:

    title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'14'}

    P_acumulado = []
    P_acumulado_labels = []

    # Plot Power Dispatch

    fig=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    for n in model.n:
         if sum(P_Ba[n])!= 0:
            P_acumulado.append(P_Ba[n])
            ax.plot(Time,P_Ba[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))
            P_acumulado_labels.append(n)
    for i in model.i:
        if sum(P[i])>0:
            P_acumulado.append(P[i])
            ax.plot(Time,P[i], linestyle = '-', lw = 2.5 ,label = '{}'.format(i))
            P_acumulado_labels.append(i)
    for w in model.w:
        if sum(Pw[w])>0:
            P_acumulado.append(Pw[w])
            ax.plot(Time,Pw[w], linestyle = '-', lw = 2.5 ,label = '{}'.format(w))
            P_acumulado_labels.append(w)
    for s in model.s:
        if sum(Ppv[s])>0:
            P_acumulado.append(Ppv[s])
            ax.plot(Time,Ppv[s], linestyle = '-', lw = 2.5 ,label = '{}'.format(s))
            P_acumulado_labels.append(s)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Power Dispatch', **title_font)

    fig1=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.stackplot(Time,P_acumulado, labels = P_acumulado_labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Power Dispatch', **title_font)

    fig2=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.stackplot(Time,Generacion['Hydro'],Generacion['Natural gas'],Generacion['Coal'],Generacion['Oil'],Generacion['Wind'],Generacion['Solar'],Generacion['BESS'], labels = ['Hydro','Natural gas','Coal','Oil','Wind','Solar','BESS'])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Power Dispatch', **title_font)

    # Plot Power Demand

    D_acumulado = []
    D_acumulado_labels = []

    plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    for b in model.b:
        D_acumulado.append(P_load[b])
        ax.plot(Time,P_load[b], linestyle = '-', lw = 2.5 ,label = '{}'.format(b))
        D_acumulado_labels.append(b)
    for n in model.n:
        D_acumulado.append(P_Ba_ch[n])
        ax.plot(Time,P_Ba_ch[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))
        D_acumulado_labels.append(n)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Power Demand', **title_font)


    fig3=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.stackplot(Time,D_acumulado, labels = D_acumulado_labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Power Demand', **title_font)

    # Plot Reserve

    R_acumulado = []
    R_acumulado_labels = []

    fig3=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    for i in model.i:
        if sum(Reserve[i])>0:
            R_acumulado.append(Reserve[i])
            ax.plot(Time,Reserve[i], linestyle = '-', lw = 2.5 ,label = '{}'.format(i))
            R_acumulado_labels.append(i)
    for n in model.n:
        if sum(Reserve_BESS[n])>0:
            R_acumulado.append(Reserve_BESS[n])
            ax.plot(Time,Reserve_BESS[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))
            R_acumulado_labels.append(n)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Primary Power Reserve', **title_font)

    fig4=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.stackplot(Time,R_acumulado, labels = R_acumulado_labels)
    ax.plot(Time,Reserve_obj, linestyle = '--', color='r', lw = 2.5 ,label = 'Power imbalance capacity')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MW]', **axis_font)
    plt.title('Primary Power Reserve', **title_font)

    # Plot Energy in BESS

    fig5=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    for n in model.n:
        ax.plot(Time,Energy_Ba[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[MWh]', **axis_font)
    plt.title('Energy in ESS', **title_font)

    # Plot RoCoF

    x = np.arange(len(Time))
    fig6=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.bar((x+1) - 0.35/2,RoCoF_Ini,0.35,label = 'Without BESS')
    ax.bar((x+1) + 0.35/2,RoCoF_Ba,0.35,label = 'With BESS')
    ax.plot(Time,RoCoF_lim, linestyle = '--', color='r', lw = 2.5 ,label = 'RoCoF limit')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)
    ax.set_xticklabels(Time)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[Hz/s]', **axis_font)
    plt.title('RoCoF', **title_font)

    # # Plot fnadir

    x = np.arange(len(Time))
    fig7=plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    ax.bar((x+1) - 0.35/2,Delta_f_Ini,0.35,label = 'Without BESS')
    ax.bar((x+1) + 0.35/2,Delta_f_Ba,0.35,label = 'With BESS')
    ax.plot(Time,Delta_f_Max, linestyle = '--', color='r', lw = 2.5 ,label = 'Delta f limit')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)
    ax.set_xticklabels(Time)
    plt.xticks(Time,**axis_font)
    plt.yticks(**axis_font)
    plt.grid(True)
    plt.xlabel('Hour', **axis_font)
    plt.ylabel('[Hz]', **axis_font)
    plt.title('Delta of frequency', **title_font)

    Figuras=[fig,fig1,fig2,fig3,fig4,fig5,fig6,fig7]

    return Figuras
