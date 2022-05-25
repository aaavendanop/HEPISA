 # -*- coding: utf-8 -*-
"""
@author: Andrés Felipe Peñaranda Bayona
"""

# In[Librerias]:

import streamlit as st
from pyomo.environ import *
import pandas as pd
import math
from datetime import *
from funciones.Import_XM_data import*
from funciones.save_files import*
from funciones.nuevo_bess import bat_param
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup

# In[System model]

def System_import(system_data,source,date1,date2,SAE,TRM):
    
    System_grid_data = {} 
    
    
    if source == 'XM (Colombia)':
        
        df_Sys_data = pd.read_excel(system_data, sheet_name='System_data', header=0, index_col=0)
        df_Bus = pd.read_excel(system_data, sheet_name='Bus', header=0, index_col=0)
        
        # GEN
        
        df_Gen_Unit = pd.read_excel(system_data, sheet_name='Gen_Unit', header=0, index_col=0)
        df_Gen_energy_cost = Precio_Oferta(date1,date2)
        df_Gen_energy_cost = df_Gen_energy_cost.div(TRM)
        df_Gen_PAP_cost = Plant_offer_data('PAP',date1,date2)
        df_Gen_PAP_cost = df_Gen_PAP_cost.div(TRM)
        df_Gen_P_max = Dispo_Comercial(date1,date2)
        df_Gen_P_min = Plant_offer_data('MO',date1,date2)
        df_Gen_R_max = Plant_offer_data('AGCP',date1,date2)
        
        G_idx = df_Gen_Unit.index.tolist()
        B_idx = df_Bus.index.tolist()    
        Gen_map = pd.DataFrame()
        Gen_map['Gen'] = G_idx
        
        for b in B_idx:
            M = []
            for g in G_idx:
                if df_Gen_Unit.loc[g,'Bus_index'] == b:
                    M.append(1)
                else:
                    M.append(0)
            Gen_map[b] = M
        
        Gen_map.set_index('Gen', inplace=True)
        Gen_map.index.name = None
        
        # ESS
        
        df_ESS = pd.read_excel(system_data, sheet_name='ESS_Unit', header=0, index_col=0)
        df_ESS = df_ESS.append(SAE)
        
        def cycle_life_loss_curve(a,b,DoD,eol):
            return (1-eol)/(a*(DoD**(-b)))
        
        DoD_values = np.linspace(0,1,10)
        
        Degradation_curve_x = DoD_values.tolist()
        Deg_x = []
        Deg_y = []
        
        for ess in df_ESS.index:
            Deg_x.append(DoD_values.tolist())
            a = df_ESS.loc[ess,'a']
            b = df_ESS.loc[ess,'b']
            eol = df_ESS.loc[ess,'eol']
            Deg_y.append(cycle_life_loss_curve(a,b,DoD_values,eol).tolist())
        
        df_ESS['Deg_x'] = Deg_x
        df_ESS['Deg_y'] = Deg_y
        
        ES_idx = df_ESS.index.tolist()
        B_idx = df_Bus.index.tolist()
        ESS_map = pd.DataFrame()
        ESS_map['ESS'] = ES_idx
        
        for b in B_idx:
            M = []
            for e in ES_idx:
                if df_ESS.loc[e,'Bus_index'] == b:
                    M.append(1)
                else:
                    M.append(0)
            ESS_map[b] = M
        
        ESS_map.set_index('ESS', inplace=True)
        ESS_map.index.name = None
        
        df_Branch = pd.read_excel(system_data, sheet_name='Branch', header=0, index_col=0)
        
        L_idx = df_Branch.index.tolist()
        B_idx = df_Bus.index.tolist()
        Branch_map = pd.DataFrame()
        Branch_map['Branch'] = L_idx
        
        for b in B_idx:
            M = []
            for l in L_idx:
                if df_Branch.loc[l,'from'] == b:
                    M.append(1)
                elif df_Branch.loc[l,'to'] == b:
                    M.append(-1)
                else:
                    M.append(0)
            Branch_map[b] = M
        
        Branch_map.set_index('Branch', inplace=True)
        Branch_map.index.name = None
        
        df_Load = pd.read_excel(system_data, sheet_name='Load', header=0, index_col=0)
        Demanda_sys = Demanda(date1,date2)
        Loads = df_Load.index.tolist()
        Total_Loads = sum(df_Load['P_MW'])
        Hours = Demanda_sys.columns.tolist()
        df_Sys_Load = pd.DataFrame()
        for h in Hours:
            V = []
            for l in Loads:
                V.append(df_Load.loc[l,'P_MW']*Demanda_sys.loc['Total',h]/Total_Loads)
            df_Sys_Load[h]=V
        df_Sys_Load = df_Sys_Load.set_index(pd.Index(Loads))
        Bus_loads = []
        for l in Loads:
            Bus_loads.append(df_Load.loc[l,'Bus_index'])
        df_Sys_Load = df_Sys_Load.set_index(pd.Index(Bus_loads))
        
        df_Sys_Holgura = Holgura_data(date1,date2)
        
        System_grid_data['Sys_data'] = df_Sys_data
        System_grid_data['Bus'] = df_Bus
        System_grid_data['Gen_Unit'] = df_Gen_Unit
        System_grid_data['Gen_energy_cost'] = df_Gen_energy_cost
        System_grid_data['Gen_PAP_cost'] = df_Gen_PAP_cost
        System_grid_data['Gen_P_max'] = df_Gen_P_max
        System_grid_data['Gen_P_min'] = df_Gen_P_min
        System_grid_data['Gen_R_max'] = df_Gen_R_max
        System_grid_data['Gen_map'] = Gen_map
        System_grid_data['ESS_Unit'] = df_ESS
        System_grid_data['ESS_map'] = ESS_map
        System_grid_data['Branch'] = df_Branch
        System_grid_data['Branch_map'] = Branch_map
        System_grid_data['Load'] = df_Load
        System_grid_data['Sys_load'] = df_Sys_Load
        System_grid_data['Sys_Reserve'] = df_Sys_Holgura
        
    elif source == 'Otro':
    
        df_Sys_data = pd.read_excel(system_data, sheet_name='System_data', header=0, index_col=0)
        df_Bus = pd.read_excel(system_data, sheet_name='Bus', header=0, index_col=0)
        
        # GEN
        
        df_Gen_Unit = pd.read_excel(system_data, sheet_name='Gen_Unit', header=0, index_col=0)
        df_Gen_energy_cost = pd.read_excel(system_data, sheet_name='Gen_Energy_cost', header=0, index_col=0)
        df_Gen_PAP_cost = pd.read_excel(system_data, sheet_name='Gen_PAP_cost', header=0, index_col=0)
        df_Gen_P_max = pd.read_excel(system_data, sheet_name='Gen_P_max', header=0, index_col=0)
        df_Gen_P_min = pd.read_excel(system_data, sheet_name='Gen_P_min', header=0, index_col=0)
        df_Gen_R_max = pd.read_excel(system_data, sheet_name='Gen_Reserve_max', header=0, index_col=0)
        
        G_idx = df_Gen_Unit.index.tolist()
        B_idx = df_Bus.index.tolist()    
        Gen_map = pd.DataFrame()
        Gen_map['Gen'] = G_idx
        
        for b in B_idx:
            M = []
            for g in G_idx:
                if df_Gen_Unit.loc[g,'Bus_index'] == b:
                    M.append(1)
                else:
                    M.append(0)
            Gen_map[b] = M
        
        Gen_map.set_index('Gen', inplace=True)
        Gen_map.index.name = None
        
        # ESS
        
        df_ESS = pd.read_excel(system_data, sheet_name='ESS_Unit', header=0, index_col=0)
        df_ESS = df_ESS.append(SAE)
        
        def cycle_life_loss_curve(a,b,DoD,eol):
            return (1-eol)/(a*(DoD**(-b)))
        
        DoD_values = np.linspace(0,1,10)
        
        Degradation_curve_x = DoD_values.tolist()
        Deg_x = []
        Deg_y = []
        
        for ess in df_ESS.index:
            Deg_x.append(DoD_values.tolist())
            a = df_ESS.loc[ess,'a']
            b = df_ESS.loc[ess,'b']
            eol = df_ESS.loc[ess,'eol']
            Deg_y.append(cycle_life_loss_curve(a,b,DoD_values,eol).tolist())
        
        df_ESS['Deg_x'] = Deg_x
        df_ESS['Deg_y'] = Deg_y
        
        ES_idx = df_ESS.index.tolist()
        B_idx = df_Bus.index.tolist()
        ESS_map = pd.DataFrame()
        ESS_map['ESS'] = ES_idx
        
        for b in B_idx:
            M = []
            for e in ES_idx:
                if df_ESS.loc[e,'Bus_index'] == b:
                    M.append(1)
                else:
                    M.append(0)
            ESS_map[b] = M
        
        ESS_map.set_index('ESS', inplace=True)
        ESS_map.index.name = None
        
        df_Branch = pd.read_excel(system_data, sheet_name='Branch', header=0, index_col=0)
        
        L_idx = df_Branch.index.tolist()
        B_idx = df_Bus.index.tolist()
        Branch_map = pd.DataFrame()
        Branch_map['Branch'] = L_idx
        
        for b in B_idx:
            M = []
            for l in L_idx:
                if df_Branch.loc[l,'from'] == b:
                    M.append(1)
                elif df_Branch.loc[l,'to'] == b:
                    M.append(-1)
                else:
                    M.append(0)
            Branch_map[b] = M
        
        Branch_map.set_index('Branch', inplace=True)
        Branch_map.index.name = None
        
        df_Load = pd.read_excel(system_data, sheet_name='Load', header=0, index_col=0)
        df_Sys_Load = pd.read_excel(system_data, sheet_name='Sys_Load', header=0, index_col=0)
        
        Loads = df_Sys_Load.index.tolist()
        Bus_loads = []
        for l in Loads:
            Bus_loads.append(df_Load.loc[l,'Bus_index'])
        df_Sys_Load = df_Sys_Load.set_index(pd.Index(Bus_loads)) 
        
        df_Sys_Holgura = pd.read_excel(system_data, sheet_name='Sys_Holgura', header=0, index_col=0)
         
        System_grid_data['Sys_data'] = df_Sys_data
        System_grid_data['Bus'] = df_Bus
        System_grid_data['Gen_Unit'] = df_Gen_Unit
        System_grid_data['Gen_energy_cost'] = df_Gen_energy_cost
        System_grid_data['Gen_PAP_cost'] = df_Gen_PAP_cost
        System_grid_data['Gen_P_max'] = df_Gen_P_max
        System_grid_data['Gen_P_min'] = df_Gen_P_min
        System_grid_data['Gen_R_max'] = df_Gen_R_max
        System_grid_data['Gen_map'] = Gen_map
        System_grid_data['ESS_Unit'] = df_ESS
        System_grid_data['ESS_map'] = ESS_map
        System_grid_data['Branch'] = df_Branch
        System_grid_data['Branch_map'] = Branch_map
        System_grid_data['Load'] = df_Load
        System_grid_data['Sys_load'] = df_Sys_Load
        System_grid_data['Sys_Reserve'] = df_Sys_Holgura
    
    return System_grid_data

# In[Modelo]:
    
def MEM_general_model(System_data,Modelo,Input_data,restr_list,Current_direction,solver):
    
    AGC_dispatch = Modelo[0]
    Ideal_dispatch = Modelo[1]
    Dispatch_AGC_Ideal = Modelo[2]
    Program_dispatch = Modelo[3]
    Co_optimization = Modelo[-1]   
    
    Number_overestimating_planes = 3
    Number_bits = 10
    
    BESS_T_capacity = 2*sum(System_data['ESS_Unit'].loc[:,'C_Potencia'])
    EFR_range = np.linspace(0,BESS_T_capacity,Number_overestimating_planes)
    EFR_range_l = EFR_range.tolist()
        
    Max_lose = max(System_data['Sys_Reserve'].loc['Total',:])
    Min_lose = min(System_data['Sys_Reserve'].loc['Total',:])
    Power_imbalance_range = np.linspace(Min_lose,Max_lose,Number_overestimating_planes)
    Power_imbalance_range = np.flip(Power_imbalance_range)
    Power_imbalance_range_l = Power_imbalance_range.tolist()
         
    #-------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    D_segments = []
    
    if len(System_data['ESS_Unit'].index) > 0:
        for i in range(len(System_data['ESS_Unit']['Deg_x'][0])-1):
            D_segments.append('S{}'.format(i))
        
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # MODEL TYPE
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    ## Simulation type
    model = ConcreteModel()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # SETS
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # Time sets
    model.t = Set(initialize=System_data['Sys_load'].columns.tolist(), ordered=True)           # Horizon simulation time
    model.tt = SetOf(model.t)
    
    # Generation units sets
    model.i = Set(initialize=System_data['Gen_Unit'].index.tolist(), ordered=True)             # Generation units set
    
    # Energy storage system sets
    model.n = Set(initialize=System_data['ESS_Unit'].index.tolist(), ordered=True)             # BESS units set
    
    # System sets
    if Program_dispatch == 1 or ('Flujos de potencia DC' in restr_list) :
        model.b = Set(initialize=System_data['Bus'].index.tolist(), ordered=True)              # Buses of system set
        model.l = Set(initialize=System_data['Branch'].index.tolist(),ordered=True)            # lines of system set
        
    # Auxiliar sets
    model.d = Set(initialize=D_segments, ordered=True)                                        # Segments of degradation curve
    model.p = RangeSet(0,Number_overestimating_planes-1)                                      # Overestimating planes
    model.m = RangeSet(0,Number_bits - 1)                                                     # Number of bits for binary expansion 
         
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETERS
    #------------------------------------------------------------------------------------------------------------------------------------------- 
    
    # CONVENTIONAL GEN MACHINES
    
    def Power_max(model,i,t):
        return System_data['Gen_P_max'].loc[i,t]
    model.P_max = Param(model.i,model.t, initialize = Power_max)                 # Maximum power capacity of each generator [MW]
    
    def Power_min_op(model,i,t):
        return System_data['Gen_P_min'].loc[i,t]
    model.P_min_op = Param(model.i,model.t, initialize = Power_min_op)           # Minimun power capacity of each generator [MW]
     
    def Power_cost(model,i,t):
        return System_data['Gen_energy_cost'].loc[i,t]
    model.P_cost = Param(model.i,model.t, initialize = Power_cost)               # Cost of energy [$COP/MWh]
    
    def PAP_cost(model,i,t):
        return System_data['Gen_PAP_cost'].loc[i,t]
    model.PAP_cost = Param(model.i,model.t, initialize = PAP_cost)               # Cost of PAP [$COP]
    
    def Status_ini(model,i):
        return System_data['Gen_Unit'].loc[i,'Initial_operation_status']
    model.onoff_t0 = Param(model.i, initialize = Status_ini)                     # Inital Status of SM (ON/OFF)
    
    def RampUP_i(model,i):
        return System_data['Gen_Unit'].loc[i,'Ramp_up']
    model.RampUP = Param(model.i, initialize = RampUP_i)                         # Ramp UP [MW/h]
    
    def RampDOWN_i(model,i):
        return System_data['Gen_Unit'].loc[i,'Ramp_down']
    model.RampDOWN = Param(model.i, initialize = RampDOWN_i)                     # Ramp DOWN [MW/h]
    
    def P_initial_i(model,i):
        return System_data['Gen_Unit'].loc[i,'Initial_power']
    model.P_i_i = Param(model.i, initialize = P_initial_i)                       # Initial power dispatch of the SM i [MW]
    
    def MinON_i(model,i):
        return System_data['Gen_Unit'].loc[i,'Min_on']
    model.MinON_t = Param(model.i, initialize = MinON_i)                         # Minimun ON time of SM [h]
    
    def MinOFF_i(model,i):
        return System_data['Gen_Unit'].loc[i,'Min_off']
    model.MinOFF_t = Param(model.i, initialize = MinOFF_i)                       # Minimun OFF time of SM [h]
    
    def RSF_UP_max(model,i,t):
        return System_data['Gen_R_max'].loc[i,t]
    model.RSF_UP_i_max = Param(model.i,model.t, initialize = RSF_UP_max)         # Maximum Secondary UP reserve capacity of each generator [MW]
    
    def RSF_DW_max(model,i,t):
        return System_data['Gen_R_max'].loc[i,t]
    model.RSF_DW_i_max = Param(model.i,model.t, initialize = RSF_DW_max)         # Maximum Secondary DOWN reserve capacity of each generator [MW]
    
    def RPF_U_i(model,i):
        return System_data['Gen_Unit'].loc[i,'PFR_service']
    model.U_RPF_i = Param(model.i, initialize = RPF_U_i)                         # Minimun ON time of SM [h]
    
    def RSF_U_i(model,i):
        return System_data['Gen_Unit'].loc[i,'SFR_service']
    model.U_RSF_i = Param(model.i, initialize = RSF_U_i)                         # Minimun ON time of SM [h]
    
    def RSF_F_i(model,i,t):
        return Input_data['RSF_UP_i'].loc[i,t] 
    
    # BESS
    
    def C_P_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'C_Potencia']
    model.C_Pot = Param(model.n, initialize=C_P_Ba_rule)                          # Power Size [MW]

    def C_E_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'C_Energia']
    model.E_max = Param(model.n, initialize=C_E_Ba_rule)                          # Energy Size [MWh]

    def C_nch_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'n_ch_eff']
    model.n_ch = Param(model.n, initialize=C_nch_Ba_rule)                         # Charge efficency of ESS [p.u.]

    def C_ndc_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'n_dc_eff']
    model.n_dc = Param(model.n, initialize=C_ndc_Ba_rule)                         # Discharge efficency of ESS [p.u.]

    def C_sdc_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'Self_discharge']
    model.s_dc = Param(model.n, initialize=C_sdc_Ba_rule)                         # Self-Discharge efficency of ESS [p.u./h]

    def C_SOCmax_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'SOC_max']
    model.SOC_max = Param(model.n, initialize=C_SOCmax_Ba_rule)                   # Maximum State of Charge of BESS [p.u.]
    
    def C_SOCmin_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'SOC_min']
    model.SOC_min = Param(model.n, initialize=C_SOCmin_Ba_rule)                   # Minimum State of Charge of BESS [p.u.]

    def C_SOCini_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'SOC_ini']
    model.SOC_ini = Param(model.n, initialize=C_SOCini_Ba_rule)                   # Initial State of Charge of BESS [p.u.]
       
    def E_cost_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'E_cost']
    model.E_ESS_cost = Param(model.n, initialize=E_cost_Ba_rule)                 # Cost of service for BESS [COP/MWh]
    
    def EOL_Ba_rule(model,n):
        return System_data['ESS_Unit'].loc[n,'eol']
    model.eol = Param(model.n, initialize=EOL_Ba_rule)    
    
    def RSF_F_n(model,n,t):
        return Input_data['RSF_UP_n'].loc[n,t] 
    
    # DEGRADATION SEGMENT CURVE
    
    def DOD_seg_max(model,n,d):
        return System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)]
    model.DOD_seg_max = Param(model.n,model.d, initialize = DOD_seg_max)
    
    def DOD_seg_min(model,n,d):
        return System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)-1]
    model.DOD_seg_min = Param(model.n,model.d, initialize = DOD_seg_min)
    
    def Degracion_pendiente_seg(model,n,d):
        if model.d.ord(d) == 1:
            return 0
        else:
            return (System_data['ESS_Unit']['Deg_y'][n][model.d.ord(d)] - System_data['ESS_Unit']['Deg_y'][n][model.d.ord(d)-1])/(System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)] - System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)-1])
    model.Degra_m_seg = Param(model.n,model.d, initialize = Degracion_pendiente_seg)
    
    def Degracion_b_seg(model,n,d):
        if model.d.ord(d) == 1:
            return 0
        else:
            return System_data['ESS_Unit']['Deg_y'][n][model.d.ord(d)-1] + ((System_data['ESS_Unit']['Deg_y'][n][model.d.ord(d)] - System_data['ESS_Unit']['Deg_y'][n][model.d.ord(d)-1])/(System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)] - System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)-1]))*-System_data['ESS_Unit']['Deg_x'][n][model.d.ord(d)-1]
    model.Degra_b_seg = Param(model.n,model.d, initialize = Degracion_b_seg)
    
    # LOAD
    
    def T_load(model,t):
        return sum(System_data['Sys_load'].loc[:,t])
    model.Total_load = Param(model.t, initialize = T_load)                        # Total demand of energy [MWh]
    
    if Program_dispatch == 1 or ('Flujos de potencia DC' in restr_list):
        def Bus_load(model,b,t):
            try:
                return System_data['Sys_load'].loc[b,t]
            except:
                return 0
        model.B_load = Param(model.b, model.t, initialize = Bus_load)                 # Load of each bus in the test system [MW]
        
    
    # RESERVE 
    
    def T_Reserve(model,t):
        return System_data['Sys_Reserve'].loc['Total',t]
    model.Total_Reserve = Param(model.t, initialize = T_Reserve)                   # Total reserve of energy [MWh]
    
    
    # SYSTEM
    
    MVA_base = System_data['Sys_data'].loc['S_base','Data']                             # Power base of the system
    delta_RPF = 0.03                                                                    # Delta of primary energy reserve [p.u.]
    Nominal_Frequency = System_data['Sys_data'].loc['N_freq','Data']                    # Nominal frequency of the system [Hz]
    Max_Frequency_desviation = System_data['Sys_data'].loc['Max_D_freq','Data']         # Maximum frequency desviation [Hz/s]
    Delta_t_EFR = System_data['Sys_data'].loc['Delta_EFR','Data']                       # Delta time of EFR [h]
    Delta_t_RSF = System_data['Sys_data'].loc['Delta_RSF','Data']                       # Delta time of RSF [h]
    P_biggest = max(System_data['Gen_Unit'].loc[:,'P_nom'])                             # Biggest SM unit [MW]
    D_Load_Damping = System_data['Sys_data'].loc['D_demand_resp','Data']/100            # Load Damping support [%/Hz]
    Minimum_Frequency_qss = System_data['Sys_data'].loc['Min_freq_qss','Data']          # Minimum qss frequency limit [Hz]
    Minimum_Frequency = System_data['Sys_data'].loc['Min_freq','Data']                  # Minimum frequency limit [Hz]
    Delta_qss_max =  Nominal_Frequency - Minimum_Frequency_qss                          # Delta of max frequency mismatch qss [Hz]
    Delta_nadir_max = Nominal_Frequency - Minimum_Frequency                             # Delta of max frequency mismatch nadir [Hz]
    t_EFR = System_data['Sys_data'].loc['T_EFR','Data']                                 # Maximum time of EFR response [s]
    t_RPF = System_data['Sys_data'].loc['T_RPF','Data']                                 # Maximum time of RPF response [s]
    t_RSF = System_data['Sys_data'].loc['T_RSF','Data']                                 # Maximum time of RSF response [s]
    
    def Inertia_rule(model,i):
        return System_data['Gen_Unit'].loc[i,'Inertia']
    model.Inertia = Param(model.i, initialize = Inertia_rule)                    # Intertia value of each SM [s]
    
    def NPower_max(model,i):
        return System_data['Gen_Unit'].loc[i,'P_nom']
    model.PG_max = Param(model.i, initialize = NPower_max)                       # Nominal power capacity of each generator [MW]
    
    # RSF
    
    Minimum_Frequency_ss = System_data['Sys_data'].loc['Min_freq_ss','Data']          # Minimum qss frequency limit [Hz]
    Delta_ss_max =  Nominal_Frequency - Minimum_Frequency_ss                                  # Delta of max frequency mismatch ss [Hz]
    
    def Reserve_demand(model,t):
        return System_data['Sys_Reserve'].loc['Total',t]
    model.T_reserve = Param(model.t, initialize = Reserve_demand)                # Total demand of energy reserve [MWh]
    
    def Max_power_imbalance(model,t):
        return System_data['Sys_Reserve'].loc['Total',t]
    model.PC_Power_imbalance = Param(model.t, initialize = Max_power_imbalance)  # Max power imbalance of the system [MW]
    
    
    # Inner-Aproximation
        
    def PL_0_value(model,p):
        return Power_imbalance_range_l[p]
    model.PL_0 = Param(model.p, initialize = PL_0_value)
    
    def EFR_0_value(model,p):
        return EFR_range_l[p]
    model.EFR_0 = Param(model.p, initialize = EFR_0_value)
    
    def A_value(model,p):
        return (((2*Power_imbalance_range_l[p])-(2*EFR_range_l[p]))*t_RPF)/(4*Delta_nadir_max)
    model.A = Param(model.p, initialize = A_value)
    
    def B_value(model,p):
        return (((2*EFR_range_l[p])-(2*Power_imbalance_range_l[p]))*t_RPF)/(4*Delta_nadir_max)
    model.B = Param(model.p, initialize = B_value)
    
    def C_value(model,p):
        return (((Power_imbalance_range_l[p]-EFR_range_l[p])**2)*t_RPF)/(4*Delta_nadir_max)
    model.C = Param(model.p, initialize = C_value)
    
    

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # VARIABLES
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # CONVENTIONAL GEN MACHINES
    
    model.P = Var(model.i, model.t, domain=NonNegativeReals)                    # Power dispatch of unit i at time t [MW]
    model.status = Var(model.i, model.t, within=Binary)                         # Status of unit i at time t {Binary}
    model.SU = Var(model.i, model.t, within=Binary)                             # Start UP status {Binary}
    model.SD = Var(model.i, model.t, within=Binary)                             # Start DOWN status {Binary}
    
    model.RPF_UP_i = Var(model.i, model.t, domain= NonNegativeReals)            # Power reseve UP for primary reserve [MW]
    model.RPF_DW_i = Var(model.i, model.t, domain= NonNegativeReals)            # Power reseve DOWN for primary reserve [MW]
    model.T_RPF_UP_i = Var(model.t, domain= NonNegativeReals)                   # Total RPF reserve UP [MW]
    model.T_RPF_DW_i = Var(model.t, domain= NonNegativeReals)                   # Total RPF reserve DOWN [MW] 
    model.RSF_DW_i = Var(model.i, model.t, domain= NonNegativeReals)            # Power reseve DOWN for primary reserve [MW]
    model.T_RSF_UP_i = Var(model.t, domain= NonNegativeReals)                   # Total RPF reserve UP [MW]
    model.T_RSF_DW_i = Var(model.t, domain= NonNegativeReals)                   # Total RPF reserve DOWN [MW]
    
    if Dispatch_AGC_Ideal != 1:
        model.RSF_UP_i = Var(model.i, model.t, domain= NonNegativeReals)        # Power reseve UP for primary reserve [MW]
    else:
        model.RSF_UP_i = Param(model.i,model.t, initialize = RSF_F_i)           # Fixed Secondary reserve power [MW]
   
    
    # BATTERY ENERGY STORAGE SYSTEM
    
    model.Pot_Ba_ch = Var(model.n, model.t, domain=NonNegativeReals)            # Power in battery charge [MW]
    model.Pot_Ba_dc = Var(model.n, model.t, domain=NonNegativeReals)            # Power in battery discharge [MW]
    model.u_ch = Var(model.n, model.t, within=Binary)                           # Status of battery charge {Binary}
    model.u_dc = Var(model.n, model.t, within=Binary)                           # Status of battery discharge [Binary]
    model.e_b = Var(model.n, model.t, domain=NonNegativeReals)                  # Energy of battery [MWh]
    
    if 'Degradación SAE' in restr_list:
        model.E_cap = Var(model.n,model.t, domain=NonNegativeReals)                 # Energy capacity of battery [MWh]
    
    model.Deg_ba_rate = Var(model.n, model.t, domain = NonNegativeReals)        # Tasa de degradacion del ESS para el instante t
    model.DOD_seg = Var(model.n, model.t, model.d, domain=NonNegativeReals)     # DOD dispatch segment k of BESS at time t [p.u]
    model.DOD_seg_b = Var(model.n, model.t, model.d, within=Binary)             # Senal binaria segmento de la curva de degradacion 
            
    model.EFR_UP_n_ch = Var(model.n, model.t, domain=NonNegativeReals)          # Charge Power for UP EFR of battery n[MW] 
    model.EFR_UP_n_dc = Var(model.n, model.t, domain=NonNegativeReals)          # Discharge Power for UP EFR of battery n[MW] 
    model.EFR_UP_n = Var(model.n, model.t, domain=NonNegativeReals)             # Total Power for UP EFR of battery n [MW] 
    model.T_EFR_UP_n = Var(model.t, domain=NonNegativeReals)                    # Total Power for UP EFR [MW]
    model.EFR_DW_n_ch = Var(model.n, model.t, domain=NonNegativeReals)          # Charge Power for DOWN EFR of battery n[MW] 
    model.EFR_DW_n_dc = Var(model.n, model.t, domain=NonNegativeReals)          # Discharge Power for DOWN EFR of battery n[MW] 
    model.EFR_DW_n = Var(model.n, model.t, domain=NonNegativeReals)             # Total Power for DOWN EFR of battery n [MW] 
    model.T_EFR_DW_n = Var(model.t, domain=NonNegativeReals)                    # Total Power for DOWN EFR [MW] 
    
    model.RSF_UP_n_ch = Var(model.n, model.t, domain=NonNegativeReals)          # Charge Power for UP RSF of battery n[MW] 
    model.RSF_UP_n_dc = Var(model.n, model.t, domain=NonNegativeReals)          # Discharge Power for UP RSF of battery n[MW] 
    model.T_RSF_UP_n = Var(model.t, domain=NonNegativeReals)                    # Total Power for UP RSF [MW]
    model.RSF_DW_n_ch = Var(model.n, model.t, domain=NonNegativeReals)          # Charge Power for DOWN RSF of battery n[MW] 
    model.RSF_DW_n_dc = Var(model.n, model.t, domain=NonNegativeReals)          # Discharge Power for DOWN RSF of battery n[MW] 
    model.RSF_DW_n = Var(model.n, model.t, domain=NonNegativeReals)             # Total Power for DOWN RSF of battery n [MW] 
    model.T_RSF_DW_n = Var(model.t, domain=NonNegativeReals)                    # Tolta Power for DOWN RSF [MW]
    
    if Dispatch_AGC_Ideal != 1:
        model.RSF_UP_n = Var(model.n, model.t, domain=NonNegativeReals)             # Total Power for UP RSF of battery n [MW] 
    else:
        model.RSF_UP_n = Param(model.n,model.t, initialize = RSF_F_n)
    
    
    # SYSTEM VARIABLES
    if Program_dispatch == 1 or ('Flujos de potencia DC' in restr_list):
        
        model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))                # Voltage angle [rad]
        model.pf = Var(model.l, model.t)                                              # Power flow through the line l [MW]
        model.P_bus = Var(model.b, model.t)                                           # Power bus balance [MW]
        model.PG_bus = Var(model.b, model.t)                                          # Power of SM in bus b [MW]
        model.PDC_bus = Var(model.b, model.t)                                         # Charge power of ESS in bus b [MW]
        model.PCH_bus = Var(model.b, model.t)                                         # Discharge power of ESS in bus b [MW]
    
        for t in model.t:
            model.theta[System_data['Sys_data'].loc['Slack_bus','Data'],t].fix(0)     # Slack angle [rad]    
    
    model.Total_inertia = Var(model.t, bounds=(0,1e6))                            # Total inertia of the system at time t [MWs]
    
    # AUXILIAR VARIABLES
    if 'Fnadir sistema' in restr_list:
        
        model.Z = Var(model.m, model.t, within = Binary, initialize=0)                # Binary variable for the expansion
        model.M = Var(model.m, model.t, domain=NonNegativeReals, initialize=0)
        model.K = Var(model.m, model.t, domain=NonNegativeReals, initialize=0)
            
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # OBJETIVE FUNCTION
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    def Objective_function(model):
        return sum(sum((model.P[i,t]*model.P_cost[i,t]) for i in model.i) for t in model.t) + sum(sum((model.SU[i,t] + model.SD[i,t])*model.PAP_cost[i,t] for i in model.i) for t in model.t) + sum(sum(model.RSF_UP_i[i,t]*model.P_cost[i,t] for i in model.i) for t in model.t) + sum(sum(((model.Deg_ba_rate[n,t]*model.E_ESS_cost[n])/(1- model.eol[n])) for n in model.n) for t in model.t)
    model.Objetivo = Objective(rule = Objective_function, sense=minimize)
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # CONSTRAINS
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # CONVENTIONAL GEN MACHINES
    
    # Power limits
      
    def P_lim_max_rule_i(model,i,t):
        return model.P[i,t] + model.RPF_UP_i[i,t] + model.RSF_UP_i[i,t] <= model.P_max[i,t]*model.status[i,t]
    model.P_max_lim_i = Constraint(model.i, model.t, rule=P_lim_max_rule_i)
   
    def P_lim_min_rule_i(model,i,t):
        return model.P[i,t] - model.RPF_DW_i[i,t] - model.RSF_DW_i[i,t] >= model.P_min_op[i,t]*model.status[i,t]
    model.P_min_lim_i = Constraint(model.i, model.t, rule=P_lim_min_rule_i)
    
    ## Start UP/DOWN signals

    def bin_cons1_rule(model,i,t):
        if t == model.t.first():
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]
        else:
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)
    
    def bin_cons2_rule(model,i,t):
        return model.SU[i,t] + model.SD[i,t] <= 1
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)
    
    ## Ramp limits
      
    def ramp_up_fn_rule(model,i,t):
        if t != model.t.first():
            return model.P[i,t] - model.P[i,t-1] <= model.RampUP[i]
        else:
            return model.P[i,t] - model.P_i_i[i] <= model.RampUP[i]
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)
    
    def ramp_dw_fn_rule(model,i,t):
        if t != model.t.first():
            return model.P[i,t-1] - model.P[i,t] <= model.RampDOWN[i]
        else:
            return  model.P_i_i[i] - model.P[i,t] <= model.RampDOWN[i]
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)
    
    # Minimum ON/OFF time
    
    def min_up_dn_time_2_rule(model,i,t):
        return sum(model.SU[i,tt] for tt in model.tt if tt >= t - model.MinON_t[i] - 1 and tt <= t) <= model.status[i,t]
    model.min_up_dn_time_2 = Constraint(model.i, model.t, rule=min_up_dn_time_2_rule)

    def min_up_dn_time_3_rule(model,i,t):
        return sum(model.SD[i,tt] for tt in model.tt if tt >= t - model.MinOFF_t[i] - 1 and tt <= t) <= 1-model.status[i,t]
    model.min_up_dn_time_3 = Constraint(model.i, model.t, rule=min_up_dn_time_3_rule)
    
    # RPF
    if 'RPF Generadores' in restr_list:
        
        def Primary_reserve_rule(model,i,t):
            return model.RPF_UP_i[i,t] >= model.P[i,t]*model.U_RPF_i[i]*delta_RPF
        model.Primary_reserve = Constraint(model.i, model.t, rule = Primary_reserve_rule) 
    
    def PRF_UP_total_reserve(model,t):
        return model.T_RPF_UP_i[t] == sum(model.RPF_UP_i[i,t] for i in model.i) 
    model.Total_RPF_UP_reserve_i = Constraint(model.t, rule = PRF_UP_total_reserve)
    
    def PRF_DW_total_reserve(model,t):
        return model.T_RPF_DW_i[t] == sum(model.RPF_DW_i[i,t] for i in model.i) 
    model.Total_RPF_DW_reserve_i = Constraint(model.t, rule = PRF_DW_total_reserve)
    
    def PRF_simetrical_offer(model,i,t):
        return model.RPF_UP_i[i,t] == model.RPF_DW_i[i,t]
    model.PRF_simetrical_offer_i = Constraint(model.i, model.t, rule = PRF_simetrical_offer)
    
    # RSF
    
    if Dispatch_AGC_Ideal != 1:
        
        def Secondary_UP_reserve_rule(model,i,t):
            return model.RSF_UP_i[i,t] <= model.RSF_UP_i_max[i,t]*model.status[i,t]
        model.Secondary_UP_reserve = Constraint(model.i, model.t, rule = Secondary_UP_reserve_rule)
        
        def Secondary_DW_reserve_rule(model,i,t):
            return model.RSF_DW_i[i,t] <= model.RSF_DW_i_max[i,t]*model.status[i,t]
        model.Secondary_DW_reserve = Constraint(model.i, model.t, rule = Secondary_DW_reserve_rule)
    
    def SRF_UP_total_reserve(model,t):
        return model.T_RSF_UP_i[t] == sum(model.RSF_UP_i[i,t] for i in model.i)
    model.Total_RSF_UP_reserve_i = Constraint(model.t, rule = SRF_UP_total_reserve)
    
    def SRF_DW_total_reserve(model,t):
        return model.T_RSF_DW_i[t] == sum(model.RSF_DW_i[i,t] for i in model.i)
    model.Total_RSF_DW_reserve_i = Constraint(model.t, rule = SRF_DW_total_reserve)
    
    def SRF_simetrical_offer(model,i,t):
        return model.RSF_UP_i[i,t] == model.RSF_DW_i[i,t]
    model.SRF_simetrical_offer_i = Constraint(model.i, model.t, rule = SRF_simetrical_offer)
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # BATTERY ENERGY STORAGE SYSTEM
    
    # Power constraints
    
    def power_ch_max_rule(model,n,t):
        return model.Pot_Ba_ch[n,t] <= model.C_Pot[n]*model.u_ch[n,t]
    model.power_c_max = Constraint(model.n, model.t, rule=power_ch_max_rule)

    def power_dc_max_rule(model,n,t):
        return model.Pot_Ba_dc[n,t] <= model.C_Pot[n]*model.u_dc[n,t]
    model.power_d_max = Constraint(model.n, model.t, rule=power_dc_max_rule)

    def sim_rule(model,n,t):
        return model.u_ch[n,t] + model.u_dc[n,t] <= 1
    model.sim = Constraint(model.n, model.t, rule=sim_rule)
    
    # # Power frequency response constraints
    
    def EFR_UP_dc_cap(model,n,t):
        return  model.EFR_UP_n_dc[n,t] <= model.C_Pot[n] - model.Pot_Ba_dc[n,t]
    model.EFR_UP_n_dc_cap = Constraint(model.n, model.t, rule=EFR_UP_dc_cap)
    
    def EFR_UP_ch_cap(model,n,t):
        return  model.EFR_UP_n_ch[n,t] <= model.Pot_Ba_ch[n,t]
    model.EFR_UP_n_ch_cap = Constraint(model.n, model.t, rule=EFR_UP_ch_cap)
    
    def EFR_UP_t_cap(model,n,t):
        return  model.EFR_UP_n[n,t] == model.EFR_UP_n_ch[n,t] + model.EFR_UP_n_dc[n,t]
    model.EFR_UP_t_cap = Constraint(model.n, model.t, rule=EFR_UP_t_cap)
    
    def EFR_DOWN_dc_cap(model,n,t):
        return  model.EFR_DW_n_dc[n,t] <= model.Pot_Ba_dc[n,t]
    model.EFR_DW_n_dc_cap = Constraint(model.n, model.t, rule=EFR_DOWN_dc_cap)
    
    def EFR_DOWN_ch_cap(model,n,t):
        return  model.EFR_DW_n_ch[n,t] <= model.C_Pot[n] - model.Pot_Ba_ch[n,t]
    model.EFR_DW_n_ch_cap = Constraint(model.n, model.t, rule=EFR_DOWN_ch_cap)
    
    def EFR_DOWN_t_cap(model,n,t):
        return  model.EFR_DW_n[n,t] == model.EFR_DW_n_ch[n,t] + model.EFR_DW_n_dc[n,t]
    model.EFR_DW_t_cap = Constraint(model.n, model.t, rule=EFR_DOWN_t_cap)
    
    def RSF_UP_dc_cap(model,n,t):
        return  model.RSF_UP_n_dc[n,t] <= model.C_Pot[n] - model.Pot_Ba_dc[n,t]
    model.RSF_UP_n_dc_cap = Constraint(model.n, model.t, rule=RSF_UP_dc_cap)
    
    def RSF_UP_ch_cap(model,n,t):
        return  model.RSF_UP_n_ch[n,t] <= model.Pot_Ba_ch[n,t]
    model.RSF_UP_n_ch_cap = Constraint(model.n, model.t, rule=RSF_UP_ch_cap)
    
    def RSF_UP_t_cap(model,n,t):
        return  model.RSF_UP_n[n,t] == model.RSF_UP_n_ch[n,t] + model.RSF_UP_n_dc[n,t]
    model.RSF_UP_t_cap = Constraint(model.n, model.t, rule=RSF_UP_t_cap)
    
    def RSF_DOWN_dc_cap(model,n,t):
        return  model.RSF_DW_n_dc[n,t] <= model.Pot_Ba_dc[n,t]
    model.RSF_DW_n_dc_cap = Constraint(model.n, model.t, rule=RSF_DOWN_dc_cap)
    
    def RSF_DOWN_ch_cap(model,n,t):
        return  model.RSF_DW_n_ch[n,t] <= model.C_Pot[n] - model.Pot_Ba_ch[n,t]
    model.RSF_DW_n_ch_cap = Constraint(model.n, model.t, rule=RSF_DOWN_ch_cap)
    
    def RSF_DOWN_t_cap(model,n,t):
        return  model.RSF_DW_n[n,t] == model.RSF_DW_n_ch[n,t] + model.RSF_DW_n_dc[n,t]
    model.RSF_DW_t_cap = Constraint(model.n, model.t, rule=RSF_DOWN_t_cap)
    

    # # Relation betwent energy status and power charging and discharging Constraint

    def energy_rule(model,n,t):
        if t == model.t.first():
            return model.e_b[n,t] == (model.E_max[n]*model.SOC_ini[n]) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])
        else:
            return model.e_b[n,t] == (model.e_b[n,t-1]*(1-model.s_dc[n])) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])
    model.energy = Constraint(model.n, model.t, rule=energy_rule)

    # # Energy limits
    if 'Degradación SAE' in restr_list:
        
        def energy_limit_rule(model,n,t):
            return model.e_b[n,t] + (model.EFR_DW_n_ch[n,t]*Delta_t_EFR) + (model.RSF_DW_n_ch[n,t]*Delta_t_RSF) <= model.E_cap[n,t]*model.SOC_max[n]
        model.energy_limit = Constraint(model.n, model.t, rule=energy_limit_rule)
    else:
        def energy_limit_rule(model,n,t):
            return model.e_b[n,t] + (model.EFR_DW_n_ch[n,t]*Delta_t_EFR) + (model.RSF_DW_n_ch[n,t]*Delta_t_RSF) <= model.E_max[n]*model.SOC_max[n]
        model.energy_limit = Constraint(model.n, model.t, rule=energy_limit_rule)

    def energy_limit_min_rule(model,n,t):
        return model.e_b[n,t] - (model.EFR_UP_n_dc[n,t]*Delta_t_EFR) - (model.RSF_UP_n_dc[n,t]*Delta_t_RSF) >= model.E_max[n]*model.SOC_min[n]
    model.energy_limit_min = Constraint(model.n, model.t, rule=energy_limit_min_rule)
    
    # ESS Degradation
    
    if 'Degradación SAE' in restr_list:
        
        def Degradacion_rule(model,n,t):
            if t == model.t.first():
                return model.E_cap[n,t] == model.E_max[n]
            else:           
                return model.E_cap[n,t] == model.E_cap[n,t-1] - (model.Deg_ba_rate[n,t]*model.E_max[n])
        model.ESS_Deg = Constraint(model.n,model.t, rule = Degradacion_rule)

    
    # Segmentos de la curva de degradacion BESS
    if 'Degradación SAE' in restr_list:
        
        def DOD_seg_base_min_rule1(model,n,t,d):
            return model.DOD_seg[n,t,d]>=model.DOD_seg_min[n,d]*model.DOD_seg_b[n,t,d]
        model.DOD_seg_lim1 = Constraint(model.n,model.t, model.d, rule=DOD_seg_base_min_rule1)
    
        def DOD_seg_base_max_rule2(model,n,t,d):
            return model.DOD_seg[n,t,d]<=model.DOD_seg_max[n,d]*model.DOD_seg_b[n,t,d]
        model.DOD_seg_lim2 = Constraint(model.n,model.t, model.d, rule=DOD_seg_base_max_rule2)
        
        def DOD_seg_bin_sum_rule2(model,n,t):
            return sum(model.DOD_seg_b[n,t,d] for d in model.d) <= 1
        model.DOD_seg_bin_lim2 = Constraint(model.n,model.t, rule=DOD_seg_bin_sum_rule2)
    
        def DOD_sum_rule(model,n,t):
            return (model.Pot_Ba_dc[n,t])/(model.E_max[n]*model.n_dc[n]) == sum(model.DOD_seg[n,t,d] for d in model.d)
        model.DOD_sum = Constraint(model.n,model.t, rule=DOD_sum_rule)
        
        def Deg_rate_rule(model,n,t):
            return model.Deg_ba_rate[n,t] == sum(((model.DOD_seg[n,t,d]*model.Degra_m_seg[n,d]) + (model.DOD_seg_b[n,t,d]*model.Degra_b_seg[n,d])) for d in model.d)
        model.Deg_rate = Constraint(model.n,model.t, rule=Deg_rate_rule)
        
    # EFR
    
    def EFR_total_UP_reserve(model,t):
        return model.T_EFR_UP_n[t] == sum(model.EFR_UP_n[n,t] for n in model.n)
    model.EFR_Total_UP_reserve_n = Constraint(model.t, rule = EFR_total_UP_reserve)
    
    def EFR_total_DW_reserve(model,t):
        return model.T_EFR_DW_n[t] == sum(model.EFR_DW_n[n,t] for n in model.n)
    model.EFR_Total_DW_reserve_n = Constraint(model.t, rule = EFR_total_DW_reserve)
    
    def EFR_simetrical_offer(model,n,t):
        return model.EFR_UP_n[n,t] == model.EFR_DW_n[n,t]
    model.EFR_simetrical_offer_n = Constraint(model.n, model.t, rule = EFR_simetrical_offer)
    
    # RSF
    
    def RSF_total_reserve(model,t):
        return model.T_RSF_UP_n[t] == sum(model.RSF_UP_n[n,t] for n in model.n)
    model.RSF_total_reserve_n = Constraint(model.t, rule = RSF_total_reserve)
    
    def RSF_total_DW_reserven(model,t):
        return model.T_RSF_DW_n[t] == sum(model.RSF_DW_n[n,t] for n in model.n)
    model.RSF_Total_DW_reserve_n = Constraint(model.t, rule = RSF_total_DW_reserven)
    
    def RSF_simetrical_offern(model,n,t):
        return model.RSF_UP_n[n,t] == model.RSF_DW_n[n,t]
    model.RSF_simetrical_offer_n = Constraint(model.n, model.t, rule = RSF_simetrical_offern)
       
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # DC POWER FLOW
    if Program_dispatch == 1 or ('Flujos de potencia DC' in restr_list):
        
        def line_flow_rule(model, t, l):
            return model.pf[l,t] == MVA_base*(1/System_data['Branch'].loc[l,"X"])*sum(model.theta[b,t]*System_data['Branch_map'].loc[l,b] for b in model.b if System_data['Branch_map'].loc[l,b] != 0)
        model.line_flow = Constraint(model.t, model.l, rule=line_flow_rule)
    
        def line_min_rule(model, t, l):
            return model.pf[l,t] >= - System_data['Branch'].loc[l,"Flowlimit"]
        model.line_min = Constraint(model.t, model.l, rule=line_min_rule)
    
        def line_max_rule(model, t, l):
            return model.pf[l,t] <= System_data['Branch'].loc[l,"Flowlimit"]
        model.line_max = Constraint(model.t, model.l, rule=line_max_rule)
    
        def Power_bus_rule(model, t, b):
            return model.P_bus[b,t] == sum(model.pf[l,t]*System_data['Branch_map'].loc[l,b] for l in model.l if System_data['Branch_map'].loc[l,b] != 0)
        model.Power_bus = Constraint(model.t, model.b, rule=Power_bus_rule)
        
        def PowerG_bus_rule(model, t, b):
            return model.PG_bus[b,t] == sum(model.P[i,t] for i in model.i if System_data['Gen_map'].loc[i,b])
        model.PowerG_bus = Constraint(model.t, model.b, rule=PowerG_bus_rule)
    
        def PowerDC_bus_rule(model, t, b):
            return model.PDC_bus[b,t] == sum(model.Pot_Ba_dc[n,t] for n in model.n if System_data['ESS_map'].loc[n,b])
        model.PowerDC_bus = Constraint(model.t, model.b, rule=PowerDC_bus_rule)
    
        def PowerCH_bus_rule(model, t, b):
            return model.PCH_bus[b,t] == sum(model.Pot_Ba_ch[n,t] for n in model.n if System_data['ESS_map'].loc[n,b])
        model.PowerCH_bus = Constraint(model.t, model.b, rule=PowerCH_bus_rule)
        
        def power_balance_rule2(model, t, b):
            return model.PG_bus[b,t] + model.PDC_bus[b,t] - model.PCH_bus[b,t] - model.B_load[b,t] == model.P_bus[b,t]
        model.power_balance2 = Constraint(model.t, model.b, rule=power_balance_rule2)
        
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # POST CONTINGENCY FREQUENCY DYNAMICS
    
    # Total inertia value

    def Total_inertia_rule(model,t):
        return model.Total_inertia[t] == (sum(model.Inertia[i]*model.PG_max[i]*model.status[i,t] for i in model.i) - P_biggest*6)  
    model.Total_Inertia_value = Constraint(model.t, rule = Total_inertia_rule)
    
    # # Frequency desviation limits

    # RoCoF limit rule    
    if 'RoCoF Sistema' in restr_list:
        def Frequency_desviation_rule(model,t):
            return model.Total_inertia[t] >= ((model.PC_Power_imbalance[t]*Nominal_Frequency)/(2*Max_Frequency_desviation))
        model.Frequency_limits1 = Constraint(model.t, rule = Frequency_desviation_rule)
        
    # Frequency qss limit rule
    if 'Fqss sistema' in restr_list:
        def Frequency_desviation_rule2(model,t):
            return ((model.PC_Power_imbalance[t] - model.T_RPF_UP_i[t] - model.T_EFR_UP_n[t])/((D_Load_Damping)*model.Total_load[t])) <= Delta_qss_max
        model.Frequency_limits2 = Constraint(model.t, rule = Frequency_desviation_rule2)
    
    # Frequency nadir limit rule
    if 'Fnadir sistema' in restr_list:     
        def binary_expansion(model,t):
            return model.T_RPF_UP_i[t] == sum(model.Z[m,t]*(2**m) for m in model.m)
        model.binary_exp = Constraint(model.t, rule = binary_expansion)
        
        def Frequency_desviation_rule3(model,p,t):
            return ((sum(model.M[m,t]*(2**m) for m in model.m)/Nominal_Frequency) - ((sum(model.K[m,t]*(2**m) for m in model.m)*t_EFR)/(4*Delta_nadir_max))) >= (model.A[p]*(model.PC_Power_imbalance[t]- model.PL_0[p])) + (model.B[p]*(model.T_EFR_UP_n[t] - model.EFR_0[p])) + (model.C[p]) - (((model.PC_Power_imbalance[t] - model.T_EFR_UP_n[t])*t_RPF*D_Load_Damping*model.Total_load[t])/4) 
        model.Frequency_limits3 = Constraint(model.p,model.t, rule = Frequency_desviation_rule3)
        
        def Aux_M_1(model,m,t):
            return (model.Total_inertia[t] - model.M[m,t]) >= 0
        model.Aux_M1 = Constraint(model.m, model.t, rule = Aux_M_1)
        
        def Aux_M_2(model,m,t):
            return (model.Total_inertia[t] - model.M[m,t]) <= 1000000*(1 - model.Z[m,t])
        model.Aux_M2 = Constraint(model.m, model.t, rule = Aux_M_2)
        
        def Aux_M_3(model,m,t):
            return model.M[m,t] >= 0
        model.Aux_M3 = Constraint(model.m, model.t, rule = Aux_M_3)
        
        def Aux_M_4(model,m,t):
            return model.M[m,t] <= 1000000*model.Z[m,t]
        model.Aux_M4 = Constraint(model.m, model.t, rule = Aux_M_4)
        
        def Aux_K_1(model,m,t):
            return (model.T_EFR_UP_n[t] - model.K[m,t]) >= 0
        model.Aux_K1 = Constraint(model.m, model.t, rule = Aux_K_1)
        
        def Aux_K_2(model,m,t):
            return (model.T_EFR_UP_n[t] - model.K[m,t]) <= 1000000*(1 - model.Z[m,t])
        model.Aux_K2 = Constraint(model.m, model.t, rule = Aux_K_2)
        
        def Aux_K_3(model,m,t):
            return model.K[m,t] >= 0
        model.Aux_K3 = Constraint(model.m, model.t, rule = Aux_K_3)
        
        def Aux_K_4(model,m,t):
            return model.K[m,t] <= 1000000*model.Z[m,t]
        model.Aux_K4 = Constraint(model.m, model.t, rule = Aux_K_4)
      
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    ## BALANCE OF ENERGY
    
    if Ideal_dispatch == 1 or Co_optimization == 1:    
        def power_balance_rule(model,t):
            return sum(model.P[i,t] for i in model.i) + sum(model.Pot_Ba_dc[n,t] for n in model.n) == model.Total_load[t] + sum(model.Pot_Ba_ch[n,t] for n in model.n)
        model.power_balance = Constraint(model.t, rule=power_balance_rule)
        
    if AGC_dispatch == 1 or Co_optimization == 1:  
        def reserve_balance_rule(model,t):
            return model.T_RSF_UP_n[t] + model.T_RSF_UP_i[t] == model.Total_Reserve[t]
        model.Reserve_balance = Constraint(model.t, rule=reserve_balance_rule)
        
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # SOLVER CONFIGURATION
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    if solver == 'CPLEX':
        from pyomo.opt import SolverFactory
        import pyomo.environ
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
        results.write()
        print("\nDisplaying Solution\n" + '-'*60)
    else:
        from pyomo.opt import SolverFactory
        import pyomo.environ
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        results.write()
        print("\nDisplaying Solution\n" + '-'*60)
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    # OUTPUT DATA
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    # VARIABLES
    
    Output_data = {}
    
    for v in model.component_objects(Objective):
        Output_data[str(v)] = pyomo_df(v)
    
    for v in model.component_objects(Var):
        sets = v.dim()
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
    
    # OTROS
    
    Opt_variables = pd.DataFrame()
    
    ## OPERATION COST
    
    SAE_Columns = ['ESS_index','Value']
    Objetive_function = []
    Operation_cost = []
    Operation_cost_SD = []
    Operation_cost_AGC = []
    SAE_incomes = {}
    SAE_op_cost = {}
    MPO_max = []
    CERE = 18.32
    
    for t in model.t:
        Precio = []
        for i in model.i:
            if model.P[i,t].value - model.P_min_op[i,t] > 0:
                Precio.append(model.P_cost[i,t])
        if len(Precio) == 0:
            MPO_max.append(0)
        else:
            MPO_max.append(max(Precio)/1000)
    
    for n in model.n:
        SAE_incomes[n] = []
        SAE_op_cost[n] = []
    
    for t in model.t:
        # SAE_Columns.append(t)
        if Dispatch_AGC_Ideal != 1:
            Objetive_function.append(sum((model.P[i,t].value*model.P_cost[i,t]) for i in model.i) + sum((model.SU[i,t].value + model.SD[i,t].value)*model.PAP_cost[i,t] for i in model.i) + sum((model.RSF_UP_i[i,t].value*model.P_cost[i,t]) for i in model.i) + sum(((model.Deg_ba_rate[n,t].value*model.E_ESS_cost[n])/(1- model.eol[n])) for n in model.n))
            Operation_cost.append(sum((model.P[i,t].value*MPO_max[t]*1000) for i in model.i) + sum((2*model.RSF_UP_i[i,t].value*CERE) for i in model.i))
            Operation_cost_SD.append(sum((model.P[i,t].value*MPO_max[t]*1000) for i in model.i))
            Operation_cost_AGC.append(sum((2*model.RSF_UP_i[i,t].value*CERE) for i in model.i))
            for n in model.n:
                SAE_incomes[n].append(sum(((model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value)*MPO_max[t]*1000) for n in model.n) + sum((2*model.RSF_UP_n[n,t].value*CERE) for n in model.n))
                SAE_op_cost[n].append(sum(((model.Deg_ba_rate[n,t].value*model.E_ESS_cost[n])/(1- model.eol[n])) for n in model.n))
        else:
            Objetive_function.append(sum((model.P[i,t].value*model.P_cost[i,t]) for i in model.i) + sum((model.SU[i,t].value + model.SD[i,t].value)*model.PAP_cost[i,t] for i in model.i) + sum((model.RSF_UP_i[i,t]*model.P_cost[i,t]) for i in model.i) + sum(((model.Deg_ba_rate[n,t].value*model.E_ESS_cost[n])/(1- model.eol[n])) for n in model.n))
            Operation_cost.append(sum((model.P[i,t].value*MPO_max[t]*1000) for i in model.i) + sum((2*model.RSF_UP_i[i,t]*CERE) for i in model.i))
            Operation_cost_SD.append(sum((model.P[i,t].value*MPO_max[t]*1000) for i in model.i))
            Operation_cost_AGC.append(sum((2*model.RSF_UP_i[i,t]*CERE) for i in model.i))
            for n in model.n:
                SAE_incomes[n].append(sum(((model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value)*MPO_max[t]*1000) for n in model.n) + sum((2*model.RSF_UP_n[n,t]*CERE) for n in model.n))
                SAE_op_cost[n].append(sum(((model.Deg_ba_rate[n,t].value*model.E_ESS_cost[n])/(1- model.eol[n])) for n in model.n))
                
    Opt_variables['Obj_f'] = Objetive_function
    Opt_variables['MPO'] = MPO_max
    Opt_variables['OP_cost'] = Operation_cost
    Opt_variables['OP_cost_SD'] = Operation_cost_SD
    Opt_variables['OP_cost_AGC'] = Operation_cost_AGC
    
    SAE_INCOMES = pd.DataFrame([[key, SAE_incomes[key]] for key in SAE_incomes.keys()], columns=SAE_Columns)
    SAE_OP_C = pd.DataFrame([[key, SAE_op_cost[key]] for key in SAE_op_cost.keys()], columns=SAE_Columns)
    SAE_INCOMES = SAE_INCOMES.set_index('ESS_index')
    SAE_OP_C = SAE_OP_C.set_index('ESS_index')
    
    ## FREQUENCY CONSTRAINTS VALUES
    
    # ROCOF
    
    def RoCoF_function(t,tf):
        return (Nominal_Frequency/(2*model.Total_inertia[t].value))*((model.T_EFR_UP_n[t].value*tf/t_EFR)+(model.T_RPF_UP_i[t].value*tf/t_RPF)-(model.PC_Power_imbalance[t]))
    
    RoCoF_max = []
    RoCoF_05 = []
    RoCoF_limit = []
    
    for t in model.t:
        RoCoF_max.append(RoCoF_function(t,0))
        RoCoF_05.append(RoCoF_function(t,0.5))
        RoCoF_limit.append(-Max_Frequency_desviation)

    Opt_variables['RoCoF_max'] = RoCoF_max
    Opt_variables['RoCoF_05'] = RoCoF_05
       
    # F_NADIR
    
    def T_nadir_function(t):
        return ((model.PC_Power_imbalance[t] - model.T_EFR_UP_n[t].value - (D_Load_Damping*model.Total_load[t]*Delta_nadir_max))*t_RPF)/model.T_RPF_UP_i[t].value
    
    def F_nadir_function(t,t_nadir):
        D_p = D_Load_Damping*model.Total_load[t]*Nominal_Frequency
        F_nadir_a = ((math.exp(-D_p*t_nadir/(2*model.Total_inertia[t].value)))-1)*(model.PC_Power_imbalance[t] + ((2*model.Total_inertia[t].value*model.T_RPF_UP_i[t].value)/(D_p*t_RPF)))
        F_nadir_b = ((model.T_RPF_UP_i[t].value*t_nadir)/t_RPF)
        F_nadir_c = model.T_EFR_UP_n[t].value*(1+(((2*model.Total_inertia[t].value)/(D_p*t_EFR))*(math.exp(-((D_p*t_nadir)/(2*model.Total_inertia[t].value)))-math.exp(-((D_p*(t_nadir - t_EFR))/(2*model.Total_inertia[t].value))))))
        return Nominal_Frequency + (((F_nadir_a + F_nadir_b + F_nadir_c)*Nominal_Frequency)/D_p)
        
    T_nadir = []
    Delta_nadir = []
    D_nadir_max = []
    
    for t in model.t:
        T_nadir.append(T_nadir_function(t))
        Delta_nadir.append(F_nadir_function(t,T_nadir_function(t)))
        D_nadir_max.append(-Delta_nadir_max)
    
    Opt_variables['t_nadir'] = T_nadir
    Opt_variables['D_nadir'] = Delta_nadir
    
    # F_QSS
    
    def F_qss_function(t):
        if ((model.T_EFR_UP_n[t].value + model.T_RPF_UP_i[t].value - model.PC_Power_imbalance[t])/(D_Load_Damping*model.Total_load[t])) > 0:
            return Nominal_Frequency
        else:
            return Nominal_Frequency + ((model.T_EFR_UP_n[t].value + model.T_RPF_UP_i[t].value - model.PC_Power_imbalance[t])/(D_Load_Damping*model.Total_load[t]))
    
    Delta_qss = []
    
    for t in model.t:
        Delta_qss.append(F_qss_function(t))
    
    Opt_variables['F_qss'] = Delta_qss
    
    Output_data['Other'] = Opt_variables
    Output_data['SAE_incomes'] = SAE_INCOMES
    Output_data['SAE_op_cost'] = SAE_OP_C 
    
    if AGC_dispatch == 1:
        file_name = 'MEM_AGC_dispatch_results'
    if Dispatch_AGC_Ideal == 1:
        file_name = 'MEM_AGC_Ideal_dispatch_results'
    if Ideal_dispatch == 1:
        file_name = 'MEM_Ideal_dispatch_results'
    if Co_optimization == 1:
        file_name = 'MEM_Co_optimization_results'
    if Program_dispatch == 1:
        file_name = 'MEM_Program_dispatch_results'
            
    with pd.ExcelWriter('{}/Resultados/{}.xlsx'.format(Current_direction,file_name)) as writer:
        for idx in Output_data.keys():
            Output_data[idx].to_excel(writer, sheet_name= idx, index = True)
        writer.save()
        # writer.close()    
    

    return Output_data

def print_results(Output_data,System_data,modelo_MEM,restr_list):
    
    st.markdown("### Resultados globales:")
    
    st.write('#### Sistema:')  
    st.write('*Función objetivo:*  {} [USD]'.format(round(sum(Output_data['Other'].loc[:,'Obj_f']))))
    st.write('*Costos de operación:*  {} [USD]'.format(round(sum(Output_data['Other'].loc[:,'OP_cost']))))
    ESS_list = Output_data['SAE_incomes'].index.tolist()
    if len(ESS_list)> 0:
        for E in ESS_list:
            st.write('#### {}({}):'.format(System_data['ESS_Unit'].loc[E,'ESS_Name'],E))    
            st.write('*Ingresos totales SAE:*  {} [USD]'.format(round(sum(Output_data['SAE_incomes'].loc[E,'Value']))))
            st.write('*Costos de operación SAE:*  {} [USD]'.format(round(sum(Output_data['SAE_op_cost'].loc[E,'Value']))))
         
    st.markdown("### Figuras:")
    
    if modelo_MEM != 'Despacho AGC':
        fig_P_disp = go.Figure()
        for i in Output_data['P'].index.tolist():
            if sum(Output_data['P'].loc[i,:])>0:
                fig_P_disp.add_trace(go.Scatter(x=Output_data['P'].columns.tolist(), y=Output_data['P'].loc[i,:], name='{}'.format(System_data['Gen_Unit'].loc[i,'Gen_Name']),line_shape='hv'))
        for n in Output_data['SAE_incomes'].index.tolist():
            if sum(Output_data['Pot_Ba_dc'].loc[n,:])>0:
                fig_P_disp.add_trace(go.Scatter(x=Output_data['Pot_Ba_dc'].columns.tolist(), y=Output_data['Pot_Ba_dc'].loc[n,:], name='{}'.format(System_data['ESS_Unit'].loc[n,'ESS_Name']),line_shape='hv'))
        fig_P_disp.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_P_disp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_P_disp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_P_disp.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Despacho mercado energía',xaxis_title='Tiempo [h]',yaxis_title='Potencia [MW]')
        
        st.write(fig_P_disp)
    
    if modelo_MEM == 'Despacho programado' or modelo_MEM == 'Co-optimización':
        fig_RPF_disp = go.Figure()
        for i in Output_data['RPF_UP_i'].index.tolist():
            if sum(Output_data['RPF_UP_i'].loc[i,:])>0:
                fig_RPF_disp.add_trace(go.Scatter(x=Output_data['RPF_UP_i'].columns.tolist(), y=Output_data['RPF_UP_i'].loc[i,:], name='{}'.format(System_data['Gen_Unit'].loc[i,'Gen_Name']),line_shape='hv'))
        for n in Output_data['SAE_incomes'].index.tolist():
            if sum(Output_data['EFR_UP_n'].loc[n,:])>0:
                fig_RPF_disp.add_trace(go.Scatter(x=Output_data['EFR_UP_n'].columns.tolist(), y=Output_data['EFR_UP_n'].loc[n,:], name='{}'.format(System_data['ESS_Unit'].loc[n,'ESS_Name']),line_shape='hv'))
        #fig_RPF_disp.add_trace(go.Scatter(x=Output_data['T_reserve'].index.tolist(), y=Output_data['T_reserve'].tolist(), name='Reserva total',line=dict(dash='dot'),line_shape='hv'))
        fig_RPF_disp.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_RPF_disp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_RPF_disp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_RPF_disp.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Reserva primaria de frecuencia',xaxis_title='Tiempo [h]',yaxis_title='Potencia [MW]')
        st.write(fig_RPF_disp)
    
    if modelo_MEM != 'Despacho Ideal':
        fig_AGC_disp = go.Figure()
        for i in Output_data['RSF_UP_i'].index.tolist():
            if sum(Output_data['RSF_UP_i'].loc[i,:])>0:
                fig_AGC_disp.add_trace(go.Scatter(x=Output_data['RSF_UP_i'].columns.tolist(), y=Output_data['RSF_UP_i'].loc[i,:], name='{}'.format(System_data['Gen_Unit'].loc[i,'Gen_Name']),line_shape='hv'))
        for n in Output_data['SAE_incomes'].index.tolist():
            if sum(Output_data['RSF_UP_n'].loc[n,:])>0:
                fig_AGC_disp.add_trace(go.Scatter(x=Output_data['RSF_UP_n'].columns.tolist(), y=Output_data['RSF_UP_n'].loc[n,:], name='{}'.format(System_data['ESS_Unit'].loc[n,'ESS_Name']),line_shape='hv'))
        fig_AGC_disp.add_trace(go.Scatter(x=Output_data['T_reserve'].index.tolist(), y=Output_data['T_reserve'].tolist(), name='Reserva total',line=dict(dash='dot'),line_shape='hv'))
        fig_AGC_disp.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_AGC_disp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_AGC_disp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_AGC_disp.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Despacho mercado AGC',xaxis_title='Tiempo [h]',yaxis_title='Potencia [MW]')
        st.write(fig_AGC_disp)
    
    
    if 'pf' in Output_data.keys():
        fig_line_flow = go.Figure()
        for l in Output_data['pf'].index.tolist():
            if sum(Output_data['pf'].loc[l,:])!=0:
                fig_line_flow.add_trace(go.Scatter(x=Output_data['pf'].columns.tolist(), y=Output_data['pf'].loc[l,:], name='{}'.format(System_data['Branch'].loc[l,'Branch_Name']),line_shape='hv'))
        fig_line_flow.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_line_flow.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_line_flow.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_line_flow.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Flujo de potencia lineas de transmisión',xaxis_title='Tiempo [h]',yaxis_title='Potencia [MW]')
        st.write(fig_line_flow)
    
    if modelo_MEM == 'Co-optimización':
        fig_Inertia = go.Figure()
        fig_Inertia.add_trace(go.Scatter(x=Output_data['Total_inertia'].index.tolist(), y=Output_data['Total_inertia'].tolist(), name='Inercia del sistema',line_shape='hv'))
        fig_Inertia.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_Inertia.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Inertia.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Inertia.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Inercia total del sistema',xaxis_title='Tiempo [h]',yaxis_title='Inercia [MWs]')    
        st.write(fig_Inertia)
    
    if modelo_MEM == 'Co-optimización':
        fig_RoCOF = go.Figure()
        fig_RoCOF.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['RoCoF_05'].values, name='RoCoF', mode='markers'))
        fig_RoCOF.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_RoCOF.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_RoCOF.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_RoCOF.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='RoCoF',xaxis_title='Tiempo [h]',yaxis_title='RoCoF [Hz/s]')    
        fig_RoCOF.update_layout(shapes=[dict(type= 'line',yref= 'y', y0= -System_data['Sys_data'].loc['Max_D_freq','Data'], y1= -System_data['Sys_data'].loc['Max_D_freq','Data'],xref= 'paper', x0= 0, x1= 1,line=dict(color='Red',dash='dot'), name='Máximo RoCoF')])
        st.write(fig_RoCOF)
    
    if modelo_MEM == 'Co-optimización':
        fig_Fnadir = go.Figure()
        fig_Fnadir.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['D_nadir'].values, name='Frecuencia nadir', mode='markers'))
        fig_Fnadir.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_Fnadir.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Fnadir.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Fnadir.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Frecuencia minima alcanza ante contingencia',xaxis_title='Tiempo [h]',yaxis_title='F nadir [Hz]')    
        fig_Fnadir.update_layout(shapes=[dict(type= 'line',yref= 'y', y0= System_data['Sys_data'].loc['Min_freq','Data'], y1= System_data['Sys_data'].loc['Min_freq','Data'],xref= 'paper', x0= 0, x1= 1,line=dict(color='Red',dash='dot'))])
        st.write(fig_Fnadir)
    
    if modelo_MEM == 'Co-optimización':
        fig_Fqss = go.Figure()
        fig_Fqss.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['F_qss'].values, name='Frecuencia en estado estable', mode='markers'))
        fig_Fqss.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_Fqss.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Fqss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Fqss.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Frecuencia en estado estable',xaxis_title='Tiempo [h]',yaxis_title='F qss [Hz]')    
        fig_Fqss.update_layout(shapes=[dict(type= 'line',yref= 'y', y0= System_data['Sys_data'].loc['Min_freq_qss','Data'], y1= System_data['Sys_data'].loc['Min_freq_qss','Data'],xref= 'paper', x0= 0, x1= 1,line=dict(color='Red',dash='dot'))])
        st.write(fig_Fqss)
    
    fig_OP_cost = go.Figure()
    fig_OP_cost.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['Obj_f'].values, name='Función Objetivo',line_shape='hv'))
    fig_OP_cost.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['OP_cost'].values, name='Costos de operación',line_shape='hv'))
    fig_OP_cost.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['OP_cost_SD'].values, name='Costos Mercado Energía',line_shape='hv'))
    fig_OP_cost.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['OP_cost_AGC'].values, name='Costos Mercado AGC',line_shape='hv'))
    fig_OP_cost.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
    fig_OP_cost.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_OP_cost.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_OP_cost.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Costos de operación del sistema',xaxis_title='Tiempo [h]',yaxis_title='Costos [USD]')
    st.write(fig_OP_cost)
    
    if modelo_MEM != 'Despacho AGC':
        fig_MPO = go.Figure()
        fig_MPO.add_trace(go.Scatter(x=Output_data['Other'].index.tolist(), y=Output_data['Other']['MPO'].values, name='MPO',line_shape='hv'))
        fig_MPO.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_MPO.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_MPO.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_MPO.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='MPO',xaxis_title='Tiempo [h]',yaxis_title='MPO [USD/kWh]')    
        st.write(fig_MPO)
    
    if len(ESS_list)> 0:
        for E in ESS_list:
            st.write('#### {}({}):'.format(System_data['ESS_Unit'].loc[E,'ESS_Name'],E)) 
            fig_ESS_operation = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ESS_operation.add_trace(go.Scatter(x=Output_data['e_b'].columns.tolist(), y=Output_data['e_b'].loc[E,:], name="Energía almacenada",line_shape='linear'),secondary_y=False)      
            fig_ESS_operation.add_trace(go.Scatter(x=Output_data['Pot_Ba_ch'].columns.tolist(), y=Output_data['Pot_Ba_ch'].loc[E,:], name="Carga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)      
            fig_ESS_operation.add_trace(go.Scatter(x=Output_data['Pot_Ba_dc'].columns.tolist(), y=Output_data['Pot_Ba_dc'].loc[E,:], name="Descarga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)
            fig_ESS_operation.update_layout(title_text="Operación del SAE",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0, traceorder='reversed'),autosize=True,plot_bgcolor='rgba(0,0,0,0)')
            fig_ESS_operation.update_xaxes(title_text='Tiempo [h]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
            fig_ESS_operation.update_yaxes(title_text='Energia [MWh]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=False)
            fig_ESS_operation.update_yaxes(title_text='Potencia [MW]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=True)    
            st.write(fig_ESS_operation)
            
            if 'Degradación SAE' in restr_list:
                fig_E_rem = go.Figure()
                fig_E_rem.add_trace(go.Scatter(x=Output_data['E_cap'].columns.tolist(), y=Output_data['E_cap'].loc[E,:], name='Energía remanente',line_shape='hv'))
                fig_E_rem.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
                fig_E_rem.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
                fig_E_rem.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
                fig_E_rem.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Capacidad de almacenamiento de energía',xaxis_title='Tiempo [h]',yaxis_title='Energía remanente [MWh]')
                st.write(fig_E_rem)

                
def math_formulation():
    return r"""
    ## Función Objetivo
    Mínimizar los costos de operación del sistema.
    $$ \begin{aligned}
        \min F = & \underbrace{\sum_{t \in \mathcal{T}}\sum_{i \in \mathcal{I}}\left(P_{i,t} \cdot C_{i,t}^{Energy}\right)}_{Costos\hspace{1mm}de\hspace{1mm}generación\hspace{1mm}de\hspace{1mm}energía}\\\\
        &  +\underbrace{\sum_{t \in \mathcal{T}}\sum_{i \in \mathcal{I}}\left(RSF_{i,t}^{up} \cdot C_{i,t}^{Energy}\right)}_{Costos\hspace{1mm}de\hspace{1mm}reserva\hspace{1mm}de\hspace{1mm}energía}\\\\
        &  +\underbrace{\sum_{t \in \mathcal{T}}\sum_{i \in \mathcal{I}}\left(SU_{i,t} + SD_{i,t}\right) \cdot C_{i,t}^{SUD}}_{Costos\hspace{1mm}de\hspace{1mm}reserva\hspace{1mm}de\hspace{1mm}energía}\\\\
        &  + \underbrace{\sum_{t \in \mathcal{T}}\sum_{n \in \mathcal{N}}\frac{\beta_{n,t}}{(1-eol_{n})} \cdot C_{n}^{storage}}_{Costos\hspace{1mm}por\hspace{1mm}degradación\hspace{1mm}del\hspace{1mm}SAE}\\\\
    \end{aligned}$$
    
    Los tres primeros términos de la función objetivo corresponden a los costos de los generadores convencionales, en donde, el primer termino representa los costos asociados a la generación de energía para
    responder a la demanda del sistema, siendo $P_{i,t}$ la potencia despachada asignada al generador $i$, $C_{i,t}^{Energy}$ el costo de la energía ofertado por dicho generador y $\Delta{t}$ es el
    intervalo de tiempo en los que se realizan los despachos; El segundo termino corresponde a los costos asociados a la asignación de reserva secundaria de los generadores, en donde $RSF_{i,t}^{up}$ es la 
    potencia asignada para la regulación secundaría de frecuencia (RSF); y el tercer termino son los costos de arranque y parada de los generadores convencionales, siendo $SU_{i,t}$ y $SD_{i,t}$ señales binarias
    que indican el arranque y parada, respectivamente, de la unidad de generación $i$ y $C_{i,t}^{SUD}$ representa el costo de arranque y parada de dicha unidad de generación. El último termino refleja los costos de operación del SAEB dada la degradación generada por los ciclos de carga y descarga de la misma, para esto se tiene en cuenta $\beta_{n,t}$ que corresponde a la 
    degradación en (p.u.) de la capacidad de almacenamiento del SAEB, $eol_{n}$ el cual es el nivel de capacidad remanente del SAEB antes de que sea necesario su reemplazo y $C_{n}^{storage}$ que
    es el costo del almacenador (baterías) dado en ($\$/MWh$).
    
    ## Restricciones
    ### Generadores convencionales
    #### Lí­mite de potencia
    
    $$ \begin{aligned}
        P_{i,t} + RPF_{i,t}^{up} + RSF_{i,t}^{up} \leq P_{i,t}^{max} \cdot x_{i,t}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        P_{i,t} - RPF_{i,t}^{dw} - RSF_{i,t}^{dw} \geq P_{i,t}^{min} \cdot x_{i,t}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    Las ecuaciones definen los rangos de potencia disponible para despacho de un generador $i$ en el instante $t$ en función de su estado de operación ($x_{i,t}$), 
    capacidad máxima ofertada ($P^{max}_{i,t}$), su mínimo técnico ($P^{min}_{i,t}$), las reservas primaria ($RPF_{i,t}^{up}$, $RPF_{i,t}^{dw}$) y secundaria ($RSF_{i,t}^{up}$, $RSF_{i,t}^{dw}$) 
    de frecuencia asignadas.
    
    #### Señales de arranque y parada
    
    $$ \begin{aligned}
        SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        SU_{i,t} + SD_{i,t} \leq 1\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    Las ecuaciones anteriores relacionan las señales de arranque ($SU_{i,t}$) y parada ($SD_{i,t}$) de cada uno de los generadores, en cada paso de tiempo del horizonte de simulación,
    con el estado de los generadores ($x_{i,t}$)
    
    #### Rampas
    
    $$ \begin{aligned}
        P_{i,t} - P_{i,t-1} \leq R_{i}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i,t}^{min} \hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        P_{i,t-1} - P_{i,t} \geq R_{i}^{dw} \cdot x_{i,t} - SD_{i,t+1} \cdot P_{i,t}^{min}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    Las ecuaciones anteriores limitan el cambio en la entrega de energía de un generador $i$ con el fin de mantener los límites de la rampa de subida ($R_{i}^{up}$) y la rampa de 
    bajada ($R_{i}^{dw}$) de los generadores, respectivamente.
    
    #### Tiempos mínimos de encendido/apagado

    $$ x_{i,t} = g_{i}^{on/off} \quad \forall t \in \left(L_{i}^{up,min}+L_{i}^{dn,min}\right),
    i \in \mathcal{I} $$

    $$ \sum_{tt=t-g_{i}^{up}+1} SU_{i,tt} \leq x_{i,tt} \quad \forall t
    \geq L_{i}^{up,min} $$

    $$ \sum_{tt=t-g_{i}^{dn}+1} SD_{i,tt} \leq 1-x_{i,tt} \quad \forall t
    \geq L_{i}^{dn,min} $$

    donde $g_{i}^{on/off}$ es el estado inicial de las unidades de generación térmica; $L_{i}^{up,min}$ y $L_{i}^{dn,min}$ son los tiempos
    mínimos de encendido y apagado de las unidades de generación térmica.
    
    #### Reserva primaria de frecuencia (RPF):

     $$ \begin{aligned}
        RPF_{i,t}^{up} \geq P_{i,t} \cdot U_{i}^{RPF} \cdot \delta^{RPF}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    $$ \begin{aligned}
        RPF_{i,t}^{up} = RPF_{i,t}^{dw}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    Para determinar la reserva primaria aportada por cada generador, y teniendo en cuenta que para el caso colombiano la reserva primaria corresponde a un porcentaje mínimo (3\%) de la potencia 
    despachada, se emplea la formulación anteriomente presentada, en donde $U_{i}^{RPF}$ indica si la unidad de generación $i$ debe suministrar reserva primaria de frecuencia y $\delta^{RPF}$ es el valor mínimo 
    de reserva en (p.u.) que debe suplir dicho generador.
    
    
    #### Reserva secundaria de frecuencia (RSF):

     $$ \begin{aligned}
        RSF_{i,t}^{up} \leq RSF_{i,t}^{up,max} \cdot x_{i,t}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    $$ \begin{aligned}
        RSF_{i,t}^{up} = RSF_{i,t}^{dw}\hspace{2mm} \forall t \in \mathcal{T},i \in \mathcal{I}
    \end{aligned}$$
    
    Por su parte, las ecuación anteriores limitan la reserva secundaria de energía asignable al generador con la oferta de holgura efectuada ($RSF_{i,t}^{up,max}$) por el agente generador

    
    ### Restricciones sistemas de almacenamiento de energí­a basados en baterías
    #### Lí­mite de potencia de los SAEB
    
    $$ \begin{aligned}
        p_{n,t}^{ch} \leq P^{max}_{n} \cdot u_{n,t}^{ch} \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        p_{n,t}^{dch} \leq P^{max}_{n} \cdot u_{n,t}^{dc} \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$ 
    
    Estas restricciones buscan limitar la potencia de carga ($P^{ch}_{n,t}$) y descarga ($P^{dch}_{n,t}$) del SAEB, respectivamente, en función de la potencia máxima del sistema
    de conversión de potencia ($P_{n}^{max}$) y la señal de carga ($u^{ch}_{n,t}$) / descarga ($u^{ch}_{n,t}$) resultante de la operación óptima del sistema
    
    #### Variables binarias de estado de los SAEB
    
    $$ \begin{aligned}
        u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    Para evitar la carga y descarga simultanea dentro de la simulación se agrega esta restricción.
    
    #### Reserva rapida de frecuencia (RRF):
        
    $$ \begin{aligned}
        RRF_{n,t}^{up,dc} \leq P^{max}_{n} - p_{n,t}^{dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RRF_{n,t}^{up,ch} \leq p_{n,t}^{ch}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
     $$ \begin{aligned}
        RRF_{n,t}^{up} = RRF_{n,t}^{up,ch} + RRF_{n,t}^{up,dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RRF_{n,t}^{dw,dc} \leq p_{n,t}^{dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RRF_{n,t}^{dw,ch} \leq P^{max}_{n} - p_{n,t}^{ch}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
     $$ \begin{aligned}
        RRF_{n,t}^{dw} = RRF_{n,t}^{dw,ch} + RRF_{n,t}^{dw,dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    Las restricciones permiten determinar la capacidad de potencia que tiene el SAEB para suministrar RRF. Debido a que los SAEB son sistemas que pueden funcionar como una
    fuente de energía y un carga, la capacidad de potencia que puede aportar para la reserva de energía se puede ver como la disponibilidad del SAEB $n$ en el tiempo $t$ para inyectar potencia a 
    la red ($RRF_{n,t}^{up,dch}$) y la capacidad que tienen de deslastrar su propia carga ($RRF_{n,t}^{up,ch}$) siendo la capacidad de reserva rápida
    total para subir la  frecuencia ($RRF_{n,t}^{up}$) igual a la suma de los dos componentes mencionados anteriormente.
    
    #### Reserva secundaria de frecuencia (RSF):
        
    $$ \begin{aligned}
        RSF_{n,t}^{up,dc} \leq P^{max}_{n} - p_{n,t}^{dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RSF_{n,t}^{up,ch} \leq p_{n,t}^{ch}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
     $$ \begin{aligned}
        RSF_{n,t}^{up} = RSF_{n,t}^{up,ch} + RSF_{n,t}^{up,dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RSF_{n,t}^{dw,dc} \leq p_{n,t}^{dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    $$ \begin{aligned}
        RSF_{n,t}^{dw,ch} \leq P^{max}_{n} - p_{n,t}^{ch}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
     $$ \begin{aligned}
        RSF_{n,t}^{dw} = RSF_{n,t}^{dw,ch} + RSF_{n,t}^{dw,dc}  \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    Las restricciones permiten determinar la capacidad de potencia que tiene el SAEB para suministrar RSF. Debido a que los SAEB son sistemas que pueden funcionar como una
    fuente de energía y un carga, la capacidad de potencia que puede aportar para la reserva de energía se puede ver como la disponibilidad del SAEB $n$ en el tiempo $t$ para inyectar potencia a 
    la red ($RSF_{n,t}^{up,dch}$) y la capacidad que tienen de deslastrar su propia carga ($RSF_{n,t}^{up,ch}$) siendo la capacidad de reserva rápida
    total para subir la  frecuencia ($RSF_{n,t}^{up}$) igual a la suma de los dos componentes mencionados anteriormente.
    
    #### Relación entre la potencia y energía de los SAEB
    
    $$ \begin{aligned}
        E_{n,t} = E_{n,t-1}\cdot (1 - \eta^{SoC}_{n}) + \left( \eta^{n,ch} \cdot p_{n,t}^{ch} -\frac{p_{n,t}^{dc}}{\eta^{n,dc}} \right)\cdot \Delta t \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N}  \hspace{10mm}
    \end{aligned}$$
    
    donde $E_{n,t}$ corresponde al nivel de energía almacenada en el SAE en el tiempo $t$ dado en MWh; $\eta^{SoC}$ es la tasa de auto-descarga horaria; $\eta^{ch}$ y $\eta^{dc}$ son las eficiencias
    de carga y descarga del SAE, respectivamente.
    
    #### Lí­mite de energí­a de los SAEB
    
    $$ \begin{aligned}
        E_{n,t} + (RRF_{n,t}^{dw,ch} \cdot \Delta_{RRF}) + (RSF_{n,t}^{dw,ch} \cdot \Delta_{RSF}) \leq E_{n,t}^{cap} \cdot SOC^{max} \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        E_{n,t} - (RRF_{n,t}^{up,dc} \cdot \Delta_{RRF}) - (RSF_{n,t}^{up,dc} \cdot \Delta_{RSF}) \geq E^{max}_{n} \cdot SOC^{min} \hspace{2mm} \forall t \in \mathcal{T}, n \in \mathcal{N} 
    \end{aligned}$$
    
    donde $E^{max}_{n}$ hace referencia a la capacidad máxima del almacenador en MWh; $E_{t}^{cap}$ es la capacidad de almacenamiento remanente del SAE en el tiempo $t$;
    $SOC^{max}$ y $SOC^{min}$ corresponden al estado de carga máximo y mínimo del SAE.  
    #### Degradación de los SAEB
        
    $$ \begin{aligned}
        DoD_{n,t,d} \leq DoD_{n,d}^{max} \cdot S_{n,t,d}^{DoD}  \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        DoD_{n,t,d} \geq DoD_{n,d}^{min} \cdot S_{n,t,d}^{DoD}  \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        \frac{p_{n,t}^{dc} \cdot \Delta t}{E^{max}_{n} \cdot \eta^{n,dc}} = \sum_{d \in \mathcal{D}}DoD_{n,t,d} \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \sum_{d \in \mathcal{D}}S_{n,t,d}^{DoD} \leq 1 \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \beta_{n,t} = \sum_{d \in \mathcal{D}}\left(\beta_{n,d}^{slope} \cdot DoD_{n,t,d}\right) + \left(\beta_{n,d}^{constant} \cdot S_{n,t,d}^{DoD}\right) \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        E_{n,t}^{cap} = E_{n,t-1}^{cap} - \beta_{n,t} \cdot E^{max}_{n} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $DoD_{t,d}$ corresponde a la profundidad de descarga del SAEB en el tiempo $t$ del segmento de la curva de degradación $d$; $S_{t,d}^{DoD}$ es la variable binaria
    que indica si la profundidad de descarga se encuentra en el segmento $d$ de la curva de degradación durante el tiempo $t$; $\beta_{d}^{slope}$ y $\beta_{d}^{constant}$
    corresponden a la pendiente y la constante de la linealización de la curva de degradación en el segmento $d$, respectivamente; $\beta_{t}$ es la tasa de pérdida de capacidad del almacenador del SAE.
    
    ### Flujo de potencia DC y pérdidas
    **Cálculo del flujo de potencia por cada línea**

    $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
    (b,r) \in \mathcal{L}, t \in \mathcal{T} $$

    donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$ representan el
    valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.

    **Límites en el flujo de potencia en las líneas**

    $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} \leq P_{b,r}^{max}
    \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$

    donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
    que conecta los nodos $b$ y $r$.
    
    ### Balance del sistema
    
    #### Balance de energía
    
    $$ \begin{aligned}
        \sum_{i \in \mathcal{I}} P_{i,t} + \sum_{n \in \mathcal{N}} p_{n,t}^{dc} = L_{t} + \sum_{n \in \mathcal{N}} p_{n,t}^{ch} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $L_{t}$ respresenta la demanda de energía total del sistema en el tiempo $t$.
    
    #### Balance de reserva
    
    $$ \begin{aligned}
        \sum_{n \in \mathcal{N}}RSF_{n,t}^{up} + \sum_{i \in \mathcal{I}}RSF_{i,t}^{up} = RE_{t} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $RE_{t}$ respresenta la demanda de holgura total del sistema en el tiempo $t$.
    
    ### Dinamica de frecuencia post-contingencia
    
    #### Inercia del sistema
    
    $$ \begin{aligned}
        H_{t}^{sis} = \sum_{i \in \mathcal{I}} H_{i} \cdot P_{i}^{nom} \cdot x_{i,t} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $H_{t}^{sis}$ respresenta la inercia equivalente del sistema; $H_{i}$ es la inercia del generador $i$; $P_{i}^{nom}$ es la potencia nominal del generador $i$
    
    #### RoCoF
    
    $$ \begin{aligned}
        \frac{f_{0} \cdot \Delta P_{t}}{2 H_{t}^{sis}} \leq RoCoF^{max}\hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $f_{0}$ representa la frecuencia nominal del sistema; $\Delta P_{t}$ es el máximo desbalance de potencia que puede tener el sistema en el tiempo $t$; $RoCoF^{max}$ representa la máxima
    tasa de desviación de frecuencia que puede tener el sistema ante una eventual contigencia.
    
    #### Fnadir
    
    $$ \begin{aligned}
        \left(\frac{H_t}{f_0}-\frac{\sum_{n \in \mathcal{N}} RRF_{n,t}^{up}\cdot T_{FFR}}{4\cdot\Delta f_{max}}\right)\cdot \sum_{i \in \mathcal{I}} RPF_{i,t}^{up}\geq\alpha-(\lambda\cdot L_t)\hspace{2mm}  \forall t \in \mathcal{T}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \alpha=\frac{\left(\Delta P_{t}-\sum_{n \in \mathcal{N}}RRF_{n,t}^{up}\right)^2\cdot T_{RPF}}{4\cdot\mathrm{\Delta}f_{max}} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \lambda=\frac{\left(\Delta P_{t}-\sum_{n \in \mathcal{N}}RRF_{n,t}^{up}\right)\cdot T_{RPF}\cdot D}{4}\hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $T_{FFR}$ y $T_{RPF}$ son tiempos máximos de respuesta de los servicios de regulación rápida y primaria de frecuencia, respectivamente; $\Delta f_{max}$ representa el delta máximo de frecuencia
    que puede alcanzar el sistema antes de activar los sistemas de deslastre de carga.
    
    #### Fqss
    
    $$ \begin{aligned}
        \frac{\Delta P_{t} -\sum_{n \in \mathcal{N}}RRF_{n,t}^{up}-\sum_{i \in \mathcal{I}} RPF_{i,t}^{up}}{D\cdot L_t}\le\mathrm{\Delta}f_{max}^{qss} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $D$ es el amortiguamiento por parte de la demanda del sistema; $\Delta f_{max}^{qss}$ representa el valor mínimo de frecuencia que debe tener el sistema cuando este esta en esta Quasi-estable-estacionario.
    
    
    """
# In[Main]

def main_MEM(data1,Current_direction):
    
    
    # --------------------------------------------------------------------------------------------------------------
    # CONSTRUCCION DEL MODELO
    # --------------------------------------------------------------------------------------------------------------
    
    # Seleccionar el tipo de modelo
    
    modelo_MEM = st.sidebar.selectbox('Seleccione el modelo de MEM que desea simular',['Despacho AGC','Despacho Ideal','Despacho AGC + Ideal','Despacho programado', 'Co-optimización'])
    
    if modelo_MEM =='Despacho AGC':
        Modelo = [1,0,0,0,0]
    elif modelo_MEM =='Despacho Ideal':
        Modelo = [0,1,0,0,0]
    elif modelo_MEM =='Despacho AGC + Ideal':
        Modelo = [0,0,1,0,0]
    elif modelo_MEM == 'Despacho programado':
        Modelo = [0,0,0,1,0]
    elif modelo_MEM =='Co-optimización':
        Modelo = [0,0,0,0,1]
    
    with st.expander("Ver formulación del problema"):
        st.write(math_formulation())
        st.write("")
                
    st.markdown("## Parámetros seleccionados para la simulación") 
    
    # Seleción de archivo con sistema
    
    st.sidebar.markdown("### Ingrese los parámetros de la simulación")
    
    source = st.sidebar.selectbox('Seleccione la fuente de información de las variables del sistema',['XM (Colombia)','Otro'])
    
    if source == 'XM (Colombia)':
        file_system = '{}/Casos_estudio/Colombia_220_2021_test_2.xlsx'.format(Current_direction)
        date1 = st.sidebar.date_input('Fecha de inicio de la simulación:',value = date.today() - timedelta(days=40), max_value = date.today() - timedelta(days=40))
        date2 = st.sidebar.date_input('Fecha final de la simulación:',value = date.today() - timedelta(days=40), min_value = date1, max_value = date.today() - timedelta(days=40))  
        
        # Ingresar TRM
        
        TRM_select=st.sidebar.selectbox('Seleccione la TRM para la simulación',['Hoy','Otra'],key='2')
        if TRM_select=="Hoy":
            URL = 'https://www.dolar-colombia.com/'
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, 'html.parser')
            rate = soup.find_all(class_="exchange-rate")
            TRM=str(rate[0])
            TRM=re.findall('\d+', TRM )
            TRM_final_x= TRM[0]+TRM[1]
            TRM_final= float(TRM_final_x)
            TRM_final_1="* La TRM seleccionada para la simulación es de: "+TRM_final_x + " COP/USD"
            st.write(TRM_final_1)
        else:
            TRM_final=st.number_input("Ingrese la TRM para la simulación [COP/USD]: ")
            
    elif source == 'Otro':
        st.set_option('deprecation.showfileUploaderEncoding', False)
        file_system = st.sidebar.file_uploader("Seleccione el archivo con el sistema a simular:", type=["csv","xlsx"])
        date1 = date.today()
        date2 = date.today()
        TRM_final = 1
    
    # Restricciones    
    
    if modelo_MEM =='Co-optimización':
        
        restr_list_compl = ['RPF Generadores', 'Degradación SAE',
                            'Flujos de potencia DC', 'RoCoF Sistema', 'Fnadir sistema', 'Fqss sistema']
    
        restr_option = st.sidebar.selectbox('¿Desea modificar las restricciones del modelo?',['No','Si'])
        
        if restr_option == 'Si': 
            st.sidebar.write('**Restricciones base**')
            st.sidebar.write('Límites de generación\n\n','Balance de potencia\n\n', 'Sistemas de Almacenamiento de Energía')
            st.sidebar.write('**Restricciones adicionales**')
            restr_list = st.sidebar.multiselect(label='', options=sorted(restr_list_compl), default=restr_list_compl)
        else:
            restr_list = restr_list_compl 
    
    elif modelo_MEM =='Despacho AGC':
        restr_list = []
    else:
        restr_list = ['RPF Generadores','Degradación SAE']
    
    
    # Agregar SAE y parametros
    
    st.sidebar.markdown("### Ingrese los parámetros del SAE a simular")
    
    SAE = pd.DataFrame()
    
    if file_system != None:
        
        SAE_option = st.sidebar.selectbox('¿Desea agregar un SAE al modelo?',['Si','No'])
      
        if SAE_option == 'Si':
      
            SAE_Name = st.sidebar.text_input("Nombre del SAE: ")
            df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)
            if modelo_MEM in ['Despacho programado','Co-optimización']:
                B_name = st.sidebar.selectbox('Ingrese el nodo de ubicación de SAE',df_Bus['Bus_name'])
                B_index = df_Bus.index[df_Bus['Bus_name'] == B_name].tolist()
            else:
                B_index = ['X']
            P = st.sidebar.number_input("Ingrese el tamaño en potencia [MW] : ",min_value=0.0)
            E = st.sidebar.number_input("Ingrese el tamaño en energía [MWh] : ",min_value=0.0)
            
            Eff,degra,autoD,DoD,costP,costE,a,b,ciclos = bat_param(data1,1)
            
            SAE['ESS_index']=['ESSP']
            SAE['ESS_Name']=[SAE_Name]
            SAE['Bus_index']=[B_index[0]]
            SAE['C_Potencia']=[P]
            SAE['C_Energia']=[E]
            SAE['n_ch_eff']=[math.sqrt(Eff)]
            SAE['n_dc_eff']=[math.sqrt(Eff)]
            SAE['Self_discharge']=[autoD]
            SAE['SOC_max']=[1]
            SAE['SOC_min']=[1-DoD]
            SAE['SOC_ini']=[1-DoD]
            SAE['N_ciclos']=[ciclos]
            SAE['eol']=[0.8]
            SAE['a']=[a]
            SAE['b']=[b]
            SAE['P_cost']=[costP]
            SAE['E_cost']=[costE]
            
            SAE.set_index('ESS_index', inplace=True)       
    
    # Seleccionar solver
    
    st.sidebar.markdown("### Simulación")
    
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'])
    
    if solver=='CPLEX':
        st.write("* El solucionador seleccionado es: "+solver)
    else:
        st.write("* El solucionador seleccionado es: "+solver)
    
    
    # Funcion para ejecutar el despacho optimo de AGC
    
    button_sent = st.sidebar.button("Simular")
    
    if button_sent:
        st.markdown("## Simulación")
        with st.spinner('Importando el modelo'):
            System_data = System_import(file_system,source,date1,date2,SAE,TRM_final)
        with st.spinner('Ejecutando la simulación'):
            if modelo_MEM =='Despacho AGC + Ideal':
                with st.spinner('Ejecutando despacho AGC'):
                    Input_data = {}
                    Modelo = [1,0,0,0,0]
                    Output_data = MEM_general_model(System_data,Modelo,Input_data,restr_list,Current_direction,solver)
                with st.spinner('Ejecutando despacho Ideal'):
                    Modelo = [0,1,1,0,0]
                    Output_data = MEM_general_model(System_data,Modelo,Output_data,restr_list,Current_direction,solver)
            
            elif modelo_MEM =='Despacho programado':
                with st.spinner('Ejecutando despacho AGC'):
                    Input_data = {}
                    Modelo = [1,0,0,0,0]
                    Output_data = MEM_general_model(System_data,Modelo,Input_data,restr_list,Current_direction,solver)
                with st.spinner('Ejecutando despacho programado'):
                    Modelo = [0,1,1,1,0]
                    Output_data = MEM_general_model(System_data,Modelo,Output_data,restr_list,Current_direction,solver)
            else:
                Input_data = {}
                Output_data = MEM_general_model(System_data,Modelo,Input_data,restr_list,Current_direction,solver)
        print_results(Output_data,System_data,modelo_MEM,restr_list)