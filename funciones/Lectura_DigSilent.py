# -*- coding: utf-8 -*-
"""
LECTURA DE ARCHIVOS DIGSILENT

@author: Andrés Felipe Peñaranda Bayona
"""

# In[Librerias]

import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import streamlit as st

# In[Funciones]

def lectura_DigSilent(Current_direction):
    
    
    # Crear bandera del de proceso de lectura
    
    file = open("{}/Datos_DigSilent/Flag_start.txt".format(Current_direction), "w")
    file.write('0')
    file.close()
    
    file = open("{}/Datos_DigSilent/Flag_end.txt".format(Current_direction), "w")
    file.write('0')
    file.close()
      
    # Lectura de bandera de proceso de lectura
    
    st.markdown('Ejecute el comando DPL en DigSilent')
    
    progress_bar = st.progress(0)
    
    file = open("{}/Datos_DigSilent/Flag_start.txt".format(Current_direction), "r")
    flag_start = file.read()
    file.close()
    
    while flag_start == '0':
        file = open("{}/Datos_DigSilent/Flag_start.txt".format(Current_direction), "r")
        flag_start = file.read()
        file.close()
        
    file = open("{}/Datos_DigSilent/Flag_end.txt".format(Current_direction), "r")
    flag_end = file.read()
    file.close()
    
    while flag_end == '0':
        file = open("{}/Datos_DigSilent/Flag_end.txt".format(Current_direction), "r")
        flag_end = file.read()
        file.close()
    
    progress_bar.progress(5)
    
    return progress_bar

def transform_DigSilent_data(Current_direction,progress_bar):
    
    # In[Lectura archivos]
        
    df_Buses = pd.read_csv('{}/Datos_DigSilent/Nodos.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Loads = pd.read_csv('{}/Datos_DigSilent/Cargas.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Gen = pd.read_csv('{}/Datos_DigSilent/Gen.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Genstat = pd.read_csv('{}/Datos_DigSilent/Genstat.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Lines = pd.read_csv('{}/Datos_DigSilent/Lineas.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Trafos_2 = pd.read_csv('{}/Datos_DigSilent/Trf2.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Trafos_3 = pd.read_csv('{}/Datos_DigSilent/Trf3.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Capacitors = pd.read_csv('{}/Datos_DigSilent/Shunt.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Switches = pd.read_csv('{}/Datos_DigSilent/Switches.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Red_ext = pd.read_csv('{}/Datos_DigSilent/Redext.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    df_Reactores = pd.read_csv('{}/Datos_DigSilent/Reactores.txt'.format(Current_direction), sep=",",header=0,encoding='latin1')
    
    progress_bar.progress(10)
    
    # In[sheet:System_data]
    
    Parameter = ['S_base','Slack_bus','N_freq','Max_D_freq','Min_freq','DB_freq','Delta_1_RF','Delta_2_RF','Delta_3_RF']
    Parameter_value = [100,2833,60,0.5,59.5,0.02,0.001388,0.004166,0.041666]
    
    System_data = pd.DataFrame()
    System_data[''] = Parameter
    System_data['Data'] = Parameter_value
    
    progress_bar.progress(15)
    
    
    # In[sheet:Bus]
    
    Bus_data = pd.DataFrame()
    Bus_data['Bus_number'] = df_Buses['Num'].tolist()
    Bus_data['Name'] = df_Buses['Nombre'].tolist()
    Bus_data['Bus_name'] = df_Buses['Subestacion'].tolist()
    Bus_data['Bus_zone'] = df_Buses['Zona'].tolist()
    
    buses_type = df_Buses['Tipo'].tolist()
    Bus_type = []
    num_buses = len(buses_type)
    for b in range(num_buses):
        if buses_type[b] == 0:
            Bus_type.append('b')
        else:
            Bus_type.append('n')
    
    Bus_data['Bus_type'] = Bus_type
    Bus_data['Bus_voltage_[kV]'] = df_Buses['[kV]'].tolist()
    Bus_data['Bus_max_voltage_[pu]'] = df_Buses['Max_V[p.u.]'].tolist()
    Bus_data['Bus_min_voltage_[pu]'] = df_Buses['Min_v[p.u.]'].tolist()
    
    outserv = df_Buses['Fuera_Servicio'].tolist()
    status = []
    lat = []
    lon = []
    num_buses = len(outserv)
    for b in range(num_buses):
        status.append(1 - outserv[b])
        lat.append(0)
        lon.append(0)
    
    Bus_data['In_service'] = status
    Bus_data['lat'] = lat
    Bus_data['lon'] = lon
    
    progress_bar.progress(20)
    
    # In[sheet:Gen]
    
    Gen_data = pd.DataFrame()
    
    num_gen = len(df_Gen)
    Gen_index = []
    num_gen_stat = len(df_Genstat)
    Gen_stat_index = []
    Gen_total_index = []
    
    for g in range(num_gen):
        Gen_index.append('G{}'.format(g+1))
        Gen_total_index.append('G{}'.format(g+1))
    
    for g in range(num_gen_stat):
        Gen_stat_index.append('GS{}'.format(g+1))
    
    for g in range(num_gen_stat):
        Gen_total_index.append(Gen_stat_index[g])
    
    Gen_data[''] = Gen_total_index
    
    Gen_name = df_Gen['Nombre'].tolist()
    Gen_total_name = df_Gen['Nombre'].tolist()
    Gen_stat_name = df_Genstat['Nombre'].tolist()
    
    for g in range(num_gen_stat):
        Gen_total_name.append(Gen_stat_name[g])
    
    Gen_data['Name'] = Gen_total_name
    
    Gen_bus_name = df_Gen['Subestacion'].tolist()
    Gen_total_bus_name = df_Gen['Subestacion'].tolist()
    Gen_stat_bus_name = df_Genstat['Subestacion'].tolist()
    
    for g in range(num_gen_stat):
        Gen_total_bus_name.append(Gen_stat_bus_name[g])
    
    Gen_data['Bus_Name'] = Gen_total_bus_name
    
    Gen_bus_number = []
    Gen_bus_name = Gen_data['Bus_Name'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for g in Gen_bus_name:
        index = bus_name.index(g)
        Gen_bus_number.append(bus_number[index])
    
    Gen_data['Bus_number'] = Gen_bus_number
    
    fuel_type = []
    
    for b in Gen_name:
        if 'Vapor' in b:
            fuel_type.append('COAL')
        elif 'Gas' in b:
            fuel_type.append('GAS')
        elif 'PCH' in b:
            fuel_type.append('PCH')
        elif 'CHP' in b:
            fuel_type.append('CHP')
        elif 'Jepirachi' in b:
            fuel_type.append('WIND')
        else:
            fuel_type.append('')
    
    for b in Gen_stat_name:
        if 'Solar' in b:
            fuel_type.append('SOLAR')
        elif 'SOLAR' in b:
            fuel_type.append('SOLAR')
        elif 'STATCOM' in b:
            fuel_type.append('STATCOM')
        else:
            fuel_type.append('WIND')
        
    Gen_data['Fuel_Type'] = fuel_type
    
    Gen_data['c'] = [0]*len(Gen_data)
    Gen_data['b'] = [0]*len(Gen_data)
    Gen_data['a'] = [0]*len(Gen_data)
    Gen_data['CSU'] = [0]*len(Gen_data)
    
    V_nom = df_Gen['V_nom[kV]'].tolist()
    V_nom_stat = df_Genstat['V_nom[kV]'].tolist()
    
    for g in range(num_gen_stat):
        V_nom.append(V_nom_stat[g])
    
    Gen_data['V_nom'] = V_nom
    
    Set_point_gen = df_Gen['Set_point[p.u.]'].tolist()
    Set_point_genstat = df_Genstat['Set_point[pu]'].tolist()
    
    for g in range(num_gen_stat):
        Set_point_gen.append(Set_point_genstat[g])
    
    Gen_data['Set_point'] = Set_point_gen
    
    S_nom_gen = df_Gen['S_nom[MVA]'].tolist()
    S_nom_genstat = df_Genstat['S_nom[MVA]'].tolist()
    
    for g in range(num_gen_stat):
        S_nom_gen.append(S_nom_genstat[g])
    
    Gen_data['S_nom'] = S_nom_gen
    
    PF_gen = df_Gen['PF[cos]'].tolist()
    PF_genstat = df_Genstat['PF[cos]'].tolist()
    
    for g in range(num_gen_stat):
        PF_gen.append(PF_genstat[g])
    
    Gen_data['PF'] = PF_gen
    
    P_gen = df_Gen['P_gen[MW]'].tolist()
    P_genstat = df_Genstat['P_gen[MW]'].tolist()
    
    for g in range(num_gen_stat):
        P_gen.append(P_genstat[g])
        
    Gen_data['Pdispath'] = P_gen
    
    Q_gen = df_Gen['Q_gen[MVAr]'].tolist()
    Q_genstat = df_Genstat['Q_gen[MVAr]'].tolist()
    
    for g in range(num_gen_stat):
        Q_gen.append(Q_genstat[g])
        
    Gen_data['Qdispath'] = Q_gen
    
    P_min_gen = df_Gen['P_min[MW]'].tolist()
    P_min_genstat = df_Genstat['P_min[MW]'].tolist()
    
    for g in range(num_gen_stat):
        P_min_gen.append(P_min_genstat[g])
    
    Gen_data['Pmin'] = P_min_gen
    
    P_max_gen = df_Gen['P_max[MW]'].tolist()
    P_max_genstat = df_Genstat['P_max[MW]'].tolist()
    
    for g in range(num_gen_stat):
        P_max_gen.append(P_max_genstat[g])
        
    Gen_data['Pmax'] = P_max_gen
    
    Q_min_gen = df_Gen['Q_min[MVAr]'].tolist()
    Q_min_genstat = df_Genstat['Q_min[MVAr]'].tolist()
    
    for g in range(num_gen_stat):
        Q_min_gen.append(Q_min_genstat[g])
        
    Gen_data['Qmin'] = Q_min_gen
    
    Q_max_gen = df_Gen['Q_max[MVAr]'].tolist()
    Q_max_genstat = df_Genstat['Q_max[MVAr]'].tolist()
    
    for g in range(num_gen_stat):
        Q_max_gen.append(Q_max_genstat[g])
        
    Gen_data['Qmax'] = Q_max_gen
    
    k1 = []
    k2 = []
    k3 = []
    
    for g in range(len(Gen_data)):
        p1_seg = (Gen_data['Pmax'][g] - Gen_data['Pmin'][g])/3
        k1.append(p1_seg)
        k2.append(p1_seg*2)
        k3.append(p1_seg*3)
    
    Gen_data['k1'] = k1
    Gen_data['k2'] = k2
    Gen_data['k3'] = k3
    
    Gen_data['Min_ON'] = [0]*len(Gen_data)
    Gen_data['Min_OFF'] = [0]*len(Gen_data)
    Gen_data['Min_OFF'] = [0]*len(Gen_data)
    Gen_data['Ramp_UP'] = [1000]*len(Gen_data)
    Gen_data['Ramp_Down'] = [1000]*len(Gen_data)
    Gen_data['IniT_ON'] = [0]*len(Gen_data)
    Gen_data['Init_off'] = [0]*len(Gen_data)
    Gen_data['IniMW'] = [0]*len(Gen_data)
    
    R = [0.5]*num_gen
    
    for g in range(num_gen_stat):
        R.append(0)
        
    Gen_data['R'] = R
    
    H = df_Gen['Inercia[s]'].tolist()
    
    for g in range(num_gen_stat):
        H.append(0)
    
    Gen_data['H'] = H
    
    R_sub = df_Gen['R_sub[p.u]'].tolist()
    
    for g in range(num_gen_stat):
        R_sub.append(0)
    
    Gen_data['R_sub'] = R_sub
    
    X_sub = df_Gen['X_sub[p.u]'].tolist()
    
    for g in range(num_gen_stat):
        X_sub.append(0)
    
    Gen_data['X_sub'] = X_sub
    
    F_escala = df_Gen['Factor_escala'].tolist()
    
    for g in range(num_gen_stat):
        F_escala.append(0)
        
    Gen_data['Factor_escala'] = F_escala
    
    Gen_data['Reservoir_name/River'] = ['']*len(Gen_data)
    Gen_data['Vmin'] = [0]*len(Gen_data)
    Gen_data['Vmax'] = [0]*len(Gen_data)
    Gen_data['Vmax_energy'] = [0]*len(Gen_data)
    Gen_data['Net_head'] = [0]*len(Gen_data)
    Gen_data['eff_turb_gen'] = [0]*len(Gen_data)
    Gen_data['C_hydro'] = [0]*len(Gen_data)
    Gen_data['Factor_conversion'] = [0]*len(Gen_data)
    Gen_data['Factor_V_to_E'] = [0]*len(Gen_data)
    
    Slack = df_Gen['Slack'].tolist()
    
    for g in range(num_gen_stat):
        Slack.append(0)
    
    Gen_data['Slack'] = Slack
    
    outserv_gen = df_Gen['Fuera_servicio'].tolist()
    outserv_genstat = df_Genstat['Fuera_servicio'].tolist()
    
    for g in range(num_gen_stat):
        outserv_gen.append(outserv_genstat[g])
        
    status = []
    num_g = len(outserv_gen)
    for g in range(num_g):
        status.append(1 - outserv_gen[g])
        
    Gen_data['In_service'] = status
        
    #### Gen Map
    
    Gen_map = {}
     
    for b in bus_number:
        Gen_map[b] = [0]*num_g
    
    for g in range(num_g):
        bus_num = Gen_bus_number[g]
        Gen_map[bus_num][g] = 1
    
    Gen_map_data = pd.DataFrame()
    Gen_map_data[''] = Gen_total_index
    
    for b in bus_number:
        Gen_map_data[b] = Gen_map[b]
    
    progress_bar.progress(30)
        
    # In[sheet:Renewable]
    
    ## Time simulation data
    
    Renewable_gen_data = pd.DataFrame()
    
    Num_hours = 24
    Hour_index = []
    
    for h in range(Num_hours):
        Hour_index.append(h+1)
    
    Renewable_gen_data[''] = Hour_index
    
    ## Renewable generation data
    
    # Generation profile
    
    Solar_generation_profile = [0,0,0,0,0,0.2,0.325,0.625,0.825,0.95,1,1,1,1,0.95,0.825,0.5,0.325,0,0,0,0,0,0]
    Wind_generation_profile = [0.586,0.485,0.453,0.586,0.500,0.459,0.504,0.402,0.517,0.652,0.712,0.725,0.726,0.714,0.655,0.682,0.697,0.691,0.758,0.780,0.725,0.707,0.682,0.636]
    
    Generation_type = np.array(fuel_type)
    
    Solar_type_index = np.where(Generation_type == 'SOLAR')[0]
    Wind_type_index = np.where(Generation_type == 'WIND')[0]
    Type_index = np.append(Solar_type_index,Wind_type_index)
    Type_index.sort()
    
    Gen_type_index = []
    Gen_type_profile = {}
    
    for g in Type_index:
        i = Gen_data.loc[g,'']
        Gen_type_index.append(i)
        Gen_type_profile[i]=[]
        
    h_i = 0
    
    for h in range(Num_hours):
        for s_i in Solar_type_index:
            i = Gen_data.loc[s_i,'']
            Gen_type_profile[i].append(Gen_data.loc[s_i,'Pmax']*Solar_generation_profile[h_i])
    
        for w_i in Wind_type_index:
            i = Gen_data.loc[w_i,'']
            Gen_type_profile[i].append(Gen_data.loc[w_i,'Pmax']*Wind_generation_profile[h_i])
        
        if h_i == 23:
            h_i = 0
        else:
            h_i = h_i + 1
    
    for g in Gen_type_index:
        Renewable_gen_data[g] = Gen_type_profile[g]
    
    progress_bar.progress(35)
              
    
    # In[sheet:Line]
    
    Line_data = pd.DataFrame()
    
    num_line = len(df_Lines)
    Line_index = []
    
    for l in range(num_line):
        Line_index.append('L{}'.format(l+1))
    
    Line_data[''] = Line_index
    Line_data['Name'] = df_Lines['Nombre'].tolist()
    
    Line_bus_number_i = []
    Line_bus_number_j = []
    Line_bus_name_i = df_Lines['Subestacion_i'].tolist()
    Line_bus_name_j = df_Lines['Subestacion_j'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for l in Line_bus_name_i:
        index = bus_name.index(l)
        Line_bus_number_i.append(bus_number[index])
    
    for l in Line_bus_name_j:
        index = bus_name.index(l)
        Line_bus_number_j.append(bus_number[index])
    
    Line_data['Bus_from'] = Line_bus_number_i
    Line_data['Bus_to'] = Line_bus_number_j
    
    Line_data['V_nom'] = df_Lines['V_nom[kV]'].tolist()
    Line_data['R'] = df_Lines['R[pu]'].tolist()
    Line_data['X'] = df_Lines['X[pu]'].tolist()
    
    Flow_limit = []
    
    for l in range(num_line):
        Flow_limit.append(df_Lines.loc[l,'V_nom[kV]']*df_Lines.loc[l,'I_nominal [kA]']*1.7320508075688)
    
    Line_data['Flowlimit'] = Flow_limit
    
    Line_data['Length'] = df_Lines['Longitud [km]'].tolist()
    Line_data['R1'] = df_Lines['R[Ohm/km]'].tolist()
    Line_data['X1'] = df_Lines['X[Ohm/km]'].tolist()
    Line_data['C1'] = df_Lines['C[uF/km]'].tolist()
    Line_data['R0'] = df_Lines['R0[Ohm/km]'].tolist()
    Line_data['X0'] = df_Lines['X0[Ohm/km]'].tolist()
    Line_data['C0'] = df_Lines['C0[uF/km]'].tolist()
    Line_data['I_nom'] = df_Lines['I_nominal [kA]'].tolist()
    Line_data['Max_load'] = df_Lines['Carga_maxima [%]'].tolist()
    
    outserv_line = df_Lines['Fuera_servicio'].tolist()    
    status = []
    num_l = len(outserv_line)
    for l in range(num_l):
        status.append(1 - outserv_line[l])
        
    Line_data['In_service'] = status
    
    progress_bar.progress(40)
    
    # In[sheet:Line]
    
    Reactor_data = pd.DataFrame()
    
    num_reactor = len(df_Reactores)
    Reactor_index = []
    
    for r in range(num_reactor):
        Reactor_index.append('RC{}'.format(r+1))
    
    Reactor_data[''] = Reactor_index
    Reactor_data['Name'] = df_Reactores['Nombre'].tolist()
    
    Reactor_bus_number_i = []
    Reactor_bus_number_j = []
    Reactor_bus_name_i = df_Reactores['Subestacion_i'].tolist()
    Reactor_bus_name_j = df_Reactores['Subestacion_j'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for r in Reactor_bus_name_i:
        index = bus_name.index(r)
        Reactor_bus_number_i.append(bus_number[index])
    
    for r in Reactor_bus_name_j:
        index = bus_name.index(r)
        Reactor_bus_number_j.append(bus_number[index])
    
    Reactor_data['Bus_from'] = Reactor_bus_number_i
    Reactor_data['Bus_to'] = Reactor_bus_number_j
    
    Reactor_data['V_nom'] = df_Reactores['V_nom[kV]'].tolist()
    Reactor_data['R'] = df_Reactores['R[pu]'].tolist()
    Reactor_data['X'] = df_Reactores['X[pu]'].tolist()
    
    Flow_limit = []
    
    for r in range(num_reactor):
        Flow_limit.append(df_Reactores.loc[r,'V_nom[kV]']*df_Reactores.loc[r,'I_nominal [kA]']*1.7320508075688)
    
    Reactor_data['Flowlimit'] = Flow_limit
    
    Reactor_data['Length'] = df_Reactores['Longitud [km]'].tolist()
    Reactor_data['R1'] = df_Reactores['R[Ohm/km]'].tolist()
    Reactor_data['X1'] = df_Reactores['X[Ohm/km]'].tolist()
    Reactor_data['C1'] = df_Reactores['C[uF/km]'].tolist()
    Reactor_data['R0'] = df_Reactores['R0[Ohm/km]'].tolist()
    Reactor_data['X0'] = df_Reactores['X0[Ohm/km]'].tolist()
    Reactor_data['C0'] = df_Reactores['C0[uF/km]'].tolist()
    Reactor_data['I_nom'] = df_Reactores['I_nominal [kA]'].tolist()
    Reactor_data['Max_load'] = df_Reactores['Carga_maxima [%]'].tolist()
    
    outserv_reactor = df_Reactores['Fuera_servicio'].tolist()    
    status = []
    num_r = len(outserv_reactor)
    for r in range(num_r):
        status.append(1 - outserv_reactor[r])
        
    Reactor_data['In_service'] = status
    
    progress_bar.progress(45)
    
    
    # In[sheet:Trf2]
    
    Trf2_data = pd.DataFrame()
    
    num_trf2 = len(df_Trafos_2)
    Trf2_index = []
    
    for t in range(num_trf2):
        Trf2_index.append('TF2_{}'.format(t+1))
    
    Trf2_data[''] = Trf2_index
    Trf2_data['Name'] = df_Trafos_2['Nombre'].tolist()
    
    Trf2_bus_number_i = []
    Trf2_bus_number_j = []
    Trf2_bus_name_i = df_Trafos_2['Subestacion_HV'].tolist()
    Trf2_bus_name_j = df_Trafos_2['Subestacion_LV'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for t in Trf2_bus_name_i:
        index = bus_name.index(t)
        Trf2_bus_number_i.append(bus_number[index])
    
    for t in Trf2_bus_name_j:
        index = bus_name.index(t)
        Trf2_bus_number_j.append(bus_number[index])
    
    Trf2_data['Bus_HV'] = Trf2_bus_number_i
    Trf2_data['Bus_LV'] = Trf2_bus_number_j
    
    Trf2_data['sn_mva'] = df_Trafos_2['S_nom[MVA]'].tolist()
    Trf2_data['vn_hv_kv'] = df_Trafos_2['HV[kV]'].tolist()
    Trf2_data['vn_lv_kv'] = df_Trafos_2['LV[kV]'].tolist()
    Trf2_data['R_pu'] = df_Trafos_2['R[pu]'].tolist()
    Trf2_data['X_pu'] = df_Trafos_2['X[pu]'].tolist()
    Trf2_data['vk_percent'] = df_Trafos_2['vk[%]'].tolist()
    Trf2_data['vkr_percent'] = df_Trafos_2['vkr[%]'].tolist()
    Trf2_data['pfe_kw'] = df_Trafos_2['Perdidas[kW]'].tolist()
    Trf2_data['i0_percent'] = df_Trafos_2['I0[%]'].tolist()
    Trf2_data['vk0_percent'] = df_Trafos_2['vk0[%]'].tolist()
    Trf2_data['vkr0_percent'] = df_Trafos_2['vkr0[%]'].tolist()
    Trf2_data['mag0_percent'] = df_Trafos_2['mag0[%]'].tolist()
    Trf2_data['mag0_rx'] = df_Trafos_2['mag0_rx'].tolist()
    Trf2_data['si0_hv_partial'] = df_Trafos_2['si0_hv[%]'].tolist()
    Trf2_data['shift_degree'] = df_Trafos_2['shift[deg]'].tolist()
    Trf2_data['max_loading_percent'] = df_Trafos_2['Maxima_carga[%]'].tolist()
    
    Tap_side_b = df_Trafos_2['Lado_tap'].tolist()
    Tap_side = []
    num_trf2 = len(Tap_side_b)
    for t in range(num_trf2):
        if Tap_side_b[t] == 0:
            Tap_side.append('hv')
        else:
            Tap_side.append('lv')
        
    Trf2_data['tap_side'] = Tap_side
    
    Trf2_data['tap_pos'] = df_Trafos_2['Pos_tap'].tolist()
    Trf2_data['tap_neutral'] = df_Trafos_2['N_tap'].tolist()
    Trf2_data['tap_max'] = df_Trafos_2['Max_tap'].tolist()
    Trf2_data['tap_min'] = df_Trafos_2['Min_tap'].tolist()
    Trf2_data['tap_step_per'] = df_Trafos_2['Aum_tap_v'].tolist()
    Trf2_data['tap_step_deg'] = df_Trafos_2['Aum_tap_deg'].tolist()
    
    
    outserv_trf2 = df_Trafos_2['Fuera_servicio'].tolist()    
    status = []
    num_trf2 = len(outserv_trf2)
    for t in range(num_trf2):
        status.append(1 - outserv_trf2[t])
        
    Trf2_data['In_service'] = status
    
    progress_bar.progress(50)
    
    
    # In[sheet:Trf3]
    
    Trf3_data = pd.DataFrame()
    
    # Filtrar datos nan
    
    nan_rows = df_Trafos_3[df_Trafos_3.isnull().any(1)]
    nan_index = nan_rows.index.tolist()
    
    for i in range(len(nan_index)):
        df_Trafos_3 = df_Trafos_3.drop([nan_index[i]],axis=0)
    
    num_trf3 = len(df_Trafos_3)
    Trf3_index = []
    
    for t in range(num_trf3):
        Trf3_index.append('TF3_{}'.format(t+1))
    
    Trf3_data[''] = Trf3_index
    Trf3_data['Name'] = df_Trafos_3['Nombre'].tolist()
    
    Trf3_bus_number_i = []
    Trf3_bus_number_j = []
    Trf3_bus_number_k = []
    Trf3_bus_name_i = df_Trafos_3['Subestacion_HV'].tolist()
    Trf3_bus_name_j = df_Trafos_3['Subestacion_MV'].tolist()
    Trf3_bus_name_k = df_Trafos_3['Subestacion_LV'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for t in Trf3_bus_name_i:
        index = bus_name.index(t)
        Trf3_bus_number_i.append(bus_number[index])
    
    for t in Trf3_bus_name_j:
        index = bus_name.index(t)
        Trf3_bus_number_j.append(bus_number[index])
    
    for t in Trf3_bus_name_k:
        index = bus_name.index(t)
        Trf3_bus_number_k.append(bus_number[index])
    
    Trf3_data['Bus_HV'] = Trf3_bus_number_i
    Trf3_data['Bus_MV'] = Trf3_bus_number_j
    Trf3_data['Bus_LV'] = Trf3_bus_number_k
    Trf3_data['sn_hv_mva'] = df_Trafos_3['S_nom_HV[MVA]'].tolist()
    Trf3_data['sn_mv_mva'] = df_Trafos_3['S_nom_MV[MVA]'].tolist()
    Trf3_data['sn_lv_mva'] = df_Trafos_3['S_nom_LV[MVA]'].tolist()
    Trf3_data['vn_hv_kv'] = df_Trafos_3['HV[kV]'].tolist()
    Trf3_data['vn_mv_kv'] = df_Trafos_3['MV[kV]'].tolist()
    Trf3_data['vn_lv_kv'] = df_Trafos_3['LV[kV]'].tolist()
    Trf3_data['R_hv_pu'] = df_Trafos_3['R_HV[pu]'].tolist()
    Trf3_data['R_mv_pu'] = df_Trafos_3['R_MV[pu]'].tolist()
    Trf3_data['R_lv_pu'] = df_Trafos_3['R_LV[pu]'].tolist()
    Trf3_data['X_hv_pu'] = df_Trafos_3['X_HV[pu]'].tolist()
    Trf3_data['X_mv_pu'] = df_Trafos_3['X_MV[pu]'].tolist()
    Trf3_data['X_lv_pu'] = df_Trafos_3['X_LV[pu]'].tolist()
    Trf3_data['vk_hv_percent'] = df_Trafos_3['Vk_HV[%]'].tolist()
    Trf3_data['vk_mv_percent'] = df_Trafos_3['Vk_MV[%]'].tolist()
    Trf3_data['vk_lv_percent'] = df_Trafos_3['Vk_LV[%]'].tolist()
    Trf3_data['vkr_hv_percent'] = df_Trafos_3['Vkr_HV[%]'].tolist()
    Trf3_data['vkr_mv_percent'] = df_Trafos_3['Vkr_MV[%]'].tolist()
    Trf3_data['vkr_lv_percent'] = df_Trafos_3['Vkr_LV[%]'].tolist()
    Trf3_data['pfe_hv_kw'] = df_Trafos_3['Perdidas_HV[kW]'].tolist()
    Trf3_data['pfe_mv_kw'] = df_Trafos_3['Perdidas_MV[kW]'].tolist()
    Trf3_data['pfe_lv_kw'] = df_Trafos_3['Perdidas_LV[kW]'].tolist()
    Trf3_data['i0_percent'] = [0]*num_trf3
    Trf3_data['shift_mv_degree'] = df_Trafos_3['Shift_MV[deg]'].tolist()
    Trf3_data['shift_lv_degree'] = df_Trafos_3['Shift_LV[deg]'].tolist()
    
    Tap_side_b = df_Trafos_3['Lado_tap'].tolist()
    Tap_side = []
    num_trf3 = len(Tap_side_b)
    for t in range(num_trf3):
        if Tap_side_b[t] == 0:
            Tap_side.append('hv')
        elif Tap_side_b[t] == 1:
            Tap_side.append('mv')
        else:
            Tap_side.append('lv')
        
    Trf3_data['tap_side'] = Tap_side
    
    Trf3_data['tap_pos'] = df_Trafos_3['Pos_tap'].tolist()
    Trf3_data['tap_neutral'] = df_Trafos_3['N_tap'].tolist()
    Trf3_data['tap_max'] = df_Trafos_3['Max_tap'].tolist()
    Trf3_data['tap_min'] = df_Trafos_3['Min_tap'].tolist()
    Trf3_data['tap_step_per'] = df_Trafos_3['Aum_tap_v'].tolist()
    Trf3_data['tap_step_deg'] = df_Trafos_3['Aum_tap_deg'].tolist()
    
    outserv_trf3 = df_Trafos_3['Fuera_servico'].tolist()    
    status = []
    num_trf3 = len(outserv_trf3)
    for t in range(num_trf3):
        status.append(1 - outserv_trf3[t])
        
    Trf3_data['In_service'] = status
    
    progress_bar.progress(55)
    
    # In[sheet:Switch]
    
    Switch_data = pd.DataFrame()
    
    nan_rows = df_Switches[df_Switches.isnull().any(1)]
    nan_index = nan_rows.index.tolist()
    
    for i in range(len(nan_index)):
        df_Switches = df_Switches.drop([nan_index[i]],axis=0)
    
    num_switch = len(df_Switches)
    switch_index = []
    
    for s in range(num_switch):
        switch_index.append('S{}'.format(s+1))
    
    Switch_data[''] = switch_index
    Switch_data['Name'] = df_Switches['Nombre'].tolist()
    
    Switch_bus_number_i = []
    Switch_bus_number_j = []
    Switch_bus_name_i = df_Switches['Subestacion_i'].tolist()
    Switch_bus_name_j = df_Switches['Subestacion_j'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for s in Switch_bus_name_i:
        index = bus_name.index(s)
        Switch_bus_number_i.append(bus_number[index])
    
    for s in Switch_bus_name_j:
        index = bus_name.index(s)
        Switch_bus_number_j.append(bus_number[index])
    
    Switch_data['Bus_from'] = Switch_bus_number_i
    Switch_data['Bus_to'] = Switch_bus_number_j
    
    Switches_types = df_Switches['Tipo'].tolist()
    Switch_type = []
    num_swwitches = len(Switches_types)
    for s in range(num_swwitches):
        if Switches_types[s] == 'Disconnector':
            Switch_type.append('DS')
        elif Switches_types[s] == 'Circuit-Breaker':
            Switch_type.append('CB')
        else:
            Switch_type.append('LS')
        
    Switch_data['Type'] = Switch_type
    
    Switch_data['V_nom'] = df_Switches['Vn[kV]'].tolist()
    Switch_data['I_nom'] = df_Switches['In[kA]'].tolist()
    Switch_data['R'] = df_Switches['R[Ohm]'].tolist()
    Switch_data['X'] = df_Switches['R[Ohm]'].tolist()
    Switch_data['Closed'] = df_Switches['Cerrado'].tolist()
    
    progress_bar.progress(60)
    # In[sheet:Branch]
    
    Branch_data = pd.DataFrame()
    
    Branch_index = []
    Branch_name = []
    Branch_from = []
    Branch_to = []
    Branch_R = []
    Branch_X = []
    Branch_Flowlimit = []
    Branch_status = []
    
    for l in range(num_line):
        Branch_index.append(Line_data.loc[l,''])
        Branch_name.append(Line_data.loc[l,'Name'])
        Branch_from.append(Line_data.loc[l,'Bus_from'])
        Branch_to.append(Line_data.loc[l,'Bus_to'])
        if Line_data.loc[l,'R'] == 0:        
            Branch_R.append(0.0000001)
            Branch_X.append(0.0000001)
        else:
            Branch_R.append(Line_data.loc[l,'R'])
            Branch_X.append(Line_data.loc[l,'X'])
        Branch_Flowlimit.append(Line_data.loc[l,'Flowlimit'])
        Branch_status.append(Line_data.loc[l,'In_service'])
    
    for r in range(num_reactor):
        Branch_index.append(Reactor_data.loc[r,''])
        Branch_name.append(Reactor_data.loc[r,'Name'])
        Branch_from.append(Reactor_data.loc[r,'Bus_from'])
        Branch_to.append(Reactor_data.loc[r,'Bus_to'])
        if Reactor_data.loc[r,'R'] == 0:        
            Branch_R.append(0.0000001)
            Branch_X.append(0.0000001)
        else:
            Branch_R.append(Reactor_data.loc[r,'R'])
            Branch_X.append(Reactor_data.loc[r,'X'])
        Branch_Flowlimit.append(Reactor_data.loc[r,'Flowlimit'])
        Branch_status.append(Reactor_data.loc[r,'In_service'])
    
    for tf2 in range(num_trf2):
        Branch_index.append(Trf2_data.loc[tf2,''])
        Branch_name.append(Trf2_data.loc[tf2,'Name'])
        Branch_from.append(Trf2_data.loc[tf2,'Bus_HV'])
        Branch_to.append(Trf2_data.loc[tf2,'Bus_LV'])
        if Trf2_data.loc[tf2,'R_pu'] == 0:
            Branch_R.append(0.0000001)
            Branch_X.append(0.0000001)     
        else:
            Branch_R.append(Trf2_data.loc[tf2,'R_pu'])
            Branch_X.append(Trf2_data.loc[tf2,'X_pu'])
        Branch_Flowlimit.append(Trf2_data.loc[tf2,'sn_mva'])
        Branch_status.append(Trf2_data.loc[tf2,'In_service'])
    
    for tf3 in range(num_trf3):
        Branch_index.append('{}(MV)'.format(Trf3_data.loc[tf3,'']))
        Branch_index.append('{}(LV)'.format(Trf3_data.loc[tf3,'']))
        Branch_name.append(Trf3_data.loc[tf3,'Name'])
        Branch_name.append(Trf3_data.loc[tf3,'Name'])
        Branch_from.append(Trf3_data.loc[tf3,'Bus_HV'])
        Branch_from.append(Trf3_data.loc[tf3,'Bus_HV'])
        Branch_to.append(Trf3_data.loc[tf3,'Bus_MV'])
        Branch_to.append(Trf3_data.loc[tf3,'Bus_LV'])
        if Trf3_data.loc[tf3,'R_hv_pu'] == 0:
            Branch_R.append(0.0000001)
            Branch_R.append(0.0000001)
        else:
            Branch_R.append(Trf3_data.loc[tf3,'R_hv_pu'])
            Branch_R.append(Trf3_data.loc[tf3,'R_lv_pu'])
        if Trf3_data.loc[tf3,'X_hv_pu'] == 0:
            Branch_X.append(0.0000001)
            Branch_X.append(0.0000001) 
        else:
            Branch_X.append(Trf3_data.loc[tf3,'X_hv_pu'])
            Branch_X.append(Trf3_data.loc[tf3,'X_lv_pu'])
        Branch_Flowlimit.append(Trf3_data.loc[tf3,'sn_hv_mva'])
        Branch_Flowlimit.append(Trf3_data.loc[tf3,'sn_lv_mva'])
        Branch_status.append(Trf3_data.loc[tf3,'In_service'])
        Branch_status.append(Trf3_data.loc[tf3,'In_service'])
    
    for s in range(num_switch):
        Branch_index.append(Switch_data.loc[s,''])
        Branch_name.append(Switch_data.loc[s,'Name'])
        Branch_from.append(Switch_data.loc[s,'Bus_from'])
        Branch_to.append(Switch_data.loc[s,'Bus_to'])
        Branch_R.append(Switch_data.loc[s,'R'])
        Branch_X.append(Switch_data.loc[s,'R'])
        Branch_Flowlimit.append(Switch_data.loc[s,'V_nom']*Switch_data.loc[s,'I_nom']*1.7320508075688)
        Branch_status.append(Switch_data.loc[s,'Closed'])
    
    Branch_data[''] = Branch_index
    Branch_data['Name'] = Branch_name
    Branch_data['from'] = Branch_from
    Branch_data['to'] = Branch_to
    Branch_data['R'] = Branch_R
    Branch_data['X'] = Branch_R
    Branch_data['Flowlimit'] = Branch_Flowlimit
    Branch_data['In_service'] = Branch_status
    
    #  Branch Map
    
    num_branch = len(Branch_data)
    
    Branch_map = {}
    
    for b in bus_number:
        Branch_map[b] = [0]*num_branch
    
    for b in range(num_branch):
        bus_num = Branch_data.loc[b,'from']
        Branch_map[bus_num][b] = 1
    
    for b in range(num_branch):
        bus_num = Branch_data.loc[b,'to']
        Branch_map[bus_num][b] = -1
    
    Branch_map_data = pd.DataFrame()
    Branch_map_data[''] = Branch_index
    
    for b in bus_number:
        Branch_map_data[b] = Branch_map[b]
    
    progress_bar.progress(65)
    
    # In[sheet:Demand]
    
    Demand_data = pd.DataFrame()
    
    num_demands = len(df_Loads)
    Demand_index = []
    
    for d in range(num_demands):
        Demand_index.append('D{}'.format(d+1))
    
    Demand_data[''] = Demand_index
    Demand_data['Name'] = df_Loads['Nombre'].tolist()
    
    Demand_bus_number_i = []
    Demand_bus_name_i = df_Loads['Subestacion'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for d in Demand_bus_name_i:
        index = bus_name.index(d)
        Demand_bus_number_i.append(bus_number[index])
    
    Demand_data['Bus_number'] = Demand_bus_number_i
    Demand_data['sn_mva'] = df_Loads['S[MVA]'].tolist()
    Demand_data['pf'] = df_Loads['PF[cos]'].tolist()
    Demand_data['p_mw'] = df_Loads['P[MW]'].tolist()
    Demand_data['q_mwar'] = df_Loads['Q[MW]'].tolist()
    Demand_data['In_service'] = [1]*num_demands

    progress_bar.progress(70)
    
    # In[sheet:Load_profile]
    
    Load_data = pd.DataFrame()
    
    Load_data[''] = Hour_index
    
    # Load profile
    
    Load_base_profile = [0.791231486,0.749314288,0.719626972,0.692725532,0.679000527,0.695543676,0.629328056,0.625875292,0.706590369,0.761232239,0.812507395,0.848680743,0.862018522,0.868192623,0.841043789,0.836407835,0.841753703,0.911400574,1,0.999214792,0.938441846,0.873753617,0.804192795,0.872548914]
    
    
    Load_bus_profile = {}
    
    for b in bus_number:
        Load_bus_profile[b] = [0]*Num_hours
        
    h_i = 0
    
    for h in range(Num_hours):
        for d in range(num_demands):
            b = Demand_data.loc[d,'Bus_number']
            Load_bus_profile[b][h] = Load_bus_profile[b][h] + (Demand_data.loc[d,'p_mw']*Load_base_profile[h])     
        if h_i == 23:
            h_i = 0
        else:
            h_i = h_i + 1
    
    for b in bus_number:
        Load_data[b] = Load_bus_profile[b]
    
    progress_bar.progress(75)
    
    # In[sheet:Capacitors]
    
    Capacitor_data = pd.DataFrame()
    
    num_capacitors = len(df_Capacitors)
    Capacitor_index = []
    
    for c in range(num_capacitors):
        Capacitor_index.append('C{}'.format(c+1))
    
    Capacitor_data[''] = Capacitor_index
    Capacitor_data['Name'] = df_Capacitors['Nombre'].tolist()
    
    Capacitor_bus_number_i = []
    Capacitor_bus_name_i = df_Capacitors['Subestacion'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for c in Capacitor_bus_name_i:
        index = bus_name.index(c)
        Capacitor_bus_number_i.append(bus_number[index])
    
    Capacitor_data['Bus_number'] = Capacitor_bus_number_i
    
    Capacitors_type = df_Capacitors['Tipo'].tolist()
    Cap_type = []
    num_capacitors = len(Capacitors_type)
    for c in range(num_capacitors):
        if Capacitors_type[c] == 1:
            Cap_type.append('Reactor')
        else:
            Cap_type.append('Capacitor')
    
    Capacitor_data['type'] = Cap_type
    Capacitor_data['vn_kv'] = df_Capacitors['Vn[kV]'].tolist()
    Capacitor_data['p_mw'] = df_Capacitors['P[MW]'].tolist()
    Capacitor_data['q_mwar'] = df_Capacitors['Q[MVAr]'].tolist()
    Capacitor_data['Max_step'] = df_Capacitors['Max_step'].tolist()
    Capacitor_data['Actual_step'] = df_Capacitors['Actual_step'].tolist()
    
    outserv_cap = df_Capacitors['Fuera_servicio'].tolist()    
    status = []
    num_cap = len(outserv_cap)
    for c in range(num_cap):
        status.append(1 - outserv_cap[c])
        
    Capacitor_data['In_service'] = status
    
    progress_bar.progress(80)
    
    # In[sheet:Red_ext]
    
    Red_data = pd.DataFrame()
    
    num_Red = len(df_Red_ext)
    Red_index = []
    
    for r in range(num_Red):
        Red_index.append('RE{}'.format(r+1))
    
    Red_data[''] = Red_index
    Red_data['Name'] = df_Red_ext['Nombre'].tolist()
    
    Red_bus_number_i = []
    Red_bus_name_i = df_Red_ext['Subestacion'].tolist()
    bus_number = Bus_data['Bus_number'].tolist()
    bus_name = Bus_data['Bus_name'].tolist()
    
    for r in Red_bus_name_i:
        index = bus_name.index(r)
        Red_bus_number_i.append(bus_number[index])
    
    Red_data['Bus_number'] = Red_bus_number_i
    Red_data['p_max_mw'] = df_Red_ext['P_max[MW]'].tolist()
    Red_data['p_min_mw'] = df_Red_ext['P_min[MW]'].tolist()
    Red_data['q_max_mvar'] = df_Red_ext['Q_max[MVAr]'].tolist()
    Red_data['q_min_mvar'] = df_Red_ext['Q_min[MVAr]'].tolist()
    
    progress_bar.progress(85)
    
    
    # In[sheet:Reserve]
    
    R1 = [273,273,273,273,273,300,300,300,307,285,285,285,285,273,285,281,285,309,399,316,357,319,320,316]
    R2 = [273,273,273,273,273,273,273,282,273,278,273,273,273,273,273,273,273,295,323,273,273,293,287,280]
    R3 = [273,273,273,273,273,273,273,273,273,273,273,273,273,273,273,273,273,273,322,317,287,273,273,273]
    
    Reserve_data = pd.DataFrame()
    Reserve_data[''] = Hour_index
    Reserve_data['R1'] = R1
    Reserve_data['R2'] = R2
    Reserve_data['R3'] = R3
    
    progress_bar.progress(90)
    
    # In[sheet:BESS]
    
    ESS_data = pd.DataFrame()
    ESS_data[''] = ['E1']
    ESS_data['Name'] = ['SAEB1']
    ESS_data['Bus_number'] = [0]
    ESS_data['C_Potencia'] = [0]
    ESS_data['C_Energia'] = [0]
    ESS_data['n_ch_eff'] = [0.9]
    ESS_data['n_dc_eff'] = [0.9]
    ESS_data['Self_discharge'] = [0.0000625]
    ESS_data['SOC_min'] = [0.2]
    ESS_data['SOC_ini'] = [0.2]
    ESS_data['CB'] = [20]
    ESS_data['N_ciclos'] = [5000]
    ESS_data['C_Rba'] = [50000]
    
    # In[sheet:BESS_map]
    
    num_ESS = len(ESS_data)
    ESS_index = ESS_data[''].tolist()
    
    ESS_map = {}
    
    for b in bus_number:
        ESS_map[b] = [0]*num_ESS
    
    ESS_map_data = pd.DataFrame()
    ESS_map_data[''] = ESS_index
    
    for b in bus_number:
        ESS_map_data[b] = ESS_map[b]
        
    # In[sheet:BESS_price]
    
    CB_base = [86.75502172,86.75502172,86.75502172,86.75502172,81.65565355,41.00800746,20.27183631,20.27183631,32.57905437,73.91846325,86.75502172,86.75502172,86.75502172,86.75502172,86.75502172,86.75502172,86.75502172,86.75502172,86.75502172,95.40659971,95.40659971,86.75502172,86.75502172,86.75502172]
    
    ESS_price_data = pd.DataFrame()
    ESS_price_data[''] = Hour_index
    ESS_price_data['CB_MWh'] = CB_base
    
    # In[sheet:Demand_response]
    
    Demand_response_data = pd.DataFrame()
    
    Demand_response_data[''] = Hour_index
    
    Demand_bus_profile = {}
    
    for b in bus_number:
        Demand_bus_profile[b] = [0]*Num_hours
        
    for b in bus_number:
        Demand_response_data[b] = Demand_bus_profile[b]
    
    progress_bar.progress(95)
    
    
    # In[sheet:ExportData]
    
    with pd.ExcelWriter('{}/Casos de estudio/DIgSILENT_System_output_data.xlsx'.format(Current_direction)) as writer:
        System_data.to_excel(writer, sheet_name='System_data', index = False)
        Bus_data.to_excel(writer, sheet_name='Bus', index = False)
        Gen_data.to_excel(writer, sheet_name='SM_Unit', index = False)
        Gen_map_data.to_excel(writer, sheet_name='SM_map', index = False)
        Renewable_gen_data.to_excel(writer, sheet_name='Renewable', index = False)
        Line_data.to_excel(writer, sheet_name='Lines', index = False)
        Trf2_data.to_excel(writer, sheet_name='Trafos_2', index = False)
        Trf3_data.to_excel(writer, sheet_name='Trafos_3', index = False)
        Switch_data.to_excel(writer, sheet_name='Switches', index = False)
        Branch_data.to_excel(writer, sheet_name='Branch', index = False)
        Branch_map_data.to_excel(writer, sheet_name='Branch_map', index = False)
        Demand_data.to_excel(writer, sheet_name='Load_data', index = False)
        Load_data.to_excel(writer, sheet_name='load', index = False)
        Reserve_data.to_excel(writer, sheet_name='Reserve', index = False)
        Capacitor_data.to_excel(writer, sheet_name='Shunt_data', index = False)
        Reactor_data.to_excel(writer, sheet_name='Reactor_data', index = False)
        ESS_data.to_excel(writer, sheet_name='ESS_Unit', index = False)
        ESS_map_data.to_excel(writer, sheet_name='ESS_map', index = False)
        ESS_price_data.to_excel(writer, sheet_name='ESS_Energy_price', index = False)
        Red_data.to_excel(writer, sheet_name='Red_ext_data', index = False)
        Demand_response_data.to_excel(writer, sheet_name='C_DR_load', index = False)
        Demand_response_data.to_excel(writer, sheet_name='PDR', index = False)
        writer.save()
        writer.close()
    
    progress_bar.progress(100)
    
    st.markdown("## Datos generales del sistema:")
    st.write(System_data)
    st.markdown("## Nodos del sistema:")
    st.write(Bus_data)
    st.markdown("## Generadores del sistema:")
    st.write(Gen_data)
    st.markdown("## Lineas del sistema:")
    st.write(Line_data)
    st.markdown("## Transformadores bi-devanados del sistema:")
    st.write(Trf2_data)
    st.markdown("## Transformadores Tri-devanados del sistema:")
    st.write(Trf3_data)
    st.markdown("## Switches del sistema:")
    st.write(Switch_data)
    st.markdown("## Carga del sistema:")
    st.write(Demand_data)
    st.markdown("## Compesadores del sistema:")
    st.write(Capacitor_data)
    # st.markdown("## Reactores del sistema:")
    # st.write(Reactor_data)
    st.markdown("## SAE del sistema:")
    st.write(ESS_data)
    st.markdown("## Redes externas del sistema:")
    st.write(Red_data)
    
    


# In[Main]

def main_DigSilent_lecture():
    
    st.markdown("<h1 style='text-align: center; color: black;'> Lectura archivos DIgSILENT </h1>", unsafe_allow_html=True)
    st.markdown("En esta sección de la herramienta el usuario podrá realizar la \
    lectura de sistemas desarrollados en DigSilent y pasarlos al formato de lectura de esta aplicación\
    .")
    
    # Direccion de archivo
    
    Current_direction = os.getcwd()
    
    # Pasos para la lectura de archivos en DigSilent
    
    st.markdown("Para realizar la lectura de manera correcta siga los siguientes pasos:")
    
    st.markdown("1 - Cargue el comando DPL en el caso de estudio que desea hacer la lectura:")
    
    info = st.checkbox("Mostrar código DPL")
    if info:
        codigo = open('{}/Datos_DigSilent/DIgSILENT_datacode.txt'.format(Current_direction),'r')
        mensaje = codigo.read()
        codigo.close()
        st.code(mensaje)
    
    st.markdown("2 - Ejecute la lectura de los archivos en la herramienta")
    
    st.markdown("3 - Cuando la herramienta lo indique ejecute el comando DPL en DIgSILENT")
        
    # Funcion para ejecutar el despacho optimo de AGC
    
    button_sent = st.sidebar.button("Ejecutar")
    
    if button_sent:
        progress_bar = lectura_DigSilent(Current_direction)
        transform_DigSilent_data(Current_direction,progress_bar)
