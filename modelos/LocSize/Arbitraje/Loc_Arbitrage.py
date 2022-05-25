# -*- coding: utf-8 -*-
"""
Dimensionamiento de los SAE para arbitraje de energía

@author: Andrés Felipe Peñanrada Bayona
"""

# In[Librerias]:
    
import streamlit as st
from pyomo.environ import *
from pyomo import environ as pym
from pyomo import kernel as pmo
import pandas as pd
from pandas import ExcelWriter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
import os
import urllib.request
import requests
from bs4 import BeautifulSoup
import re
import random
from datetime import date
from datetime import datetime
from datetime import timedelta
from array import *

# Import functions
from funciones.nuevo_bess import bat_param
from funciones.read_download_files import *
import funciones.pyomoio as po
import funciones.pydataxm as pxm

import base64
from io import BytesIO

# In[Price_data]

def update_system_data(date1,date2,TRM):
    
    with st.spinner('Obteniedo datos de precios de XM'):
            
        consult = pxm.ReadDB()
        df1 = consult.request_data("PrecBolsNaci", 0, date1, date2)
        
        df1 = df1.drop(['Id','Values_code','Date'], axis=1)
        PBN = []
        for d in df1.index:
            for h in df1.columns:
                PBN.append((float(df1.loc[d,h])*1000)/TRM)
        
    return PBN

# In[Modelo]

def dim_arbitrage(MPO,SAEB,Pro_time,solver,Current_direction):
    
    with st.spinner('Ejecutando el modelo de optimización'):
        
        Horizon_time = len(MPO)  
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # MODEL TYPE
        #-------------------------------------------------------------------------------------------------------------------------------------------
         
        ## Simulation type
        model = ConcreteModel()
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # SETS
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        model.t = RangeSet(0,len(MPO)-1)                                           # Horizon simulation time 
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # PARAMETERS
        #------------------------------------------------------------------------------------------------------------------------------------------- 
        
        def Power_cost(model,t):
            return MPO[t]
        model.C_E = Param(model.t, initialize = Power_cost)                        # Cost of energy [USD/MWh]
        
        # //------//------//------//------//------//------//------//------//------//------//------//------//------//------//------//------//------
    
        model.n_ch = Param(initialize=SAEB[0])                                      # Charge efficency of ESS [p.u.]
        model.n_dc = Param(initialize=SAEB[1])                                      # Discharge efficency of ESS [p.u.]
        model.s_dc = Param(initialize=SAEB[2])                                      # Self-Discharge efficency of ESS [p.u.]
        model.deg_rate = Param(initialize = SAEB[3])                                # Degradation rate of ESS at 80% of DoD [p.u.]
        model.deg_slope = Param(initialize = SAEB[3]/0.8)                           # Slope of cycle life loss 
        model.SOC_max = Param(initialize=SAEB[4])                                   # Maximum State of Charge of ESS  [p.u.]
        model.SOC_min = Param(initialize=SAEB[5])                                   # Minimum State of Charge of ESS  [p.u.]
        model.SOC_ini = Param(initialize=SAEB[6])                                   # Initial State of Charge of ESS  [p.u.]
        model.CC_P = Param(initialize=SAEB[7])                                      # Cost of Power capacity [USD/MW]
        model.CC_E = Param(initialize=SAEB[8])                                      # Cost of Energy capacity [USD/MWh]
        model.P_limit = Param(initialize=SAEB[9])                                   # Maximum size of PCS  [MW]
        model.E_limit = Param(initialize=SAEB[10])                                  # Maximum size of Storage  [MWh]
        model.C_limit = Param(initialize=SAEB[11])                                  # Maximum cost of Storage  [USD]
        Delta_t = 1
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # VARIABLES
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        model.E_max = Var(domain = NonNegativeReals, initialize=0)                  # Energy max capacity [MWh]
        model.P_max = Var(domain = NonNegativeReals, initialize=0)                  # Power max capacity [MW] 
        model.Pot_ch = Var(model.t, domain=NonNegativeReals)                        # Power in battery charge [MW]
        model.Pot_dc = Var(model.t, domain=NonNegativeReals)                        # Power in battery discharge [MW]
        model.u_ch = Var(model.t, within=Binary, initialize=0)                      # Status of battery charge {Binary}
        model.u_dc = Var(model.t, within=Binary, initialize=0)                      # Status of battery discharge [Binary]
        model.E = Var(model.t, domain=NonNegativeReals)                             # Energy of battery [MWh]
        model.E_cap = Var(model.t, domain=NonNegativeReals)                         # Energy capacity of battery [MWh]
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # OBJETIVE FUNCTION
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        def Cost_rule(model):
            return sum(model.C_E[t]*(model.Pot_dc[t] - model.Pot_ch[t]) for t in model.t) - sum((model.CC_E*model.deg_slope*model.Pot_dc[t]*Delta_t/(model.n_dc*0.2)) for t in model.t) - (((model.CC_E*model.E_max) + (model.CC_P*model.P_max))*Horizon_time/(Pro_time*8760))
        model.Objetivo = Objective(rule = Cost_rule, sense=maximize)
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # CONSTRAINS
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        ## ESS constraints
        
        def power_c_max_rule(model,t):
            return model.Pot_ch[t] <= model.P_limit*model.u_ch[t]
        model.power_c_max = Constraint(model.t, rule=power_c_max_rule)
        
        def power_c_max_rule_2(model,t):
            return model.Pot_ch[t] <= model.P_max
        model.power_c_max_2 = Constraint(model.t, rule=power_c_max_rule_2)
        
        def power_d_max_rule(model,t):
            return model.Pot_dc[t] <= model.P_limit*model.u_dc[t]
        model.power_d_max = Constraint(model.t, rule=power_d_max_rule)
        
        def power_d_max_rule_2(model,t):
            return model.Pot_dc[t] <= model.P_max
        model.power_d_max_2 = Constraint(model.t, rule=power_d_max_rule_2)
        
        def sim_rule(model,t):
            return model.u_ch[t] + model.u_dc[t] <= 1
        model.sim = Constraint(model.t, rule=sim_rule)
        
        def energy_rule(model,t):
            if t == model.t.first():
                return model.E[t] == (model.E_max*model.SOC_ini) + (model.n_ch*model.Pot_ch[t]) - ((model.Pot_dc[t])/model.n_dc) 
            else:
                return model.E[t] == (model.E[t-1]*(1-model.s_dc)) + (model.n_ch*model.Pot_ch[t]) - (model.Pot_dc[t]/model.n_dc)    
        model.energy = Constraint(model.t, rule=energy_rule)
        
        # Energy limits
            
        def energy_limit_max_rule(model,t):
            return model.E[t] <= model.E_cap[t]*model.SOC_max 
        model.energy_limit_max = Constraint(model.t, rule=energy_limit_max_rule)
        
        def energy_limit_max_rule2(model):
            return model.E_max <= model.E_limit 
        model.energy_limit_max2 = Constraint(rule=energy_limit_max_rule2)
        
        def energy_limit_min_rule(model,t):
            return model.E[t] >= model.E_max*model.SOC_min
        model.energy_limit_min = Constraint(model.t, rule=energy_limit_min_rule)
        
        # ESS Degradation
        
        def Degradacion_rule(model,t):
            if t == model.t.first():
                return model.E_cap[t] == model.E_max
            else:           
                return model.E_cap[t] == model.E_cap[t-1] - (model.deg_slope*model.Pot_dc[t-1]*Delta_t/model.n_dc)
        model.ESS_Deg = Constraint(model.t, rule = Degradacion_rule)
        
        # Economic constraints:
        
        def CAPEX_max_rule(model):
            return ((model.CC_E*model.E_max) + (model.CC_P*model.P_max)) <= model.C_limit
        model.CAPEX_max = Constraint(rule=CAPEX_max_rule)
        
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # SOLVER CONFIGURATION
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        def pyomo_postprocess(options=None, instance=None, results=None):
            model.Objetivo.display()
            
        
        if solver == 'CPLEX':
            from pyomo.opt import SolverFactory
            import pyomo.environ
            opt = SolverManagerFactory('neos')
            results = opt.solve(model, opt='cplex')
            #sends results to stdout
            results.write()
            print("\nDisplaying Solution\n" + '-'*60)
            pyomo_postprocess(None, model, results)
        else:
            from pyomo.opt import SolverFactory
            import pyomo.environ
            opt = SolverFactory('glpk')
            results = opt.solve(model)
            results.write()
            print("\nDisplaying Solution\n" + '-'*60)
            pyomo_postprocess(None, model, results)
               
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # OUTPUT DATA
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        Output_data = {}
        
        for o in model.component_data_objects(Objective):
            Output_data[str(o)] = pd.DataFrame()
            Output_data[str(o)]['value'] = [o()]
        
        for v in model.component_objects(Var):
            sets = po._get_onset_names(v)
            if len(sets) >= 2:
                Output_data[str(v)] = po.get_entity(model, str(v)).unstack()
            elif len(sets) == 1:
                Output_data[str(v)] = po.get_entity(model, str(v))
            else:
                Output_data[str(v)] = pd.DataFrame()
                Output_data[str(v)]['value'] = [v.value]
                
        
        for p in model.component_objects(Param):
            sets = po._get_onset_names(p)
            if len(sets) >= 2:
                Output_data[str(p)] = po.get_entity(model, str(p)).unstack()
            elif len(sets) == 1:
                Output_data[str(p)] = po.get_entity(model, str(p))
            else:
                Output_data[str(p)] = pd.DataFrame()
                Output_data[str(p)]['value'] = [p.value]
        
        with pd.ExcelWriter('{}/Resultados/Size_BESS_Arbitrage.xlsx'.format(Current_direction)) as writer:
            for idx in Output_data.keys():
                Output_data[idx].to_excel(writer, sheet_name= idx, index = True)
            writer.save()
            # writer.close()
        
        # OTROS
        
        Output_data['i_arb'] = sum(model.C_E[t]*(model.Pot_dc[t].value - model.Pot_ch[t].value) for t in model.t) 
        Output_data['c_deg'] = sum((model.CC_E*model.deg_slope*model.Pot_dc[t].value*Delta_t/(model.n_dc*0.2)) for t in model.t)
        Output_data['c_cap_eq'] = (((model.CC_E*Horizon_time)/(Pro_time*8760))*model.E_max.value) + (((model.CC_P*Horizon_time)/(Pro_time*8760))*model.P_max.value)
        Output_data['c_cap'] = (model.CC_E*model.E_max.value) + (model.CC_P*model.P_max.value)
        
        return Output_data

# In[Graficas]

def figures(Output_data):
    
    
    fig_ESS_operation = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ESS_operation.add_trace(go.Scatter(x=Output_data['E'].index.tolist(), y=Output_data['E'].values, name="Energía almacenada",line_shape='linear'),secondary_y=False)      
    fig_ESS_operation.add_trace(go.Scatter(x=Output_data['Pot_ch'].index.tolist(), y=Output_data['Pot_ch'].values, name="Carga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)      
    fig_ESS_operation.add_trace(go.Scatter(x=Output_data['Pot_dc'].index.tolist(), y=Output_data['Pot_dc'].values, name="Descarga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)
    fig_ESS_operation.update_layout(title_text="Operación del SAE",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0, traceorder='reversed'),autosize=True,plot_bgcolor='rgba(0,0,0,0)')
    fig_ESS_operation.update_xaxes(title_text='Tiempo [h]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_ESS_operation.update_yaxes(title_text='Energia [MWh]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=False)
    fig_ESS_operation.update_yaxes(title_text='Potencia [MW]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=True)
    
    st.write(fig_ESS_operation)
    
    fig_ESS_operation_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ESS_operation_price.add_trace(go.Scatter(x=Output_data['C_E'].index.tolist(), y=Output_data['C_E'].values, name="Costo de energía",line_shape='vh'),secondary_y=False)      
    fig_ESS_operation_price.add_trace(go.Scatter(x=Output_data['Pot_ch'].index.tolist(), y=Output_data['Pot_ch'].values, name="Carga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)      
    fig_ESS_operation_price.add_trace(go.Scatter(x=Output_data['Pot_dc'].index.tolist(), y=Output_data['Pot_dc'].values, name="Descarga", line=dict(dash='dot'),line_shape='vh'),secondary_y=True)
    fig_ESS_operation_price.update_layout(title_text="Operación del SAE VS Costo de energía",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0, traceorder='reversed'),autosize=True,plot_bgcolor='rgba(0,0,0,0)')
    fig_ESS_operation_price.update_xaxes(title_text='Tiempo [h]', showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_ESS_operation_price.update_yaxes(title_text='Costo de energía [USD/MWh]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=False)
    fig_ESS_operation_price.update_yaxes(title_text='Potencia [MW]',showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True,secondary_y=True)
    
    st.write(fig_ESS_operation_price)
    
    fig_E_rem = go.Figure()
    fig_E_rem.add_trace(go.Scatter(x=Output_data['E_cap'].index.tolist(), y=Output_data['E_cap'].values, name='Energía remanente',line_shape='linear'))
    fig_E_rem.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
    fig_E_rem.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_E_rem.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_E_rem.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Capacidad de almacenamiento de energía',xaxis_title='Tiempo [h]',yaxis_title='Energía remanente [MWh]')
    
    st.write(fig_E_rem)

# In[Math]

def math_formulation():
    return r"""
    ## Función Objetivo
    Máximizar los ingresos del Sistema de Almacenamiento de Energía (SAE) por medio del arbitraje de energía eléctrica.
    $$ \begin{aligned}
        \max F = & \underbrace{\sum_{t \in \mathcal{T}}C_{t}^{energy}\cdot (p_{t}^{dc} - p_{t}^{ch})}_{Ingresos\hspace{1mm}por\hspace{1mm}arbitraje}\\\\
        &\quad  - \underbrace{\sum_{t \in \mathcal{T}}\frac{\mu \cdot p_{t}^{dc} \cdot \Delta t}{(1-eol) \cdot \eta^{dc}} \cdot C^{storage}}_{Costos\hspace{1mm}por\hspace{1mm}degradación\hspace{1mm}del\hspace{1mm}SAE}\\\\
        &\quad  - \underbrace{P^{SAEB} \cdot \frac{C^{pcs} \cdot S_{time}}{L_{time}} + E^{SAEB} \cdot \frac{C^{storage} \cdot S_{time}}{L_{time}}}_{Costo\hspace{1mm}equivalente\hspace{1mm}por\hspace{1mm} SAE} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $C_{t}^{energy}$ corresponde al costo de la energía en el instante de tiempo $t$; $p_{t}^{dc}$ y $p_{t}^{ch}$ son la potencia de descarga y carga del SAE, respectivamente;
    $\mu$ es la tasa de degradación del SAE; $eol$ correspone a la capacidad de almacenamiento de energía remanente máxima, en pu, antes de tener que reemplazar el almacenador;
    $P^{SAEB}$ es la capacidad óptima del Sistema de Conversión de Potencia (PCS, por sus siglas en inglés) del SAE dada en MW; $E^{SAEB}$ es la capacidad de almacenamiento de energía óptima dada en MWh; $C^{pcs}$ y $C^{storage}$ corresponden a
    los costos del PCS y del almacenador, respectivamente; $L_{time}$ es el tiempo de estimado de operación del proyecto en horas y $S_{time}$ corresponde al total de horas del horizonte de simulación
    
    ## Restricciones
    ### Restricciones sistemas de almacenamiento de energí­a basados en baterías
    #### Lí­mite de potencia de los SAEB
    
    $$ \begin{aligned}
        p_{t}^{ch} \leq P^{limit} \cdot u_{t}^{ch} \hspace{2mm} \forall t \in \mathcal{T} 
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        p_{t}^{ch} \leq P^{SAEB} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        p_{t}^{dc} \leq P^{limit} \cdot u_{t}^{dc} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        p_{t}^{dc} \leq P^{SAEB} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $P^{limit}$ hace referencia a la capacidad máxima del sistema del conversión de potencia; $u_{t}^{ch}$ y $u_{t}^{dc}$ corresponden a señales binarias las cuales indican
    la carga y descarga del SAE, respectivamente.
    
    #### Variables binarias de estado de los SAEB
    
    $$ \begin{aligned}
        u_{t}^{ch} + u_{t}^{dc} \leq 1 \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    #### Relación entre la potencia y energía de los SAEB
    
    $$ \begin{aligned}
        e_{t} = e_{t-1}\cdot (1 - \eta^{SoC}) + \left( \eta^{ch} \cdot p_{t}^{ch} -\frac{p_{t}^{dc}}{\eta^{dc}} \right)\cdot \Delta t \hspace{2mm} \forall t \in \mathcal{T} \hspace{10mm}
    \end{aligned}$$
    
    donde $e_{t}$ corresponde al nivel de energía almacenada en el SAE en el tiempo $t$ dado en MWh; $\eta^{SoC}$ es la tasa de auto-descarga horaria; $\eta^{ch}$ y $\eta^{dc}$ son las eficiencias
    de carga y descarga del SAE, respectivamente.
    
    #### Lí­mite de energí­a de los SAEB
    
    $$ \begin{aligned}
        E^{SAEB} \leq E^{limit}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        e_{t} \leq E_{t}^{cap} \cdot SOC^{max} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        e_{t} \geq E^{SAEB} \cdot SOC^{min} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $E^{limit}$ hace referencia a la capacidad máxima del almacenador en MWh; $E_{t}^{cap}$ es la capacidad de almacenamiento remanente del SAE en el tiempo $t$;
    $SOC^{max}$ y $SOC^{min}$ corresponden al estado de carga máximo y mínimo del SAE.  
    #### Degradación de los SAEB
    
    $$ \begin{aligned}
        E_{t}^{cap} = E_{t-1}^{cap} - \left(\frac{\mu \cdot p_{t}^{dc} \cdot \Delta t}{\eta^{dc}} \right) \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    """
   
 

# In[Main]

def main_Dim_Arbitrage(data1,Current_direction):
    
    with st.expander("Ver formulación del problema"):
        st.write(math_formulation())
        st.write("")
                
    st.markdown("## Parámetros seleccionados para la simulación") 
    st.sidebar.markdown("### Ingrese los parámetros del SAE a simular")
    
    # SelecciÃ³n de tecnlogÃ­a de SAE
    Eff,degra,autoD,DoD,costP,costE,a,b,ciclos = bat_param(data1,1)
    
    limits = st.sidebar.selectbox('Seleccione el tipo de limitación para el dimensionamiento:', ('Sin limitación','Potencia', 'Presupuesto','Área'))
    
    if limits == 'Sin limitación':
        Plimit = 1000000
        Elimit = 1000000
        Climit = 10000000000000000
    elif limits == 'Potencia':
        Plimit = st.sidebar.number_input("Ingrese el capacidad máxima de transporte de energía [MW] : ",min_value=0.0)
        Elimit = 4*Plimit
        Climit = 10000000000000000
    elif limits == 'Presupuesto':
        Plimit = 1000000
        Elimit = 1000000
        Climit = st.sidebar.number_input("Ingrese el costo capital máximo del SAE [USD] : ",min_value=0.0)
    else:
        Container_energy_capacity = 4
        Container_area = 28.1
        size_factor = (Container_energy_capacity/Container_area)
        user_area = st.sidebar.number_input("Ingrese el área máxima disponible para el SAE [m^2] : ",min_value=0.0)
        Elimit = user_area*size_factor
        Plimit = Elimit
        Climit = 10000000000000000
    
    Pro_time = st.sidebar.number_input("Ingrese el tiempo estimado de operación del proyecto [año(s)] : ",min_value=1,max_value=30)
    
    SAEB = []
    
    SAEB.append(math.sqrt(Eff))
    SAEB.append(math.sqrt(Eff))
    SAEB.append(autoD)
    SAEB.append(degra)
    SAEB.append(1)
    SAEB.append(1-DoD)
    SAEB.append(1-DoD)
    SAEB.append(costP)
    SAEB.append(costE)
    SAEB.append(Plimit)
    SAEB.append(Elimit)
    SAEB.append(Climit)
    
    # Horizonte de simulacion
    
    st.sidebar.markdown("### Ingrese los parámetros de la simulación")
    
    Country = st.sidebar.selectbox('Seleccione la fuente de información de las variables del mercado de energía:', ('XM (Colombia)', 'Otro'))
    
    if Country == 'XM (Colombia)':
        
        Fechas = pd.DataFrame()
        index_fechas = ['Fecha de inicio de la simulación:', 'Fecha final', 'Total dias']
        values_fechas = []
        
        date1 = st.sidebar.date_input('Fecha de inicio de la simulación:',value = date.today() - timedelta(days=2), max_value = date.today() - timedelta(days=2))
        date2 = st.sidebar.date_input('Fecha final de la simulación:',value = date.today() - timedelta(days=2), min_value = date1, max_value = date.today() - timedelta(days=2))    
        
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
     
        
    elif Country == 'Otro':
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        file_prices = st.sidebar.file_uploader("Seleccione el archivo con historicos de precios de bolsa (USD/MWh):", type=["csv","xlsx"])
    
    # Seleccionar solver
    
    st.sidebar.markdown("### Simulación")
    
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')
    if solver=='CPLEX':
        st.write("* El solucionador seleccionado es: "+solver)
    else:
        st.write("* El solucionador seleccionado es: "+solver)
    
    # Direccion de archivo
    
    Current_direction = os.getcwd()
    
    # Funcion para ejecutar el despacho optimo de AGC
    
    button_sent = st.sidebar.button("Simular")
    
    if button_sent:
        
        st.markdown("## Simulación")
        
        if Country == 'XM (Colombia)':
            MPO = update_system_data(date1,date2,TRM_final)
        elif Country == 'Otro':
            df_values = pd.read_excel(file_prices, sheet_name='Precios', header=0, index_col=0)
            MPO = df_values['Price']
        
        Output_data = dim_arbitrage(MPO,SAEB,Pro_time,solver,Current_direction)
        
        st.markdown("### Resultados:")
        st.write('El tamaño óptimo del SAE es:  {} [MW] / {} [MWh]'.format(Output_data['P_max'].loc[0,'value'],Output_data['E_max'].loc[0,'value']))
        st.write('Ingresos netos:  {} [USD]'.format(Output_data['Objetivo'].loc[0,'value']))
        st.write('Ingresos por arbitraje:  {} [USD]'.format(Output_data['i_arb']))
        st.write('Costos por degradación del SAE:  {} [USD]'.format(Output_data['c_deg']))
        st.write('Costo capital equivalente del SAE:  {} [USD]'.format(Output_data['c_cap_eq']))
        st.write('Costo capital real del SAE:  {} [USD]'.format(Output_data['c_cap']))
        
        st.markdown("### Figuras:")
        
        figures(Output_data)
            
