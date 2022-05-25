# -*- coding: utf-8 -*-
"""
Operación optmia del SAE para arbitraje de energía

@author: Andrés Felipe Peñanranda Bayona
"""

# In[Librerias]:
    
import streamlit as st
from pyomo.environ import *
from pyomo import environ as pym
from pyomo import kernel as pmo
import funciones.pyomoio as po
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

from funciones.read_download_files import *
import funciones.pydataxm as pxm
from funciones.nuevo_bess import bat_param
from funciones.save_files import*

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

def dim_arbitrage(MPO,SAEB,solver,Current_direction):
    
    with st.spinner('Ejecutando el modelo de optimización'):
    
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # MODEL TYPE
        #-------------------------------------------------------------------------------------------------------------------------------------------
         
        ## Simulation type
        model = ConcreteModel()
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # SETS
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        D_segments = []
        for i in range(len(SAEB[11])-1):
            D_segments.append('S{}'.format(i))
            
        model.t = RangeSet(0,len(MPO)-1)                                           # Horizon simulation time 
        model.d = Set(initialize=D_segments, ordered=True)                     # Segments of degradation curve
        
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
        model.P_max = Param(initialize=SAEB[9])                                     # Maximum size of PCS  [MW]
        model.E_max = Param(initialize=SAEB[10])                                    # Maximum size of Storage  [MWh]
        model.E_ini = Param(initialize=SAEB[6]*SAEB[10])                            # Initial State of Charge of BESS
        model.eol = Param(initialize=SAEB[13])                                      # End of life [p.u.]
                        
        Delta_t = 1
        
        # //------//------//------//------//------//------//------//------//------//------//------//------//------//------//------//------//------
        
        def DOD_seg_max(model,d):
            return SAEB[11][model.d.ord(d)]
        model.DOD_seg_max = Param(model.d, initialize = DOD_seg_max)
        
        def DOD_seg_min(model,d):
            return SAEB[11][model.d.ord(d)-1]
        model.DOD_seg_min = Param(model.d, initialize = DOD_seg_min)
        
        def Degracion_pendiente_seg(model,d):
            if model.d.ord(d) == 1:
                return 0
            else:
                return (SAEB[12][model.d.ord(d)] - SAEB[12][model.d.ord(d)-1])/(SAEB[11][model.d.ord(d)] - SAEB[11][model.d.ord(d)-1])
        model.Degra_m_seg = Param(model.d, initialize = Degracion_pendiente_seg)
        
        def Degracion_b_seg(model,d):
            if model.d.ord(d) == 1:
                return 0
            else:
                return SAEB[12][model.d.ord(d)-1] + ((SAEB[12][model.d.ord(d)] - SAEB[12][model.d.ord(d)-1])/(SAEB[11][model.d.ord(d)] - SAEB[11][model.d.ord(d)-1]))*-SAEB[11][model.d.ord(d)-1]
        model.Degra_b_seg = Param(model.d, initialize = Degracion_b_seg)
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # VARIABLES
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        model.Pot_ch = Var(model.t, domain=NonNegativeReals)                        # Power in battery charge [MW]
        model.Pot_dc = Var(model.t, domain=NonNegativeReals)                        # Power in battery discharge [MW]
        model.u_ch = Var(model.t, within=Binary, initialize=0)                      # Status of battery charge {Binary}
        model.u_dc = Var(model.t, within=Binary, initialize=0)                      # Status of battery discharge [Binary]
        model.E = Var(model.t, domain=NonNegativeReals)                             # Energy of battery [MWh]
        model.E_cap = Var(model.t, domain=NonNegativeReals)                         # Energy capacity of battery [MWh]
        model.Deg_ba_rate = Var(model.t, domain = NonNegativeReals)                 # Tasa de degradacion del ESS para el instante t
        model.DOD_seg = Var(model.t, model.d, domain=NonNegativeReals)              # DOD dispatch segment k of BESS at time t [p.u]
        model.DOD_seg_b = Var(model.t, model.d, within=Binary)                      # Senal binaria segmento de la curva de degradacion 
            
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # OBJETIVE FUNCTION
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        def Cost_rule(model):
            return sum(model.C_E[t]*(model.Pot_dc[t] - model.Pot_ch[t]) for t in model.t) - sum((model.CC_E*model.Deg_ba_rate[t]/(1-model.eol)) for t in model.t)
        model.Objetivo = Objective(rule = Cost_rule, sense=maximize)
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # CONSTRAINS
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        ## ESS constraints
        
        def power_c_max_rule(model,t):
            return model.Pot_ch[t] <= model.P_max*model.u_ch[t]
        model.power_c_max = Constraint(model.t, rule=power_c_max_rule)
        
        def power_d_max_rule(model,t):
            return model.Pot_dc[t] <= model.P_max*model.u_dc[t]
        model.power_d_max = Constraint(model.t, rule=power_d_max_rule)
        
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

        
        def energy_limit_min_rule(model,t):
            return model.E[t] >= model.E_max*model.SOC_min
        model.energy_limit_min = Constraint(model.t, rule=energy_limit_min_rule)
        
        # ESS Degradation
        
        def Degradacion_rule(model,t):
            if t == model.t.first():
                return model.E_cap[t] == model.E_max
            else:           
                return model.E_cap[t] == model.E_cap[t-1] - (model.Deg_ba_rate[t]*model.E_ini)
        model.ESS_Deg = Constraint(model.t, rule = Degradacion_rule)
    
        
        # Segmentos de la curva de degradacion BESS
        
        def DOD_seg_base_min_rule1(model,t,d):
            return model.DOD_seg[t,d]>=model.DOD_seg_min[d]*model.DOD_seg_b[t,d]
        model.DOD_seg_lim1 = Constraint(model.t, model.d, rule=DOD_seg_base_min_rule1)

        def DOD_seg_base_max_rule2(model,t,d):
            return model.DOD_seg[t,d]<=model.DOD_seg_max[d]*model.DOD_seg_b[t,d]
        model.DOD_seg_lim2 = Constraint(model.t, model.d, rule=DOD_seg_base_max_rule2)
        
        def DOD_seg_bin_sum_rule2(model,t):
            return sum(model.DOD_seg_b[t,d] for d in model.d) <= 1
        model.DOD_seg_bin_lim2 = Constraint(model.t, rule=DOD_seg_bin_sum_rule2)

        def DOD_sum_rule(model,t):
            return (model.Pot_dc[t]*Delta_t)/(model.E_ini*model.n_dc) == sum(model.DOD_seg[t,d] for d in model.d)
        model.DOD_sum = Constraint(model.t, rule=DOD_sum_rule)
        
        def Deg_rate_rule(model,t):
            return model.Deg_ba_rate[t] == sum(((model.DOD_seg[t,d]*model.Degra_m_seg[d]) + (model.DOD_seg_b[t,d]*model.Degra_b_seg[d])) for d in model.d)
        model.Deg_rate = Constraint(model.t, rule=Deg_rate_rule)
    
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------
        # SOLVER CONFIGURATION
        #-------------------------------------------------------------------------------------------------------------------------------------------
        
        def pyomo_postprocess(options=None, instance=None, results=None):
            model.Objetivo.display()
            
        
        if solver == 'CPLEX':
            from pyomo.opt import SolverFactory
            import pyomo.environ
            opt = SolverFactory()
            results = opt.solve(model)
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
        
        
        with pd.ExcelWriter('{}/Resultados/Op_ESS_Arbitrage.xlsx'.format(Current_direction)) as writer:
            for idx in Output_data.keys():
                Output_data[idx].to_excel(writer, sheet_name= idx, index = True)
            writer.save()
            # writer.close()
        
        # OTROS
        
        Output_data['i_arb'] = sum(model.C_E[t]*(model.Pot_dc[t].value - model.Pot_ch[t].value) for t in model.t) 
        Output_data['c_deg'] = sum((model.CC_E*model.Deg_ba_rate[t].value/(1-model.eol)) for t in model.t)
        
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
    Máximizar los ingresos del SAE por medio del arbitraje de energía eléctrica.
    $$ \begin{aligned}
        \max F = & \underbrace{\sum_{t \in \mathcal{T}}C_{t}^{energy}\cdot (p_{t}^{dc} - p_{t}^{ch})}_{Ingresos\hspace{1mm}por\hspace{1mm}arbitraje}\\\\
        &\quad  - \underbrace{\sum_{t \in \mathcal{T}}\frac{\beta_{t}}{(1-eol)} \cdot C^{storage}}_{Costos\hspace{1mm}por\hspace{1mm}degradación\hspace{1mm}del\hspace{1mm}SAE}\\\\
    \end{aligned}$$
    
    donde $C_{t}^{energy}$ corresponde al costo de la energía en el instante de tiempo $t$; $p_{t}^{dc}$ y $p_{t}^{ch}$ son la potencia de descarga y carga del SAE, respectivamente;
    $\mu$ es la tasa de degradación del SAE; $eol$ correspone a la capacidad de almacenamiento de energía remanente máxima, en pu, antes del reemplazo del almacenador.
    
    ## Restricciones
    ### Restricciones sistemas de almacenamiento de energí­a basados en baterías
    #### Lí­mite de potencia de los SAEB
    
    $$ \begin{aligned}
        p_{t}^{ch} \leq P^{limit} \cdot u_{t}^{ch} \hspace{2mm} \forall t \in \mathcal{T} 
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        p_{t}^{dc} \leq P^{limit} \cdot u_{t}^{dc} \hspace{2mm} \forall t \in \mathcal{T}
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
        e_{t} \leq E_{t}^{cap} \cdot SOC^{max} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        e_{t} \geq E^{SAEB} \cdot SOC^{min} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $E^{limit}$ hace referencia a la capacidad máxima del almacenador en MWh; $E_{t}^{cap}$ es la capacidad de almacenamiento remanente del SAE en el tiempo $t$;
    $SOC^{max}$ y $SOC^{min}$ corresponden al estado de carga máximo y mínimo del SAE.  
    #### Degradación de los SAEB
        
    $$ \begin{aligned}
        DoD_{t,d} \leq DoD_{d}^{max} \cdot S_{t,d}^{DoD}  \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        DoD_{t,d} \geq DoD_{d}^{min} \cdot S_{t,d}^{DoD}  \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$ 
    
    $$ \begin{aligned}
        \frac{p_{t}^{dc} \cdot \Delta t}{E^{SAEB} \cdot \eta^{dc}} = \sum_{d \in \mathcal{D}}DoD_{t,d} \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \sum_{d \in \mathcal{D}}S_{t,d}^{DoD} \leq 1 \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        \beta_{t} = \sum_{d \in \mathcal{D}}\left(\beta_{d}^{slope} \cdot DoD_{t,d}\right) + \left(\beta_{d}^{constant} \cdot S_{t,d}^{DoD}\right) \hspace{2mm} \forall t \in \mathcal{T}, d \in \mathcal{D}
    \end{aligned}$$
    
    $$ \begin{aligned}
        E_{t}^{cap} = E_{t-1}^{cap} - \beta_{t} \cdot E^{SAEB} \hspace{2mm} \forall t \in \mathcal{T}
    \end{aligned}$$
    
    donde $DoD_{t,d}$ corresponde a la profundidad de descarga del SAEB en el tiempo $t$ del segmento de la curva de degradación $d$; $S_{t,d}^{DoD}$ es la variable binaria
    que indica si la profundidad de descarga se encuentra en el segmento $d$ de la curva de degradación durante el tiempo $t$; $\beta_{d}^{slope}$ y $\beta_{d}^{constant}$
    corresponden a la pendiente y la constante de la linealización de la curva de degradación en el segmento $d$, respectivamente; $\beta_{t}$ es la tasa de pérdida de capacidad del almacenador del SAE.
    
    """

def degradation_curve(a,b,DoD):
    return a*(DoD**(-b))

def cycle_life_loss_curve(a,b,DoD,eol):
    return (1-eol)/(degradation_curve(a,b,DoD))

def degradation_curve_figure(a,b,size):
    
    DoD = np.linspace(0,1,10)
    
    fig_Deg_curve = go.Figure()
    fig_Deg_curve.add_trace(go.Scatter(x=DoD, y=degradation_curve(a,b,DoD), name='Degradacion',line_shape='linear'))
    fig_Deg_curve.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
    fig_Deg_curve.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    fig_Deg_curve.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
    if size == 0:
        fig_Deg_curve.update_layout(autosize=False, width=350, height=200, margin=dict(l=10, r=10, b=10, t=15), font={'size': 10})
    else:
        fig_Deg_curve.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Curva de degradacion',xaxis_title='DoD [p.u.]',yaxis_title='NC [Ciclos]')
    
    return fig_Deg_curve
        
 

# In[Main]

def main_Op_Arbitrage(data1,Current_direction):
    
    
    with st.expander("Ver formulación del problema"):
        st.write(math_formulation())
        st.write("")
                
    st.markdown("## Parámetros seleccionados para la simulación")
    st.markdown("El SAE seleccionado tiene las siguientes caracterá­sticas:")  
    st.sidebar.markdown("### Ingrese los parámetros del SAEB a simular")
    
    # SelecciÃ³n de tecnlogÃ­a de SAE
    
    Eff,degra,autoD,DoD,costP,costE,a,b,ciclos = bat_param(data1,1)
    eol = 0.8
    DoD_values = np.linspace(0,1,10)
    Degradation_curve_x = DoD_values.tolist()
    Degradation_curve_y = cycle_life_loss_curve(a,b,DoD_values,eol).tolist()
    
    P_SAE = st.sidebar.number_input("Ingrese el tamaño del PCS [MW] : ")
    E_SAE = st.sidebar.number_input("Ingrese el tamaño del Almacenamiento [MWh] : ")
    
    SAEB = []
    
    SAEB.append(Eff)
    SAEB.append(Eff)
    SAEB.append(autoD)
    SAEB.append(degra)
    SAEB.append(1)
    SAEB.append(1-DoD)
    SAEB.append(1-DoD)
    SAEB.append(costP)
    SAEB.append(costE)
    SAEB.append(P_SAE)
    SAEB.append(E_SAE)
    SAEB.append(Degradation_curve_x)
    SAEB.append(Degradation_curve_y)
    SAEB.append(eol)
    
    # Horizonte de simulacion
    
    st.sidebar.markdown("### Ingrese los parámetros de la simulación")
    
    Country = st.sidebar.selectbox('Seleccione el país en el cual desea realizar la simulación:', ('', 'XM (Colombia)', 'Otro'))
    
    if Country == 'XM (Colombia)':
        
        Fechas = pd.DataFrame()
        index_fechas = ['Fecha de inicio', 'Fecha final', 'Total dias']
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
            TRM_final_1="La TRM seleccionada para la simulación es de: "+TRM_final_x + " COP/USD"
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
        st.write("El solucionador seleccionado es: "+solver)
    else:
        st.write("El solucionador seleccionado es: "+solver)
    
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
        
        Output_data = dim_arbitrage(MPO,SAEB,solver,Current_direction)
        
        st.markdown("### Resultados:")
        st.write('Ingresos netos:  {} [USD]'.format(Output_data['Objetivo'].loc['1','1']))
        st.write('Ingresos por arbitraje:  {} [USD]'.format(Output_data['i_arb']))
        st.write('Costos por degradación del SAE:  {} [USD]'.format(Output_data['c_deg']))
        
        st.markdown("### Figuras:")
        
        figures(Output_data)