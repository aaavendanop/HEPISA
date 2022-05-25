import streamlit as st
from os import path
from os import remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import requests
from bs4 import BeautifulSoup
import re

from despacho import *

def dashboard_XMDoc(data1, s_study):

    if s_study == 'Versión 1':

        st.markdown("<h1 style='text-align: center; color: black;'>Despacho Colombiano uninodal con SAE (Versión 1)</h1>", unsafe_allow_html=True)
        # image = Image.open('arbitrage.jpeg')
        # st.image(image, caption='', use_column_width=True)
        st.markdown('En esta sección de la herramienta el usuario podrá analizar la \
        operación de un SAE bajo el esquema actual de despacho económico uninodal de XM. \
        La formulación esta basada en la versión incial del documento de XM, publicado \
        en el mes de Noviembre de 2019.')
        st.markdown('## Parámetros seleccionados para la simulación')
        st.sidebar.markdown('### Ingrese los parámetros de simulación')

    elif s_study == 'Versión 2':

        st.markdown("<h1 style='text-align: center; color: black;'>Despacho Colombiano uninodal con SAE (Versión 2)</h1>", unsafe_allow_html=True)
        # image = Image.open('arbitrage.jpeg')
        # st.image(image, caption='', use_column_width=True)
        st.markdown('En esta sección de la herramienta el usuario podrá analizar la \
        operación de un SAE bajo el esquema actual de despacho económico uninodal de XM. \
        La formulación esta basada en la segunda versión del documento de XM, publicado \
        en el mes de Septiembre de 2020.')
        st.markdown('## Parámetros seleccionados para la simulación')
        st.sidebar.markdown('### Ingrese los parámetros de simulación')

# Selección de tecnlogía de SAE
    technology = st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE', data1.index, key='1')
    if technology == 'New':
        st.markdown('Ingrese las características del SAE a simular:')
        Eff = st.text_input('Ingrese la eficiencia del SAE [pu]: ', key='1')
        degra = st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ', key='2')
        autoD = st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ', key='3')
        DoD = st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ', key='4')
        costP = st.text_input('Ingrese el costo por potencia [USD/MW]: ', key='5')
        costE = st.text_input('Ingrese el costo por energía [USD/MWh]: ', key='6')
    else:
        st.markdown('El SAE seleccionado tiene las siguientes características:')
        st.write(data1.loc[technology])
        Eff = data1.iloc[data1.index.get_loc(technology), 0]
        degra = data1.iloc[data1.index.get_loc(technology), 1]
        autoD = data1.iloc[data1.index.get_loc(technology), 2]
        DoD = data1.iloc[data1.index.get_loc(technology), 3]
        costP = data1.iloc[data1.index.get_loc(technology), 4]
        costE = data1.iloc[data1.index.get_loc(technology), 5]

    # Selección fecha
    fecha = st.sidebar.date_input('Fecha de simulación')
    st.write('La fecha de simulación seleccionada fue: ' + str(fecha))

    # Descarga y lectura de archivos
    file_ofertas, file_demanda, file_MPO = descarga_archivos(fecha)


#Ingrese el tamaño del SAE
    Pot_max = st.sidebar.number_input('Potencia [MW]')
    E_max = st.sidebar.number_input('Energía [MWh]')
    st.write('Potencia del SAE seleccionada: ' + str(Pot_max) + ' [MW]')
    st.write('Energía del SAE seleccionada: ' + str(E_max) + ' [MWh]')

# SoC MT
    if s_study == 'Versión 2':
        SoC_MT = st.sidebar.number_input('Estado de carga, mínimo técnico')
        st.write('Mínimo técnico seleccionado: '+ str(SoC_MT) + ' [p.u.]')


# tiempos de descarga previo y tiempos de descarga
    tdp = st.sidebar.number_input('Tiempo de descarga previo', min_value=1, max_value=25)
    td = st.sidebar.number_input('Tiempo de descarga', min_value=1, max_value=25)
    st.write('Hora(s) de descarga previa seleccionada: ' + str(tdp) + ' [h]')
    st.write('Hora(s) de descarga seleccionada: ' + str(td) + ' [h]')

# Ingresar tiempo de simulación
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')

# Seleccionar solver
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')
    if solver=='CPLEX':
        st.write('El solucionador seleccionado es: '+solver)
    else:
        st.write('El solucionador seleccionado es: '+solver)

    if s_study == 'Versión 1':

        if st.checkbox('Formulación Matemática'):
            st.write('### Función Objetivo')
            st.write(r"""
                    $$ \begin{aligned}
                        \min{} \sum_{t} \sum_{R} Pofe_{[R]}\cdot V_{GenRec[R][t]} +
                        \sum_t \sum_{RT} PAP_{[RT]}\cdot B_{Arr[RT][t]} + \sum_t CROest\cdot V_{Rac[t]} + \\
                        \sum_t \sum_s PC_{[s]}{[t]}\cdot V_{PC[s][t]} +
                        \sum_{tpd} \sum_{s} CROest\cdot V_{DoC[s][tpd]}
                    \end{aligned} $$
                    """)
            st.write('### Restricciones')
            st.write('#### Balance de generación demanda considerando almacenamiento')
            st.write(r"""
                    $$ \sum_R V_{GenRec[R][t]} + V_{Rac[t]} + \sum_s V_{PD[s][t]} = Dem_{[t]} +
                    \sum_s V_{PC[s][t]} \hspace{2mm} \forall t $$ """)
            st.write('#### Balance del almacenamiento')
            st.write(r"""
                    $$ V_{SoC[s][t]} = V_{SoC[s][t-1]}\cdot \eta_{SoC[s]} +
                    \eta_{c[s]}\cdot \dfrac{V_{PC[s][t]}}{Cap_{[s]}} - \dfrac{1}{\eta_{d[s]}} \cdot
                    \dfrac{V_{PD[s][t]}}{Cap_{[s]}} \hspace{2mm} \forall s, t $$ """)
            st.write('#### Balance de Estado de Carga')
            st.write(r"""
                    $$ V_{DoC[s][t]} = SoC_{max[s]} - V_{SoC[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Capacidad máxima de almacenamiento')
            st.write(r"""
                    $$ SoC_{min[s]} \leq V_{SoC[s][t]} \leq SoC_{max[s]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Causalidad de la carga/descarga')
            st.write(r"""
                    $$ B_{PC[s][t]} + B_{PD[s][t]} \leq 1 \hspace{2mm} \forall
                    s, t $$ """)
            st.write(r"""
                    $$ 0 \leq V_{PC[s][t]} \leq PC_{max[s][t]} \cdot B_{PC[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write(r"""
                    $$ 0 \leq V_{PD[s][t]} \leq PD_{max[s][t]} \cdot B_{PD[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Carga y descarga requerida')
            st.write(r"""
                    $$ V_{PD[s][td]} = PD_{des[s][t]} \hspace{2mm} \forall
                    s, td $$ """)
            st.write(r"""
                    $$ V_{PC[s][td]} \geq PC_{des[s][td]} \hspace{2mm} \forall
                    s, td $$ """)

    elif s_study == 'Versión 2':

        if st.checkbox('Formulación Matemática'):
            st.write('### Función Objetivo')
            st.write(r"""
                    $$ \begin{aligned}
                        \min{} \sum_{t} \sum_{r} Pofe_{[R]}\cdot V_{GenRec[r][t]} +
                        \sum_t \sum_{rt} PAP_{[rt]}\cdot B_{Arr[rt][t]} + \sum_t CROest\cdot V_{Rac[t]} + \\
                        \sum_t \sum_s PC_{[s]}{[t]}\cdot V_{PC[s][t]} +
                        \sum_{tpd} \sum_{s} CROest\cdot Cap_{[s][tpd]} \cdot V_{SoD[s][tpd]}
                    \end{aligned} $$
                    """)
            st.write('### Restricciones')
            st.write('#### Balance de generación demanda considerando almacenamiento')
            st.write(r"""
                    $$ \sum_r V_{GenRec[r][t]} + V_{Rac[t]} + \sum_s V_{PD[s][t]} \cdot ECS_{[s][t]} =
                    Dem_{[t]} + \sum_s V_{PC[s][t]} \cdot ECS_{[s][t]} \hspace{2mm} \forall t $$
                    """)
            st.write('#### Balance del almacenamiento')
            st.write(r"""
                    $$ V_{SoC[s][t]} = V_{SoC_E[s][t-1]} + ECS_{[s][t]} \cdot \left(
                    \eta_{c[s]}\cdot \dfrac{V_{PC[s][t]}}{Cap_{[s]}} - \dfrac{1}{\eta_{d[s]}} \cdot
                    \dfrac{V_{PD[s][t]}}{Cap_{[s]}} \right) \hspace{2mm} \forall s, t $$
                    """)
            st.write('#### Afectación del estado de carga por eficiencia de almacenamiento')
            st.write(r"""
                    $$ -(B_{PC[s][t]} + B_{PD[s][t]}) + V_{SoC[s][t]}\cdot (1 - \eta_{SoC[s]})
                    \leq V_{SoC_E[s][t]} \hspace{2mm} \forall s, t $$
                    """)
            st.write(r"""
                    $$ V_{SoC[s][t]} \cdot (1 - \eta_{SoC[s]}) + (B_{PC[s][t]} + B_{PD[s][t]})
                    \geq V_{SoC_E[s][t]} \hspace{2mm} \forall s, t $$
                    """)
            st.write(r"""
                    $$ -(1 - B_{PC[s][t]}) + V_{SoC[s][t]} \leq V_{SoC_E[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write(r"""
                    $$ V_{SoC[s][t]} + (1 - B_{PC[s][t]}) \geq V_{SoC_E[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write(r"""
                    $$ -(1 - B_{PD[s][t]}) + V_{Soc[s][t]} \leq V_{SoC_E[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write(r"""
                    $$ V_{SoC[s][t]} + (1 - B_{PD[s][t]}) \geq V_{SoC_E[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Balance de Estado de Carga')
            st.write(r"""
                    $$ V_{SoD[s][t]} = 1 - V_{SoC[s][t]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Mínimo y máximo Estado de Carga del almacenamiento')
            st.write(r"""
                    $$ SoC_{min[s]} \leq V_{SoC[s][t]} \leq SoC_{max[s]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Mínimo técnico del sistema de almacenamiento')
            st.write(r"""
                    $$ V_{SoC[s][t]} \geq SoC_{MT[s]} \hspace{2mm} \forall
                    s, t $$ """)
            st.write('#### Causalidad de la carga/descarga')
            st.write(r"""
                    $$ B_{PC[s][t]} + B_{PD[s][t]} \leq 1 \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$ """)
            st.write(r"""
                    $$ 0 \leq V_{PC[s][t]} \leq PC_{max[s][t]} \cdot B_{PC[s][t]} \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$ """)
            st.write(r"""
                    $$ 0 \leq V_{PD[s][t]} \leq PD_{max[s][t]} \cdot B_{PD[s][t]} \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$ """)
            st.write('#### Carga y descarga requerida')
            st.write(r"""
                    $$V_{PD[s][t]} = \begin{cases}
                    PD_{req[s][tr]} & \forall s, tr, ECS_{[s][t]} = 1 \\
                    0 & \forall s, t \neq tr, ECS_{[s][t]} = 1
                    \end{cases}$$
                    """)
            st.write(r"""
                    $$ V_{PC[s][td]} \geq PC_{des[s][td]} \hspace{2mm} \forall
                    s, td $$ """)

        # st.write("""
        #     Función Objetivo:
        #     $\sum_{t}\sum_{R} = Pofe_{[R]}\cdot V_GenRec_{[R][t]}$
        #         """)

# Correr función de optimización

    if s_study == 'Versión 1':

        def run_despacho():

            opt_results=opt_despacho(fecha, file_ofertas, file_demanda, file_MPO, Pot_max, E_max, Eff, Eff, autoD, 1-DoD, tdp, td, solver)
            #Imprimir resultados
            st.title('Resultados:')
            SOC, tiempo = opt_results
            st.write('Tiempo de simulación: ' + str(tiempo))

            axis_font = {'fontname':'Microsoft Sans Serif', 'size':'12'}

            fig, ax = plt.subplots()
            ax.plot(SOC)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9 ])
            ax.legend(['Estado de Carga'], bbox_to_anchor=(0.65, -0.18), fancybox=True, shadow=True, ncol=10)
            plt.xticks(**axis_font)
            plt.yticks(**axis_font)
            plt.grid(True)
            plt.xlabel('Tiempo [h]', **axis_font)
            plt.ylabel('[MWh]', **axis_font)
            st.pyplot(fig)

    elif s_study == 'Versión 2':

        def run_despacho():

            opt_results = opt_despacho_2(fecha, file_ofertas, file_demanda, Pot_max, E_max, Eff, Eff, autoD, 1-DoD, SoC_MT, tdp, td, solver)
            #Imprimir resultados
            st.title('Resultados:')
            SOC, tiempo = opt_results
            st.write('Tiempo de simulación: ' + str(tiempo))

            axis_font = {'fontname':'Microsoft Sans Serif', 'size':'12'}

            fig, ax = plt.subplots()
            ax.plot(SOC)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9 ])
            ax.legend(['Estado de Carga'], bbox_to_anchor=(0.65, -0.18), fancybox=True, shadow=True, ncol=10)
            plt.xticks(**axis_font)
            plt.yticks(**axis_font)
            plt.grid(True)
            plt.xlabel('Tiempo [h]', **axis_font)
            plt.ylabel('[MWh]', **axis_font)
            st.pyplot(fig)

#############Simulation button###############################################
    button_sent = st.sidebar.button('Simular')
    if button_sent:
        # if path.exists('Resultados/resultados_size_loc.xlsx'):
        #     remove('Resultados/resultados_size_loc.xlsx')
        run_despacho()
        