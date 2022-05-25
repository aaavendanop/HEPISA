# coding: utf-8
#Librería de intersaz
import streamlit as st
# Librerías para maniular datos
from os import path
from os import remove
# Librearias para manejo de datos
import pandas as pd
import numpy as np
# Librarías para graficas
import matplotlib.pyplot as plt
import pydeck as pdk
# Libraries for reading data form internet
import requests
from bs4 import BeautifulSoup
import re
# Importing optimization functions
from Loc_dim_OpC import opt_dim

def dashboard_DLOp(data1, info_, ope_fact, con_fact):

    if info_ == 'Ope':

        st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento \
        y localización de SAE basado en reducción de costos de operación</h1>", unsafe_allow_html=True)
        # image = Image.open('arbitrage.jpeg')
        # st.image(image, caption='', use_column_width=True)
        st.markdown('En esta sección de la herramienta el usuario podrá encontrar el \
        tamaño óptimo de SAE para cada nodo del sistema (en términos de potencia y energía) \
        con el fin de reducir los costos de operación. Para este fin, el usuario deberá ingresar un archivo con \
        los datos del sistema de potencia a analizar, el horizonte de tiempo de la simulación,\
        la tecnología de SAE y el tipo de solver.')

    elif info_ == 'Con':

        st.markdown("<h1 style='text-align: center; color: black;'> Dimensionamiento y localización de SAE basado en reducción \
            de costos por congestión </h1>", unsafe_allow_html=True)

    elif info_ == 'Both':

        st.markdown("<h1 style='text-align: center; color: black;'> Dimensionamiento y localización de SAE basado en reducción \
            de costos de operación y costos por congesstión</h1>", unsafe_allow_html=True)

    st.markdown('## Parámetros seleccionados para la simulación')
    st.sidebar.markdown('### Ingrese los parámetros de simulación')

# Selección de tecnlogía de SAE
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data1.index,key='1')
    if technology=='New':
        st.markdown('Ingrese las características del SAEB a simular:')
        Eff=st.text_input('Ingrese la eficiencia del SAEB [pu]: ',key='1')
        degra=st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ',key='2')
        autoD=st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ',key='3')
        DoD=st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ',key='4')
        costP=st.text_input('Ingrese el costo por potencia [USD/MW]: ',key='5')
        costE=st.text_input('Ingrese el costo por energía [USD/MWh]: ',key='6')
    else:
        st.markdown('El SAE seleccionado tiene las siguientes características:')
        st.dataframe(data1.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))
        Eff=data1.iloc[data1.index.get_loc(technology),0]
        degra=data1.iloc[data1.index.get_loc(technology),1]
        autoD=data1.iloc[data1.index.get_loc(technology),2]
        DoD=data1.iloc[data1.index.get_loc(technology),3]
        costP=data1.iloc[data1.index.get_loc(technology),4]
        costE=data1.iloc[data1.index.get_loc(technology),5]

# Seleción de archivo con precios
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_system = st.sidebar.file_uploader('Seleccione el archivo con el sistema a simular:', type=['csv','xlsx'])

# Costo por Congestion

    con_cost = 0

    if info_ == 'Con' or info_ == 'Both':
        con_cost = st.sidebar.number_input('Ingrese el costo por congestión [$/MWh]')
        st.write('El costo por congestión seleccionado fue: ' + str(con_cost) + ' [$/MWh]')

# Ingresar tiempo de simulación
    time_sim = st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')

# Seleccionar solver
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')
    if solver=='CPLEX':
        st.write('El solucionador seleccionado es: '+solver)
    else:
        st.write('El solucionador seleccionado es: '+solver)

# Formulación

    if st.checkbox('Formulación Matemática'):

        if info_ == 'Ope':

            st.write('### Función Objetivo')
            st.write(r"""
                    $$ \begin{aligned}
                        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                        \left( p_{i,b,t}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
                        C_{i}^{up}\cdot SD_{i,t}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +
                        \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}} \sum_{t \in \mathcal{T}}
                        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \\
                        \underbrace{\sum_{n\in \mathcal{N}} \sum_{b\in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}
                    \end{aligned}$$
                """)

        if info_ == 'Con':

            st.write('### Función Objetivo')
            st.write(r"""
                    $$ \begin{aligned}
                        \min \underbrace{\sum_{(b, r) \in \mathcal{L}}
                        \sum_{t \in \mathcal{T}} \left(\left[p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf}\right]
                        - P_{b,r}^{max} \right) \cdot C_{t}^{con}}_{Costos\hspace{1mm}por\hspace{1mm}congestión}
                    \end{aligned}$$
                """)

        if info_ == 'Both':

            st.write('### Función Objetivo')
            st.write(r"""
                    $$ \begin{aligned}
                        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                        \left( p_{i,b,t}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
                        C_{i}^{up}\cdot SD_{i,t}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +
                        \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}} \sum_{t \in \mathcal{T}}
                        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \\
                        \underbrace{\sum_{n\in \mathcal{N}} \sum_{b\in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} + \underbrace{\sum_{(b, r) \in \mathcal{L}}
                        \sum_{t \in \mathcal{T}} \left(\left[p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf}\right]
                        - P_{b,r}^{max} \right) \cdot C_{t}^{con}}_{Costos\hspace{1mm}por\hspace{1mm}congestión}
                    \end{aligned}$$
                """)

        st.write('### Restricciones')
        st.write('#### Restricciones del sistema')
        st.write('Balance de potencia')
        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}}p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}}p_{j,b,t}^{hyd} +
                    \sum_{w \in \mathcal{W}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}
                    \left(p_{b,r,t}^{pf} + \frac{1}{2}q_{b,r,t}^{pf}\right) \\ + \sum_{n\in \mathcal{N}_b}
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{f} \hspace{2mm} \forall
                    b \in \mathcal{B}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write('#### Restricciones Generación térmica')
        st.write('Límites en la capacidad de generación térmica')
        st.write(r"""
                $$ P_{i}^{min} \leq p_{i,b,t}^{th} \leq P_{i}^{max} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$ """)
        st.write('Rampas de generadores térmicos')
        st.write(r"""
                $$ p_{i,t+1}^{th} - p_{i,t}^{th} \leq R_{i}^{up} \cdot x_{i,t} t + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$ """)
        st.write(r"""
                $$ p_{i,t}^{th} - p_{i,t+1}^{th} \leq R_{i}^{dn} \cdot x_{i,t} t + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$ """)
        st.write('Variables binarias de operación de unidades térmicas')
        st.write(r"""
                $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$ """)
        st.write(r"""
                $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$ """)
        st.write('Tiempos mínimos de encendido y apagado de generadores térmicos')
        st.write(r"""
                $$ x_{i,t} = g_{i,t}^{on/off} \hspace{2mm} \forall
                t \in \left(L_{i}^{up,min}+L_{i}^{dn,min}\right), i \in \mathcal{I} $$ """)
        st.write(r"""
                $$ \sum_{tt=t-g_{i}^{up}+1}^{t} SU_{i,tt} \leq x_{i,tt} \hspace{2mm} \forall
                t \geq L_{i}^{up,min} $$ """)
        st.write(r"""
                $$ \sum_{tt=t-g_{i}^{dn}+1}^{t} SD_{i,tt} \leq 1-x_{i,tt} \hspace{2mm} \forall
                t \geq L_{i}^{dn,min} $$ """)
        st.write('#### Restricciones Generación hidráulica')
        st.write('Límites en la capacidad de generación hidráulica')
        st.write(r"""
                $$ P_{j}^{min} \leq p_{j,b,t}^{hyd} \leq P_{j}^{max} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$ """)
        st.write('Unidades hidráulicas de generación')
        st.write(r"""
                $$ Q_{j}^{min} \leq q_{j,t} \leq Q_{j}^{max} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T}$$
                """)
        st.write(r"""
                $$ V_{j}^{min} \leq v_{j,t} \leq V_{j}^{max} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ 0 \leq s_{j,t} \leq Q_{j,t} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ v_{j,t} = v_{j,t-1} + 3600 \Delta t \left(I_{t} - \sum_{j \in \mathcal{J}} q_{j,t} -
                s_{j,t} \right) \hspace{2mm} j \in \mathcal{J}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ P_{j,t}^{hyd} = H_{j} \cdot q_{j,t} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$ """)
        st.write('#### Restricciones generación renovable')
        st.write('Curvas de generación de unidades renovables')
        st.write('Límites de generación en unidades renovables')
        st.write(r"""
                $$ p_{w,b,t}^{ren} \leq P_{w,t}^{f} \hspace{2mm} \forall
                w \in \mathcal{W}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)
        st.write('#### Restricciones flujo de potencia DC y pérdidas')
        st.write('Cálculo del flujo de potencia por cada línea')
        st.write(r"""
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
                \left(b,r \right) \in \mathcal{L}, t \in \mathcal{T} $$ """)
        st.write('Cálculo de las pérdidas eléctricas de cada línea')
        st.write(r"""
                $$ q_{b,r,t}^{pf} = G_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right)^2 \hspace{2mm}
                \forall \left(s,r,l \right) \in \mathcal{L}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ \delta_{b,r}^{+} + \delta_{b,r}^{-} = \sum_{k=1}^{K} \delta_{b,r}(k) \hspace{2mm}
                k = 1,...,K $$ """)
        st.write(r"""
                $$ lpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}
                k = 1, ... , K $$ """)
        st.write(r"""
                $$ q_{b,r,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \alpha_{b,r}(k)\cdot \delta_{b,r}(k)
                \hspace{2mm} \forall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$ """)
        st.write('Límites en el flujo de potencia en las líneas')
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} + \frac{1}{2} \cdot q_{b,r,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$ """)
        st.write('#### Restricciones sistemas de almacenamiento de energía basados en baterías')
        st.write('Variables binarias de estado de los SAEB')
        st.write(r"""
                $$ u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \forall
                n \in \mathcal{N}, t \in \mathcal{T} $$ """)
        st.write('Relación entre la potencia y energía de los SAEB')
        st.write(r"""
                $$ e_{n,b,t} = e_{n,b,t-1}\cdot \eta_{n}^{SoC} + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -
                \frac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
                b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$ """)
        st.write('Límite de energía de los SAEB')
        st.write(r"""
                $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)
        st.write('Límite de potencia de los SAEB')
        st.write(r"""
                $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,t}^{ch} \hspace{2mm} \forall
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ p_{n,b,t}^{ch} \leq P_{n,b} \hspace{2mm} \forall
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ p_{n,b,t}^{dc} \leq Z \cdot u_{n,t}^{dc} \hspace{2mm} \forall
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)
        st.write(r"""
                $$ p_{n,b,t}^{dc} \leq P_{n,b} \hspace{2mm} \forall
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ """)

# Correr función de optimización
    def run_dim_size_OpC():
        opt_results = opt_dim(file_system, Eff, DoD, 0.2, time_sim, 20, costP, costE, ope_fact, con_fact, con_cost, solver)
        #Imprimir resultados
        st.title('Resultados:')
        st.write('Tiempo de simulación: ' + str(opt_results[3]))
        Power=pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Power_size', header=0, index_col=0)
        Power=Power.rename(columns={0: 'Potencia [MW]'})
        Energy= pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Energy_size', header=0, index_col=0)
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})
        Size=result = pd.concat([Power,Energy], axis=1)
        for i in range(1,len(Size)+1):
            if Size.loc[i-1,'Potencia [MW]']>0:
                Size=Size.rename(index={i-1:'b'+'%i'%i})
            else:
                Size=Size.drop([i-1])
        st.dataframe(Size.style.format('{:.2f}'))
        file_results=opt_results[2]
        return file_results
#############Simulation button###############################################
    button_sent = st.sidebar.button('Simular')
    if button_sent:
        if path.exists('Resultados/resultados_size_loc.xlsx'):
            remove('Resultados/resultados_size_loc.xlsx')
        run_dim_size_OpC()

################################ PLOTS  #####################################
    button_plot= st.sidebar.checkbox('Graficar',value=False)
    if button_plot:
        #Imprimir resultados
        st.title('Resultados:')
        Power=pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Power_size', header=0, index_col=0)
        Power=Power.rename(columns={0: 'Potencia [MW]'})
        Energy= pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Energy_size', header=0, index_col=0)
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})
        Size= pd.concat([Power,Energy], axis=1)
        for i in range(1,len(Size)+1):
            if Size.loc[i-1,'Potencia [MW]']>0:
                Size=Size.rename(index={i-1:'b'+'%i'%i})
            else:
                Size=Size.drop([i-1])
        st.dataframe(Size.style.format('{:.2f}'))
        ######
        #Selección de tipo de gráfica
        tipo = st.sidebar.selectbox('Seleccione el tipo de gráfica que desea visualizar',['Despacho SAE','SOC SAE','Flujo en lineas','Mapa'])
        colores = ['#53973A','#4A73B2','#B7C728','#77ABBD','#FF7000','#1f77b4', '#aec7e8',
                    '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a','#d62728', '#ff9896',
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94','#e377c2', '#f7b6d2',
                    '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']

        flag_mapa=True
        if tipo =='Despacho SAE':
            sheet='BESS_Ch_Power'
            title='Despacho de los SAE'
            label='Potencia [MW]'
            id='b'
            plot2=True
            sheet2='BESS_Dc_Power'
            id1='Pc_'
            id2='Pd_'
        elif tipo =='SOC SAE':
            sheet='BESS_Energy'
            title='Estado de carga de los SAE'
            label='Energía [MWh]'
            id='b'
            id1=''
            plot2=False
        elif tipo =='Flujo en lineas':
            sheet='pf'
            title='Flujo de potencia en las lineas'
            label='Potencia [MW]'
            id='L'
            id1=''
            plot2=False
        else:

#################################################################################
##########################    MAP     ##########################################
            # DATA_URL = {
            #     'AIRPORTS': 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/line/airports.json',
            #     'FLIGHT_PATHS': 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/line/heathrow-flights.json',  # noqa
            # }
            # LINES=json.load('lines.json')

            map_data_1 = pd.read_excel(file_system, sheet_name='Bus', header=0,usecols=['lat','lon'] )
            map_data_2 = pd.read_excel(file_system, sheet_name='Branch', header=0,usecols=['start','end'] )
            a_list=[0]*len(Power)
            colores=[0]*len(Power)
            for i in range(1,len(Power)+1):
                if Power.loc[i-1,'Potencia [MW]']>0:
                    a_list[i-1]=40000
                    colores[i-1]=[10, 230, 120]
                else:
                    a_list[i-1]=15000
                    colores[i-1]=[230, 158, 10]
            # a_list=[1,1,1,1,1,1,1,1,1,1,1,1,5,1,1]
            map_data_1['exits_radius']=a_list
            map_data_1['color']=colores
            midpoint = (np.average(map_data_1['lat']), np.average(map_data_1['lon']))
            df=map_data_1
            st.pydeck_chart(pdk.Deck(
             map_style='mapbox://styles/mapbox/light-v9',
             initial_view_state=pdk.ViewState(
                 latitude= midpoint[0],
                 longitude= midpoint[1],
                 zoom=4,
                 # pitch=50,  # inclinación del mapa
             ),
             layers=[
                    # pdk.Layer(
                    # 'LineLayer',
                    # # LINES,
                    # DATA_URL['FLIGHT_PATHS'],
                    # # map_data_2,
                    # get_source_position='start',
                    # get_target_position='end',
                    # get_color=[10, 230, 120],
                    # get_width=10,
                    # highlight_color=[255, 255, 0],
                    # picking_radius=10,
                    # auto_highlight=True,
                    # pickable=True,
                    # ),

                    pdk.Layer(
                     'ScatterplotLayer',
                     data=df,
                     get_position='[lon, lat]',
                     get_color='color',
                     get_radius='exits_radius',
                    ),
             ],
            ))

            flag_mapa=False
####################################################################################

        if flag_mapa:
            # Lectura de datos
            df_results = pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name=sheet, header=0, index_col=0)
            results = df_results.to_numpy()
            results = np.absolute(results)

            # Lectura de otros datos de ser necesario (Caso carga/ descarga)
            if plot2:
                df_results2 = pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name=sheet2, header=0, index_col=0)
                results2 = df_results2.to_numpy()
                results2 = np.absolute(results2)
                st.sidebar.markdown('Seleccione la/las variable(s) que desea visualizar:')
                elemento1=st.sidebar.checkbox(id1,value=True,key=0)
                elemento2=st.sidebar.checkbox(id2,value=True,key=0)

            # checkbox para seleccionar/deseleccionar todos los elementos
            st.sidebar.markdown('Seleccione el/los elemento(s) que desea visualizar:')
            Todas=st.sidebar.checkbox('Todas',value=True,key=0)

            Flag= np.ones(len(results))

            if Todas:
                for i in range(0,len(results)):
                    Flag[i]=st.sidebar.checkbox(id+'%s'%(i+1),value=True,key=i+1)
            else:
                for i in range(0,len(results)):
                    Flag[i]=st.sidebar.checkbox(id+'%s'%(i+1),value=False,key=i+1)

            plt.figure(figsize=(10,6))
            for i in range(1,len(results) + 1):
                if Flag[i-1]:
                    if plot2:
                        if elemento1:
                            plt.step(list(range(0,len(df_results.columns))),results[i-1,:], label=id1+id+'%s'%i)
                        if elemento2:
                            plt.step(list(range(0,len(df_results2.columns))),results2[i-1,:], label=id2+id+'%s'%i)
                    else:
                        plt.step(list(range(0,len(df_results.columns))),results[i-1,:], label=id1+id+'%s'%i)

            plt.xlabel('Tiempo [h]')
            plt.ylabel(label).set_fontsize(15)
            plt.ylabel(label).set_fontsize(15)
            plt.legend(fontsize='x-large')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=8, fancybox=True, shadow=True)
            # plt.xlabel('Tiempo [h]').set_fontsize(15)
            plt.xticks(size = 'x-large')
            plt.yticks(size = 'x-large')
            plt.title(title, fontsize=20)
            st.pyplot()
