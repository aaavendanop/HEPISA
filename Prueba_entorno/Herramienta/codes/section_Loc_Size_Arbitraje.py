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
# Libraries for reading data form internet
import requests
from bs4 import BeautifulSoup
import re
# Importing optimization functions
from arbitraje import opt

def dashboard_DLA(data1):

    st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización de SAE basado en Arbitraje</h1>", unsafe_allow_html=True)
    # image = Image.open('arbitrage.jpeg')
    # st.image(image, caption='', use_column_width=True)
    st.markdown('En esta sección de la herramienta el usuario podrá encontrar el \
    tamaño óptimo del SAE (en términos de potencia y energía) para realizar la\
    función de arbitraje en un mercado uninodal. Para este fin, el usuario deberá ingresar un archivo con \
    el histórico de precios a analizar, el horizonte de tiempo de la simulación, la TMR,\
    la tecnología de SAE y el tipo de solver.')
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

    # Seleeción de archivo con precios

    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_prices = st.sidebar.file_uploader('Seleccione el archivo con históricos de precios de bolsa:', type=['csv','xlsx'])

    # Ingresar TRM
    TRM_select=st.sidebar.selectbox('Seleccione la TRM para la simulación',['Hoy','Otra'],key='2')
    if TRM_select=='Hoy':
        URL = 'https://www.dolar-colombia.com/'
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        rate = soup.find_all(class_='exchange-rate')
        TRM=str(rate[0])
        TRM=re.findall('\d+', TRM )
        TRM_final=TRM[0]+','+TRM[1]+'.'+TRM[2]
        TRM_final_1='La TRM seleccionada para la simulación es de: '+TRM_final + ' COP'
        st.write(TRM_final_1)
    else:
        TRM_final=st.text_input('Ingrese la TRM para la simulación: ')

    # Ingresar tiempo de simulación
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=10000000)
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')

    # Seleccionar solver
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')
    if solver=='CPLEX':
        st.write('El solucionador seleccionado es: '+solver)
    else:
        st.write('El solucionador seleccionado es: '+solver)

    # Correr función de optimización
    def run_arbitraje():
        opt_results=opt(file_prices,Eff,degra,autoD,DoD,DoD,
        costP,costE,30000000,TRM_final,time_sim,solver)
        File='Resultados/resultados_size_loc_arbitraje.xlsx'
        #Imprimir resultados
        st.title('Resultados:')
        Power=pd.read_excel(File, sheet_name='Power_size', header=0, index_col=0)
        Power=Power.rename(columns={0: 'Potencia [MW]'})
        Energy= pd.read_excel(File, sheet_name='Energy_size', header=0, index_col=0)
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})
        Size= pd.concat([Power,Energy], axis=1)
        st.dataframe(Size.style.format('{:.2f}'))

    #############Simulation button###############################################
    button_sent = st.sidebar.button('Simular')
    if button_sent:
        if path.exists('Resultados/resultados_size_loc_arbitraje.xlsx'):
            remove('Resultados/resultados_size_loc_arbitraje.xlsx')
        run_arbitraje()

    ################################ PLOTS  #####################################
    button_plot= st.sidebar.checkbox('Graficar',value=False)
    if button_plot:
        File='Resultados/resultados_size_loc_arbitraje.xlsx'
        #Imprimir resultados
        st.title('Resultados:')
        Power=pd.read_excel(File, sheet_name='Power_size', header=0, index_col=0)
        Power=Power.rename(columns={0: 'Potencia [MW]'})
        Energy= pd.read_excel(File, sheet_name='Energy_size', header=0, index_col=0)
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})
        Size= pd.concat([Power,Energy], axis=1)
        st.dataframe(Size.style.format('{:.2f}'))
        ######
        #Selección de tipo de gráfica##########################################
        tipo = st.sidebar.selectbox('Seleccione el tipo de gráfica que desea visualizar',['Despacho SAE','SOC SAE','Precios','Ingresos'])
        colores = ['#53973A','#4A73B2','#B7C728','#77ABBD','#FF7000','#1f77b4','#aec7e8',
                    '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a','#d62728','#ff9896',
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94','#e377c2', '#f7b6d2',
                    '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']

        if tipo =='Despacho SAE':
            sheet='BESS_Ch_Power'
            title='Despacho de los SAE'
            label='Potencia [MW]'
            plot2=True
            sheet2='BESS_Dc_Power'
            id1='Pc_'
            id2='Pd_'
        elif tipo =='SOC SAE':
            sheet='BESS_Energy'
            title='Estado de carga del SAE'
            label='Energía [MWh]'
            id1='SOC'
            plot2=False
        elif tipo =='Ingresos':
            sheet='Ingresos'
            title='Ingresos'
            label='MCOP'
            id1='Ingresos'
            plot2=False
            Suma=pd.read_excel(File, sheet_name=sheet, header=0, index_col=0).sum(axis = 0, skipna = True)
            st.write('El total de ingresos neto durante el horizonte de tiempo fue de: '+'%i'%Suma)
        else:
            File=file_prices
            sheet='Precios'
            title='Precio de bolsa'
            label='Precio [COP/kWh]'
            plot2=False
            id1='Precio de bolsa'

        df_results = pd.read_excel(File, sheet_name=sheet, header=0, index_col=0)
        results = df_results.to_numpy()
        # results = np.absolute(results)

        # Lectura de otros datos de ser necesario (Caso carga/ descarga)
        if plot2:
            df_results2 = pd.read_excel(File, sheet_name=sheet2, header=0, index_col=0)
            results2 = df_results2.to_numpy()
            results2 = np.absolute(results2)
            st.sidebar.markdown('Seleccione la/las variable(s) que desea visualizar:')
            elemento1=st.sidebar.checkbox(id1,value=True,key=0)
            elemento2=st.sidebar.checkbox(id2,value=True,key=0)
        st.sidebar.markdown('Seleccione el rango de tiempo que desea visualizar:')
        values = st.sidebar.slider('',0, time_sim, (0, time_sim))

        plt.figure(figsize=(10,6))
        if plot2:
            if elemento1 and elemento2:
                plt.step(list(range(values[0],values[1])),results[values[0]:values[1]], label=id1)
                plt.step(list(range(values[0],values[1])),-results2[values[0]:values[1]], label=id2)
            if elemento2 and not(elemento1):
                plt.step(list(range(values[0],values[1])),results2[values[0]:values[1]], label=id2)
            if elemento1 and not(elemento2):
                plt.step(list(range(values[0],values[1])),results2[values[0]:values[1]], label=id1)
        else:
            plt.step(list(range(values[0],values[1])),results[values[0]:values[1]], label=id1)

        plt.xlabel('Tiempo [h]')
        plt.ylabel(label).set_fontsize(15)
        plt.ylabel(label).set_fontsize(15)
        plt.legend(fontsize='x-large')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=8, fancybox=True, shadow=True)
        # plt.xlabel('Tiempo [h]').set_fontsize(15)
        plt.xticks(size = 'x-large')
        plt.yticks(size = 'x-large')
        plt.title(title, fontsize=20)
        st.pyplot()
