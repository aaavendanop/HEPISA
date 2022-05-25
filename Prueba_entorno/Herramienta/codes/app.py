# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# import SessionState
import os
from os import remove
from os import path
import pydeck as pdk
import math
import json
# Libraries for reading data form internet
import requests
from bs4 import BeautifulSoup
import re

# Importing optimization functions
from arbitraje import opt
from Loc_dim_OpC import opt_dim
from despacho import *
from Regulacion_F_UC_V6 import UC_ESS_with_Freq

# Importing dashboard functions
from section_Loc_Size_Arbitraje import dashboard_DLA
from section_Loc_Size_operacion import dashboard_DLOp
from section_Eco import dashboard_Eco
from section_XM_document import dashboard_XMDoc
####Cargando Datos######

actual_path = os.getcwd()

aux = actual_path.split('/')
last_element = aux[-1]
aux.remove(last_element)
before_path = '/'.join(aux)

DATA_URL_1=(os.path.join(before_path, 'data_base/Tech.csv'))
DATA_URL_2=(os.path.join(before_path, 'data_base/inputs_eco.csv'))
DATA_URL_3=(os.path.join(before_path, 'data_base/inputs_eco_opex.csv'))
# @st.cache(persist=True)
def load_data1():
    data1=pd.read_csv(DATA_URL_1)
    data1=data1.set_index('Tech')
    return data1
data1=load_data1()

# @st.cache(persist=True)
def load_data2():
    data2=pd.read_csv(DATA_URL_2)
    data2=data2.set_index('Tech')
    return data2
data2=load_data2()

# @st.cache(persist=True)
def load_data3():
    data3=pd.read_csv(DATA_URL_3)
    data3=data3.set_index('Tech')
    return data3
data3=load_data3()

st.sidebar.title('Energy Storage Analysis Tool')
st.sidebar.subheader('Elija el tipo de análisis que quiere efectuar:')
study = st.sidebar.selectbox('',('','Dimensionamiento y Localización', 'Análisis financiero',
        'Documento XM', 'Regulación de Frecuencia'))

if study =='':
    st.title('Energy Storage Analysis Tool')
    image = Image.open('ess.jpg')
    st.image(image, caption='',use_column_width=True)
    st.markdown('Descripción general de la herramienta')

################################################################################
######################### Sección de Localización ##############################
################################################################################
if (study == 'Dimensionamiento y Localización'):

    s_study = st.sidebar.selectbox('Tipo de dimensionamiento y localización:', ('', 'Arbitraje', 'Costo Operación/Congestión'))

    if s_study == 'Arbitraje':
        dashboard_DLA(data1)

    elif s_study == 'Costo Operación/Congestión':
        ss_study = st.sidebar.selectbox('Para minimizar costos de:', ('', 'Operación', 'Congestión', 'Ambos'))

        if ss_study == 'Operación':

            ope_fact = 1
            con_fact = 0
            info_ = 'Ope'
            dashboard_DLOp(data1, info_, ope_fact, con_fact)

        elif ss_study == 'Congestión':

            ope_fact = 0
            con_fact = 1
            info_ = 'Con'
            dashboard_DLOp(data1, info_, ope_fact, con_fact)

        elif ss_study == 'Ambos':

            ope_fact = 1
            con_fact = 1
            info_ = 'Both'
            dashboard_DLOp(data1, info_, ope_fact, con_fact)

################################################################################
########################Sección de Modelo económico#############################
################################################################################

if (study=='Análisis financiero'):

    dashboard_Eco(data2,data3)

################################################################################
############################## Documento XM ####################################
################################################################################

if study == 'Documento XM':

    s_study = st.sidebar.selectbox('Versión del documento:', ('', 'Versión 1', 'Versión 2'))

    if s_study == 'Versión 1':
        dashboard_XMDoc(data1, s_study)
    elif s_study == 'Versión 2':
        dashboard_XMDoc(data1, s_study)

    ################################################################################
    ###############################Regulación de frecuencia#########################
    ################################################################################

if (study=='Regulación de Frecuencia'):
    st.markdown("<h1 style='text-align: center; color: black;'>Regulación \
    de Frecuencia</h1>", unsafe_allow_html=True)
    # image = Image.open('arbitrage.jpeg')
    # st.image(image, caption='', use_column_width=True)
    st.markdown('En esta sección de la herramienta el usuario podrá analizar \
    los beneficios de los SAE en la prestación de la regulación de frecuencia.')
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
        st.write(data1.loc[technology])
        Eff=data1.iloc[data1.index.get_loc(technology),0]
        degra=data1.iloc[data1.index.get_loc(technology),1]
        autoD=data1.iloc[data1.index.get_loc(technology),2]
        DoD=data1.iloc[data1.index.get_loc(technology),3]
        costP=data1.iloc[data1.index.get_loc(technology),4]
        costE=data1.iloc[data1.index.get_loc(technology),5]

# Seleción de archivo con sistema
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_system = st.sidebar.file_uploader('Seleccione el archivo con el sistema a simular:', type=['csv','xlsx'])

# Ingresar tiempo de simulación
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')

# Seleccionar solver
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')
    if solver=='CPLEX':
        st.write('El solucionador seleccionado es: '+solver)
    else:
        st.write('El solucionador seleccionado es: '+solver)

# Correr función de optimización
    def run_frecuencia():
        opt_results=UC_ESS_with_Freq(file_system,time_sim,solver)
        return opt_results

#############Simulation button###############################################

    button_sent = st.sidebar.button('Simular')
    if button_sent:

        reserva=run_frecuencia()
        st.write(reserva[6])
