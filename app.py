# coding: utf-8
import streamlit as st
import pandas as pd
import os

# import numpy as np
# import plotly.express as px
# import time
from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# import matplotlib.pyplot as plt
# import SessionState
# from os import remove
# from os import path
# import pydeck as pdk
# import math
# import json
# # Libraries for reading data form internet
# import requests
# from bs4 import BeautifulSoup
# import re

# Importing optimization functions

# from despacho import *


# Importing dashboard functions
from modelos.LocSize.Arbitraje.Loc_Arbitrage import main_Dim_Arbitrage
from modelos.Ope_SAE.Op_Arb.Op_Arbitrage import main_Op_Arbitrage
from modelos.Ope_SAE.MEM.MEM_model import main_MEM
from Secciones.section_Loc_Size_operacion import dashboard_DLOp
from Secciones.section_Loc_Size_restr import dashboard_DLOr
from Secciones.section_Eco import dashboard_Eco
from Secciones.section_XM_document import dashboard_XMDoc

# Importing other functions

from funciones.Lectura_DigSilent import main_DigSilent_lecture
from funciones.guias_uso import *

#### Configuración de la página
st.set_page_config(page_title="HEPISA", page_icon='Imagenes/icon_app.png', layout='centered', initial_sidebar_state='auto')

####Cargando Datos######

actual_path = os.getcwd()

DATA_URL_1=(os.path.join(actual_path, 'Tablas/Tech.csv'))
DATA_URL_2=(os.path.join(actual_path, 'Tablas/inputs_eco.csv'))
DATA_URL_3=(os.path.join(actual_path, 'Tablas/inputs_eco_opex.csv'))

@st.cache
def load_data1():
    data1=pd.read_csv(DATA_URL_1)
    data1=data1.set_index('Tech')
    return data1

data1=load_data1()

@st.cache
def load_data2():
    data2=pd.read_csv(DATA_URL_2)
    data2=data2.set_index('Tech')
    return data2
data2=load_data2()

@st.cache
def load_data3():
    data3=pd.read_csv(DATA_URL_3)
    data3=data3.set_index('Tech')
    return data3
data3=load_data3()

################## DASHBOARD INICIO#############################################

st.sidebar.image('Imagenes/logo-geb.png', use_column_width='auto' )
st.sidebar.title("Herramienta para el Planeamiento e Integración de Sistemas de Almacenamiento -HEPISA")
st.sidebar.subheader("Elija el tipo de análisis que quiere efectuar:")
study = st.sidebar.selectbox('',('INICIO','Dimensionamiento y Localización','Operación del SAE (MEM-SC, XM)','Análisis financiero','Análisis eléctricos'))

if study =='INICIO':
    st.markdown("<h1 style='text-align: center; color: black;'>Herramienta para el Planeamiento e Integración de Sistemas de Almacenamiento - HEPISA</h1>", unsafe_allow_html=True)
    st.markdown("HEPISA permite realizar análisis ralacionados con la integración de sistemas \
        de almacenamiento de energía (SAE) a los sistemas de potencia. La herramienta cuenta con modulos \
        para la localización y dimensionamiento, el análisis de la interacción con el mercado de energía,\
        la prestación de servicios complementarios, el análisis financiero, entre otros. ")
    
    
    # image = Image.open('ess.jpg')
    # st.image(image, caption='',use_column_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        # st.markdown("<h3 style='text-align: center; color: green;'>Localización y dimensionamiento</h3>", unsafe_allow_html=True)
        st.image("Imagenes/3.1.png",use_column_width=True)
    with col2:
        # st.markdown("<h3 style='text-align: center; color: green;'>Análisis de beneficios técnicos</h3>", unsafe_allow_html=True)
        st.image("Imagenes/3.2.png",use_column_width=True)
        # st.image("Imagenes/despacho.png",use_column_width=True)
    with col3:
        # st.markdown("<h3 style='text-align: center; color: green;'>Análisis económico y financiero</h3>", unsafe_allow_html=True)
        st.image("Imagenes/3.3.png",use_column_width=True)
        # st.image("Imagenes/economico.png",use_column_width=True)
        
    st.markdown("Para consultar archivos anexos relacionados con casos de estudio, formato de ingreso de datos, entre otros, haga click [aquí](https://github.com/aaavendanop/HEPISA) para acceder al repositorio de la herramienta.")
        
################################################################################
######################### Sección de Localización ##############################
################################################################################


if (study == 'Dimensionamiento y Localización'):
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización</h1>", unsafe_allow_html=True)
        st.markdown("En esta sección, el usuario puede seleccionar entre localizar y dimensionar SAE para \
        prestar la función de arbitraje, reducción de costos de operación del sistema o para aliviar  \
            el costo por restricciones. Para esto, seleccione la opción de su preferencia en el menú \
                 *Tipo de dimensionamiento y localización* de la parte izquierda. En cada una de las opciones \
                se dará mayor información sobre el respectivo módulo.")
    s_study = st.sidebar.selectbox('Tipo de dimensionamiento y localización:', ('', 'Arbitraje', 'Minimizar costo de operación', 'Minimizar costo de restricciones'))

    if s_study == 'Arbitraje':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización de SAE basado en Arbitraje</h1>", unsafe_allow_html=True) 
            st.markdown("En esta sección de la herramienta el usuario podrá encontrar el \
            tamaño óptimo teórico del SAE (en términos de potencia y energía) para realizar la\
            función de arbitraje en un mercado uninodal. Para este fin, el usuario deberá ingresar \
            el histórico de precios a analizar, el horizonte de tiempo de la simulación, la TMR,\
            la tecnología de SAE y el tipo de solucionador/solver.")

        guias(study, s_study)

        main_Dim_Arbitrage(data1,actual_path)

    elif s_study == 'Minimizar costo de operación':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización de SAE para minimizar costo de operación</h1>", unsafe_allow_html=True) 
            st.markdown("En esta sección de la herramienta el usuario podrá encontrar el \
        tamaño óptimo de SAE para cada nodo del sistema (en términos de potencia y energía) \
        con el fin de reducir los costos de operación. Para este fin, el usuario deberá ingresar un archivo con \
        los datos del sistema de potencia a analizar, el horizonte de tiempo de la simulación,\
        la tecnología de SAE y el tipo de solucionador/solver, por ejemplo, CPLEX (cálculo en servidor NEOS) o GLPK (cálculo local).")

        guias(study, s_study)

        dashboard_DLOp(data1)

    elif s_study == 'Minimizar costo de restricciones':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización de SAE para minimizar costo de restricciones</h1>", unsafe_allow_html=True) 
            st.markdown("En esta sección de la herramienta el usuario podrá encontrar el \
        tamaño óptimo de SAE para cada nodo del sistema (en términos de potencia y energía) \
        con el fin de reducir los costos de restricciones. Para este fin, el usuario deberá ingresar un archivo con \
        los datos del sistema de potencia a analizar, el horizonte de tiempo de la simulación,\
        la tecnología de SAE y el tipo de solucionador/solver, por ejemplo, CPLEX (cálculo en servidor NEOS) o GLPK (cálculo local).")

        guias(study, s_study)

        dashboard_DLOr(data1)

################################################################################
##################Sección de Modelo MEM Colombiano #############################
################################################################################
if (study=='Operación del SAE (MEM-SC, XM)'):
    
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<h1 style='text-align: center; color: black;'>Operación del SAE</h1>", unsafe_allow_html=True)
        st.markdown("En esta sección, el usuario puede analizar la operación del SAE en diferentes aplicaciones y esquemas de mercados.\
                 Para esto, seleccione la opción de su preferencia en el menú \
                 *Tipo de simulación* de la parte izquierda. En cada una de las opciones \
                 se dará mayor información sobre el respectivo módulo.")
    s_study = st.sidebar.selectbox('Tipo de simulación:', ('', 'Arbitraje', 'Mercado Energía Mayorista','Procedimientos operación SAE (XM)'))
 
    if s_study == 'Arbitraje':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Operación del SAE en Arbitraje</h1>", unsafe_allow_html=True) 
            st.markdown("En esta sección de la herramienta el usuario podrá simular la \
            operación óptima teórica del SAE (asumiendo que se conoce perfectamente el comportamiento del precio de energía) para realizar la\
            función de arbitraje en un mercado uninodal. Para este fin, el usuario deberá ingresar \
            el histórico de precios a analizar, el horizonte de tiempo de la simulación, la TMR,\
            la tecnología de SAE y el tipo de solucionador/solver.")

        guias(study, s_study)

        main_Op_Arbitrage(data1,actual_path)
        
    elif s_study == 'Mercado Energía Mayorista':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Mercado de energía mayorista</h1>", unsafe_allow_html=True) 
            st.markdown("En esta sección de la herramienta el usuario podrá simular la participación del SAE en diferentes esquemas\
                        de mercado de energía mayorista, como tambien diferentes servicios complementarios como la regulación primaria\
                        y secundaria de frecuencia. Para este fin, el usuario deberá ingresar un archivo con \
                        los datos del sistema de potencia a analizar, el horizonte de tiempo de la simulación,\
                        la tecnología de SAE y el tipo de solucionador/solver, por ejemplo, CPLEX (cálculo en servidor NEOS) o GLPK (cálculo local).")

        image2 = Image.open('{}/MEM_SAE/Esquema.JPG'.format(actual_path))
        st.image(image2, caption='',use_column_width=True)

        guias(study, s_study)

        main_MEM(data1,actual_path)

    elif s_study == 'Procedimientos operación SAE (XM)':
        with placeholder.container():
            st.markdown("<h1 style='text-align: center; color: black;'>Procedimientos operación SAE (XM)</h1>", unsafe_allow_html=True)
            st.markdown("En esta sección, el usuario podrá seleccionar entre los dos documentos públicados hasta la fecha para la operación \
                        de SAE en el sistema eléctrico colombiano. Para este fin, el usuario deberá ingresar las características técnicas de \
                        cada SAE, los tiempos previos a bloques de descarga y los tiempos en donde el SAE debe descargarse, la fecha que se \
                        quiera simular y el tipo de solucionador/solver, por ejemplo, CPLEX (cálculo en servidor NEOS) o GLPK (cálculo local). ")

        ss_study = st.sidebar.selectbox('Versión del documento:', ('', 'Versión 1', 'Versión 2'))

        if ss_study == 'Versión 1':

            with placeholder.container():
                st.markdown("<h1 style='text-align: center; color: black;'>Despacho colombiano uninodal con SAE (Versión 1)</h1>",
                            unsafe_allow_html=True)
                st.markdown("En esta sección de la herramienta el usuario podrá analizar la operación de un SAE bajo el esquema actual de \
                            despacho económico uninodal de XM. La formulación esta basada en la versión incial del documento de XM, publicado \
                            en el mes de Noviembre de 2019.")

            guias(study, ss_study)

            dashboard_XMDoc(data1, ss_study)

        elif ss_study == 'Versión 2':

            with placeholder.container():
                st.markdown("<h1 style='text-align: center; color: black;'>Despacho colombiano uninodal con SAE (Versión 2)</h1>",
                            unsafe_allow_html=True)
                st.markdown("En esta sección de la herramienta el usuario podrá analizar la operación de un SAE bajo el esquema actual de \
                            despacho económico uninodal de XM. La formulación esta basada en la segunda versión del documento de XM, publicado \
                            en el mes de Septiembre de 2020.")

            guias(study, ss_study)

            dashboard_XMDoc(data1, ss_study)

################################################################################
########################Sección de Modelo económico#############################
################################################################################

if (study=='Análisis financiero'):

    dashboard_Eco(data2,data3)

################################################################################
######################   Sección Flujo de potencia #############################
################################################################################

if (study=='Análisis eléctricos'):
    
    
    st.markdown("<h1 style='text-align: center; color: black;'>Simulacion de flujos de potencia </h1>", unsafe_allow_html=True)
    st.markdown("En esta sección de la herramienta el usuario podrá realizar la \
    simulación del flujo de potencia de cualquier sistema. Dentro de la herramienta se puede realizar\
    flujos AC, flujos optimos AC, flujos DC y flujos optimos DC. Además, esta herramienta cuenta con un modulo\
    para hacer la lectura de sistemas elaborados en DigSilent")
    
    st.sidebar.markdown("### Selecione el tipo de simulación:")
    FP_simulation = study=st.sidebar.selectbox('',('','Lectura archivos DIgSILENT','Flujos de potencia'))
    
    if FP_simulation == 'Lectura archivos DIgSILENT':
        main_DigSilent_lecture()
        
    if FP_simulation == 'Flujos de potencia':
        introduction = 0
        main_PF_SR()

st.sidebar.subheader("")
st.sidebar.subheader("")
st.sidebar.subheader("")
st.sidebar.subheader("Equipo de desarrollo de la herramienta:")
st.sidebar.subheader("")
st.sidebar.image('Imagenes/unal.png', use_column_width='auto' )
st.sidebar.subheader("")
st.sidebar.image('Imagenes/logo-uniandes.png', use_column_width='auto' )
st.sidebar.subheader("")
st.sidebar.image('Imagenes/logo-ceiba.png', use_column_width='auto' )

st.subheader("")
st.subheader("")
st.subheader("Equipo de desarrollo de la herramienta:")
# st.image('Imagenes/participantes.png', use_column_width='auto' )

col1, col2, col3, col4 = st.columns(4)
with col1:
    # st.markdown("<h3 style='text-align: center; color: green;'>Localización y dimensionamiento</h3>", unsafe_allow_html=True)
    st.image("Imagenes/logo-geb.png",use_column_width=True)

with col2:
    # st.markdown("<h3 style='text-align: center; color: green;'>Análisis económico y financiero</h3>", unsafe_allow_html=True)
    st.image("Imagenes/logo-ceiba.png",use_column_width=True)
    # st.image("Imagenes/economico.png",use_column_width=True)
with col3:
    # st.markdown("<h3 style='text-align: center; color: green;'>Análisis de beneficios técnicos</h3>", unsafe_allow_html=True)
    st.image("Imagenes/unal.png",use_column_width=True)
    # st.image("Imagenes/despacho.png",use_column_width=True)
with col4:
    # st.markdown("<h3 style='text-align: center; color: green;'>Análisis económico y financiero</h3>", unsafe_allow_html=True)
    st.image("Imagenes/logo_uniandes.jpg",use_column_width=True)
    # st.image("Imagenes/economico.png",use_column_width=True)

