# coding: utf-8
import os
import subprocess as sp
from virtualenv import cli_run

actual_dir = os.getcwd()

### Create folders

os.mkdir('Herramienta')
os.mkdir('Herramienta/Casos de estudio')
os.mkdir('Herramienta/Casos de estudio/archivos_despacho')
os.mkdir('Herramienta/Casos de estudio/archivos_despacho/scrap_files')
os.mkdir('Herramienta/Casos de estudio/Proyecciones_Eco')
os.mkdir('Herramienta/requerimientos')
os.mkdir('Herramienta/codes')
os.mkdir('Herramienta/data_base')
os.mkdir('Herramienta/Resultados')

### Create files

## requeriments.txt

requirements = open('Herramienta/requerimientos/requirements.txt', 'w')
requirements.write('brotlipy==0.7.0' + os.linesep)
requirements.write('wordcloud==1.8.1' + os.linesep)
requirements.write('PyUtilib==6.0.0' + os.linesep)
requirements.write('Pyomo==5.7' + os.linesep)
requirements.write('keyring==21.2.1' + os.linesep)
requirements.write('cryptography==2.9.2' + os.linesep)
requirements.write('numpy==1.18.5' + os.linesep)
requirements.write('pandas==1.0.5' + os.linesep)
requirements.write('plotly==4.12.0' + os.linesep)
requirements.write('matplotlib==3.3.2' + os.linesep)
requirements.write('Cython==0.29.21' + os.linesep)
requirements.write('streamlit==0.71.0'+ os.linesep)
requirements.write('pydeck==0.5.0b1' + os.linesep)
requirements.write('lxml==4.5.2' + os.linesep)
requirements.write('beautifulsoup4==4.9.3' + os.linesep)
requirements.write('brotli==1.0.9' + os.linesep)
requirements.write('ipaddr==2.2.0' + os.linesep)
requirements.write('ordereddict==1.1' + os.linesep)
requirements.write('Pillow==8.0.1' + os.linesep)
requirements.write('protobuf==3.14.0' + os.linesep)
requirements.write('pyOpenSSL==19.1.0' + os.linesep)
requirements.write('wincertstore==0.2')
requirements.close()

## Tech.csv

tech = open('Herramienta/data_base/Tech.csv', 'w')
tech.write('Tech,Eficiencia [pu],Degradación [p.u./ciclo],Autodescarga [%/h], DoD [pu],Costo por potencia [USD],Costo por \
    energía [USD]' + os.linesep)
tech.write('Li-Ion,0.95,0.00002,0.0000625,0.8,70000,209000' + os.linesep)
tech.write('PbA,0.9,0.0001,0.0042,0.8,150000,600000' + os.linesep)
tech.write('Sodio-sulfuro,0.85,0.00005,0.0042,0.2,200000,700000' + os.linesep)
tech.write('VRFB,0.8,0.000016,0.0021,1,3700000,500000' + os.linesep)
tech.write('Celda de Combustible,0.5,0.0002,0.042,1,15000,2000000' + os.linesep)
tech.write('Central de Bombeo,0.85,0,0.042,1,50000,2000000' + os.linesep)
tech.write('Volante,0.95,0.000002,2.1,1,5000000,300000' + os.linesep)
tech.write('CAES,0.75,0.000015,0.042,1,400000,1800000' + os.linesep)
tech.write('SMES,0.99,0.000002,0.625,1,10000000,3500000' + os.linesep)
tech.write('SCES,0.95,0.0000004,0.42,1,20000000,200000' + os.linesep)
tech.write('Thermal,0.6,0,0.042,1,60000,300000' + os.linesep)
tech.write('New,,,,,,')
tech.close()

## input_eco_opex.csv

inputs_eco_opex = open('Herramienta/data_base/inputs_eco_opex.csv', 'w')
inputs_eco_opex.write('Tech, O&M Fijo [USD/MW/año], O&M Variable[USD/MWh]' + os.linesep)
inputs_eco_opex.write('Li-Ion, 10000, 0.3' + os.linesep)
inputs_eco_opex.write('New, ,')
inputs_eco_opex.close()

## inputs_eco.

inputs_eco = open('Herramienta/data_base/inputs_eco.csv', 'w')
inputs_eco.write('Tech, costo baterías [USD/MWh], costo electrónica de potencia [USD/MW], costos de equipos AC [USD/MW], Costo de \
    construcción y puesta en servicio [USD/MWh], costo del predio [USD], costos de licenciamiento [USD], tarifa de \
    conexión [USD/MW]' + os.linesep)
inputs_eco.write('Li-Ion, 209000, 70000, 100000, 101000, 250000, 50000, 30000' + os.linesep)
inputs_eco.write('New, ,,,,,,')
inputs_eco.close()

### pictures files

# pic_1 = open('Herramienta/ess.jpg')
# pic_1.write("")

### Install libraries

# os.system('pip install -r Herramienta/requerimientos/requirements.txt')

### Create dashboard functions codes

## app.py

app = open('Herramienta/codes/app.py', 'w')
app.write("# coding: utf-8\n\
import streamlit as st\n\
import pandas as pd\n\
import numpy as np\n\
import plotly.express as px\n\
import time\n\
from PIL import Image\n\
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n\
import matplotlib.pyplot as plt\n\
# import SessionState\n\
import os\n\
from os import remove\n\
from os import path\n\
import pydeck as pdk\n\
import math\n\
import json\n\
# Libraries for reading data form internet\n\
import requests\n\
from bs4 import BeautifulSoup\n\
import re\n\
\n\
# Importing optimization functions\n\
from arbitraje import opt\n\
from Loc_dim_OpC import opt_dim\n\
from despacho import *\n\
from Regulacion_F_UC_V6 import UC_ESS_with_Freq\n\
\n\
# Importing dashboard functions\n\
from section_Loc_Size_Arbitraje import dashboard_DLA\n\
from section_Loc_Size_operacion import dashboard_DLOp\n\
from section_Eco import dashboard_Eco\n\
from section_XM_document import dashboard_XMDoc\n\
####Cargando Datos######\n\
\n\
actual_path = os.getcwd()\n\
\n\
aux = actual_path.split('/')\n\
last_element = aux[-1]\n\
aux.remove(last_element)\n\
before_path = '/'.join(aux)\n\
\n\
DATA_URL_1=(os.path.join(before_path, 'data_base/Tech.csv'))\n\
DATA_URL_2=(os.path.join(before_path, 'data_base/inputs_eco.csv'))\n\
DATA_URL_3=(os.path.join(before_path, 'data_base/inputs_eco_opex.csv'))\n\
# @st.cache(persist=True)\n\
def load_data1():\n\
    data1=pd.read_csv(DATA_URL_1)\n\
    data1=data1.set_index('Tech')\n\
    return data1\n\
data1=load_data1()\n\
\n\
# @st.cache(persist=True)\n\
def load_data2():\n\
    data2=pd.read_csv(DATA_URL_2)\n\
    data2=data2.set_index('Tech')\n\
    return data2\n\
data2=load_data2()\n\
\n\
# @st.cache(persist=True)\n\
def load_data3():\n\
    data3=pd.read_csv(DATA_URL_3)\n\
    data3=data3.set_index('Tech')\n\
    return data3\n\
data3=load_data3()\n\
\n\
st.sidebar.title('Energy Storage Analysis Tool')\n\
st.sidebar.subheader('Elija el tipo de análisis que quiere efectuar:')\n\
study = st.sidebar.selectbox('',('','Dimensionamiento y Localización', 'Análisis financiero',\n\
        'Documento XM', 'Regulación de Frecuencia'))\n\
\n\
if study =='':\n\
    st.title('Energy Storage Analysis Tool')\n\
    image = Image.open('ess.jpg')\n\
    st.image(image, caption='',use_column_width=True)\n\
    st.markdown('Descripción general de la herramienta')\n\
\n\
################################################################################\n\
######################### Sección de Localización ##############################\n\
################################################################################\n\
if (study == 'Dimensionamiento y Localización'):\n\
\n\
    s_study = st.sidebar.selectbox('Tipo de dimensionamiento y localización:', ('', 'Arbitraje', 'Costo Operación/Congestión'))\n\
\n\
    if s_study == 'Arbitraje':\n\
        dashboard_DLA(data1)\n\
\n\
    elif s_study == 'Costo Operación/Congestión':\n\
        ss_study = st.sidebar.selectbox('Para minimizar costos de:', ('', 'Operación', 'Congestión', 'Ambos'))\n\
\n\
        if ss_study == 'Operación':\n\
\n\
            ope_fact = 1\n\
            con_fact = 0\n\
            info_ = 'Ope'\n\
            dashboard_DLOp(data1, info_, ope_fact, con_fact)\n\
\n\
        elif ss_study == 'Congestión':\n\
\n\
            ope_fact = 0\n\
            con_fact = 1\n\
            info_ = 'Con'\n\
            dashboard_DLOp(data1, info_, ope_fact, con_fact)\n\
\n\
        elif ss_study == 'Ambos':\n\
\n\
            ope_fact = 1\n\
            con_fact = 1\n\
            info_ = 'Both'\n\
            dashboard_DLOp(data1, info_, ope_fact, con_fact)\n\
\n\
################################################################################\n\
########################Sección de Modelo económico#############################\n\
################################################################################\n\
\n\
if (study=='Análisis financiero'):\n\
\n\
    dashboard_Eco(data2,data3)\n\
\n\
################################################################################\n\
############################## Documento XM ####################################\n\
################################################################################\n\
\n\
if study == 'Documento XM':\n\
\n\
    s_study = st.sidebar.selectbox('Versión del documento:', ('', 'Versión 1', 'Versión 2'))\n\
\n\
    if s_study == 'Versión 1':\n\
        dashboard_XMDoc(data1, s_study)\n\
    elif s_study == 'Versión 2':\n\
        dashboard_XMDoc(data1, s_study)\n\
\n\
    ################################################################################\n\
    ###############################Regulación de frecuencia#########################\n\
    ################################################################################\n\
\n\
if (study=='Regulación de Frecuencia'):\n\
    st.markdown(\u0022<h1 style='text-align: center; color: black;'>Regulación \u005c\n\
    de Frecuencia</h1>\u0022, unsafe_allow_html=True)\n\
    # image = Image.open('arbitrage.jpeg')\n\
    # st.image(image, caption='', use_column_width=True)\n\
    st.markdown('En esta sección de la herramienta el usuario podrá analizar \u005c\n\
    los beneficios de los SAE en la prestación de la regulación de frecuencia.')\n\
    st.markdown('## Parámetros seleccionados para la simulación')\n\
    st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
# Selección de tecnlogía de SAE\n\
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data1.index,key='1')\n\
    if technology=='New':\n\
        st.markdown('Ingrese las características del SAEB a simular:')\n\
        Eff=st.text_input('Ingrese la eficiencia del SAEB [pu]: ',key='1')\n\
        degra=st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ',key='2')\n\
        autoD=st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ',key='3')\n\
        DoD=st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ',key='4')\n\
        costP=st.text_input('Ingrese el costo por potencia [USD/MW]: ',key='5')\n\
        costE=st.text_input('Ingrese el costo por energía [USD/MWh]: ',key='6')\n\
    else:\n\
        st.markdown('El SAE seleccionado tiene las siguientes características:')\n\
        st.write(data1.loc[technology])\n\
        Eff=data1.iloc[data1.index.get_loc(technology),0]\n\
        degra=data1.iloc[data1.index.get_loc(technology),1]\n\
        autoD=data1.iloc[data1.index.get_loc(technology),2]\n\
        DoD=data1.iloc[data1.index.get_loc(technology),3]\n\
        costP=data1.iloc[data1.index.get_loc(technology),4]\n\
        costE=data1.iloc[data1.index.get_loc(technology),5]\n\
\n\
# Seleción de archivo con sistema\n\
    st.set_option('deprecation.showfileUploaderEncoding', False)\n\
    file_system = st.sidebar.file_uploader('Seleccione el archivo con el sistema a simular:', type=['csv','xlsx'])\n\
\n\
# Ingresar tiempo de simulación\n\
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)\n\
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')\n\
\n\
# Seleccionar solver\n\
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')\n\
    if solver=='CPLEX':\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
    else:\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
\n\
# Correr función de optimización\n\
    def run_frecuencia():\n\
        opt_results=UC_ESS_with_Freq(file_system,time_sim,solver)\n\
        return opt_results\n\
\n\
#############Simulation button###############################################\n\
\n\
    button_sent = st.sidebar.button('Simular')\n\
    if button_sent:\n\
\n\
        reserva=run_frecuencia()\n\
        st.write(reserva[6])\n\
")
app.close()

## section_XM_document.py

section_XM_document = open('Herramienta/codes/section_XM_document.py', 'w')
section_XM_document.write("import streamlit as st\n\
from os import path\n\
from os import remove\n\
import pandas as pd\n\
import numpy as np\n\
import matplotlib.pyplot as plt\n\
import pydeck as pdk\n\
import requests\n\
from bs4 import BeautifulSoup\n\
import re\n\
\n\
from despacho import *\n\
\n\
def dashboard_XMDoc(data1, s_study):\n\
\n\
    if s_study == 'Versión 1':\n\
\n\
        st.markdown(\u0022<h1 style='text-align: center; color: black;'>Despacho Colombiano uninodal con SAE (Versión 1)</h1>\u0022, unsafe_allow_html=True)\n\
        # image = Image.open('arbitrage.jpeg')\n\
        # st.image(image, caption='', use_column_width=True)\n\
        st.markdown('En esta sección de la herramienta el usuario podrá analizar la \u005c\n\
        operación de un SAE bajo el esquema actual de despacho económico uninodal de XM. \u005c\n\
        La formulación esta basada en la versión incial del documento de XM, publicado \u005c\n\
        en el mes de Noviembre de 2019.')\n\
        st.markdown('## Parámetros seleccionados para la simulación')\n\
        st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
    elif s_study == 'Versión 2':\n\
\n\
        st.markdown(\u0022<h1 style='text-align: center; color: black;'>Despacho Colombiano uninodal con SAE (Versión 2)</h1>\u0022, unsafe_allow_html=True)\n\
        # image = Image.open('arbitrage.jpeg')\n\
        # st.image(image, caption='', use_column_width=True)\n\
        st.markdown('En esta sección de la herramienta el usuario podrá analizar la \u005c\n\
        operación de un SAE bajo el esquema actual de despacho económico uninodal de XM. \u005c\n\
        La formulación esta basada en la segunda versión del documento de XM, publicado \u005c\n\
        en el mes de Septiembre de 2020.')\n\
        st.markdown('## Parámetros seleccionados para la simulación')\n\
        st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
# Selección de tecnlogía de SAE\n\
    technology = st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE', data1.index, key='1')\n\
    if technology == 'New':\n\
        st.markdown('Ingrese las características del SAE a simular:')\n\
        Eff = st.text_input('Ingrese la eficiencia del SAE [pu]: ', key='1')\n\
        degra = st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ', key='2')\n\
        autoD = st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ', key='3')\n\
        DoD = st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ', key='4')\n\
        costP = st.text_input('Ingrese el costo por potencia [USD/MW]: ', key='5')\n\
        costE = st.text_input('Ingrese el costo por energía [USD/MWh]: ', key='6')\n\
    else:\n\
        st.markdown('El SAE seleccionado tiene las siguientes características:')\n\
        st.write(data1.loc[technology])\n\
        Eff = data1.iloc[data1.index.get_loc(technology), 0]\n\
        degra = data1.iloc[data1.index.get_loc(technology), 1]\n\
        autoD = data1.iloc[data1.index.get_loc(technology), 2]\n\
        DoD = data1.iloc[data1.index.get_loc(technology), 3]\n\
        costP = data1.iloc[data1.index.get_loc(technology), 4]\n\
        costE = data1.iloc[data1.index.get_loc(technology), 5]\n\
\n\
    # Selección fecha\n\
    fecha = st.sidebar.date_input('Fecha de simulación')\n\
    st.write('La fecha de simulación seleccionada fue: ' + str(fecha))\n\
\n\
    # Descarga y lectura de archivos\n\
    file_ofertas, file_demanda, file_MPO = descarga_archivos(fecha)\n\
\n\
\n\
#Ingrese el tamaño del SAE\n\
    Pot_max = st.sidebar.number_input('Potencia [MW]')\n\
    E_max = st.sidebar.number_input('Energía [MWh]')\n\
    st.write('Potencia del SAE seleccionada: ' + str(Pot_max) + ' [MW]')\n\
    st.write('Energía del SAE seleccionada: ' + str(E_max) + ' [MWh]')\n\
\n\
# SoC MT\n\
    if s_study == 'Versión 2':\n\
        SoC_MT = st.sidebar.number_input('Estado de carga, mínimo técnico')\n\
        st.write('Mínimo técnico seleccionado: '+ str(SoC_MT) + ' [p.u.]')\n\
\n\
\n\
# tiempos de descarga previo y tiempos de descarga\n\
    tdp = st.sidebar.number_input('Tiempo de descarga previo', min_value=1, max_value=25)\n\
    td = st.sidebar.number_input('Tiempo de descarga', min_value=1, max_value=25)\n\
    st.write('Hora(s) de descarga previa seleccionada: ' + str(tdp) + ' [h]')\n\
    st.write('Hora(s) de descarga seleccionada: ' + str(td) + ' [h]')\n\
\n\
# Ingresar tiempo de simulación\n\
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)\n\
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')\n\
\n\
# Seleccionar solver\n\
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')\n\
    if solver=='CPLEX':\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
    else:\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
\n\
    if s_study == 'Versión 1':\n\
\n\
        if st.checkbox('Formulación Matemática'):\n\
            st.write('### Función Objetivo')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \u005cbegin{aligned}\n\
                        \min{} \sum_{t} \sum_{R} Pofe_{[R]}\cdot V_{GenRec[R][t]} +\n\
                        \sum_t \sum_{RT} PAP_{[RT]}\cdot B_{Arr[RT][t]} + \sum_t CROest\cdot V_{Rac[t]} + \u005c\u005c\n\
                        \sum_t \sum_s PC_{[s]}{[t]}\cdot V_{PC[s][t]} +\n\
                        \sum_{tpd} \sum_{s} CROest\cdot V_{DoC[s][tpd]}\n\
                    \end{aligned} $$\n\
                    \u0022\u0022\u0022)\n\
            st.write('### Restricciones')\n\
            st.write('#### Balance de generación demanda considerando almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \sum_R V_{GenRec[R][t]} + V_{Rac[t]} + \sum_s V_{PD[s][t]} = Dem_{[t]} +\n\
                    \sum_s V_{PC[s][t]} \hspace{2mm} \u005cforall t $$ \u0022\u0022\u0022)\n\
            st.write('#### Balance del almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} = V_{SoC[s][t-1]}\cdot \eta_{SoC[s]} +\n\
                    \eta_{c[s]}\cdot \dfrac{V_{PC[s][t]}}{Cap_{[s]}} - \dfrac{1}{\eta_{d[s]}} \cdot\n\
                    \dfrac{V_{PD[s][t]}}{Cap_{[s]}} \hspace{2mm} \u005cforall s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Balance de Estado de Carga')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{DoC[s][t]} = SoC_{max[s]} - V_{SoC[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Capacidad máxima de almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ SoC_{min[s]} \leq V_{SoC[s][t]} \leq SoC_{max[s]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Causalidad de la carga/descarga')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ B_{PC[s][t]} + B_{PD[s][t]} \leq 1 \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ 0 \leq V_{PC[s][t]} \leq PC_{max[s][t]} \cdot B_{PC[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ 0 \leq V_{PD[s][t]} \leq PD_{max[s][t]} \cdot B_{PD[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Carga y descarga requerida')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{PD[s][td]} = PD_{des[s][t]} \hspace{2mm} \u005cforall\n\
                    s, td $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{PC[s][td]} \geq PC_{des[s][td]} \hspace{2mm} \u005cforall\n\
                    s, td $$ \u0022\u0022\u0022)\n\
\n\
    elif s_study == 'Versión 2':\n\
\n\
        if st.checkbox('Formulación Matemática'):\n\
            st.write('### Función Objetivo')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \u005cbegin{aligned}\n\
                        \min{} \sum_{t} \sum_{r} Pofe_{[R]}\cdot V_{GenRec[r][t]} +\n\
                        \sum_t \sum_{rt} PAP_{[rt]}\cdot B_{Arr[rt][t]} + \sum_t CROest\cdot V_{Rac[t]} + \u005c\u005c\n\
                        \sum_t \sum_s PC_{[s]}{[t]}\cdot V_{PC[s][t]} +\n\
                        \sum_{tpd} \sum_{s} CROest\cdot Cap_{[s][tpd]} \cdot V_{SoD[s][tpd]}\n\
                    \end{aligned} $$\n\
                    \u0022\u0022\u0022)\n\
            st.write('### Restricciones')\n\
            st.write('#### Balance de generación demanda considerando almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \sum_r V_{GenRec[r][t]} + V_{Rac[t]} + \sum_s V_{PD[s][t]} \cdot ECS_{[s][t]} =\n\
                    Dem_{[t]} + \sum_s V_{PC[s][t]} \cdot ECS_{[s][t]} \hspace{2mm} \u005cforall t $$\n\
                    \u0022\u0022\u0022)\n\
            st.write('#### Balance del almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} = V_{SoC_E[s][t-1]} + ECS_{[s][t]} \cdot \left(\n\
                    \eta_{c[s]}\cdot \dfrac{V_{PC[s][t]}}{Cap_{[s]}} - \dfrac{1}{\eta_{d[s]}} \cdot\n\
                    \dfrac{V_{PD[s][t]}}{Cap_{[s]}} \u005cright) \hspace{2mm} \u005cforall s, t $$\n\
                    \u0022\u0022\u0022)\n\
            st.write('#### Afectación del estado de carga por eficiencia de almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ -(B_{PC[s][t]} + B_{PD[s][t]}) + V_{SoC[s][t]}\cdot (1 - \eta_{SoC[s]})\n\
                    \leq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall s, t $$\n\
                    \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} \cdot (1 - \eta_{SoC[s]}) + (B_{PC[s][t]} + B_{PD[s][t]})\n\
                    \geq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall s, t $$\n\
                    \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ -(1 - B_{PC[s][t]}) + V_{SoC[s][t]} \leq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} + (1 - B_{PC[s][t]}) \geq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ -(1 - B_{PD[s][t]}) + V_{Soc[s][t]} \leq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} + (1 - B_{PD[s][t]}) \geq V_{SoC_E[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Balance de Estado de Carga')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoD[s][t]} = 1 - V_{SoC[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Mínimo y máximo Estado de Carga del almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ SoC_{min[s]} \leq V_{SoC[s][t]} \leq SoC_{max[s]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Mínimo técnico del sistema de almacenamiento')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{SoC[s][t]} \geq SoC_{MT[s]} \hspace{2mm} \u005cforall\n\
                    s, t $$ \u0022\u0022\u0022)\n\
            st.write('#### Causalidad de la carga/descarga')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ B_{PC[s][t]} + B_{PD[s][t]} \leq 1 \hspace{2mm} \u005cforall\n\
                    s, t, ECS_{[s][t]} = 1 $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ 0 \leq V_{PC[s][t]} \leq PC_{max[s][t]} \cdot B_{PC[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t, ECS_{[s][t]} = 1 $$ \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ 0 \leq V_{PD[s][t]} \leq PD_{max[s][t]} \cdot B_{PD[s][t]} \hspace{2mm} \u005cforall\n\
                    s, t, ECS_{[s][t]} = 1 $$ \u0022\u0022\u0022)\n\
            st.write('#### Carga y descarga requerida')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$V_{PD[s][t]} = \u005cbegin{cases}\n\
                    PD_{req[s][tr]} & \u005cforall s, tr, ECS_{[s][t]} = 1 \u005c\u005c\n\
                    0 & \u005cforall s, t \u005cneq tr, ECS_{[s][t]} = 1\n\
                    \end{cases}$$\n\
                    \u0022\u0022\u0022)\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ V_{PC[s][td]} \geq PC_{des[s][td]} \hspace{2mm} \u005cforall\n\
                    s, td $$ \u0022\u0022\u0022)\n\
\n\
        # st.write(\u0022\u0022\u0022\n\
        #     Función Objetivo:\n\
        #     $\sum_{t}\sum_{R} = Pofe_{[R]}\cdot V_GenRec_{[R][t]}$\n\
        #         \u0022\u0022\u0022)\n\
\n\
# Correr función de optimización\n\
\n\
    if s_study == 'Versión 1':\n\
\n\
        def run_despacho():\n\
\n\
            opt_results=opt_despacho(fecha, file_ofertas, file_demanda, file_MPO, Pot_max, E_max, Eff, Eff, autoD, 1-DoD, tdp, td, solver)\n\
            #Imprimir resultados\n\
            st.title('Resultados:')\n\
            SOC, tiempo = opt_results\n\
            st.write('Tiempo de simulación: ' + str(tiempo))\n\
\n\
            axis_font = {'fontname':'Microsoft Sans Serif', 'size':'12'}\n\
\n\
            fig, ax = plt.subplots()\n\
            ax.plot(SOC)\n\
            box = ax.get_position()\n\
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9 ])\n\
            ax.legend(['Estado de Carga'], bbox_to_anchor=(0.65, -0.18), fancybox=True, shadow=True, ncol=10)\n\
            plt.xticks(**axis_font)\n\
            plt.yticks(**axis_font)\n\
            plt.grid(True)\n\
            plt.xlabel('Tiempo [h]', **axis_font)\n\
            plt.ylabel('[MWh]', **axis_font)\n\
            st.pyplot(fig)\n\
\n\
    elif s_study == 'Versión 2':\n\
\n\
        def run_despacho():\n\
\n\
            opt_results = opt_despacho_2(fecha, file_ofertas, file_demanda, Pot_max, E_max, Eff, Eff, autoD, 1-DoD, SoC_MT, tdp, td, solver)\n\
            #Imprimir resultados\n\
            st.title('Resultados:')\n\
            SOC, tiempo = opt_results\n\
            st.write('Tiempo de simulación: ' + str(tiempo))\n\
\n\
            axis_font = {'fontname':'Microsoft Sans Serif', 'size':'12'}\n\
\n\
            fig, ax = plt.subplots()\n\
            ax.plot(SOC)\n\
            box = ax.get_position()\n\
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9 ])\n\
            ax.legend(['Estado de Carga'], bbox_to_anchor=(0.65, -0.18), fancybox=True, shadow=True, ncol=10)\n\
            plt.xticks(**axis_font)\n\
            plt.yticks(**axis_font)\n\
            plt.grid(True)\n\
            plt.xlabel('Tiempo [h]', **axis_font)\n\
            plt.ylabel('[MWh]', **axis_font)\n\
            st.pyplot(fig)\n\
\n\
#############Simulation button###############################################\n\
    button_sent = st.sidebar.button('Simular')\n\
    if button_sent:\n\
        # if path.exists('Resultados/resultados_size_loc.xlsx'):\n\
        #     remove('Resultados/resultados_size_loc.xlsx')\n\
        run_despacho()\n\
        ")
section_XM_document.close()

## section_Loc_Size_operacion.py

section_Loc_Size_operacion = open('Herramienta/codes/section_Loc_Size_operacion.py' , 'w')
section_Loc_Size_operacion.write("# coding: utf-8\n\
#Librería de intersaz\n\
import streamlit as st\n\
# Librerías para maniular datos\n\
from os import path\n\
from os import remove\n\
# Librearias para manejo de datos\n\
import pandas as pd\n\
import numpy as np\n\
# Librarías para graficas\n\
import matplotlib.pyplot as plt\n\
import pydeck as pdk\n\
# Libraries for reading data form internet\n\
import requests\n\
from bs4 import BeautifulSoup\n\
import re\n\
# Importing optimization functions\n\
from Loc_dim_OpC import opt_dim\n\
\n\
def dashboard_DLOp(data1, info_, ope_fact, con_fact):\n\
\n\
    if info_ == 'Ope':\n\
\n\
        st.markdown(\u0022<h1 style='text-align: center; color: black;'>Dimensionamiento \u005c\n\
        y localización de SAE basado en reducción de costos de operación</h1>\u0022, unsafe_allow_html=True)\n\
        # image = Image.open('arbitrage.jpeg')\n\
        # st.image(image, caption='', use_column_width=True)\n\
        st.markdown('En esta sección de la herramienta el usuario podrá encontrar el \u005c\n\
        tamaño óptimo de SAE para cada nodo del sistema (en términos de potencia y energía) \u005c\n\
        con el fin de reducir los costos de operación. Para este fin, el usuario deberá ingresar un archivo con \u005c\n\
        los datos del sistema de potencia a analizar, el horizonte de tiempo de la simulación,\u005c\n\
        la tecnología de SAE y el tipo de solver.')\n\
\n\
    elif info_ == 'Con':\n\
\n\
        st.markdown(\u0022<h1 style='text-align: center; color: black;'> Dimensionamiento y localización de SAE basado en reducción \u005c\n\
            de costos por congestión </h1>\u0022, unsafe_allow_html=True)\n\
\n\
    elif info_ == 'Both':\n\
\n\
        st.markdown(\u0022<h1 style='text-align: center; color: black;'> Dimensionamiento y localización de SAE basado en reducción \u005c\n\
            de costos de operación y costos por congesstión</h1>\u0022, unsafe_allow_html=True)\n\
\n\
    st.markdown('## Parámetros seleccionados para la simulación')\n\
    st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
# Selección de tecnlogía de SAE\n\
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data1.index,key='1')\n\
    if technology=='New':\n\
        st.markdown('Ingrese las características del SAEB a simular:')\n\
        Eff=st.text_input('Ingrese la eficiencia del SAEB [pu]: ',key='1')\n\
        degra=st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ',key='2')\n\
        autoD=st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ',key='3')\n\
        DoD=st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ',key='4')\n\
        costP=st.text_input('Ingrese el costo por potencia [USD/MW]: ',key='5')\n\
        costE=st.text_input('Ingrese el costo por energía [USD/MWh]: ',key='6')\n\
    else:\n\
        st.markdown('El SAE seleccionado tiene las siguientes características:')\n\
        st.dataframe(data1.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))\n\
        Eff=data1.iloc[data1.index.get_loc(technology),0]\n\
        degra=data1.iloc[data1.index.get_loc(technology),1]\n\
        autoD=data1.iloc[data1.index.get_loc(technology),2]\n\
        DoD=data1.iloc[data1.index.get_loc(technology),3]\n\
        costP=data1.iloc[data1.index.get_loc(technology),4]\n\
        costE=data1.iloc[data1.index.get_loc(technology),5]\n\
\n\
# Seleción de archivo con precios\n\
    st.set_option('deprecation.showfileUploaderEncoding', False)\n\
    file_system = st.sidebar.file_uploader('Seleccione el archivo con el sistema a simular:', type=['csv','xlsx'])\n\
\n\
# Costo por Congestion\n\
\n\
    con_cost = 0\n\
\n\
    if info_ == 'Con' or info_ == 'Both':\n\
        con_cost = st.sidebar.number_input('Ingrese el costo por congestión [$/MWh]')\n\
        st.write('El costo por congestión seleccionado fue: ' + str(con_cost) + ' [$/MWh]')\n\
\n\
# Ingresar tiempo de simulación\n\
    time_sim = st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=100000)\n\
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')\n\
\n\
# Seleccionar solver\n\
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')\n\
    if solver=='CPLEX':\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
    else:\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
\n\
# Formulación\n\
\n\
    if st.checkbox('Formulación Matemática'):\n\
\n\
        if info_ == 'Ope':\n\
\n\
            st.write('### Función Objetivo')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \u005cbegin{aligned}\n\
                        \min \u005cunderbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}\n\
                        \left( p_{i,b,t}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +\n\
                        C_{i}^{up}\cdot SD_{i,t}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +\n\
                        \u005cunderbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}} \sum_{t \in \mathcal{T}}\n\
                        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \u005c\u005c\n\
                        \u005cunderbrace{\sum_{n\in \mathcal{N}} \sum_{b\in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +\n\
                        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}\n\
                    \end{aligned}$$\n\
                \u0022\u0022\u0022)\n\
\n\
        if info_ == 'Con':\n\
\n\
            st.write('### Función Objetivo')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \u005cbegin{aligned}\n\
                        \min \u005cunderbrace{\sum_{(b, r) \in \mathcal{L}}\n\
                        \sum_{t \in \mathcal{T}} \left(\left[p_{b,r,t}^{pf} + \u005cfrac{1}{2} q_{b,r,t}^{pf}\u005cright]\n\
                        - P_{b,r}^{max} \u005cright) \cdot C_{t}^{con}}_{Costos\hspace{1mm}por\hspace{1mm}congestión}\n\
                    \end{aligned}$$\n\
                \u0022\u0022\u0022)\n\
\n\
        if info_ == 'Both':\n\
\n\
            st.write('### Función Objetivo')\n\
            st.write(r\u0022\u0022\u0022\n\
                    $$ \u005cbegin{aligned}\n\
                        \min \u005cunderbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}\n\
                        \left( p_{i,b,t}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +\n\
                        C_{i}^{up}\cdot SD_{i,t}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +\n\
                        \u005cunderbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}} \sum_{t \in \mathcal{T}}\n\
                        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \u005c\u005c\n\
                        \u005cunderbrace{\sum_{n\in \mathcal{N}} \sum_{b\in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +\n\
                        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\u005cright)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} + \u005cunderbrace{\sum_{(b, r) \in \mathcal{L}}\n\
                        \sum_{t \in \mathcal{T}} \left(\left[p_{b,r,t}^{pf} + \u005cfrac{1}{2} q_{b,r,t}^{pf}\u005cright]\n\
                        - P_{b,r}^{max} \u005cright) \cdot C_{t}^{con}}_{Costos\hspace{1mm}por\hspace{1mm}congestión}\n\
                    \end{aligned}$$\n\
                \u0022\u0022\u0022)\n\
\n\
        st.write('### Restricciones')\n\
        st.write('#### Restricciones del sistema')\n\
        st.write('Balance de potencia')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ \u005cbegin{aligned}\n\
                    \sum_{i \in \mathcal{I}_{b}}p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}}p_{j,b,t}^{hyd} +\n\
                    \sum_{w \in \mathcal{W}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}\n\
                    \left(p_{b,r,t}^{pf} + \u005cfrac{1}{2}q_{b,r,t}^{pf}\u005cright) \u005c\u005c + \sum_{n\in \mathcal{N}_b}\n\
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \u005cright) = D_{b,t}^{f} \hspace{2mm} \u005cforall\n\
                    b \in \mathcal{B}, t \in \mathcal{T}\n\
                \end{aligned}$$\n\
                \u0022\u0022\u0022)\n\
        st.write('#### Restricciones Generación térmica')\n\
        st.write('Límites en la capacidad de generación térmica')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ P_{i}^{min} \leq p_{i,b,t}^{th} \leq P_{i}^{max} \hspace{2mm} \u005cforall\n\
                t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$ \u0022\u0022\u0022)\n\
        st.write('Rampas de generadores térmicos')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{i,t+1}^{th} - p_{i,t}^{th} \leq R_{i}^{up} \cdot x_{i,t} t + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}\n\
                \u005cforall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{i,t}^{th} - p_{i,t+1}^{th} \leq R_{i}^{dn} \cdot x_{i,t} t + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}\n\
                \u005cforall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$ \u0022\u0022\u0022)\n\
        st.write('Variables binarias de operación de unidades térmicas')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \u005cforall\n\
                t \in \mathcal{T}, i \in \mathcal{I} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \u005cforall\n\
                t \in \mathcal{T}, i \in \mathcal{I} $$ \u0022\u0022\u0022)\n\
        st.write('Tiempos mínimos de encendido y apagado de generadores térmicos')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ x_{i,t} = g_{i,t}^{on/off} \hspace{2mm} \u005cforall\n\
                t \in \left(L_{i}^{up,min}+L_{i}^{dn,min}\u005cright), i \in \mathcal{I} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ \sum_{tt=t-g_{i}^{up}+1}^{t} SU_{i,tt} \leq x_{i,tt} \hspace{2mm} \u005cforall\n\
                t \geq L_{i}^{up,min} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ \sum_{tt=t-g_{i}^{dn}+1}^{t} SD_{i,tt} \leq 1-x_{i,tt} \hspace{2mm} \u005cforall\n\
                t \geq L_{i}^{dn,min} $$ \u0022\u0022\u0022)\n\
        st.write('#### Restricciones Generación hidráulica')\n\
        st.write('Límites en la capacidad de generación hidráulica')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ P_{j}^{min} \leq p_{j,b,t}^{hyd} \leq P_{j}^{max} \hspace{2mm} \u005cforall\n\
                t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$ \u0022\u0022\u0022)\n\
        st.write('Unidades hidráulicas de generación')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ Q_{j}^{min} \leq q_{j,t} \leq Q_{j}^{max} \hspace{2mm} \u005cforall\n\
                j \in \mathcal{J}, t \in \mathcal{T}$$\n\
                \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ V_{j}^{min} \leq v_{j,t} \leq V_{j}^{max} \hspace{2mm} \u005cforall\n\
                j \in \mathcal{J}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ 0 \leq s_{j,t} \leq Q_{j,t} \hspace{2mm} \u005cforall\n\
                j \in \mathcal{J}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ v_{j,t} = v_{j,t-1} + 3600 \Delta t \left(I_{t} - \sum_{j \in \mathcal{J}} q_{j,t} -\n\
                s_{j,t} \u005cright) \hspace{2mm} j \in \mathcal{J}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ P_{j,t}^{hyd} = H_{j} \cdot q_{j,t} \hspace{2mm} \u005cforall\n\
                j \in \mathcal{J}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('#### Restricciones generación renovable')\n\
        st.write('Curvas de generación de unidades renovables')\n\
        st.write('Límites de generación en unidades renovables')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{w,b,t}^{ren} \leq P_{w,t}^{f} \hspace{2mm} \u005cforall\n\
                w \in \mathcal{W}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('#### Restricciones flujo de potencia DC y pérdidas')\n\
        st.write('Cálculo del flujo de potencia por cada línea')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \u005cright) \hspace{2mm} \u005cforall\n\
                \left(b,r \u005cright) \in \mathcal{L}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('Cálculo de las pérdidas eléctricas de cada línea')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ q_{b,r,t}^{pf} = G_{b,r} \cdot \left(\delta_{b} - \delta_{r} \u005cright)^2 \hspace{2mm}\n\
                \u005cforall \left(s,r,l \u005cright) \in \mathcal{L}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ \delta_{b,r}^{+} + \delta_{b,r}^{-} = \sum_{k=1}^{K} \delta_{b,r}(k) \hspace{2mm}\n\
                k = 1,...,K $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ \alpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}\n\
                k = 1, ... , K $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ q_{b,r,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \u005calpha_{b,r}(k)\cdot \delta_{b,r}(k)\n\
                \hspace{2mm} \u005cforall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('Límites en el flujo de potencia en las líneas')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} + \u005cfrac{1}{2} \cdot q_{b,r,t}^{pf} \leq P_{b,r}^{max}\n\
                \hspace{2mm} \u005cforall l \in \mathcal{L}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('#### Restricciones sistemas de almacenamiento de energía basados en baterías')\n\
        st.write('Variables binarias de estado de los SAEB')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('Relación entre la potencia y energía de los SAEB')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ e_{n,b,t} = e_{n,b,t-1}\cdot \eta_{n}^{SoC} + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -\n\
                \u005cfrac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \u005cright)\cdot \Delta t \hspace{2mm} \u005cforall\n\
                b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$ \u0022\u0022\u0022)\n\
        st.write('Límite de energía de los SAEB')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write('Límite de potencia de los SAEB')\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,t}^{ch} \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{n,b,t}^{ch} \leq P_{n,b} \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{n,b,t}^{dc} \leq Z \cdot u_{n,t}^{dc} \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
        st.write(r\u0022\u0022\u0022\n\
                $$ p_{n,b,t}^{dc} \leq P_{n,b} \hspace{2mm} \u005cforall\n\
                n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$ \u0022\u0022\u0022)\n\
\n\
# Correr función de optimización\n\
    def run_dim_size_OpC():\n\
        opt_results = opt_dim(file_system, Eff, DoD, 0.2, time_sim, 20, costP, costE, ope_fact, con_fact, con_cost, solver)\n\
        #Imprimir resultados\n\
        st.title('Resultados:')\n\
        st.write('Tiempo de simulación: ' + str(opt_results[3]))\n\
        Power=pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Power_size', header=0, index_col=0)\n\
        Power=Power.rename(columns={0: 'Potencia [MW]'})\n\
        Energy= pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Energy_size', header=0, index_col=0)\n\
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})\n\
        Size=result = pd.concat([Power,Energy], axis=1)\n\
        for i in range(1,len(Size)+1):\n\
            if Size.loc[i-1,'Potencia [MW]']>0:\n\
                Size=Size.rename(index={i-1:'b'+'%i'%i})\n\
            else:\n\
                Size=Size.drop([i-1])\n\
        st.dataframe(Size.style.format('{:.2f}'))\n\
        file_results=opt_results[2]\n\
        return file_results\n\
#############Simulation button###############################################\n\
    button_sent = st.sidebar.button('Simular')\n\
    if button_sent:\n\
        if path.exists('Resultados/resultados_size_loc.xlsx'):\n\
            remove('Resultados/resultados_size_loc.xlsx')\n\
        run_dim_size_OpC()\n\
\n\
################################ PLOTS  #####################################\n\
    button_plot= st.sidebar.checkbox('Graficar',value=False)\n\
    if button_plot:\n\
        #Imprimir resultados\n\
        st.title('Resultados:')\n\
        Power=pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Power_size', header=0, index_col=0)\n\
        Power=Power.rename(columns={0: 'Potencia [MW]'})\n\
        Energy= pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name='Energy_size', header=0, index_col=0)\n\
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})\n\
        Size= pd.concat([Power,Energy], axis=1)\n\
        for i in range(1,len(Size)+1):\n\
            if Size.loc[i-1,'Potencia [MW]']>0:\n\
                Size=Size.rename(index={i-1:'b'+'%i'%i})\n\
            else:\n\
                Size=Size.drop([i-1])\n\
        st.dataframe(Size.style.format('{:.2f}'))\n\
        ######\n\
        #Selección de tipo de gráfica\n\
        tipo = st.sidebar.selectbox('Seleccione el tipo de gráfica que desea visualizar',['Despacho SAE','SOC SAE','Flujo en lineas','Mapa'])\n\
        colores = ['#53973A','#4A73B2','#B7C728','#77ABBD','#FF7000','#1f77b4', '#aec7e8',\n\
                    '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a','#d62728', '#ff9896',\n\
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94','#e377c2', '#f7b6d2',\n\
                    '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']\n\
\n\
        flag_mapa=True\n\
        if tipo =='Despacho SAE':\n\
            sheet='BESS_Ch_Power'\n\
            title='Despacho de los SAE'\n\
            label='Potencia [MW]'\n\
            id='b'\n\
            plot2=True\n\
            sheet2='BESS_Dc_Power'\n\
            id1='Pc_'\n\
            id2='Pd_'\n\
        elif tipo =='SOC SAE':\n\
            sheet='BESS_Energy'\n\
            title='Estado de carga de los SAE'\n\
            label='Energía [MWh]'\n\
            id='b'\n\
            id1=''\n\
            plot2=False\n\
        elif tipo =='Flujo en lineas':\n\
            sheet='pf'\n\
            title='Flujo de potencia en las lineas'\n\
            label='Potencia [MW]'\n\
            id='L'\n\
            id1=''\n\
            plot2=False\n\
        else:\n\
\n\
#################################################################################\n\
##########################    MAP     ##########################################\n\
            # DATA_URL = {\n\
            #     'AIRPORTS': 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/line/airports.json',\n\
            #     'FLIGHT_PATHS': 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/line/heathrow-flights.json',  # noqa\n\
            # }\n\
            # LINES=json.load('lines.json')\n\
\n\
            map_data_1 = pd.read_excel(file_system, sheet_name='Bus', header=0,usecols=['lat','lon'] )\n\
            map_data_2 = pd.read_excel(file_system, sheet_name='Branch', header=0,usecols=['start','end'] )\n\
            a_list=[0]*len(Power)\n\
            colores=[0]*len(Power)\n\
            for i in range(1,len(Power)+1):\n\
                if Power.loc[i-1,'Potencia [MW]']>0:\n\
                    a_list[i-1]=40000\n\
                    colores[i-1]=[10, 230, 120]\n\
                else:\n\
                    a_list[i-1]=15000\n\
                    colores[i-1]=[230, 158, 10]\n\
            # a_list=[1,1,1,1,1,1,1,1,1,1,1,1,5,1,1]\n\
            map_data_1['exits_radius']=a_list\n\
            map_data_1['color']=colores\n\
            midpoint = (np.average(map_data_1['lat']), np.average(map_data_1['lon']))\n\
            df=map_data_1\n\
            st.pydeck_chart(pdk.Deck(\n\
             map_style='mapbox://styles/mapbox/light-v9',\n\
             initial_view_state=pdk.ViewState(\n\
                 latitude= midpoint[0],\n\
                 longitude= midpoint[1],\n\
                 zoom=4,\n\
                 # pitch=50,  # inclinación del mapa\n\
             ),\n\
             layers=[\n\
                    # pdk.Layer(\n\
                    # 'LineLayer',\n\
                    # # LINES,\n\
                    # DATA_URL['FLIGHT_PATHS'],\n\
                    # # map_data_2,\n\
                    # get_source_position='start',\n\
                    # get_target_position='end',\n\
                    # get_color=[10, 230, 120],\n\
                    # get_width=10,\n\
                    # highlight_color=[255, 255, 0],\n\
                    # picking_radius=10,\n\
                    # auto_highlight=True,\n\
                    # pickable=True,\n\
                    # ),\n\
\n\
                    pdk.Layer(\n\
                     'ScatterplotLayer',\n\
                     data=df,\n\
                     get_position='[lon, lat]',\n\
                     get_color='color',\n\
                     get_radius='exits_radius',\n\
                    ),\n\
             ],\n\
            ))\n\
\n\
            flag_mapa=False\n\
####################################################################################\n\
\n\
        if flag_mapa:\n\
            # Lectura de datos\n\
            df_results = pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name=sheet, header=0, index_col=0)\n\
            results = df_results.to_numpy()\n\
            results = np.absolute(results)\n\
\n\
            # Lectura de otros datos de ser necesario (Caso carga/ descarga)\n\
            if plot2:\n\
                df_results2 = pd.read_excel('Resultados/resultados_size_loc.xlsx', sheet_name=sheet2, header=0, index_col=0)\n\
                results2 = df_results2.to_numpy()\n\
                results2 = np.absolute(results2)\n\
                st.sidebar.markdown('Seleccione la/las variable(s) que desea visualizar:')\n\
                elemento1=st.sidebar.checkbox(id1,value=True,key=0)\n\
                elemento2=st.sidebar.checkbox(id2,value=True,key=0)\n\
\n\
            # checkbox para seleccionar/deseleccionar todos los elementos\n\
            st.sidebar.markdown('Seleccione el/los elemento(s) que desea visualizar:')\n\
            Todas=st.sidebar.checkbox('Todas',value=True,key=0)\n\
\n\
            Flag= np.ones(len(results))\n\
\n\
            if Todas:\n\
                for i in range(0,len(results)):\n\
                    Flag[i]=st.sidebar.checkbox(id+'%s'%(i+1),value=True,key=i+1)\n\
            else:\n\
                for i in range(0,len(results)):\n\
                    Flag[i]=st.sidebar.checkbox(id+'%s'%(i+1),value=False,key=i+1)\n\
\n\
            plt.figure(figsize=(10,6))\n\
            for i in range(1,len(results) + 1):\n\
                if Flag[i-1]:\n\
                    if plot2:\n\
                        if elemento1:\n\
                            plt.step(list(range(0,len(df_results.columns))),results[i-1,:], label=id1+id+'%s'%i)\n\
                        if elemento2:\n\
                            plt.step(list(range(0,len(df_results2.columns))),results2[i-1,:], label=id2+id+'%s'%i)\n\
                    else:\n\
                        plt.step(list(range(0,len(df_results.columns))),results[i-1,:], label=id1+id+'%s'%i)\n\
\n\
            plt.xlabel('Tiempo [h]')\n\
            plt.ylabel(label).set_fontsize(15)\n\
            plt.ylabel(label).set_fontsize(15)\n\
            plt.legend(fontsize='x-large')\n\
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=8, fancybox=True, shadow=True)\n\
            # plt.xlabel('Tiempo [h]').set_fontsize(15)\n\
            plt.xticks(size = 'x-large')\n\
            plt.yticks(size = 'x-large')\n\
            plt.title(title, fontsize=20)\n\
            st.pyplot()\n\
")
section_Loc_Size_operacion.close()

## section_Loc_Size_Arbitraje.py

section_Loc_Size_Arbitraje = open('Herramienta/codes/section_Loc_Size_Arbitraje.py', 'w')
section_Loc_Size_Arbitraje.write("#Librería de intersaz\n\
import streamlit as st\n\
# Librerías para maniular datos\n\
from os import path\n\
from os import remove\n\
# Librearias para manejo de datos\n\
import pandas as pd\n\
import numpy as np\n\
# Librarías para graficas\n\
import matplotlib.pyplot as plt\n\
# Libraries for reading data form internet\n\
import requests\n\
from bs4 import BeautifulSoup\n\
import re\n\
# Importing optimization functions\n\
from arbitraje import opt\n\
\n\
def dashboard_DLA(data1):\n\
\n\
    st.markdown(\u0022<h1 style='text-align: center; color: black;'>Dimensionamiento y Localización de SAE basado en Arbitraje</h1>\u0022, unsafe_allow_html=True)\n\
    # image = Image.open('arbitrage.jpeg')\n\
    # st.image(image, caption='', use_column_width=True)\n\
    st.markdown('En esta sección de la herramienta el usuario podrá encontrar el \u005c\n\
    tamaño óptimo del SAE (en términos de potencia y energía) para realizar la\u005c\n\
    función de arbitraje en un mercado uninodal. Para este fin, el usuario deberá ingresar un archivo con \u005c\n\
    el histórico de precios a analizar, el horizonte de tiempo de la simulación, la TMR,\u005c\n\
    la tecnología de SAE y el tipo de solver.')\n\
    st.markdown('## Parámetros seleccionados para la simulación')\n\
    st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
    # Selección de tecnlogía de SAE\n\
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data1.index,key='1')\n\
    if technology=='New':\n\
        st.markdown('Ingrese las características del SAEB a simular:')\n\
        Eff=st.text_input('Ingrese la eficiencia del SAEB [pu]: ',key='1')\n\
        degra=st.text_input('Ingrese el porcentaje de degradación por ciclo [%/ciclo]: ',key='2')\n\
        autoD=st.text_input('Ingrese el valor de autodescarga por hora [%/h]: ',key='3')\n\
        DoD=st.text_input('Ingrese la profundidad de descarga (DoD) [pu]: ',key='4')\n\
        costP=st.text_input('Ingrese el costo por potencia [USD/MW]: ',key='5')\n\
        costE=st.text_input('Ingrese el costo por energía [USD/MWh]: ',key='6')\n\
    else:\n\
        st.markdown('El SAE seleccionado tiene las siguientes características:')\n\
        st.dataframe(data1.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))\n\
        Eff=data1.iloc[data1.index.get_loc(technology),0]\n\
        degra=data1.iloc[data1.index.get_loc(technology),1]\n\
        autoD=data1.iloc[data1.index.get_loc(technology),2]\n\
        DoD=data1.iloc[data1.index.get_loc(technology),3]\n\
        costP=data1.iloc[data1.index.get_loc(technology),4]\n\
        costE=data1.iloc[data1.index.get_loc(technology),5]\n\
\n\
    # Seleeción de archivo con precios\n\
\n\
    st.set_option('deprecation.showfileUploaderEncoding', False)\n\
    file_prices = st.sidebar.file_uploader('Seleccione el archivo con históricos de precios de bolsa:', type=['csv','xlsx'])\n\
\n\
    # Ingresar TRM\n\
    TRM_select=st.sidebar.selectbox('Seleccione la TRM para la simulación',['Hoy','Otra'],key='2')\n\
    if TRM_select=='Hoy':\n\
        URL = 'https://www.dolar-colombia.com/'\n\
        page = requests.get(URL)\n\
        soup = BeautifulSoup(page.content, 'html.parser')\n\
        rate = soup.find_all(class_='exchange-rate')\n\
        TRM=str(rate[0])\n\
        TRM=re.findall('\d+', TRM )\n\
        TRM_final=TRM[0]+','+TRM[1]+'.'+TRM[2]\n\
        TRM_final_1='La TRM seleccionada para la simulación es de: '+TRM_final + ' COP'\n\
        st.write(TRM_final_1)\n\
    else:\n\
        TRM_final=st.text_input('Ingrese la TRM para la simulación: ')\n\
\n\
    # Ingresar tiempo de simulación\n\
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [h]:', min_value=1, max_value=10000000)\n\
    st.write('El horizonte de simulación es de: '+str(time_sim)+'h')\n\
\n\
    # Seleccionar solver\n\
    solver=st.sidebar.selectbox('Seleccione el tipo de Solver',['CPLEX','GLPK'],key='1')\n\
    if solver=='CPLEX':\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
    else:\n\
        st.write('El solucionador seleccionado es: '+solver)\n\
\n\
    # Correr función de optimización\n\
    def run_arbitraje():\n\
        opt_results=opt(file_prices,Eff,degra,autoD,DoD,DoD,\n\
        costP,costE,30000000,TRM_final,time_sim,solver)\n\
        File='Resultados/resultados_size_loc_arbitraje.xlsx'\n\
        #Imprimir resultados\n\
        st.title('Resultados:')\n\
        Power=pd.read_excel(File, sheet_name='Power_size', header=0, index_col=0)\n\
        Power=Power.rename(columns={0: 'Potencia [MW]'})\n\
        Energy= pd.read_excel(File, sheet_name='Energy_size', header=0, index_col=0)\n\
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})\n\
        Size= pd.concat([Power,Energy], axis=1)\n\
        st.dataframe(Size.style.format('{:.2f}'))\n\
\n\
    #############Simulation button###############################################\n\
    button_sent = st.sidebar.button('Simular')\n\
    if button_sent:\n\
        if path.exists('Resultados/resultados_size_loc_arbitraje.xlsx'):\n\
            remove('Resultados/resultados_size_loc_arbitraje.xlsx')\n\
        run_arbitraje()\n\
\n\
    ################################ PLOTS  #####################################\n\
    button_plot= st.sidebar.checkbox('Graficar',value=False)\n\
    if button_plot:\n\
        File='Resultados/resultados_size_loc_arbitraje.xlsx'\n\
        #Imprimir resultados\n\
        st.title('Resultados:')\n\
        Power=pd.read_excel(File, sheet_name='Power_size', header=0, index_col=0)\n\
        Power=Power.rename(columns={0: 'Potencia [MW]'})\n\
        Energy= pd.read_excel(File, sheet_name='Energy_size', header=0, index_col=0)\n\
        Energy=Energy.rename(columns={0: 'Energía [MWh]'})\n\
        Size= pd.concat([Power,Energy], axis=1)\n\
        st.dataframe(Size.style.format('{:.2f}'))\n\
        ######\n\
        #Selección de tipo de gráfica##########################################\n\
        tipo = st.sidebar.selectbox('Seleccione el tipo de gráfica que desea visualizar',['Despacho SAE','SOC SAE','Precios','Ingresos'])\n\
        colores = ['#53973A','#4A73B2','#B7C728','#77ABBD','#FF7000','#1f77b4','#aec7e8',\n\
                    '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a','#d62728','#ff9896',\n\
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94','#e377c2', '#f7b6d2',\n\
                    '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d','#17becf', '#9edae5']\n\
\n\
        if tipo =='Despacho SAE':\n\
            sheet='BESS_Ch_Power'\n\
            title='Despacho de los SAE'\n\
            label='Potencia [MW]'\n\
            plot2=True\n\
            sheet2='BESS_Dc_Power'\n\
            id1='Pc_'\n\
            id2='Pd_'\n\
        elif tipo =='SOC SAE':\n\
            sheet='BESS_Energy'\n\
            title='Estado de carga del SAE'\n\
            label='Energía [MWh]'\n\
            id1='SOC'\n\
            plot2=False\n\
        elif tipo =='Ingresos':\n\
            sheet='Ingresos'\n\
            title='Ingresos'\n\
            label='MCOP'\n\
            id1='Ingresos'\n\
            plot2=False\n\
            Suma=pd.read_excel(File, sheet_name=sheet, header=0, index_col=0).sum(axis = 0, skipna = True)\n\
            st.write('El total de ingresos neto durante el horizonte de tiempo fue de: '+'%i'%Suma)\n\
        else:\n\
            File=file_prices\n\
            sheet='Precios'\n\
            title='Precio de bolsa'\n\
            label='Precio [COP/kWh]'\n\
            plot2=False\n\
            id1='Precio de bolsa'\n\
\n\
        df_results = pd.read_excel(File, sheet_name=sheet, header=0, index_col=0)\n\
        results = df_results.to_numpy()\n\
        # results = np.absolute(results)\n\
\n\
        # Lectura de otros datos de ser necesario (Caso carga/ descarga)\n\
        if plot2:\n\
            df_results2 = pd.read_excel(File, sheet_name=sheet2, header=0, index_col=0)\n\
            results2 = df_results2.to_numpy()\n\
            results2 = np.absolute(results2)\n\
            st.sidebar.markdown('Seleccione la/las variable(s) que desea visualizar:')\n\
            elemento1=st.sidebar.checkbox(id1,value=True,key=0)\n\
            elemento2=st.sidebar.checkbox(id2,value=True,key=0)\n\
        st.sidebar.markdown('Seleccione el rango de tiempo que desea visualizar:')\n\
        values = st.sidebar.slider('',0, time_sim, (0, time_sim))\n\
\n\
        plt.figure(figsize=(10,6))\n\
        if plot2:\n\
            if elemento1 and elemento2:\n\
                plt.step(list(range(values[0],values[1])),results[values[0]:values[1]], label=id1)\n\
                plt.step(list(range(values[0],values[1])),-results2[values[0]:values[1]], label=id2)\n\
            if elemento2 and not(elemento1):\n\
                plt.step(list(range(values[0],values[1])),results2[values[0]:values[1]], label=id2)\n\
            if elemento1 and not(elemento2):\n\
                plt.step(list(range(values[0],values[1])),results2[values[0]:values[1]], label=id1)\n\
        else:\n\
            plt.step(list(range(values[0],values[1])),results[values[0]:values[1]], label=id1)\n\
\n\
        plt.xlabel('Tiempo [h]')\n\
        plt.ylabel(label).set_fontsize(15)\n\
        plt.ylabel(label).set_fontsize(15)\n\
        plt.legend(fontsize='x-large')\n\
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=8, fancybox=True, shadow=True)\n\
        # plt.xlabel('Tiempo [h]').set_fontsize(15)\n\
        plt.xticks(size = 'x-large')\n\
        plt.yticks(size = 'x-large')\n\
        plt.title(title, fontsize=20)\n\
        st.pyplot()\n\
")
section_Loc_Size_Arbitraje.close()

## section_ECO.py

section_Eco = open('Herramienta/codes/section_Eco.py', 'w')
section_Eco.write("#Librería de intersaz\n\
import streamlit as st\n\
# Librerías para maniular datos\n\
from os import path\n\
from os import remove\n\
# Librearias para manejo de datos\n\
import pandas as pd\n\
import numpy as np\n\
# Librarías para graficas\n\
import matplotlib.pyplot as plt\n\
import pydeck as pdk\n\
# Libraries for reading data form internet\n\
import requests\n\
from bs4 import BeautifulSoup\n\
import re\n\
# Importing optimization functions\n\
from Loc_dim_OpC import opt_dim\n\
\n\
def dashboard_Eco(data2,data3):\n\
\n\
    st.markdown(\u0022<h1 style='text-align: center; color: black;'>Análisis financiero</h1>\u0022, unsafe_allow_html=True)\n\
    # image = Image.open('arbitrage.jpeg')\n\
    # st.image(image, caption='', use_column_width=True)\n\
    st.markdown('En esta sección de la herramienta es posible realizar un análisis \u005c\n\
    financiero para evaluar la conveniencia económica de implementar proyectos con \u005c\n\
    uso de baterías para el almacenamiento de energía eléctrica en aplicaciones \u005c\n\
    como arbitraje, reducción de congestiones de red, entre otros. Así, se contemplan\u005c\n\
    los costos teóricos asociados al tipo de batería, y es necesario introducir \u005c\n\
    por parámetro la información relacionada a los ingresos esperados, la capacidad\u005c\n\
    de las baterías, el tiempo de operación del proyecto, la política de capital \u005c\n\
    de trabajo, el costo de oportunidad, etc. Esto con el fin de obtener los \u005c\n\
    indicadores financieros VPN y TIR, los cuales permitirán analizar la viabilidad\u005c\n\
    del proyecto, así como el comportamiento del Flujo de Caja Libre.')\n\
    st.markdown('## Parámetros seleccionados para la simulación')\n\
    st.sidebar.markdown('### Ingrese los parámetros de simulación')\n\
\n\
# Ingresar tiempo de simulación\n\
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [Años]:', min_value=1, max_value=1000000)\n\
    st.write('El horizonte de simulación es de: '+str(time_sim)+' año(s)')\n\
\n\
# Selección de tecnología de SAE\n\
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data2.index,key='2')\n\
    if technology=='New':\n\
        st.markdown('Ingrese los costos de CAPEX asociados al SAEB que quiere simular:')\n\
        costP=st.text_input('Costo por potencia [USD/MW]: ',key='5')\n\
        costE=st.text_input('Costo por energía (Baterías) [USD/MWh]: ',key='6')\n\
        costAC=st.text_input('Costo por equipos AC [USD/MW]: ',key='7')\n\
        costMano=st.text_input('Costo de construcción y puesta en servicio [USD/MW]: ',key='8')\n\
        costPredio=st.text_input('Costo del predio [USD]: ',key='10')\n\
        costLic=st.text_input('Costo de licenciamiento [USD]: ',key='11')\n\
        costCon=st.text_input('Tarifa de conexión [USD]: ',key='12')\n\
        st.markdown('Ingrese los costos de OPEX asociados al SAEB que quiere simular:')\n\
        OM_fix=st.text_input('Costo O&M fijo [USD/kW/año]: ',key='13')\n\
        OM_var=st.text_input('Costo O&M variable [cUSD/kWh]: ',key='14')\n\
\n\
    else:\n\
        st.markdown('El SAE seleccionado tiene los siguientes costos de CAPEX asociados:')\n\
        st.dataframe(data2.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))\n\
        st.markdown('El SAE seleccionado tiene los siguientes costos de OPEX asociados:')\n\
        st.dataframe(data3.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))\n\
        costP=data2.iloc[data2.index.get_loc(technology),1]\n\
        costE=data2.iloc[data2.index.get_loc(technology),0]\n\
        costAC=data2.iloc[data2.index.get_loc(technology),2]\n\
        costMano=data2.iloc[data2.index.get_loc(technology),3]\n\
        costPredio=data2.iloc[data2.index.get_loc(technology),4]\n\
        costLic=data2.iloc[data2.index.get_loc(technology),5]\n\
        costCon=data2.iloc[data2.index.get_loc(technology),6]\n\
\n\
        OM_fix=data3.iloc[data3.index.get_loc(technology),0]\n\
        OM_var=data3.iloc[data3.index.get_loc(technology),1]\n\
\n\
#Ingrese el tamaño del SAE\n\
    st.sidebar.markdown('Ingrese el tamaño del SAE a simular:')\n\
    E_max = st.sidebar.number_input('Energía [MWh]')\n\
    Pot_max = st.sidebar.number_input('Potencia [MW]')\n\
\n\
## Politica de Working Capital\n\
    politica=st.sidebar.selectbox('Seleccione la política de capital de trabajo',['Meses','Rotación en días'],key='2')\n\
    if politica== 'Meses':\n\
        meses=st.sidebar.number_input('Ingrese el número de meses:', min_value=1, max_value=1000000)\n\
    if politica== 'Rotación en días':\n\
        dias=st.sidebar.number_input('Ingrese el número de días:', min_value=1, max_value=365)\n\
\n\
###Costo de oportunidad\n\
\n\
    Coportunidad=st.sidebar.number_input('Ingrese el costo de oportunidad efectivo anual en COP (%):', min_value=1, max_value=10000000000)\n\
    Coportunidad=Coportunidad/100\n\
# Seleción de archivo con proyección de TRM y IPP\n\
    st.set_option('deprecation.showfileUploaderEncoding', False)\n\
    file_TRM = st.sidebar.file_uploader('Seleccione el archivo con proyecciones de TRM:', type=['csv','xlsx'])\n\
\n\
# Seleción de archivo con Ingresos\n\
    st.set_option('deprecation.showfileUploaderEncoding', False)\n\
    file_Ingresos = st.sidebar.file_uploader('Seleccione el archivo con los ingresos esperados del proyecto:', type=['csv','xlsx'])\n\
\n\
#https://totoro.banrep.gov.co/analytics/saw.dll?Download&Format=excel2007&Extension=.xls&BypassCache=true&lang=es&path=%2Fshared%2FSeries%20Estad%C3%ADsticas_T%2F1.%20IPC%20base%202008%2F1.2.%20Por%20a%C3%B1o%2F1.2.2.IPC_Total%20nacional%20-%20IQY\n\
\n\
\n\
    def run_VF(value,n,x):\n\
        # OM_fix*(1+x)\n\
        y=[0]*(n+1)\n\
        for i in range(0,n+1):\n\
            y[i]=float(value)*np.power(x+1,i)\n\
        return y\n\
\n\
    if st.sidebar.button('Simular'):\n\
        TRM_proy = pd.read_excel(file_TRM, sheet_name='TRM', header=0,usecols=['Proyección TRM'])\n\
        ingresos =pd.read_excel(file_Ingresos, sheet_name='Ingresos', header=0,usecols=['Ingresos [USD]'])\n\
        y_OM_fix=run_VF(OM_fix,time_sim,0.05)\n\
        y_OM_var=run_VF(OM_var,time_sim,0.05)\n\
        y_costE=run_VF(costE,time_sim,0.05)\n\
        st.title('Resultados:')\n\
        cambio=[0]*len(y_OM_fix)\n\
        if len(y_OM_fix)>10:\n\
            cambio[10]=1\n\
        flujos=[0]*len(y_OM_fix)\n\
        flujos[0]=(y_costE[0]*E_max+costP*Pot_max \u005c\n\
        +costAC*Pot_max+costMano*E_max+costPredio+costLic+costCon*Pot_max)*-1\n\
\n\
        egresos=[0]*len(y_OM_fix)\n\
        for i in range (1,len(y_OM_fix)):\n\
            egresos[i]=-(y_OM_fix[i]*Pot_max+y_OM_var[i]*E_max+y_costE[i]*E_max*cambio[i])\n\
            flujos[i]=ingresos.loc[i,'Ingresos [USD]']+egresos[i]\n\
\n\
        flujos_COP=[0]*len(y_OM_fix)\n\
        # TRM_periodo=TRM_proy.values.tolist()\n\
\n\
        ingresos_COP=[0]*len(y_OM_fix)\n\
        for i in range (0,len(y_OM_fix)):\n\
            flujos_COP[i]=flujos[i]*TRM_proy.loc[i,'Proyección TRM']/1000000000\n\
            ingresos_COP[i]=ingresos.loc[i,'Ingresos [USD]']*TRM_proy.loc[i,'Proyección TRM']/1000000000\n\
\n\
        egresos_COP=[0]*len(y_OM_fix)\n\
        for i in range (1,len(y_OM_fix)):\n\
            egresos_COP[i]=egresos[i]*TRM_proy.loc[i,'Proyección TRM']/1000000000\n\
        egresos_COP[0]=flujos_COP[0]\n\
\n\
# Valor presente neto\n\
        VPN=np.npv(Coportunidad, flujos_COP)\n\
        st.write('El VPN es: ' + '%i'%VPN + ' MM COP')\n\
# TIR or TVR\n\
\n\
        def Negativos(Flujo):\n\
            count=0\n\
            for i in range(0,len(Flujo)):\n\
                if Flujo[i]<0:\n\
                    count=count+1\n\
            return count\n\
\n\
        if Negativos(flujos_COP)==1:\n\
            tir=np.irr(flujos_COP)\n\
            st.write('La TIR es: ' + '%f'%round(tir*100,2) + ' %')\n\
        else:\n\
            st.write('La TIR no puede ser calculada debido a que hay más de un cambio de signo en los flujos.')\n\
            tvr=np.mirr(flujos_COP,Coportunidad,Coportunidad)\n\
            st.write('La TVR es: ' +'%f'%round(tvr*100,2)+ ' %')\n\
\n\
        fig=plt.figure(figsize=(15,9))\n\
        ax = fig.add_axes([0,0,1,1])\n\
        ax.bar(list(range(0,len(y_OM_fix))),(ingresos_COP),label='ingresos', width = 0.5, color='g')\n\
        ax.bar(list(range(0,len(y_OM_fix))),(egresos_COP),label='egresos', width = 0.5,color='r')\n\
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)\n\
        plt.xlabel('Time [Años]')\n\
        plt.ylabel('[MM COP]')\n\
        plt.ylim([-100, 80])\n\
        st.pyplot(fig)\n\
\n\
####\n\
        file_TRM_hist='1.1.1.TCM_Serie histórica IQY.xlsx'\n\
        TRM_hist = pd.read_excel(file_TRM_hist, sheet_name='TRM')\n\
        TRM_hist = TRM_hist.dropna()\n\
        # st.write(TRM_hist.describe())\n\
        # st.dataframe(TRM_hist.TRM)\n\
        TRM_hist.Fecha = pd.to_datetime(TRM_hist.Fecha,format='%d-%m-%Y')\n\
        # set the column as the index\n\
        TRM_hist.set_index('Fecha', inplace=True)\n\
\n\
        fig1=plt.figure(figsize=(15,9))\n\
        plt.plot(TRM_hist)\n\
        st.pyplot(fig1)\n\
        # st.write(TRM_hist)\n\
\n\
###### Movimiento Browniano#####\n\
\n\
        # Calculo de retornos logaritmicos\n\
        log_returns = np.log(TRM_hist / TRM_hist.shift(periods=1))\n\
        log_returns=log_returns.dropna()\n\
        # st.dataframe(log_returns.tail(10)*100)\n\
        # st.write()\n\
        # Parametros para el movimiento Browniano\n\
        ## Media diaria y media anual\n\
        time_mb=10 #  Ventana de tiempo en años\n\
        media_d=log_returns.iloc[-time_mb*365:].mean()\n\
        # st.write(media_d*100)\n\
        media_m=(1+media_d.iloc[0])**30-1\n\
        media_a=(1+media_d)**365-1\n\
        # st.write(media_a*100)\n\
        ### desviación\n\
        std_d=log_returns.iloc[-time_mb*365:].std()\n\
        # st.write(std_d*100)\n\
        std_m=std_d.iloc[0]*(30)**(1/2)\n\
        std_a=std_d*(365)**(1/2)\n\
        # st.write(std_a*100)\n\
        ###\n\
        def gen_paths(S0, r, sigma, T, M, I):\n\
            ''' Generates Monte Carlo paths for geometric Brownian Motion.\n\
            Parameters\n\
            ==========\n\
            S0 : float\n\
            iniial stock/index value\n\
            r : float\n\
            constant short rate\n\
            sigma : float\n\
            constant volatility\n\
            T : float\n\
            final time horizon\n\
            M : int\n\
            number of time steps/intervals\n\
            I : int\n\
            number of paths to be simulated\n\
            Returns\n\
            =======\n\
            paths : ndarray, shape (M + 1, I)\n\
            simulated paths given the parameters\n\
            '''\n\
            dt= float(T) /M\n\
            paths = np.zeros((M + 1, I), np.float64)\n\
            paths[0] = S0\n\
            for t in range(1, M + 1):\n\
                rand=np.random.standard_normal(I)\n\
                rand=(rand-rand.mean()) / rand.std()\n\
                paths[t]=paths[t-1]* np.exp((r- 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)\n\
            return paths\n\
        aaa=gen_paths(TRM_hist.iloc[-1], media_m, std_m, 2, 24, 500)\n\
\n\
        fig=plt.figure(figsize=(15,9))\n\
        ax = fig.add_axes([0,0,1,1])\n\
        ax.plot(aaa)\n\
        plt.xlabel('Time [Meses]')\n\
        plt.ylabel('[USD/COP]')\n\
        # plt.ylim([0, 6000])\n\
        st.pyplot(fig)\n\
\n\
        aaa=aaa.mean(axis = 1)\n\
        # st.write(aaa)\n\
        p_1=aaa[1:13].mean(axis = 0)\n\
        p_2=aaa[13:].mean(axis = 0)\n\
        # st.write(p_1)\n\
        # st.write(p_2)\n\
        prediction=[0]*(time_sim+1);\n\
\n\
        for i in range(2,time_sim+1):\n\
            prediction[i]=p_2\n\
\n\
        prediction[0]=TRM_hist.iloc[-1]\n\
        prediction[1]=p_1\n\
        # st.write(prediction)\n\
\n\
        fig=plt.figure(figsize=(15,9))\n\
        ax = fig.add_axes([0,0,1,1])\n\
        ax.step(range(0,time_sim+1),prediction)\n\
        plt.xlabel('Time [Años]')\n\
        plt.ylabel('[USD/COP]')\n\
        # plt.ylim([0, 6000])\n\
        st.pyplot(fig)\n\
\n\
        # regresión a la media\n\
\n\
        ## AJuste por ARIMA\n\
\n\
        # fig2=plt.plot(aaa[:, :15]) #Graficamos solo 15 caminos, todos los datos renglones o time steps, 15 columnas\n\
        # plt.grid(True)\n\
        # plt.xlabel('time steps')\n\
        # plt.ylabel('index level')\n\
        # st.pyplot(fig2)\n\
\n\
        ######\n\
")
section_Eco.close()

### Optimization functions codes

## arbitraje.py

arbitraje = open('Herramienta/codes/arbitraje.py', 'w')
arbitraje.write("import pandas as pd\n\
from pandas import ExcelWriter\n\
from pyomo.environ import *\n\
from pyomo import environ as pym\n\
from pyomo import kernel as pmo\n\
from pyomo.opt import SolverFactory\n\
import numpy as np\n\
import os\n\
\n\
# para poder corer GLPK desde una API\n\
import pyutilib.subprocess.GlobalData\n\
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False\n\
###\n\
\n\
def opt(file_prices,txt_eff,txt_deg,txt_auto,txt_SOC_ini,txt_SOC_min,txt_cost_P,txt_cost_E,txt_inv,txt_trm,txt_time_sim,combo):\n\
\n\
    # Carga de archivos de precio\n\
\n\
    df_Precios = pd.read_excel(file_prices, sheet_name = 'Precios', header = 0, index_col = 0)\n\
    # Caracteristicas del sistema de almacenamiento de energia\n\
\n\
    Big_number = 1e20\n\
    Eficiencia = float (txt_eff)      # Eficiencia del sistema de almacenamiento de energia [%]\n\
    Degradacion = float (txt_deg)     # [%Capacidad/ciclo]\n\
    Autodescarga = float (txt_auto)   # [%Capacidad/hora]\n\
    SOC_ini = 1-float (txt_SOC_ini)     # Estado de carga inicial del sistema de almacenamiento de energia [%]\n\
    SOC_min = 0.2                    # Estado de carga minimo del sistema de almacenamiento de energia [%]\n\
    Costo_Potencia = float (txt_cost_P)  # Costo del sistema de potencia del sistema de almacenamiento de energia [$USD/MW]\n\
    Costo_Energia = float (txt_cost_E)   # Costo del sistema de Energia (Baterias) del sistema de almacenamiento de energia [$USD/MWh]\n\
    Inv_limit=float (txt_inv)         # límite de inversión\n\
    TRM = float (txt_trm)             # Tasa de cambio COP/USD\n\
    # ---------------------------------------------------------------------------\n\
\n\
    model = ConcreteModel()\n\
\n\
    # ---------------------------------------------------------------------------\n\
    # Sets:\n\
    N_horas = int (txt_time_sim)\n\
    model.t = RangeSet(1,N_horas)\n\
\n\
    # ---------------------------------------------------------------------------\n\
    # Parameters from interface:\n\
\n\
    def P_init(model,t):\n\
        return df_Precios.loc[t,'Precio']\n\
    model.P = Param(model.t, initialize=P_init)\n\
\n\
    model.Eficiencia = Param(initialize = Eficiencia)\n\
    model.Degradacion = Param(initialize = Degradacion)\n\
    model.SOCmin = Param(initialize = SOC_min)\n\
    model.Autodescarga = Param(initialize = Autodescarga)\n\
    model.CostoEnergia = Param(initialize = Costo_Energia)\n\
\n\
    # ---------------------------------------------------------------------------\n\
    # Variables:\n\
\n\
    #model.Degra= Var(model.t,domain = NonNegativeReals)\n\
\n\
    model.Carga = Var(model.t, domain = Binary)\n\
    model.Descarga = Var(model.t, domain = Binary)\n\
\n\
    model.EnergiaComprada = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
    model.EnergiaVendida = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
\n\
    model.CapacidadEnergia = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
    model.NivelEnergia = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
\n\
    model.E_max = Var(domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
    model.C_Pot= Var(domain = NonNegativeReals, bounds=(0,1e6), initialize=0)\n\
\n\
    model.EnergiaAcumulada = Var(model.t,domain = NonNegativeReals, initialize=0)\n\
\n\
    # ---------------------------------------------------------------------------\n\
    # Funcion Objetivo:\n\
\n\
    def Objetivo_rule(model):\n\
        return  sum(model.P[t]*1000*(model.EnergiaVendida[t]-model.EnergiaComprada[t]) \u005c\n\
                for t in model.t) - model.C_Pot*Costo_Potencia*TRM - Costo_Energia*model.E_max*TRM -\u005c\n\
                1e-4*sum(model.Carga[t]+model.Descarga[t] for t in model.t)\n\
    model.Objetivo = Objective(rule = Objetivo_rule, sense = maximize)\n\
\n\
    # ---------------------------------------------------------------------------\n\
    # Restricciones:\n\
\n\
    ##### Investment limit\n\
    def Investment_limit_rule(model):\n\
        return model.C_Pot*Costo_Potencia + Costo_Energia*model.E_max <= Inv_limit\n\
    model.IL = Constraint(model.t, rule = Investment_limit_rule)\n\
\n\
    def Limite_energia_comprada_rule(model,t):\n\
        if t == 1:\n\
            return model.EnergiaComprada[t] <= model.E_max*(1 - SOC_ini)/model.Eficiencia\n\
        else:\n\
            return model.EnergiaComprada[t] <= (model.CapacidadEnergia[t-1] - model.NivelEnergia[t-1])/model.Eficiencia\n\
\n\
    model.LEC = Constraint(model.t, rule = Limite_energia_comprada_rule)\n\
\n\
    def Limite_energia_vendida_rule(model,t):\n\
        if t == 1:\n\
            return model.EnergiaVendida[t] <= model.E_max*(SOC_ini-SOC_min)*model.Eficiencia\n\
        else:\n\
            return model.EnergiaVendida[t] <= model.NivelEnergia[t-1]*model.Eficiencia\n\
\n\
    model.LEV = Constraint(model.t, rule = Limite_energia_vendida_rule)\n\
\n\
    def Limite_Nivel_Energia_rule(model,t):\n\
        return model.NivelEnergia[t]<=model.E_max\n\
    model.LNE = Constraint(model.t, rule = Limite_Nivel_Energia_rule)\n\
\n\
    def Limite_Capacidad_Energia_rule(model,t):\n\
        return model.CapacidadEnergia[t]<=model.E_max\n\
    model.LCE = Constraint(model.t, rule = Limite_Capacidad_Energia_rule)\n\
\n\
    ####### Constraints for variables Carga, Descarga\n\
    def Capacidad_maxima_inversor_c_rule(model,t):\n\
        return model.EnergiaComprada[t] <= Big_number * model.Carga[t]\n\
    model.CMIc = Constraint(model.t, rule = Capacidad_maxima_inversor_c_rule)\n\
\n\
    def Capacidad_maxima_inversor_v_rule(model,t):\n\
        return model.EnergiaVendida[t] <= Big_number * model.Descarga[t]\n\
    model.CMIv = Constraint(model.t, rule = Capacidad_maxima_inversor_v_rule)\n\
\n\
    def Capacidad_maxima_inversor_c_2_rule(model,t):\n\
        return model.EnergiaComprada[t] <= model.C_Pot\n\
    model.CMIc2 = Constraint(model.t, rule = Capacidad_maxima_inversor_c_2_rule)\n\
\n\
    def Capacidad_maxima_inversor_v_2_rule(model,t):\n\
        return model.EnergiaVendida[t] <= model.C_Pot\n\
    model.CMIv2 = Constraint(model.t, rule = Capacidad_maxima_inversor_v_2_rule)\n\
\n\
    def Simultaneidad_rule(model,t):\n\
        return model.Carga[t]+model.Descarga[t]<=1\n\
    model.Simultaneidad = Constraint(model.t, rule = Simultaneidad_rule)\n\
\n\
    def Nivel_de_carga_rule(model,t):\n\
        if t == 1:\n\
            return model.NivelEnergia[t] == model.E_max*SOC_ini\n\
        else:\n\
            return model.NivelEnergia[t] == (1-model.Autodescarga)*model.NivelEnergia[t-1] + (model.EnergiaComprada[t]*model.Eficiencia) - ((model.EnergiaVendida[t])/model.Eficiencia)\n\
    model.NDC = Constraint(model.t, rule = Nivel_de_carga_rule)\n\
\n\
    ####### Degradacion del sistema de almacenamiento de energia:\n\
    ####### basado en informacion presentada en: Impact of battery degradation on energy arbitrage revenue of grid-level energy storage\n\
\n\
    def Energia_Vendida_Acumulada_rule(model,t):\n\
        if t == 1:\n\
            return model.EnergiaAcumulada[t] == model.EnergiaVendida[t]\n\
        else:\n\
            return model.EnergiaAcumulada[t] == model.EnergiaAcumulada[t-1] + model.EnergiaVendida[t]\n\
    model.EVA = Constraint(model.t, rule = Energia_Vendida_Acumulada_rule)\n\
\n\
    def Degradacion_rule(model,t):\n\
        if t == 1:\n\
            return model.CapacidadEnergia[t] == model.E_max\n\
        else:\n\
            return model.CapacidadEnergia[t] == model.E_max - model.Degradacion*model.EnergiaAcumulada[t-1]\n\
    model.DR = Constraint(model.t, rule = Degradacion_rule)\n\
\n\
    #### --------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    def SOC_minimo_rule(model,t):\n\
        return model.NivelEnergia[t] >= model.E_max* model.SOCmin\n\
    model.SOCm = Constraint(model.t, rule = SOC_minimo_rule)\n\
\n\
    def pyomo_postprocess(options=None, instance=None, results=None):\n\
        model.Objetivo.display()\n\
        model.E_max.display()\n\
        model.C_Pot.display()\n\
    # ---------------------------------------------------------------------------\n\
    # Configuracion:\n\
    solver_selected=combo\n\
    if solver_selected== 'CPLEX':\n\
        opt = SolverManagerFactory('neos')\n\
        results = opt.solve(model, opt='cplex')\n\
        #sends results to stdout\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
    else:\n\
        opt = SolverFactory('glpk')\n\
        results = opt.solve(model)\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
\n\
    #################################################################################\n\
    #######################Creación de Archivo Excel#################################\n\
    #################################################################################\n\
\n\
    V_Pot_Ba_ch = np.ones(len(model.t))\n\
    V_Pot_Ba_dc = np.ones(len(model.t))\n\
    V_e_b = np.ones(len(model.t))\n\
    V_cost = model.Objetivo.expr\n\
    V_E_size = model.E_max.value\n\
    V_P_size = model.C_Pot.value\n\
    V_capacity=np.ones(len(model.t))\n\
    V_E_acum=np.ones(len(model.t))\n\
\n\
    for t in model.t:\n\
        V_Pot_Ba_ch[t-1] = model.EnergiaComprada[t].value\n\
\n\
    for t in model.t:\n\
        V_Pot_Ba_dc[t-1] = model.EnergiaVendida[t].value\n\
\n\
    for t in model.t:\n\
        V_e_b[t-1] = model.NivelEnergia[t].value\n\
\n\
    for t in model.t:\n\
        V_capacity[t-1] = model.CapacidadEnergia[t].value\n\
\n\
    for t in model.t:\n\
        V_E_acum[t-1] = model.EnergiaAcumulada[t].value\n\
\n\
    df_Pot_Ba_ch = pd.DataFrame(V_Pot_Ba_ch)\n\
    df_Pot_Ba_dc = pd.DataFrame(V_Pot_Ba_dc)\n\
    df_e_b = pd.DataFrame(V_e_b)\n\
    df_capacity=pd.DataFrame(V_capacity)\n\
    df_E_acum=pd.DataFrame(V_E_acum)\n\
\n\
    df_E_size = pd.DataFrame(V_E_size, index=['1'], columns=['Energía [MWh]'])\n\
    # df_E_size  = df_E_size.drop(['2'], axis=0)\n\
    df_P_size = pd.DataFrame(V_P_size, index=['1'], columns=['Potencia [MW]'])\n\
    # df_P_size  = df_P_size.drop(['2'], axis=0)\n\
    df_cost = pd.DataFrame(V_cost, index=['1','2'], columns=['Cost'])\n\
    df_cost  = df_cost.drop(['2'], axis=0)\n\
\n\
    mydir = os.getcwd()\n\
    name_file = 'Resultados/resultados_size_loc_arbitraje.xlsx'\n\
\n\
    path = os.path.join(mydir, name_file)\n\
\n\
    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')\n\
\n\
    df_Pot_Ba_ch.to_excel(writer, sheet_name='BESS_Ch_Power', index=True)\n\
    df_Pot_Ba_dc.to_excel(writer, sheet_name='BESS_Dc_Power', index=True)\n\
    df_e_b.to_excel(writer, sheet_name='BESS_Energy', index=True)\n\
    df_capacity.to_excel(writer, sheet_name='Capacity', index=True)\n\
    df_E_acum.to_excel(writer, sheet_name='E_acumulada', index=True)\n\
\n\
    df_E_size.to_excel(writer, sheet_name='Energy_size', index=True)\n\
    df_P_size.to_excel(writer, sheet_name='Power_size', index=True)\n\
    df_cost.to_excel(writer, sheet_name='cost', index=True)\n\
\n\
    mapping = {0 : 'Precio'}\n\
    df_Pot_Ba_ch=df_Pot_Ba_ch.rename(columns=mapping)\n\
    df_Pot_Ba_dc=df_Pot_Ba_dc.rename(columns=mapping)\n\
    Precios=df_Precios.reset_index()\n\
    df_Ingresos= Precios* (df_Pot_Ba_dc-df_Pot_Ba_ch)/1000\n\
    df_Ingresos=df_Ingresos.drop(['index'], axis=1)\n\
    df_Ingresos=df_Ingresos.dropna()\n\
\n\
    df_Ingresos.to_excel(writer, sheet_name='Ingresos', index=True)\n\
    # y=df_Ingresos.sum(axis = 0, skipna = True)\n\
\n\
    writer.save()\n\
    writer.close()\n\
\n\
    ##########################################################################\n\
\n\
    return model.E_max.value, model.C_Pot.value\n\
")
arbitraje.close()

## despacho.py

despacho = open('Herramienta/codes/despacho.py', 'w')
despacho.write("from pyomo.environ import *\n\
import numpy as np\n\
import pandas as pd\n\
import math\n\
import os\n\
import time\n\
import datetime\n\
from datetime import timedelta, datetime\n\
import urllib.request\n\
\n\
import matplotlib.pyplot as plt\n\
\n\
def pyomo_df(element):\n\
\n\
    data = value(element)\n\
    df_data = pd.DataFrame(data, index=['1','2'], columns=['1'])\n\
    df_data = df_data.drop(['2'], axis=0)\n\
    df_data = df_data.rename(columns={'1':fecha[0]})\n\
    return df_data\n\
\n\
def pyomo1_df(element):\n\
\n\
    data = {i: value(v) for i, v in element.items()}\n\
    keys = data.items()\n\
    idx = pd.MultiIndex.from_tuples(keys)\n\
    df_data = pd.DataFrame(data, index=[0]).stack().loc[0]\n\
    return df_data\n\
\n\
def pyomo2_df(element):\n\
\n\
    if len(element) != 0:\n\
        data = {(i, j): value(v) for (i, j), v in element.items()}\n\
        keys, values = zip(*data.items())\n\
        idx = pd.MultiIndex.from_tuples(keys)\n\
        df_data = pd.DataFrame(data, index=[0], ).stack().loc[0]\n\
        df_data = df_data.rename(columns={1:fecha[0]})\n\
    else:\n\
        df_data = 0\n\
    return df_data\n\
\n\
def descarga_archivos(fecha):\n\
\n\
    mydir = os.getcwd()\n\
    aux = mydir.split('/')\n\
    last_element = aux[-1]\n\
    aux.remove(last_element)\n\
    before_path = '/'.join(aux)\n\
\n\
    day1 = int(fecha.day)\n\
    month1 = int(fecha.month)\n\
    year1 = int(fecha.year)\n\
\n\
    if month1 < 10:\n\
        month1 = '0{}'.format(month1)\n\
\n\
    if day1 < 10:\n\
        day1 = '0{}'.format(day1)\n\
\n\
    url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year1,month1,month1,day1)\n\
    response_oferta = urllib.request.urlopen(url_oferta)\n\
    data_oferta = response_oferta.read()\n\
\n\
    with open(os.path.join(before_path , 'Casos de estudio/archivos_despacho/scrap_files/oferta.txt'), 'wb') as archivo:\n\
        archivo.write(data_oferta)\n\
        archivo.close()\n\
\n\
    ## Pronostico Demanda\n\
\n\
    fecha_dem = str(year1) + '-' + str(month1) + '-' + str(day1)\n\
    fecha_dem = datetime.strptime(fecha_dem, '%Y-%m-%d').date()\n\
\n\
    while fecha_dem.weekday() != 0:\n\
        fecha_dem = fecha_dem - timedelta(days=1)\n\
\n\
    year_dem = fecha_dem.year\n\
    month_dem = fecha_dem.month\n\
    day_dem = fecha_dem.day\n\
\n\
    if month_dem < 10:\n\
        month_dem = '0{}'.format(month_dem)\n\
\n\
    if day_dem < 10:\n\
        day_dem = '0{}'.format(day_dem)\n\
\n\
    url_dem = 'http://www.xm.com.co/pronosticooficial/{}-{}/PRON_SIN{}{}.txt'.format(year_dem,month_dem,month_dem,day_dem)\n\
    response_dem = urllib.request.urlopen(url_dem)\n\
    data_dem = response_dem.read()\n\
\n\
    with open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/pronostico_dem.txt'), 'wb') as archivo:\n\
        archivo.write(data_dem)\n\
        archivo.close()\n\
\n\
    ## MPO\n\
\n\
    url_MPO = 'http://www.xm.com.co/predespachoideal/{}-{}/iMAR{}{}_NAL.TXT'.format(year1,month1,month1,day1)\n\
    response_MPO = urllib.request.urlopen(url_MPO)\n\
    data_MPO = response_MPO.read()\n\
\n\
    with open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/MPO.txt'), 'wb') as archivo:\n\
        archivo.write(data_MPO)\n\
        archivo.close()\n\
\n\
    #### Lectura de archivos\n\
\n\
    agents_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/oferta.txt'), encoding='utf8')\n\
    agents_all_of_it = agents_file.read()\n\
    agents_file.close()\n\
\n\
    load_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/pronostico_dem.txt'), encoding='utf8')\n\
    load_all_of_it = load_file.read()\n\
    load_file.close()\n\
\n\
    MPO_file = open(os.path.join(before_path, 'Casos de estudio/archivos_despacho/scrap_files/MPO.txt'), encoding='utf8')\n\
    MPO_all_of_it = MPO_file.read()\n\
    MPO_file.close()\n\
\n\
    return agents_all_of_it, load_all_of_it, MPO_all_of_it\n\
\n\
\n\
def opt_despacho(fecha, file_ofer, file_load, file_MPO, txt_Pot_max, txt_Ene, txt_eff_ch, txt_eff_dc, txt_eff_SoC, txt_SoC_min, txt_tdp, txt_td, combo):\n\
\n\
    ################################################# Load files ################################################################\n\
\n\
    StartTime = time.time()\n\
\n\
    year = str(fecha.year)\n\
    month = str(fecha.month)\n\
    day = str(fecha.day)\n\
\n\
    load_all_of_it = file_load\n\
    agents_all_of_it = file_ofer\n\
    MPO_all_of_it = file_MPO\n\
\n\
    df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\u005cn')])\n\
    dic_OFEI = df_OFEI.to_dict('dict')\n\
\n\
    none_val, agents_glb = list(dic_OFEI.items())[0]\n\
\n\
    if int(month) == 12:\n\
        year = int(year) - 1\n\
\n\
    date_forecast_load = year + '-' + month + '-' + day\n\
    date_forecast_load = datetime.strptime(date_forecast_load, '%Y-%m-%d').date()\n\
\n\
    ##\n\
\n\
    days_pron = []\n\
\n\
    for i in range(1,8):\n\
        tomorrow = str(date_forecast_load + timedelta(days=i-1))\n\
        days_pron.append(tomorrow)\n\
\n\
    df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\u005cn')])\n\
\n\
    if int(year) >= 2020 and int(month) > 2:\n\
        del df_PRON_DEM[0]\n\
        df_PRON_DEM.columns -= 1\n\
        for i in range(1, len(days_pron) + 1):\n\
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)\n\
    else:\n\
        for i in range(1, len(days_pron) + 1):\n\
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)\n\
\n\
    del df_PRON_DEM['t']\n\
\n\
    df_PRON_DEM = df_PRON_DEM.dropna()\n\
    df_PRON_DEM.index += 1\n\
\n\
    for i in range(1,8):\n\
        tomorrow = str(date_forecast_load + timedelta(days=i-1))\n\
        df_PRON_DEM[str(tomorrow)] = df_PRON_DEM[str(tomorrow)].astype(float)\n\
\n\
    nul_val = []\n\
\n\
    for key, value in agents_glb.items():\n\
        if value == str(''):\n\
            nul_val.append(key)\n\
\n\
    for i in nul_val:\n\
        del(agents_glb[i])\n\
\n\
    def extract_str(element, agents_glb):\n\
        idx = ', ' + str(element) + ', '\n\
        dicc = {}\n\
        for key, value in agents_glb.items():\n\
            if value.find(idx) >= 0:\n\
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')\n\
        return dicc\n\
\n\
    def extract_num(element, agents_glb):\n\
        idx = ', ' + str(element) + ', '\n\
        dicc = {}\n\
        for key, value in agents_glb.items():\n\
            if value.find(idx) >= 0:\n\
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]\n\
        return dicc\n\
\n\
    D = extract_num('D', agents_glb)\n\
    P = extract_num('P', agents_glb)\n\
    CONF = extract_num('CONF', agents_glb)\n\
    C = extract_str('C', agents_glb)\n\
    PAPUSD = extract_num('PAPUSD1', agents_glb)\n\
    PAP = extract_num('PAP1', agents_glb)\n\
    MO = extract_num('MO', agents_glb)\n\
    AGCP = extract_num('AGCP', agents_glb)\n\
    AGCU = extract_num('AGCU', agents_glb)\n\
    PRU = extract_num('PRU', agents_glb)\n\
    CNA = extract_num('CNA', agents_glb)\n\
\n\
    ##\n\
\n\
    df_agente_precio = pd.DataFrame(columns=('Planta','Precio'))\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in P.items():\n\
            if key[:] == key1[:] or key[:-1] == key1[:]:\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
\n\
    df_agente_precio = df_agente_precio.drop(df_agente_precio[df_agente_precio['Planta']=='GECELCA32'].index)\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in P.items():\n\
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):\n\
                df_agente_precio.loc[len(df_agente_precio)+2] = key, value1[0]\n\
\n\
    df_agente_precio.index = df_agente_precio['Planta']\n\
    df_agente_precio = df_agente_precio.drop(['Planta'],axis=1)\n\
\n\
    ##\n\
    df_disponibilidad_maxima = pd.DataFrame.from_dict(D,orient='index').transpose()\n\
    df_disponibilidad_maxima.index += 1\n\
    df_disponibilidad_maxima = df_disponibilidad_maxima.transpose()\n\
    ##\n\
\n\
    df_disponibilidad_minima = pd.DataFrame.from_dict(MO,orient='index')\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in MO.items():\n\
            if key[:-1] == key1[:]:\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
\n\
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['ALBAN'])\n\
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['GUATRON'])\n\
    # df_disponibilidad_minima = df_disponibilidad_minima.drop(['PAGUA'])\n\
\n\
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()\n\
    df_disponibilidad_minima.index += 1\n\
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()\n\
\n\
    ##\n\
    df_PAP = pd.DataFrame(columns=('Planta','PAP'))\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in PAP.items():\n\
            if key[:] == key1[:] or key[:-1] == key1[:]:\n\
                df_PAP.loc[len(df_PAP)] = key, value1[0]\n\
\n\
    df_PAP = df_PAP.drop(df_PAP[df_PAP['Planta']=='GECELCA32'].index)\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in PAP.items():\n\
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):\n\
                df_PAP.loc[len(df_PAP)+2] = key, value1[0]\n\
\n\
    df_PAP.index = df_PAP['Planta']\n\
    df_PAP = df_PAP.drop(['Planta'],axis=1)\n\
\n\
    ##\n\
    df_con_max = df_agente_precio.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan,int(0))\n\
    del df_con_max['union']\n\
\n\
    df_disponibilidad_maxima = df_con_max.drop(['Precio'], axis=1)\n\
\n\
    df_precio = df_con_max\n\
    for i in range(1,25):\n\
        del df_precio[i]\n\
\n\
    ##\n\
\n\
    df_con_min = df_PAP.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan,int(0))\n\
    for i in range(1,25):\n\
        del df_con_min[i]\n\
    del df_con_min['union']\n\
    df_con_min = df_con_min.rename(columns={0:'PAP'})\n\
    df_con_all_min = df_con_min.merge(df_disponibilidad_minima, how='outer', indicator='union',left_index=True, right_index=True).replace(np.nan,int(0))\n\
    del df_con_all_min['union']\n\
\n\
    df_disponibilidad_minima = df_con_all_min.drop(['PAP'], axis=1)\n\
\n\
    df_PAP = df_con_all_min\n\
    for i in range(1,25):\n\
        del df_PAP[i]\n\
\n\
    df_MPO = pd.DataFrame([x.split(',') for x in MPO_all_of_it.split('\u005cn')])\n\
    df_MPO = df_MPO.dropna()\n\
    df_MPO = df_MPO.drop([0], axis=1)\n\
\n\
    ReadingTime = time.time() - StartTime\n\
\n\
    ################################################ Sets Definitions ################################################\n\
    StartTime = time.time()\n\
    model = ConcreteModel()\n\
    ##\n\
    dim = df_MPO.shape\n\
\n\
    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)              ## scheduling periods\n\
    # model.td = RangeSet(5,7)                                                        ## discharge periods\n\
    model.tdp = RangeSet(16,18)                                                       ## previous discharge periods\n\
    model.i = Set(initialize=df_disponibilidad_maxima.index.tolist(), ordered=True) ## Units\n\
    model.s = RangeSet(1)                                                         ## Batteries in the system\n\
    model.umbral = RangeSet(1,4)                                                    ## Umbrales costo de escasez\n\
\n\
    ########################################## Parameters definitions ####################################\n\
\n\
    ###\n\
    a = {}\n\
    aa = {}\n\
    k = 0\n\
    kk = 0\n\
\n\
    for i in range(1,dim[0]+1):\n\
        for j in range(1,dim[1]+1):\n\
            k += 1\n\
            aa[k] = int(df_MPO.iloc[i-1,j-1])\n\
\n\
    def MPO_init(model):\n\
        return aa\n\
    model.MPO = Param(model.t, initialize=MPO_init)\n\
\n\
    for i in range(1,dim[0]+1):\n\
        for j in range(1,dim[1]+1):\n\
            kk += 1\n\
            max_number = max(aa.items())[1]\n\
            a[kk] = (int(df_MPO.iloc[i-1,j-1]) ** 2)/max_number\n\
\n\
    def PC_init(model):\n\
        return a\n\
    model.PC = Param(model.t, initialize=PC_init)\n\
\n\
    model.PC_max = Param(model.s, model.t, initialize=int(txt_Pot_max))                ## Power capacity max charging\n\
    model.PD_max = Param(model.s, model.t, initialize=int(txt_Pot_max))                ## Power capacity max discharging\n\
    model.Eficiencia_descarga = Param(model.s, initialize=float(txt_eff_ch))           ## Discharge efficiency\n\
    model.Eficiencia_carga = Param(model.s, initialize=float(txt_eff_dc))              ## Charge efficiency\n\
    model.Eficiencia_SoC = Param(model.s, initialize=float(txt_eff_SoC))               ## Storage efficiency\n\
    model.E_max = Param(model.s, initialize=float(txt_Ene))                            ## Energy storage max limit\n\
    model.SoC_min = Param(model.s, initialize=float(txt_SoC_min)*float(txt_Ene)) ## Minimum state of charge\n\
    # model.SoC_max = Param(model.s, initialize=float(txt_SoC_max.get())*float(txt_Ene.get())) ## Maximum state of charge\n\
    model.PCdes = Param(model.s, model.t, initialize=0)        ## Charging power capacity on discharge periods\n\
    model.PDdes = Param(model.s, model.t, initialize=30)        ## Discharging power capacity on discharge periods\n\
\n\
    td_i = txt_td\n\
    td_f = txt_td\n\
    tdp_i = txt_tdp\n\
    tdp_f = txt_tdp\n\
\n\
    CRO_est = {1: 1480.31,\n\
               2: 2683.49,\n\
               3: 4706.20,\n\
               4: 9319.71}\n\
\n\
    def CRO_init(model):\n\
        return CRO_est\n\
    model.CRO_est = Param(model.umbral, initialize=CRO_init)    ## Costo incremental de racionamiento de energía\n\
\n\
    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################\n\
    model.status = Var(model.i, model.t, within=Binary, initialize=0)          ## Commitment of unit i at time t\n\
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)     ## Power dispatch of unit i at time t\n\
    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)            ## Binary Status of battery charge\n\
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)            ## Binary Status of battery discharge\n\
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)  ## Power in battery charge\n\
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)  ## Power in battery discharge\n\
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                  ## Energy of battery\n\
    model.V_DoC = Var(model.s, model.t, domain=NonNegativeReals)                  ## Discharge state\n\
    model.V_Rac = Var(model.t, bounds=(0,2400))  ## Energy of rationing\n\
\n\
    ModelingTime = time.time() - StartTime\n\
\n\
    ###################################################### MODEL ######################################################\n\
    StartTime = time.time()\n\
\n\
    ## Objective function\n\
\n\
    def cost_rule(model):\n\
        return sum(model.P[i,t] * df_precio.loc[i,'Precio'] for i in model.i for t in model.t) + \u005c\n\
                sum(df_PAP.loc[i,'PAP'] * model.status[i,t] for i in model.i for t in model.t) + \u005c\n\
                sum((CRO_est[1] * 1000) * model.V_DoC[s,t] for s in model.s for t in model.t if t >= tdp_i and t <= tdp_f) + \u005c\n\
                sum((CRO_est[1] * 1000) * model.V_Rac[t] for t in model.t) + \u005c\n\
                sum(model.PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)\n\
    model.cost = Objective(rule=cost_rule, sense=minimize)\n\
\n\
    ###################################################### CONSTRAINTS ######################################################\n\
    ## Power limits\n\
\n\
    def P_lim_max_rule(model,i,t):\n\
        return model.P[i,t] <= df_disponibilidad_maxima.loc[i,t] * model.status[i,t]\n\
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)\n\
\n\
    def P_lim_min_rule(model,i,t):\n\
        return model.P[i,t] >= df_disponibilidad_minima.loc[i,t] * model.status[i,t]\n\
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)\n\
\n\
    ## Power Balance\n\
\n\
    def power_balance_rule(model,t,s):\n\
        return sum(model.P[i,t] for i in model.i) + model.V_Rac[t] + \u005c\n\
                model.V_PD[s,t] == df_PRON_DEM.loc[t,str(fecha)] + model.V_PC[s,t]\n\
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)\n\
\n\
    ##### Batteries\n\
\n\
    ## Causalidad de la carga/descarga\n\
\n\
    def sim_rule(model,s,t):\n\
        return model.B_PC[s,t] + model.B_PD[s,t] <= 1\n\
    model.sim = Constraint(model.s, model.t, rule=sim_rule)\n\
\n\
    def power_c_max_rule(model,s,t):\n\
        return model.V_PC[s,t] <= model.PC_max[s,t] * model.B_PC[s,t]\n\
    model.power_c_max = Constraint(model.s, model.t, rule=power_c_max_rule)\n\
\n\
    def power_d_max_rule(model,s,t):\n\
        return model.V_PD[s,t] <= model.PD_max[s,t] * model.B_PD[s,t]\n\
    model.power_d_max = Constraint(model.s, model.t, rule=power_d_max_rule)\n\
\n\
    ## Balance almacenamiento\n\
\n\
    def energy_rule(model,s,t):\n\
        if t == 1:\n\
            return model.V_SoC[s,t] == model.SoC_min[s] + (model.Eficiencia_carga[s] * model.V_PC[s,t] - (model.V_PD[s,t]) * (1/model.Eficiencia_descarga[s]))\n\
        else:\n\
            return model.V_SoC[s,t] == model.V_SoC[s,t-1] * (1 - model.Eficiencia_SoC[s]) + (model.Eficiencia_carga[s] * model.V_PC[s,t] - model.V_PD[s,t] * (1/model.Eficiencia_descarga[s]))\n\
    model.energy = Constraint(model.s, model.t, rule=energy_rule)\n\
\n\
    ## Balance de Estado de Carga\n\
\n\
    def energy_balance_rule(model,s,t):\n\
        return model.V_DoC[s,t] == model.E_max[s] - model.V_SoC[s,t]\n\
    model.energy_balance = Constraint(model.s, model.t, rule=energy_balance_rule)\n\
\n\
    ## Capacidad mínima y máxima de almacenamiento\n\
\n\
    def energy_min_limit_rule(model,s,t):\n\
        return model.V_SoC[s,t] >= model.SoC_min[s]\n\
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)\n\
\n\
    def energy_max_limit_rule(model,s,t):\n\
        return model.V_SoC[s,t] <= model.E_max[s]\n\
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)\n\
\n\
    ## Carga y descarga requerida\n\
\n\
    def power_required_dh_rule(model,s,t):\n\
        if t >= td_i and t <= td_f:\n\
            return model.V_PD[s,t] == model.PDdes[s,t]\n\
        else:\n\
            return Constraint.Skip\n\
    model.power_required_dh = Constraint(model.s, model.t, rule=power_required_dh_rule)\n\
\n\
    # def power_required_ch_rule(model,s,t):\n\
    #     if t >= td_i and t <= td_f:\n\
    #         return model.V_PC[s,t] >= model.PCdes[s,t] * model.B_PC[s,t]\n\
    #     else:\n\
    #         return Constraint.Skip\n\
    # model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)\n\
\n\
    # Configuracion:\n\
\n\
    solver_selected = combo\n\
\n\
    if solver_selected== 'CPLEX':\n\
        opt = SolverManagerFactory('neos')\n\
        results = opt.solve(model, opt='cplex')\n\
    else:\n\
        opt = SolverFactory('glpk')\n\
        results = opt.solve(model)\n\
\n\
    SolvingTime = time.time() - StartTime\n\
\n\
    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)\n\
\n\
    tiempo = timedelta(seconds=TotalTime)\n\
\n\
    df_SOC = pyomo1_df(model.V_PD)\n\
\n\
    return df_SOC, tiempo\n\
\n\
def opt_despacho_2(fecha, file_ofer, file_load, txt_Pot_max, txt_Ene, txt_eff_ch, txt_eff_dc, txt_eff_SoC, txt_SoC_min, txt_SoC_MT, txt_tdp, txt_td, combo):\n\
\n\
    ##################################### Load file ####################################\n\
\n\
    StartTime = time.time()\n\
\n\
    type_sim = 1 #predespacho ideal\n\
\n\
    day1 = int(fecha.day)\n\
    month1 = int(fecha.month)\n\
    year1 = int(fecha.year)\n\
\n\
    if month1 < 10:\n\
        month1 = '0{}'.format(month1)\n\
\n\
    if day1 < 10:\n\
        day1 = '0{}'.format(day1)\n\
\n\
    load_all_of_it = file_load\n\
    agents_all_of_it = file_ofer\n\
\n\
    #### Organización archivos\n\
\n\
    ## Oferta\n\
\n\
    df_OFEI = pd.DataFrame([x.split(';') for x in agents_all_of_it.split('\u005cn')])\n\
    dic_OFEI = df_OFEI.to_dict('dict')\n\
\n\
    none_val, agents_glb = list(dic_OFEI.items())[0]\n\
\n\
    nul_val = []\n\
\n\
    for key, value in agents_glb.items():\n\
        if value == str(''):\n\
            nul_val.append(key)\n\
\n\
    for i in nul_val:\n\
        del(agents_glb[i])\n\
\n\
    ## Pronostico Demanda\n\
\n\
    days_pron = []\n\
\n\
    date_forecast_load = str(year1) + '-' + str(month1) + '-' + str(day1)\n\
    date_forecast_load = datetime.strptime(date_forecast_load, '%Y-%m-%d').date()\n\
\n\
    for i in range(1,8):\n\
        dates = str(date_forecast_load + timedelta(days = i - 1))\n\
        days_pron.append(dates)\n\
\n\
    df_PRON_DEM = pd.DataFrame([x.split(',') for x in load_all_of_it.split('\u005cn')])\n\
\n\
    if year1 >= 2020 and int(month1) > 2:\n\
        del df_PRON_DEM[0]\n\
        df_PRON_DEM.columns -= 1\n\
        for i in range(1, len(days_pron) + 1):\n\
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)\n\
    else:\n\
        for i in range(1, len(days_pron) + 1):\n\
            df_PRON_DEM.rename(columns={0: 't', i: str(days_pron[i-1])}, inplace=True)\n\
\n\
    del df_PRON_DEM['t']\n\
\n\
    df_PRON_DEM = df_PRON_DEM.dropna()\n\
    df_PRON_DEM.index += 1\n\
\n\
    for i in range(1,8):\n\
        dates = str(date_forecast_load + timedelta(days = i - 1))\n\
        df_PRON_DEM[str(dates)] = df_PRON_DEM[str(dates)].astype(float)\n\
\n\
    #### Funciones para extraer diccionarios con cada componente de archivos globales\n\
\n\
    ## Extracción strings\n\
\n\
    def extract_str(element, agents_glb):\n\
        idx = ', ' + str(element) + ', '\n\
        dicc = {}\n\
        for key, value in agents_glb.items():\n\
            if value.find(idx) >= 0:\n\
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')\n\
        return dicc\n\
\n\
    ## Extracción números\n\
\n\
    def extract_num(element, agents_glb):\n\
        idx = ', ' + str(element) + ', '\n\
        dicc = {}\n\
        for key, value in agents_glb.items():\n\
            if value.find(idx) >= 0:\n\
                dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]\n\
        return dicc\n\
\n\
    #### Extracción de componentes\n\
\n\
    D = extract_num('D', agents_glb)\n\
    P = extract_num('P', agents_glb)\n\
    CONF = extract_num('CONF', agents_glb)\n\
    C = extract_str('C', agents_glb)\n\
    PAPUSD = extract_num('PAPUSD1', agents_glb)\n\
    PAP = extract_num('PAP1', agents_glb)\n\
    MO = extract_num('MO', agents_glb)\n\
    AGCP = extract_num('AGCP', agents_glb)\n\
    AGCU = extract_num('AGCU', agents_glb)\n\
    PRU = extract_num('PRU', agents_glb)\n\
    CNA = extract_num('CNA', agents_glb)\n\
\n\
    #### Desempate de ofertas\n\
\n\
    def Desempate_ofertas(P):\n\
        for p in P.keys():\n\
            same_price = []\n\
            for u in P.keys():\n\
                if P[p][0] == P[u][0]:\n\
                    same_price.append(u)\n\
            N = len(same_price)\n\
            Delta_price = 0\n\
            for n in range(N):\n\
                oferente = random.choice(same_price)\n\
                P[oferente][0] = P[oferente][0] + Delta_price\n\
                Delta_price += 0.1\n\
        return P\n\
\n\
    ####\n\
\n\
    df_agente_precio = pd.DataFrame(columns=('Planta','Precio'))\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in P.items():\n\
            if key[:] == key1[:] or key[:-1] == key1[:]:\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):\n\
                df_agente_precio.loc[len(df_agente_precio)] = key, value1[0]\n\
\n\
    df_agente_precio = df_agente_precio.drop(df_agente_precio[df_agente_precio['Planta'] == 'GECELCA32'].index)\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in P.items():\n\
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):\n\
                df_agente_precio.loc[len(df_agente_precio) + 2] = key, value1[0]\n\
\n\
    df_agente_precio.index = df_agente_precio['Planta']\n\
    df_agente_precio = df_agente_precio.drop(['Planta'], axis=1)\n\
\n\
    ##\n\
\n\
    df_disponibilidad_maxima = pd.DataFrame.from_dict(D, orient='index').transpose()\n\
    df_disponibilidad_maxima.index += 1\n\
    df_disponibilidad_maxima = df_disponibilidad_maxima.transpose()\n\
\n\
    ##\n\
\n\
    df_disponibilidad_minima = pd.DataFrame.from_dict(MO, orient='index')\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in MO.items():\n\
            if key[:-1] == key1[:]:\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-1] == str('ALTOANCHICAYA') or key[:-1] == str('BAJOANCHICAYA')) and key1[:] == str('ALBAN'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-2] == str('GUADALUPE') or key[:-1] == str('TRONERAS')) and key1[:] == str('GUATRON'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
            elif (key[:-1] == str('LAGUACA') or key[:-1] == str('PARAISO')) and key1[:] == str('PAGUA'):\n\
                df_disponibilidad_minima.loc[key] = MO[key1]\n\
\n\
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['ALBAN'])\n\
    df_disponibilidad_minima = df_disponibilidad_minima.drop(['GUATRON'])\n\
    # df_disponibilidad_minima = df_disponibilidad_minima.drop(['PAGUA'])\n\
\n\
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()\n\
    df_disponibilidad_minima.index += 1\n\
    df_disponibilidad_minima = df_disponibilidad_minima.transpose()\n\
\n\
    if type_sim == 1:\n\
        df_disponibilidad_minima = df_disponibilidad_minima.clip(upper=0)\n\
    else:\n\
        df_disponibilidad_minima = df_disponibilidad_minima\n\
\n\
    ##\n\
\n\
    df_PAP = pd.DataFrame(columns=('Planta','PAP'))\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in PAP.items():\n\
            if key[:] == key1[:] or key[:-1] == key1[:]:\n\
                df_PAP.loc[len(df_PAP)] = key, value1[0]\n\
\n\
    df_PAP = df_PAP.drop(df_PAP[df_PAP['Planta'] == 'GECELCA32'].index)\n\
\n\
    for key, value in D.items():\n\
        for key1, value1 in PAP.items():\n\
            if key[:] == str('GECELCA32') and key1[:] == str('GECELCA32'):\n\
                df_PAP.loc[len(df_PAP) + 2] = key, value1[0]\n\
\n\
    df_PAP.index = df_PAP['Planta']\n\
    df_PAP = df_PAP.drop(['Planta'], axis=1)\n\
\n\
    ##\n\
\n\
    df_con_max = df_agente_precio.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))\n\
    del df_con_max['union']\n\
\n\
    df_disponibilidad_maxima = df_con_max.drop(['Precio'], axis=1)\n\
\n\
    df_precio = df_con_max\n\
\n\
    for i in range(1,25):\n\
        del df_precio[i]\n\
\n\
    ##\n\
\n\
    df_con_min = df_PAP.merge(df_disponibilidad_maxima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))\n\
\n\
    for i in range(1,25):\n\
        del df_con_min[i]\n\
\n\
    del df_con_min['union']\n\
\n\
    df_con_min = df_con_min.rename(columns={0:'PAP'})\n\
    df_con_all_min = df_con_min.merge(df_disponibilidad_minima, how='outer', indicator='union', left_index=True, right_index=True).replace(np.nan, int(0))\n\
    del df_con_all_min['union']\n\
\n\
    df_disponibilidad_minima = df_con_all_min.drop(['PAP'], axis=1)\n\
\n\
    df_PAP = df_con_all_min\n\
\n\
    for i in range(1,25):\n\
        del df_PAP[i]\n\
\n\
    ## Demanda SIN\n\
\n\
    mydir = os.getcwd()\n\
    files_path = os.path.join(mydir, 'Casos de estudio/archivos_despacho')\n\
\n\
    if year1 == 2019 and int(month1) <= 6 :\n\
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME1_2019.xlsx'),\n\
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)\n\
    elif year1 == 2019 and int(month1) >= 7:\n\
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME2_2019.xlsx'),\n\
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)\n\
    elif year1 == 2020 and int(month1) <= 6:\n\
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME1_2020.xlsx'),\n\
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)\n\
    elif year1 == 2020 and int(month1) >= 7:\n\
        df_demanda = pd.read_excel(os.path.join(files_path, 'Demanda_Comercial_Por_Comercializador_SEME2_2020.xlsx'),\n\
                            sheet_name='Demanda_Comercial_Por_Comercial', header=0)\n\
    else:\n\
        print('No existen archivos de demanda para esa fecha')\n\
\n\
    df_demanda = df_demanda.drop([0], axis=0)\n\
    df_demanda = df_demanda.fillna(0)\n\
    a = df_demanda.loc[1,:]\n\
    for i in range(28):\n\
        df_demanda = df_demanda.rename(columns={'Unnamed: {}'.format(i):a[i]})\n\
    df_demanda = df_demanda.drop(['Codigo Comercializador','Mercado','Version'], 1)\n\
    df_demanda = df_demanda.drop([1], axis=0)\n\
    df_demanda = df_demanda.reset_index(drop=True)\n\
\n\
    df_demanda_fecha = pd.DataFrame()\n\
\n\
    fecha_SIN = str(year1) + '-' + str(month1) + '-' + str(day1)\n\
\n\
    for i in range(len(df_demanda)):\n\
        if df_demanda.loc[i,'Fecha'] == fecha_SIN:\n\
            df_demanda_fecha = df_demanda_fecha.append(df_demanda.loc[i,:], ignore_index=True)\n\
\n\
    df_demanda_fecha = df_demanda_fecha.drop(['Fecha'], 1)\n\
    df_demanda_fecha = df_demanda_fecha.rename(columns={'0':0})\n\
    df_demanda_fecha = df_demanda_fecha.reindex(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])\n\
\n\
    df_demanda_fecha = df_demanda_fecha.sum(axis=0)\n\
    df_demanda_fecha.index += 1\n\
\n\
    for i in range(len(df_demanda_fecha)):\n\
        df_demanda_fecha.loc[i+1] = round(df_demanda_fecha.loc[i+1] / 1000, 2)\n\
\n\
    ReadingTime = time.time() - StartTime\n\
\n\
    ################################################ Sets Definitions ################################################\n\
    StartTime = time.time()\n\
\n\
    model = ConcreteModel()\n\
    ##\n\
\n\
    model.t = Set(initialize=df_PRON_DEM.index.tolist(), ordered=True)              ## scheduling periods\n\
    model.tr = Set(initialize=[txt_td])                                                        ## discharge periods\n\
    model.tdp = Set(initialize=[txt_tdp])                                                     ## previous discharge periods\n\
    model.i = Set(initialize=df_disponibilidad_maxima.index.tolist(), ordered=True) ## Units\n\
    model.s = RangeSet(1)                                                           ## Batteries in the system\n\
    ########################################## Parameters definitions ####################################\n\
\n\
    model.PC_max = Param(model.s, initialize=int(txt_Pot_max))           ## Power capacity max charging\n\
    model.PD_max = Param(model.s, initialize=int(txt_Pot_max))           ## Power capacity max discharging\n\
    model.Eficiencia_descarga = Param(model.s, initialize=float(txt_eff_dc))      ## Discharge efficiency\n\
    model.Eficiencia_carga = Param(model.s, initialize=float(txt_eff_ch))         ## Charge efficiency\n\
    model.Eficiencia_SoC = Param(model.s, initialize=float(txt_eff_SoC))       ## Storage efficiency\n\
    model.E_max = Param(initialize=int(txt_Ene))                     ## Energy storage max limit\n\
    SoC_min = float(txt_SoC_min)                                                  ## Minimum state of charge\n\
    SoC_max = 1                                                     ## Maximum state of charge\n\
    SoC_MT = float(txt_SoC_MT)                                                   ## Minimum technical state of charge\n\
    model.PCreq = Param(initialize=0)            ## Charging power capacity on discharge periods\n\
    model.PDreq = Param(initialize=30)            ## Discharging power capacity on discharge periods\n\
    K_e = 1                                                         ## Scalling factor [$/MWh]\n\
\n\
    td_i = int(txt_td)\n\
    td_f = int(txt_td)\n\
    tpd_i = int(txt_tdp)\n\
    tpd_f = int(txt_tdp)\n\
\n\
    ## PC definitions\n\
\n\
    PC = round((K_e * df_PRON_DEM.loc[:,str(fecha)]) / max(df_PRON_DEM.loc[:,str(fecha)]),4)\n\
\n\
    ## costo de racionamiento\n\
\n\
    cro = {'CRO1': 1480.31,\n\
            'CRO2': 2683.49,\n\
            'CRO3': 4706.20,\n\
            'CRO4': 9319.71}\n\
\n\
    ## Estado de conexión del SAEB al SIN\n\
\n\
    model.ECS = Param(model.s, model.t, initialize=1, mutable=True)\n\
\n\
    # model.ECS[1,7] = 0\n\
\n\
    ###################################################### VARIABLES ############################################################################################################ VARIABLES ######################################################\n\
    model.status = Var(model.i, model.t, within=Binary, initialize=0)           ## Commitment of unit i at time t\n\
    model.P = Var(model.i, model.t, domain=NonNegativeReals, initialize=0)      ## Power dispatch of unit i at time t\n\
    model.V_Rac = Var(model.t, bounds=(0, 2400))   ## Energy of rationing\n\
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)               ## Start-Up cost of uit i\n\
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)               ## Shut-Down cost of unit i\n\
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)               ## Start-Up status of unit i\n\
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)               ## Shut-Down status of unit i\n\
\n\
    model.B_PC = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery charge\n\
    model.B_PD = Var(model.s, model.t, within=Binary, initialize=0)             ## Binary Status of battery discharge\n\
    model.V_PC = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery charge\n\
    model.V_PD = Var(model.s, model.t, initialize=0, domain=NonNegativeReals)   ## Power in battery discharge\n\
    model.V_SoC = Var(model.s, model.t, domain=NonNegativeReals)                ## Energy of battery\n\
    model.V_SoD = Var(model.s, model.t, domain=NonNegativeReals)                ## Discharge state\n\
    model.V_SoC_E = Var(model.s, model.t, domain=NonNegativeReals)              ## State of Charge with storage efficiency\n\
\n\
    ModelingTime = time.time() - StartTime\n\
\n\
    ###################################################### MODEL ######################################################\n\
    StartTime = time.time()\n\
\n\
    ## Objective function\n\
\n\
    def cost_rule(model):\n\
        return sum(model.P[i,t] * df_precio.loc[i,'Precio'] for i in model.i for t in model.t) + \u005c\n\
                sum(model.costSD[i,t] + model.costSU[i,t] for i in model.i for t in model.t) + \u005c\n\
                sum((cro['CRO1'] * 1000) * model.V_SoD[s,t] * model.E_max for s in model.s for t in model.t if t == tpd_i) + \u005c\n\
                sum((cro['CRO1'] * 1000) * model.V_Rac[t] for t in model.t) + \u005c\n\
                sum(PC[t] * model.V_PC[s,t] for s in model.s for t in model.t)\n\
    model.cost = Objective(rule=cost_rule, sense=minimize)\n\
\n\
    ###################################################### CONSTRAINTS ######################################################\n\
\n\
    #### Dispath constraints\n\
\n\
    ## Power limits\n\
\n\
    def P_lim_max_rule(model,i,t):\n\
        return model.P[i,t] <= df_disponibilidad_maxima.loc[i,t] * model.status[i,t]\n\
    model.P_lim_max = Constraint(model.i, model.t, rule=P_lim_max_rule)\n\
\n\
    def P_lim_min_rule(model,i,t):\n\
        return model.P[i,t] >= df_disponibilidad_minima.loc[i,t] * model.status[i,t]\n\
    model.P_lim_min = Constraint(model.i, model.t, rule=P_lim_min_rule)\n\
\n\
    ## PAP cost\n\
\n\
    def CostSUfn_init(model,i,t):\n\
        return model.costSU[i,t] == df_PAP.loc[i,'PAP'] * model.SU[i,t]\n\
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)\n\
\n\
    def CostSDfn_init(model,i,t):\n\
        return model.costSD[i,t] == df_PAP.loc[i,'PAP'] * model.SD[i,t]\n\
    model.CostSDfn = Constraint(model.i, model.t, rule=CostSDfn_init)\n\
\n\
    ## Integer Constraint\n\
\n\
    def bin_cons1_rule(model,i,t):\n\
        if t == model.t.first():\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] #- model.onoff_to[i,t]\n\
        else:\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]\n\
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)\n\
\n\
    def bin_cons2_rule(model,i,t):\n\
        return model.SU[i,t] + model.SD[i,t] <= 1\n\
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)\n\
\n\
    ## Power Balance\n\
\n\
    def power_balance_rule(model,t,s):\n\
        return sum(model.P[i,t] for i in model.i) + model.V_Rac[t] + \u005c\n\
                model.V_PD[s,t] * model.ECS[s,t] == df_demanda_fecha[t] + model.V_PC[s,t] * model.ECS[s,t]\n\
    model.power_balance = Constraint(model.t, model.s, rule=power_balance_rule)\n\
\n\
    ##### Batteries\n\
\n\
    ## Balance almacenamiento\n\
\n\
    def energy_rule(model,s,t):\n\
        if t == 1:\n\
            return model.V_SoC[s,t] == SoC_min + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] * (1 / model.E_max) - \u005c\n\
                    model.V_PD[s,t] * (1 / model.Eficiencia_descarga[s]) * (1 / model.E_max))\n\
        else:\n\
            return model.V_SoC[s,t] == model.V_SoC_E[s,t-1] + model.ECS[s,t] * (model.Eficiencia_carga[s] * model.V_PC[s,t] * (1 / model.E_max) - \u005c\n\
                    model.V_PD[s,t] * (1 / model.Eficiencia_descarga[s]) * (1 / model.E_max))\n\
    model.energy = Constraint(model.s, model.t, rule=energy_rule)\n\
\n\
    ## Afectación del estado de carga por eficiencia de almacenamiento\n\
\n\
    def efe_storage_1_rule(model,s,t):\n\
        return -(model.B_PC[s,t] + model.B_PD[s,t]) + model.V_SoC[s,t] * (1 - model.Eficiencia_SoC[s]) <= model.V_SoC_E[s,t]\n\
    model.efe_storage_1 = Constraint(model.s, model.t, rule=efe_storage_1_rule)\n\
\n\
    def efe_storage_2_rule(model,s,t):\n\
        return model.V_SoC[s,t] * (1 - model.Eficiencia_SoC[s]) + (model.B_PC[s,t] + model.B_PD[s,t]) >= model.V_SoC_E[s,t]\n\
    model.efe_storage_2 = Constraint(model.s, model.t, rule=efe_storage_2_rule)\n\
\n\
    def efe_storage_3_rule(model,s,t):\n\
        return -(1 - model.B_PC[s,t]) + model.V_SoC[s,t] <= model.V_SoC_E[s,t]\n\
    model.efe_storage_3 = Constraint(model.s, model.t, rule=efe_storage_3_rule)\n\
\n\
    def efe_storage_4_rule(model,s,t):\n\
        return model.V_SoC[s,t] + (1 - model.B_PC[s,t]) >= model.V_SoC_E[s,t]\n\
    model.efe_storage_4 = Constraint(model.s, model.t, rule=efe_storage_4_rule)\n\
\n\
    def efe_storage_5_rule(model,s,t):\n\
        return -(1 - model.B_PD[s,t]) + model.V_SoC[s,t] <= model.V_SoC_E[s,t]\n\
    model.efe_storage_5 = Constraint(model.s, model.t, rule=efe_storage_5_rule)\n\
\n\
    def efe_storage_6_rule(model,s,t):\n\
        return model.V_SoC[s,t] + (1 - model.B_PD[s,t]) >= model.V_SoC_E[s,t]\n\
    model.efe_storage_6 = Constraint(model.s, model.t, rule=efe_storage_6_rule)\n\
\n\
    ## Balance de Estado de Carga\n\
\n\
    def energy_balance_rule(model,s,t):\n\
        return model.V_SoD[s,t] == 1 - model.V_SoC[s,t]\n\
    model.energy_balance = Constraint(model.s, model.t, rule=energy_balance_rule)\n\
\n\
    ## Capacidad mínima y máxima de almacenamiento\n\
\n\
    def energy_min_limit_rule(model,s,t):\n\
        return model.V_SoC[s,t] >= SoC_min\n\
    model.energy_min_limit = Constraint(model.s, model.t, rule=energy_min_limit_rule)\n\
\n\
    def energy_max_limit_rule(model,s,t):\n\
        return model.V_SoC[s,t] <= SoC_max\n\
    model.energy_max_limit = Constraint(model.s, model.t, rule=energy_max_limit_rule)\n\
\n\
    ## mínimo técnico del sistema de almacenamiento\n\
\n\
    def energy_min_tec_rule(model,s,t):\n\
        return model.V_SoC[s,t] >= SoC_MT\n\
    model.energy_min_tec = Constraint(model.s, model.t, rule=energy_min_tec_rule)\n\
\n\
    ## Causalidad de la carga/descarga\n\
\n\
    def sim_rule(model,s,t):\n\
        if model.ECS[s,t].value == 1:\n\
            return model.B_PC[s,t] + model.B_PD[s,t] <= 1\n\
        else:\n\
            return Constraint.Skip\n\
    model.sim = Constraint(model.s, model.t, rule=sim_rule)\n\
\n\
    def power_c_max_rule(model,s,t):\n\
        if model.ECS[s,t].value == 1:\n\
            return model.V_PC[s,t] <= model.PC_max[s] * model.B_PC[s,t]\n\
        else:\n\
            return Constraint.Skip\n\
    model.power_c_max = Constraint(model.s, model.t, rule=power_c_max_rule)\n\
\n\
    def power_d_max_rule(model,s,t):\n\
        if model.ECS[s,t].value == 1:\n\
            return model.V_PD[s,t] <= model.PD_max[s] * model.B_PD[s,t]\n\
        else:\n\
            return Constraint.Skip\n\
    model.power_d_max = Constraint(model.s, model.t, rule=power_d_max_rule)\n\
\n\
    ## Carga y descarga requerida\n\
\n\
    def power_required_dc_rule(model,s,t):\n\
        if t == td_i and model.ECS[s,t].value == 1:\n\
            return model.V_PD[s,t] == model.PDreq\n\
        else:\n\
            return model.V_PD[s,t] == 0\n\
    model.power_required_dc = Constraint(model.s, model.t, rule=power_required_dc_rule)\n\
\n\
    def power_required_ch_rule(model,s,t):\n\
        if t == td_i and model.ECS[s,t].value == 1:\n\
            return model.V_PC[s,t] >= model.PCreq\n\
        else:\n\
            return Constraint.Skip\n\
    model.power_required_ch = Constraint(model.s, model.t, rule=power_required_ch_rule)\n\
\n\
    # Configuracion:\n\
\n\
    solver_selected = combo\n\
\n\
    if solver_selected== 'CPLEX':\n\
        opt = SolverManagerFactory('neos')\n\
        results = opt.solve(model, opt='cplex')\n\
    else:\n\
        opt = SolverFactory('glpk')\n\
        results = opt.solve(model)\n\
\n\
    SolvingTime = time.time() - StartTime\n\
\n\
    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)\n\
\n\
    tiempo = timedelta(seconds=TotalTime)\n\
\n\
    df_SOC = pyomo1_df(model.V_PD)\n\
\n\
    return df_SOC, tiempo\n\
")
despacho.close()

## Loc_dim_OpC.py

Loc_dim_OpC = open('Herramienta/codes/Loc_dim_OpC.py', 'w')
Loc_dim_OpC.write("import pandas as pd\n\
from pandas import ExcelWriter\n\
from pyomo.environ import *\n\
from pyomo import environ as pym\n\
from pyomo import kernel as pmo\n\
import math\n\
import time\n\
from datetime import datetime, timedelta\n\
import os\n\
import numpy as np\n\
\n\
# para poder corer GLPK desde una API\n\
import pyutilib.subprocess.GlobalData\n\
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False\n\
###\n\
\n\
def opt_dim(file_system, txt_eff, txt_SOC_min, txt_SOC_ini, txt_time_sim, txt_re_inv, txt_C_Pot, txt_C_Bat, txt_ope_fact, txt_con_fact,\n\
            txt_con_cost, combo):\n\
\n\
    StartTime = time.time()\n\
\n\
    df_System_data = pd.read_excel(file_system, sheet_name='System_data', header=0, index_col=0)\n\
    df_SM_Unit = pd.read_excel(file_system, sheet_name='SM_Unit', header=0, index_col=0)\n\
    df_SM_map = pd.read_excel(file_system, sheet_name='SM_map', header=0, index_col=0)\n\
    df_Renewable = pd.read_excel(file_system, sheet_name='Renewable', header=0, index_col=0)\n\
    df_Branch = pd.read_excel(file_system, sheet_name='Branch', header=0, index_col=0)\n\
    df_line_map = pd.read_excel(file_system, sheet_name='line_map', header=0, index_col=0)\n\
    df_load = pd.read_excel(file_system, sheet_name='load', header=0, index_col=0)\n\
    df_Bus = pd.read_excel(file_system, sheet_name='Bus', header=0, index_col=0)\n\
\n\
    #Total time steps\n\
    N_horas = int(txt_time_sim)\n\
    ReadingTime = time.time() - StartTime\n\
\n\
    ###### Index definitions ######\n\
\n\
    C_hydro = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'HYDRO'].index))\n\
    C_hydro = C_hydro.tolist()\n\
    C_acpm = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'ACPM'].index))\n\
    C_acpm = C_acpm.tolist()\n\
    C_diesel = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'DIESEL'].index))\n\
    C_diesel = C_diesel.tolist()\n\
    C_coal = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'COAL'].index))\n\
    C_coal = C_coal.tolist()\n\
    C_combustoleo = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'COMBUSTOLEO'].index))\n\
    C_combustoleo = C_combustoleo.tolist()\n\
    C_gas = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'GAS'].index))\n\
    C_gas = C_gas.tolist()\n\
    C_wind = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'WIND'].index))\n\
    C_wind = C_wind.tolist()\n\
    C_solar = df_SM_Unit.index.get_indexer_for((df_SM_Unit[df_SM_Unit.Fuel_Type == 'SOLAR'].index))\n\
    C_solar = C_solar.tolist()\n\
    C_thermal = C_acpm + C_diesel + C_coal + C_combustoleo + C_gas\n\
\n\
    thermal_idx = df_SM_Unit.index[C_thermal]\n\
    hydro_idx = df_SM_Unit.index[C_hydro]\n\
    solar_idx = df_SM_Unit.index[C_solar]\n\
    wind_idx = df_SM_Unit.index[C_wind]\n\
\n\
    ################################################ Sets Definitions ################################################\n\
    StartTime = time.time()\n\
\n\
    model = ConcreteModel()\n\
\n\
    model.t = RangeSet(1,N_horas)                                               ## Time Set\n\
    model.tt = SetOf(model.t)                                              ## Time Set relations intertemporal\n\
    model.i = Set(initialize=thermal_idx, ordered=True)                    ## Thermal generators set\n\
    model.w = Set(initialize=wind_idx, ordered=True)                       ## Wind generators set\n\
    model.so = Set(initialize=solar_idx, ordered=True)                     ## Solar generators set\n\
    model.j = Set(initialize=hydro_idx, ordered=True)                      ## Hydro generators set\n\
    model.b = Set(initialize=df_Bus.index.tolist(), ordered=True)          ## Bus system set\n\
    model.l = Set(initialize=df_Branch.index.tolist(), ordered=True)       ## Lines Set\n\
    model.k = Set(initialize=['k1','k2','k3'], ordered=True)               ## Index number of linearized segments\n\
\n\
    ########################################## Parameters definitions ####################################\n\
\n\
    model.delta_theta = Param(initialize=15*math.pi/180)\n\
    L = 3\n\
    def alpha_init(model):\n\
        return sum(model.delta_theta * (2*l - 1) / L for l in range(1,L))\n\
    model.alpha = Param(initialize=alpha_init)\n\
\n\
\n\
    # Piecewise Linear Approximation of Quadratic Cost Curve (Upper Approximation)\n\
\n\
    ## slope segment k generator i\n\
    def slope_init(model,i,k):\n\
        if model.k.ord(k) == 1:\n\
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,'k1']+df_SM_Unit.loc[i,'Pmin'])+df_SM_Unit.loc[i,'b'])\n\
        else:\n\
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,k]+df_SM_Unit.loc[i,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[i,'b'])\n\
    model.slope = Param(model.i, model.k, initialize=slope_init)\n\
\n\
    ## Minimum production cost of unit i at Pmin\n\
    def fg_min_init(model,i):\n\
        return (df_SM_Unit.loc[i,'a']*df_SM_Unit.loc[i,'Pmin']*df_SM_Unit.loc[i,'Pmin']+df_SM_Unit.loc[i,'b']*df_SM_Unit.loc[i,'Pmin'] + df_SM_Unit.loc[i,'c'])\n\
    model.fg_min = Param(model.i, initialize=fg_min_init)\n\
\n\
    #### Hydro\n\
\n\
    def slope_Hydro_init(model,j,k):\n\
        if model.k.ord(k) == 1:\n\
            return (df_SM_Unit.loc[j,'a']*(df_SM_Unit.loc[j,'k1']+df_SM_Unit.loc[j,'Pmin'])+df_SM_Unit.loc[j,'b'])\n\
        else:\n\
            return (df_SM_Unit.loc[j,'a']*(df_SM_Unit.loc[j,k]+df_SM_Unit.loc[j,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[j,'b'])\n\
    model.slope_j = Param(model.j, model.k, initialize=slope_Hydro_init)\n\
\n\
    def fg_min_j_init(model,j):\n\
        return (df_SM_Unit.loc[j,'a']*df_SM_Unit.loc[j,'Pmin']*df_SM_Unit.loc[j,'Pmin']+df_SM_Unit.loc[j,'b']*df_SM_Unit.loc[j,'Pmin'] + df_SM_Unit.loc[j,'c'])\n\
    model.fg_min_j = Param(model.j, initialize=fg_min_j_init)\n\
\n\
    #Parameters for minimum up time constraints\n\
\n\
    def onoff_t0_init(model,i):\n\
        if df_SM_Unit.loc[i,'IniT_ON'] > 0:\n\
            a = 1\n\
        else:\n\
            a = 0\n\
        return a\n\
    model.onoff_t0 = Param(model.i, initialize=onoff_t0_init)\n\
\n\
    def L_up_min_init(model,i):\n\
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_ON']-df_SM_Unit.loc[i,'IniT_ON'])*model.onoff_t0[i])\n\
    model.L_up_min = Param(model.i, initialize=L_up_min_init)\n\
\n\
    def L_down_min_init(model,i):\n\
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_OFF']-df_SM_Unit.loc[i,'IniT_off'])*(1-model.onoff_t0[i]))\n\
    model.L_down_min = Param(model.i, initialize=L_down_min_init)\n\
\n\
    ## Conductance of each line\n\
\n\
    def Conductance_init(model,l):\n\
        return (df_Branch.loc[l,'R'])/(df_Branch.loc[l,'R']*df_Branch.loc[l,'R'] + df_Branch.loc[l,'X']*df_Branch.loc[l,'X'])\n\
    model.Conductance = Param(model.l, rule=Conductance_init)\n\
\n\
    # Sbase [MVA]\n\
    S_base = df_System_data.loc['S_base'][0]\n\
    model.MVA_base = Param(initialize=S_base)\n\
\n\
    re_inv = int(txt_re_inv)\n\
    discount = 0\n\
    Costo_potencia = int(txt_C_Pot)\n\
    Costo_energia = int(txt_C_Bat)\n\
    Costo_potencia_eqv = round(Costo_potencia*(1-discount/100)/(365*re_inv),2)\n\
    Costo_energia_eqv = round(Costo_energia*(1-discount/100)/(365*re_inv),2)\n\
\n\
    ####Parameters BESS\n\
    Big_number = 1e20\n\
    model.Costo_potencia = Param(initialize=Costo_potencia_eqv) ## Costo del inversor de potencia de la batería\n\
    model.Costo_energia = Param(initialize=Costo_energia_eqv)  ## Costo de los modulos de baterías\n\
    Eficiencia = float(txt_eff)        ## Eficiencia global del BESS\n\
    SOC_min = 1-float(txt_SOC_min)       ## Minimum State of Charge of BESS\n\
    SOC_ini = float(txt_SOC_ini)       ## Initial State of Charge of BESS\n\
\n\
    model.Eficiencia_descarga = Param(initialize=Eficiencia)      ## Eficiencia global del BESS\n\
    model.Eficiencia_carga = Param(initialize=Eficiencia)\n\
\n\
    #################################### Define Variables ##############################\n\
\n\
    model.status = Var(model.i, model.t, within=Binary, initialize=0)\n\
    model.status_j = Var(model.j, model.t, within=Binary, initialize=0)\n\
    model.P_i = Var(model.i, model.t, domain=NonNegativeReals)\n\
    model.P_j = Var(model.j, model.t, domain=NonNegativeReals)\n\
    model.P_seg_i = Var(model.i, model.t, model.k, domain=NonNegativeReals)\n\
    model.P_seg_j = Var(model.j, model.t, model.k, domain=NonNegativeReals)\n\
\n\
    model.pcost = Var(model.t, domain=NonNegativeReals)\n\
    model.pcost_j = Var(model.t, domain=NonNegativeReals)\n\
    model.costgen = Var(model.i, model.t, domain=NonNegativeReals)\n\
    model.costgen_j = Var(model.j, model.t, domain=NonNegativeReals)\n\
\n\
    model.Pw = Var(model.w, model.t, domain=NonNegativeReals)\n\
    model.Ps = Var(model.so, model.t, domain=NonNegativeReals)\n\
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)\n\
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)\n\
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)\n\
    model.costSD = Var(model.i, model.t, domain=NonNegativeReals)\n\
    ## model.cost = Var(doc='total operation cost')\n\
\n\
    model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))\n\
    model.pf = Var(model.l, model.t)\n\
\n\
    model.theta_sr = Var(model.l, model.t)\n\
    model.theta_sr_pos = Var(model.l, model.t)\n\
    model.theta_sr_neg = Var(model.l, model.t)\n\
    model.theta_sr_abs = Var(model.l, model.t)\n\
    model.line_losses = Var(model.l, model.t, domain=Reals, bounds=(-1e6,1e6), initialize=0)\n\
\n\
    model.abs_var = Var(model.l, model.t, domain=Binary, initialize=0)\n\
    model.line_losses_pos = Var(model.l, model.t, initialize=0)\n\
    model.line_losses_neg = Var(model.l, model.t, domain=NonNegativeReals, initialize=0)\n\
    model.line_losses_abs = Var(model.l, model.t, domain=NonNegativeReals, initialize=0)\n\
\n\
    # positive and negative pf (new)\n\
\n\
    model.pf_pos = Var(model.l, model.t, domain=NonNegativeReals)\n\
    model.pf_neg = Var(model.l, model.t, domain=NonNegativeReals)\n\
\n\
    ############ BESS#######\n\
    model.u_ch = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery charge\n\
    model.u_dc = Var(model.b, model.t, within=Binary, initialize=0)                     ## Status of battery discharge\n\
    model.Pot_Ba_ch = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery charge\n\
    model.Pot_Ba_dc = Var(model.b, model.t, bounds=(0,1e6), initialize=0)               ## Power in battery discharge\n\
    model.e_b = Var(model.b, model.t, domain=NonNegativeReals, initialize=0)            ## Energy of battery\n\
    model.C_Pot = Var(model.b, domain = NonNegativeReals, bounds=(0,1e6), initialize=0) ## Power Size\n\
    model.E_max = Var(model.b, domain = NonNegativeReals, bounds=(0,1e12), initialize=0) ## Energy Size\n\
\n\
    ###slack\n\
\n\
    Slack_bus = df_System_data.loc['Slack_bus'][0]\n\
\n\
    for t in model.t:\n\
        model.theta[Slack_bus,t].fix(0)\n\
\n\
    ModelingTime = time.time() - StartTime\n\
\n\
    ###################################################### MODEL ######################################################\n\
\n\
    StartTime = time.time()\n\
\n\
    ## Objective function definition\n\
\n\
    ## Thermal\n\
\n\
    def P_sum_rule(model,i,t):\n\
        return model.P_i[i,t] == model.status[i,t]*df_SM_Unit.loc[i,'Pmin'] + sum(model.P_seg_i[i,t,k] for k in model.k)\n\
    model.P_sum = Constraint(model.i, model.t, rule=P_sum_rule)\n\
\n\
    def costgen_rule(model, i, t):\n\
        return model.costgen[i,t] == model.status[i,t]*model.fg_min[i] + sum(model.P_seg_i[i,t,k]*model.slope[i,k] for k in model.k)\n\
    model.costgen_fn = Constraint(model.i, model.t, rule=costgen_rule)\n\
\n\
    def pcost_rule(model, t):\n\
        return model.pcost[t] == sum(model.costgen[i,t] + model.costSU[i,t] + model.costSD[i,t]  for i in model.i)\n\
    model.costfnperiod = Constraint(model.t, rule=pcost_rule)\n\
\n\
    ## Hydro\n\
\n\
    def P_Hydro_sum_rule(model,j,t):\n\
        return model.P_j[j,t] == model.status_j[j,t] * df_SM_Unit.loc[j,'Pmin'] + sum(model.P_seg_j[j,t,k] for k in model.k)\n\
    model.P_Hydro_sum = Constraint(model.j, model.t, rule=P_Hydro_sum_rule)\n\
\n\
    def costgen_Hydro_rule(model, j, t):\n\
        return model.costgen_j[j,t] == model.status_j[j,t] * model.fg_min_j[j] + sum(model.P_seg_j[j,t,k]*model.slope_j[j,k] for k in model.k)\n\
    model.costgen_Hydro_fn = Constraint(model.j, model.t, rule=costgen_Hydro_rule)\n\
\n\
    def pcost_Hydro_rule(model, t):\n\
        return model.pcost_j[t] == sum(model.costgen_j[j,t] for j in model.j)\n\
    model.costfnperiod_Hydro = Constraint(model.t, rule=pcost_Hydro_rule)\n\
\n\
    ## Objective function\n\
\n\
    def cost_rule(model):\n\
        return int(txt_ope_fact) * (sum(model.pcost[t] for t in model.t) + sum(model.pcost_j[t] for t in model.t)) + \u005c\n\
            sum(model.C_Pot[b] * model.Costo_potencia + model.E_max[b] * model.Costo_energia for b in model.b) + \u005c\n\
            int(txt_con_fact) * (sum(model.pf_pos[l,t] + model.pf_neg[l,t] + 0.5 * model.line_losses_pos[l,t] +\n\
                                0.5 * model.line_losses_neg[l,t] - df_Branch.loc[l,'Flowlimit'] for l in model.l for t in model.t)) * float(txt_con_cost)\n\
    model.cost = Objective(rule=cost_rule, sense=minimize)\n\
\n\
    ##### Constraints\n\
\n\
    ## Startup/down cost\n\
\n\
    def CostSUfn_init(model,i,t):\n\
        return model.costSU[i,t] == df_SM_Unit.loc[i,'CSU']*model.SU[i,t]\n\
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)\n\
\n\
    def CostSDfn_init(model,i,t):\n\
        return model.costSD[i,t] == df_SM_Unit.loc[i,'CSU']*model.SD[i,t]\n\
    model.CostSDfn = Constraint(model.i, model.t, rule=CostSDfn_init)\n\
\n\
    ## Power limits of generators\n\
\n\
    # Thermal\n\
\n\
    def P_lim_min_rule(model,i,t):\n\
        return model.P_i[i,t] >= df_SM_Unit.loc[i,'Pmin']*model.status[i,t]\n\
    model.P_min_lim = Constraint(model.i, model.t, rule=P_lim_min_rule)\n\
\n\
    def P_lim_max_rule(model,i,t):\n\
        return model.P_i[i,t] <= df_SM_Unit.loc[i,'Pmax']*model.status[i,t]\n\
    model.P_max_lim = Constraint(model.i, model.t, rule=P_lim_max_rule)\n\
\n\
    def P_seg_base_max_rule(model,i,t,k):\n\
        return model.P_seg_i[i,t,k]<=(df_SM_Unit.loc[i,'k2']-df_SM_Unit.loc[i,'k1'])*model.status[i,t];\n\
    model.P_seg_base_max = Constraint(model.i,model.t,model.k, rule=P_seg_base_max_rule)\n\
\n\
    # Hydro\n\
\n\
    def P_Hydro_lim_min_rule(model,j,t):\n\
        return model.P_j[j,t] >= df_SM_Unit.loc[j,'Pmin'] * model.status_j[j,t]\n\
    model.P_Hydro_lim_min = Constraint(model.j, model.t, rule=P_Hydro_lim_min_rule)\n\
\n\
    def P_Hydro_lim_max_rule(model,j,t):\n\
        return model.P_j[j,t] <= df_SM_Unit.loc[j,'Pmax'] * model.status_j[j,t]\n\
    model.P_Hydro_lim_max = Constraint(model.j, model.t, rule=P_Hydro_lim_max_rule)\n\
\n\
    def P_Hydro_seg_lim_max_rule(model,j,t,k):\n\
        return model.P_seg_j[j,t,k] <= (df_SM_Unit.loc[j,'k2']-df_SM_Unit.loc[j,'k1']) * model.status_j[j,t]\n\
    model.P_Hydro_seg_lim_max = Constraint(model.j, model.t, model.k, rule=P_Hydro_seg_lim_max_rule)\n\
\n\
    # Wind\n\
\n\
    def Wind_lim_max_rule(model,w,t):\n\
        return model.Pw[w,t] <= df_Renewable.loc[t,w]\n\
    model.Wind_max_lim = Constraint(model.w, model.t, rule=Wind_lim_max_rule)\n\
\n\
    # Solar\n\
\n\
    def maxPs_rule(model, so, t):\n\
        return model.Ps[so,t] <= df_Renewable.loc[t,so]\n\
    model.maxPs = Constraint(model.so, model.t, rule=maxPs_rule)\n\
\n\
    ## Integer Constraint\n\
\n\
    def bin_cons1_rule(model,i,t):\n\
        if t == model.t.first():\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]\n\
        else:\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]\n\
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)\n\
\n\
    def bin_cons2_rule(model,i,t):\n\
        return model.SU[i,t] + model.SD[i,t] <= 1\n\
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)\n\
\n\
    # ## Min up_dn time\n\
\n\
    # def min_up_dn_time_1_rule(model,i,t):\n\
    #     if model.L_up_min[i] + model.L_down_min[i] > 0 and t < model.L_up_min[i] + model.L_down_min[i]:\n\
    #         return model.status[i,t] == model.onoff_t0[i]\n\
    #     else:\n\
    #         return Constraint.Skip\n\
    # model.min_up_dn_time_1 = Constraint(model.i, model.t, rule=min_up_dn_time_1_rule)\n\
\n\
    # def min_up_dn_time_2_rule(model,i,t):\n\
    #     return sum(model.SU[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_ON']+1 and tt <= t) <= model.status[i,t]\n\
    # model.min_up_dn_time_2 = Constraint(model.i, model.t, rule=min_up_dn_time_2_rule)\n\
\n\
    # def min_up_dn_time_3_rule(model,i,t):\n\
    #     return sum(model.SD[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_OFF']+1 and tt <= t) <= 1-model.status[i,t]\n\
    # model.min_up_dn_time_3 = Constraint(model.i, model.t, rule=min_up_dn_time_3_rule)\n\
\n\
    ## ramp constraints\n\
\n\
    def ramp_up_fn_rule(model,i,t):\n\
        if t > 1:\n\
            return model.P_i[i,t] - model.P_i[i,t-1] <= df_SM_Unit.loc[i,'Ramp_Up']*model.status[i,t-1] + df_SM_Unit.loc[i,'Pmin']*(model.status[i,t]-model.status[i,t-1]) + df_SM_Unit.loc[i,'Pmax']*(1-model.status[i,t])\n\
        else:\n\
            return Constraint.Skip\n\
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)\n\
\n\
    def ramp_dw_fn_rule(model,i,t):\n\
        if t > 1:\n\
            return model.P_i[i,t-1] - model.P_i[i,t] <= df_SM_Unit.loc[i,'Ramp_Down']*model.status[i,t] + df_SM_Unit.loc[i,'Pmin']*(model.status[i,t-1]-model.status[i,t]) + df_SM_Unit.loc[i,'Pmax']*(1-model.status[i,t-1])\n\
        else:\n\
            return Constraint.Skip\n\
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)\n\
\n\
    ## Angle definition\n\
\n\
    def theta_sr_fn_rule(model,l,t):\n\
        return model.theta_sr[l,t] == sum(model.theta[b,t]*df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)\n\
    model.theta_sr_fn = Constraint(model.l, model.t, rule=theta_sr_fn_rule)\n\
\n\
    def abs_definition_1_rule(model,l,t):\n\
        return model.theta_sr[l,t] <= Big_number * model.abs_var[l,t]\n\
    model.abs_definition_1 = Constraint(model.l, model.t, rule=abs_definition_1_rule)\n\
\n\
    def abs_definition_2_rule(model,l,t):\n\
        return model.theta_sr[l,t] >= -(1 - model.abs_var[l,t]) * 20\n\
    model.ans_definition_2 = Constraint(model.l, model.t, rule=abs_definition_2_rule)\n\
\n\
    def theta_positive_1_rule(model,l,t):\n\
        return model.theta_sr_pos[l,t] <= Big_number * (model.abs_var[l,t])\n\
    model.theta_positive_1 = Constraint(model.l, model.t, rule=theta_positive_1_rule)\n\
\n\
    def theta_positive_2_rule(model,l,t):\n\
        return model.theta_sr_pos[l,t] <= - model.theta_sr[l,t]\n\
    model.theta_positive_2 = Constraint(model.l, model.t, rule=theta_positive_2_rule)\n\
\n\
    def theta_negative_1_rule(model,l,t):\n\
        return model.theta_sr_neg[l,t] <= Big_number * (1 - model.abs_var[l,t])\n\
    model.theta_negative_1 = Constraint(model.l, model.t, rule=theta_negative_1_rule)\n\
\n\
    def theta_negative_2_rule(model,l,t):\n\
        return model.theta_sr_neg[l,t] <= model.theta_sr[l,t]\n\
    model.theta_negative_2 = Constraint(model.l, model.t, rule=theta_negative_2_rule)\n\
\n\
    def q_rule(model,l,t):\n\
        return model.theta_sr_neg[l,t] <= 0\n\
    model.q = Constraint(model.l, model.t, rule=q_rule)\n\
\n\
    def qq_rule(model,l,t):\n\
        return model.theta_sr_pos[l,t] <= 0\n\
    model.qq = Constraint(model.l, model.t, rule=qq_rule)\n\
\n\
    def e_rule(model,l,t):\n\
        return model.theta_sr[l,t] == - model.theta_sr_pos[l,t] - - model.theta_sr_neg[l,t]\n\
    model.e = Constraint(model.l, model.t, rule=e_rule)\n\
\n\
    def total_rule(model,l,t):\n\
        return model.theta_sr_abs[l,t] == - model.theta_sr_pos[l,t] + - model.theta_sr_neg[l,t]\n\
    model.total = Constraint(model.l, model.t, rule=total_rule)\n\
\n\
    ####DC transmission network security constraint\n\
\n\
    def line_flow_rule(model, l, t):\n\
        return model.pf[l,t] == model.MVA_base*(1/df_Branch.loc[l,'X'])*(- model.theta_sr_pos[l,t] - - model.theta_sr_neg[l,t])\n\
    model.line_flow = Constraint(model.l, model.t, rule=line_flow_rule)\n\
\n\
    def line_min_rule(model, l, t):\n\
        return model.pf[l,t] + 0.5 * model.line_losses[l,t] >= - df_Branch.loc[l,'Flowlimit']\n\
    model.line_min = Constraint(model.l, model.t, rule=line_min_rule)\n\
\n\
    def line_max_rule(model, l, t):\n\
        return model.pf[l,t] + 0.5 * model.line_losses[l,t] <= df_Branch.loc[l,'Flowlimit']\n\
    model.line_max = Constraint(model.l, model.t, rule=line_max_rule)\n\
\n\
    # positive and negative constraint powerflow (new)\n\
\n\
    def line_flow_abs_rule(model,l,t):\n\
        return model.pf[l,t] == model.pf_pos[l,t] - model.pf_neg[l,t]\n\
    model.line_flow_abs = Constraint(model.l, model.t, rule=line_flow_abs_rule)\n\
\n\
    # Losses\n\
\n\
    def losses_rule(model,l,t):\n\
        return model.line_losses[l,t] == model.MVA_base * model.alpha * model.Conductance[l] * (model.theta_sr[l,t])\n\
    model.losses = Constraint(model.l, model.t, rule=losses_rule)\n\
\n\
    def losses_abs_rule(model, l, t):\n\
        return model.line_losses[l,t] == model.line_losses_pos[l,t] - model.line_losses_neg[l,t]\n\
    model.losses_abs = Constraint(model.l, model.t, rule=losses_abs_rule)\n\
\n\
    #--------------------------------DC PF------------------------------------------\n\
\n\
    #contribución al balance de los BESS\n\
\n\
    def power_balance_rule(model,t,b):\n\
        return sum(model.P_i[i,t] for i in model.i if df_SM_map.loc[i,b]) +sum(model.P_j[j,t] for j in model.j if df_SM_map.loc[j,b]) +\u005c\n\
            sum(model.Ps[so,t] for so in model.so if df_SM_map.loc[so,b]) + sum(model.Pw[w,t] for w in model.w if df_SM_map.loc[w,b]) -\u005c\n\
            sum((model.pf[l,t] + 0.5 * model.line_losses[l,t])*df_line_map.loc[l,b] for l in model.l if df_line_map.loc[l,b] != 0) +\u005c\n\
            model.Pot_Ba_dc[b,t] - model.Pot_Ba_ch[b,t] == df_load.loc[t,b]\n\
    model.power_balance = Constraint(model.t, model.b, rule=power_balance_rule)\n\
\n\
    #--------------------------------BESS------------------------------------------\n\
\n\
    ############################################################################################################################\n\
\n\
    ## power charging Constraints\n\
\n\
    def power_c_max_rule(model,b,t):\n\
        return model.Pot_Ba_ch[b,t] <= Big_number*model.u_ch[b,t]\n\
    model.power_c_max = Constraint(model.b, model.t, rule=power_c_max_rule)\n\
\n\
    def power_c_max_2_rule(model,b,t):\n\
        return model.Pot_Ba_ch[b,t] <= model.C_Pot[b]\n\
    model.power_c_max_2 = Constraint(model.b, model.t, rule=power_c_max_2_rule)\n\
\n\
    ## power dischraging Constraints\n\
\n\
    def power_d_max_rule(model,b,t):\n\
        return model.Pot_Ba_dc[b,t] <= Big_number * model.u_dc[b,t]\n\
    model.power_d_max = Constraint(model.b, model.t, rule=power_d_max_rule)\n\
\n\
    def power_d_max_2_rule(model,b,t):\n\
        return model.Pot_Ba_dc[b,t] <= model.C_Pot[b]\n\
    model.power_d_max_2 = Constraint(model.b, model.t, rule=power_d_max_2_rule)\n\
\n\
    ## Simultaneous charging and discharging Constraint\n\
\n\
    def sim_rule(model,b,t):\n\
        return model.u_ch[b,t] + model.u_dc[b,t] <= 1\n\
    model.sim = Constraint(model.b, model.t, rule=sim_rule)\n\
\n\
    ## relation betwent energy status and power charging and discharging Constraint\n\
\n\
    def energy_rule(model,b,t):\n\
        if t == 1:\n\
            return model.e_b[b,t] == model.E_max[b] * SOC_ini + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga\n\
        else:\n\
            return model.e_b[b,t] == model.e_b[b,t-1] + model.Eficiencia_carga*model.Pot_Ba_ch[b,t] - (model.Pot_Ba_dc[b,t])/model.Eficiencia_descarga\n\
    model.energy = Constraint(model.b, model.t, rule=energy_rule)\n\
\n\
    ### Energy limits\n\
\n\
    def energy_limit_rule(model,b,t):\n\
        return model.e_b[b,t] <= model.E_max[b]\n\
    model.energy_limit = Constraint(model.b, model.t, rule=energy_limit_rule)\n\
\n\
    def energy_limit_min_rule(model,b,t):\n\
        return model.e_b[b,t] >= model.E_max[b] * SOC_min\n\
    model.energy_limit_min = Constraint(model.b, model.t, rule=energy_limit_min_rule)\n\
\n\
    ###Solution#####\n\
\n\
    def pyomo_postprocess(options=None, instance=None, results=None):\n\
        model.C_Pot.display()\n\
        model.E_max.display()\n\
        model.pf.display()\n\
        model.P_i.display()\n\
        model.P_j.display()\n\
    # Configuracion:\n\
\n\
    solver_selected = combo\n\
\n\
    if solver_selected == 'CPLEX':\n\
#        if __name__ == '__main__':\n\
        # This emulates what the pyomo command-line tools does\n\
        opt = SolverManagerFactory('neos')\n\
        results = opt.solve(model, opt='cplex')\n\
        #sends results to stdout\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
    else:\n\
        opt = SolverFactory('glpk')\n\
        results = opt.solve(model)\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
\n\
    SolvingTime = time.time() - StartTime\n\
\n\
    TotalTime = round(ReadingTime)+round(ModelingTime)+round(SolvingTime)\n\
\n\
    tiempo = timedelta(seconds=TotalTime)\n\
\n\
    print('Reading DATA time:',round(ReadingTime,3), '[s]')\n\
    print('Modeling time:',round(ModelingTime,3), '[s]')\n\
    print('Solving time:',round(SolvingTime,3), '[s]')\n\
    print('Total time:', tiempo)\n\
\n\
    #################################################################################\n\
    #######################Creación de Archivo Excel#################################\n\
    #################################################################################\n\
\n\
    V_pf = np.ones((len(model.l),len(model.t)))\n\
    V_gen_i = np.ones((len(model.i),len(model.t)))\n\
    V_gen_j = np.ones((len(model.j),len(model.t)))\n\
    V_gen_w = np.ones((len(model.w),len(model.t)))\n\
    V_gen_so = np.ones((len(model.so),len(model.t)))\n\
    V_losses = np.ones((len(model.l),len(model.t)))\n\
    V_Pot_Ba_ch = np.ones((len(model.b),len(model.t)))\n\
    V_Pot_Ba_dc = np.ones((len(model.b),len(model.t)))\n\
    V_e_b = np.ones((len(model.b),len(model.t)))\n\
    V_cost = model.cost.value()\n\
    V_E_size = np.ones(len(model.E_max))\n\
    V_P_size = np.ones(len(model.C_Pot))\n\
\n\
# Se multiplica por 0.5 por formulación de pérdidas\n\
    for l in model.l:\n\
        for t in model.t:\n\
            V_pf[model.l.ord(l)-1, t-1] = model.pf[l,t].value + 0.5 * model.line_losses[l,t].value\n\
\n\
    for i in model.i:\n\
        for t in model.t:\n\
            V_gen_i[model.i.ord(i)-1, t-1] = model.P_i[i,t].value\n\
\n\
    for j in model.j:\n\
        for t in model.t:\n\
            V_gen_j[model.j.ord(j)-1, t-1] = model.P_j[j,t].value\n\
\n\
    for w in model.w:\n\
        for t in model.t:\n\
            V_gen_w[model.w.ord(w)-1, t-1] = model.Pw[w,t].value\n\
\n\
    for so in model.so:\n\
        for t in model.t:\n\
            V_gen_so[model.so.ord(so)-1, t-1] = model.Ps[so,t].value\n\
\n\
    for l in model.l:\n\
        for t in model.t:\n\
            V_losses[model.l.ord(l)-1, t-1] = 0.5 * model.line_losses[l,t].value\n\
\n\
    for b in model.b:\n\
        for t in model.t:\n\
            V_Pot_Ba_ch[model.b.ord(b)-1, t-1] = model.Pot_Ba_ch[b,t].value\n\
\n\
    for b in model.b:\n\
        for t in model.t:\n\
            V_Pot_Ba_dc[model.b.ord(b)-1, t-1] = model.Pot_Ba_dc[b,t].value\n\
\n\
    for b in model.b:\n\
        for t in model.t:\n\
            V_e_b[model.b.ord(b)-1, t-1] = model.e_b[b,t].value\n\
\n\
    for b in model.b:\n\
        V_E_size[model.b.ord(b)-1] = model.E_max[b].value\n\
\n\
    for b in model.b:\n\
        V_P_size[model.b.ord(b)-1] = model.C_Pot[b].value\n\
\n\
\n\
    df_pf = pd.DataFrame(V_pf)\n\
    df_gen_i = pd.DataFrame(V_gen_i)\n\
    df_gen_j = pd.DataFrame(V_gen_j)\n\
    df_gen_w = pd.DataFrame(V_gen_w)\n\
    df_gen_so = pd.DataFrame(V_gen_so)\n\
    df_losses = pd.DataFrame(V_losses)\n\
    df_Pot_Ba_ch = pd.DataFrame(V_Pot_Ba_ch)\n\
    df_Pot_Ba_dc = pd.DataFrame(V_Pot_Ba_dc)\n\
    df_e_b = pd.DataFrame(V_e_b)\n\
    df_E_size= pd.DataFrame(V_E_size)\n\
    df_P_size  = pd.DataFrame(V_P_size)\n\
    df_cost = pd.DataFrame(V_cost, index=['1','2'], columns=['Cost'])\n\
    df_cost  = df_cost.drop(['2'], axis=0)\n\
\n\
    mydir = os.getcwd()\n\
    name_file = 'Resultados/resultados_size_loc.xlsx'\n\
\n\
    path = os.path.join(mydir, name_file)\n\
\n\
    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')\n\
\n\
    df_pf.to_excel(writer, sheet_name='pf', index=True)\n\
    df_gen_i.to_excel(writer, sheet_name='gen_Thermal', index=True)\n\
    df_gen_j.to_excel(writer, sheet_name='gen_Hydro', index=True)\n\
    df_gen_w.to_excel(writer, sheet_name='gen_Wind', index=True)\n\
    df_gen_so.to_excel(writer, sheet_name='gen_Solar', index=True)\n\
    df_losses.to_excel(writer, sheet_name='Losses', index=True)\n\
    df_Pot_Ba_ch.to_excel(writer, sheet_name='BESS_Ch_Power', index=True)\n\
    df_Pot_Ba_dc.to_excel(writer, sheet_name='BESS_Dc_Power', index=True)\n\
    df_e_b.to_excel(writer, sheet_name='BESS_Energy', index=True)\n\
    df_E_size.to_excel(writer, sheet_name='Energy_size', index=True)\n\
    df_P_size.to_excel(writer, sheet_name='Power_size', index=True)\n\
    df_cost.to_excel(writer, sheet_name='cost', index=True)\n\
    writer.save()\n\
    writer.close()\n\
    ##########################################################################\n\
\n\
    return model.C_Pot, model.E_max, name_file, tiempo\n\
")
Loc_dim_OpC.close()

## Regulacion_F_UC_V6.py

Regulacion_F_UC_V6 = open('Herramienta/codes/Regulacion_F_UC_V6.py', 'w')
Regulacion_F_UC_V6.write("# -*- coding: utf-8 -*-\n\
\u0022\u0022\u0022\n\
Created on Mon Jun  8 09:20:08 2020\n\
\n\
UNIT COMMITMENT WITH ESS AND FREQUENCY CONSTRAINS\n\
\n\
@author: Andres Felipe Peñaranda Bayona\n\
\u0022\u0022\u0022\n\
\n\
# In[Librerias]\n\
\n\
from pyomo.environ import *\n\
from pyomo import environ as pym\n\
from pyomo import kernel as pmo\n\
import pandas as pd\n\
from pandas import ExcelWriter\n\
import matplotlib.pyplot as plt\n\
import numpy as np\n\
import math\n\
\n\
# In[Study case]\n\
\n\
def UC_ESS_with_Freq(system_data,Simulation_hours,opt_option):\n\
\n\
\n\
    df_System_data = pd.read_excel(system_data, sheet_name='System_data', header=0, index_col=0)\n\
    df_bus = pd.read_excel(system_data, sheet_name='Bus', header=0, index_col=0)\n\
    df_SM_Unit = pd.read_excel(system_data, sheet_name='SM_Unit', header=0, index_col=0)\n\
    df_SM_map = pd.read_excel(system_data, sheet_name='SM_map', header=0, index_col=0)\n\
    df_Renewable = pd.read_excel(system_data, sheet_name='Renewable', header=0, index_col=0)\n\
    df_branch = pd.read_excel(system_data, sheet_name='Branch', header=0, index_col=0)\n\
    df_line_map = pd.read_excel(system_data, sheet_name='line_map', header=0, index_col=0)\n\
    df_load = pd.read_excel(system_data, sheet_name='load', header=0, index_col=0)\n\
    df_Reserve = pd.read_excel(system_data, sheet_name='Reserve', header=0, index_col=0)\n\
    df_ESS_Unit = pd.read_excel(system_data, sheet_name='ESS_Unit', header=0, index_col=0)\n\
    df_ESS_map = pd.read_excel(system_data, sheet_name='ESS_map', header=0, index_col=0)\n\
    df_E_price = pd.read_excel(system_data, sheet_name='ESS_Energy_price', header=0, index_col=0)\n\
    df_C_DR = pd.read_excel(system_data, sheet_name ='C_DR_load', header=0, index_col=0)\n\
    df_P_C_DR = pd.read_excel(system_data, sheet_name ='PDR', header=0, index_col=0)\n\
\n\
\n\
    # In[Model]\n\
\n\
    ## Simulation type\n\
\n\
    model = ConcreteModel()\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # SETS\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
\n\
\n\
    model.t = RangeSet(1,Simulation_hours)                                        # Horizon simulation time\n\
    model.tt = SetOf(model.t)\n\
\n\
    I = []\n\
    for i in df_SM_Unit.index.tolist():\n\
        if df_SM_Unit.loc[i,'Fuel_Type'] != 'SOLAR' and df_SM_Unit.loc[i,'Fuel_Type'] != 'WIND':\n\
            I.append(i)\n\
    model.i = Set(initialize=I)                                                   # Generation units set\n\
\n\
    model.b = Set(initialize=df_bus.index.tolist(), ordered=True)                 # Buses of system set\n\
    model.k = Set(initialize=['k1', 'k2', 'k3'],ordered=True)                     # Segments of generation cost for generation units\n\
\n\
    W = []\n\
    for i in df_SM_Unit.index.tolist():\n\
        if df_SM_Unit.loc[i,'Fuel_Type'] == 'WIND':\n\
            W.append(i)\n\
    model.w = Set(initialize=W)                                                   # Wind generation units set\n\
\n\
    S = []\n\
    for i in df_SM_Unit.index.tolist():\n\
        if df_SM_Unit.loc[i,'Fuel_Type'] == 'SOLAR':\n\
            S.append(i)\n\
    model.s = Set(initialize=S)                                                   # Pv generation units set\n\
\n\
    model.l = Set(initialize=df_line_map.index.tolist(),ordered=True)             # lines of system set\n\
    model.n = Set(initialize=df_ESS_map.index.tolist(), ordered=True)             # Energy storage sistem units set\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # PARAMETERS\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    model.MVA_base = Param(initialize=df_System_data.loc['S_base','Data'])        # Power base of the system\n\
\n\
    model.Nominal_Frequency = Param(initialize=df_System_data.loc['N_freq','Data'])  # Nominal frequency of the system [Hz]\n\
    model.Max_Frequency_desviation = Param(initialize=df_System_data.loc['Max_D_freq','Data'])  # Maximum frequency desviation [Hz/s]\n\
    model.Minimum_Frequency = Param(initialize=df_System_data.loc['Min_freq','Data'])    # Minimum frequency limit [Hz]\n\
    model.DB_Frequency = Param(initialize=df_System_data.loc['DB_freq','Data'])   # Dead Band governors frequency [Hz]\n\
\n\
    model.Delta_t1 = Param(initialize=df_System_data.loc['Delta_1_RF','Data'])    # Time delta of post-contingency inertia response [h]\n\
    model.Delta_t2 = Param(initialize=df_System_data.loc['Delta_2_RF','Data'])    # Time delta of primary frequency response [h]\n\
    model.Delta_t3 = Param(initialize=df_System_data.loc['Delta_3_RF','Data'])    # Time delta of second frequency response [h]\n\
\n\
\n\
    def Power_max(model,i):\n\
        return df_SM_Unit.loc[i,'Pmax']\n\
    model.PG_max = Param(model.i, initialize = Power_max)                         # Maximum power capacity of each generator [MW]\n\
\n\
    def Power_min(model,i):\n\
        return df_SM_Unit.loc[i,'Pmin']\n\
    model.PG_min = Param(model.i, initialize = Power_min)                         # Minimum power capacity of each generator [MW]\n\
\n\
    def Power_seg_max(model,i,k):\n\
        return df_SM_Unit.loc[i,'k2']-df_SM_Unit.loc[i,'k1']\n\
    model.P_seg_max = Param(model.i, model.k, initialize = Power_seg_max)         # Maximum power capacity of each segment k of each generator [MW]\n\
\n\
    def Cost_seg(model,i,k):\n\
        if model.k.ord(k) == 1:\n\
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,'k1']+df_SM_Unit.loc[i,'Pmin'])+df_SM_Unit.loc[i,'b'])\n\
        else:\n\
            return (df_SM_Unit.loc[i,'a']*(df_SM_Unit.loc[i,k]+df_SM_Unit.loc[i,model.k[model.k.ord(k)-1]])+df_SM_Unit.loc[i,'b'])\n\
    model.C_seg = Param(model.i, model.k, initialize = Cost_seg)                  # Cost of power of each segment k of each generator [$/MW]\n\
\n\
    def fg_min_init(model,i):\n\
        return ((df_SM_Unit.loc[i,'a']*df_SM_Unit.loc[i,'Pmin']*df_SM_Unit.loc[i,'Pmin'])+(df_SM_Unit.loc[i,'b']*df_SM_Unit.loc[i,'Pmin'])+(df_SM_Unit.loc[i,'c']))\n\
    model.FG_min= Param(model.i, initialize=fg_min_init)                          # Price of generate minimum power for each generator [$]\n\
\n\
    def onoff_t0_init(model,i):\n\
        if df_SM_Unit.loc[i,'IniT_ON'] > 0:\n\
            a = 1\n\
        else:\n\
            a = 0\n\
        return a\n\
    model.onoff_t0 = Param(model.i, initialize=onoff_t0_init)                     # Initial ON/OFF signal [Binary]\n\
\n\
    def L_up_min_init(model,i):\n\
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_ON']-df_SM_Unit.loc[i,'IniT_ON'])*model.onoff_t0[i])\n\
    model.L_up_min = Param(model.i, initialize=L_up_min_init)                     # Minimum time for each generator when it is ON\n\
\n\
    def L_down_min_init(model,i):\n\
        return min(len(model.t), (df_SM_Unit.loc[i,'Min_OFF']-df_SM_Unit.loc[i,'IniT_off'])*(1-model.onoff_t0[i]))\n\
    model.L_down_min = Param(model.i, initialize=L_down_min_init)                 # Minimum time for each generator when it is OFF\n\
\n\
    def Reserve_max_rule(model,i):\n\
        return df_SM_Unit.loc[i,'R']*df_SM_Unit.loc[i,'Pmax']\n\
    model.Reserve_Max = Param(model.i, initialize=Reserve_max_rule)               # Maximum power of reserve for each generator [MW]\n\
\n\
    def CR_rule(model,i):\n\
        return df_SM_Unit.loc[i,'b']\n\
    model.C_Reserve = Param(model.i, initialize=CR_rule)                          # Cost of Primary reserve [$/MW]\n\
\n\
    def Ramp_power(model,i):\n\
        return df_SM_Unit.loc[i,'Ramp_Up']\n\
    model.Ramp = Param(model.i, initialize=Ramp_power)                            # Ramp power for each generator [MW/s]\n\
\n\
\n\
    def Power_wind(model,w,t):\n\
        return df_Renewable.loc[t,w]\n\
    model.Pw_max = Param(model.w, model.t, initialize = Power_wind)               # Power dispatch of wind turbine w at time t\n\
\n\
    def Power_solar(model,s,t):\n\
        return df_Renewable.loc[t,s]\n\
    model.Ppv_max = Param(model.s, model.t, initialize = Power_solar)             # Power dispatch of pv systems s at time t\n\
\n\
\n\
    def C_P_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'C_Potencia']\n\
    model.C_Pot = Param(model.n, initialize=C_P_Ba_rule)                          # Power Size [MW]\n\
\n\
    def C_E_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'C_Energia']\n\
    model.E_max = Param(model.n, initialize=C_E_Ba_rule)                          # Energy Size [MWh]\n\
\n\
    def C_nch_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'n_ch_eff']\n\
    model.n_ch = Param(model.n, initialize=C_nch_Ba_rule)                         # Charge efficency of ESS [p.u.]\n\
\n\
    def C_ndc_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'n_dc_eff']\n\
    model.n_dc = Param(model.n, initialize=C_ndc_Ba_rule)                         # Discharge efficency of ESS [p.u.]\n\
\n\
    def C_sdc_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'Self_discharge']\n\
    model.s_dc = Param(model.n, initialize=C_sdc_Ba_rule)                         # Self-Discharge efficency of ESS [p.u./h]\n\
\n\
    def C_SOCmin_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'SOC_min']\n\
    model.SOC_min = Param(model.n, initialize=C_SOCmin_Ba_rule)                   # Minimum State of Charge of BESS\n\
\n\
    def C_SOCini_Ba_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'SOC_ini']\n\
    model.SOC_ini = Param(model.n, initialize=C_SOCini_Ba_rule)                   # Initial State of Charge of BESS\n\
\n\
    def C_D_Ba_rule(model,t):\n\
        return df_E_price.loc[t,'CB_MWh']\n\
    model.C_D_Ba = Param(model.t, initialize=C_D_Ba_rule)                         # Cost of Energy Storage System Degradation [$/MWh]\n\
\n\
    def Delta_energy_rule(model,n):\n\
        return df_ESS_Unit.loc[n,'C_Potencia']*(model.Delta_t1 + model.Delta_t2 + 0.5*model.Delta_t3)\n\
    model.Delta_energy = Param(model.n, initialize=Delta_energy_rule)             # Delta energy for contingency supply [MWh]\n\
\n\
\n\
    def Bus_load(model,b,t):\n\
        return df_load.loc[t,b]\n\
    model.P_load = Param(model.b, model.t, initialize = Bus_load)                 # Load of each bus in the test system [MW]\n\
\n\
\n\
    def CDR_rule(model,b,t):\n\
        return df_C_DR.loc[t,b]\n\
    model.C_DR = Param(model.b, model.t, initialize=CDR_rule)                     # Cost of power reduction of customer [$/MW]\n\
\n\
    def DR_Max_rule(model,b,t):\n\
        return df_load.loc[t,b]*(1-df_P_C_DR.loc[t,b])\n\
    model.Pot_DR_Max = Param(model.b, model.t, initialize=DR_Max_rule)            # Maximum power reduction for each consumer\n\
\n\
\n\
    def Power_imbalance_value(model,t):\n\
        return df_Reserve.loc[t,'R1']\n\
    model.Power_imbalance = Param(model.t,initialize=Power_imbalance_value)       # Power imbalance at contingency [MW]\n\
\n\
    def Inertia_rule(model,i):\n\
        return df_SM_Unit.loc[i,'H']\n\
    model.Inertia = Param(model.i, initialize = Inertia_rule)                     # Inertia value of each generator [s]\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # VARIABLES\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    model.P = Var(model.i, model.t, domain=NonNegativeReals)                      # Power dispatch of unit i at time t [MW]\n\
    model.status = Var(model.i, model.t, within=Binary)                           # Status [ON/OFF] for each generator [Binary]\n\
    model.P_seg = Var(model.i, model.t, model.k, domain=NonNegativeReals)         # Power dispatch segment k of i at time t [MW]\n\
    model.costSU = Var(model.i, model.t, domain=NonNegativeReals)                 # Start UP cost for each generator [$]\n\
    model.SU = Var(model.i, model.t, within=Binary, initialize=0)                 # Star UP signal [Binary]\n\
    model.SD = Var(model.i, model.t, within=Binary, initialize=0)                 # Star DOWN signal [Binary]\n\
    model.Reserve = Var(model.i, model.t, domain= NonNegativeReals)               # Power resever for primary reserve [MW]\n\
\n\
    model.Pw = Var(model.w, model.t, domain=NonNegativeReals)                     # Power dispatch of unit w at time t [MW]\n\
    model.Ppv = Var(model.s, model.t, domain=NonNegativeReals)                    # Power dispatch of unit p at time t [MW]\n\
\n\
    model.Pot_DR = Var(model.b, model.t, domain = NonNegativeReals)               # Power reduction of customer [MW]\n\
\n\
    model.theta = Var(model.b, model.t, bounds=(-math.pi,math.pi))                # Voltage angle [rad]\n\
    model.pf = Var(model.l, model.t)                                              # Power flow through the line l [MW]\n\
    model.P_bus = Var(model.b, model.t)                                           # Power bus balance [MW]\n\
    model.PG_bus = Var(model.b, model.t)                                          # Power of SM in bus b [MW]\n\
    model.PW_bus = Var(model.b, model.t)                                          # Power of wind turbine in bus b [MW]\n\
    model.PPV_bus = Var(model.b, model.t)                                         # Power of Pv systems in bus b [MW]\n\
    model.PDC_bus = Var(model.b, model.t)                                         # Charge power of ESS in bus b [MW]\n\
    model.PCH_bus = Var(model.b, model.t)                                         # Discharge power of ESS in bus b [MW]\n\
\n\
    for t in model.t:\n\
        model.theta[df_System_data.loc['Slack_bus','Data'],t].fix(0)              # Slack angle [rad]\n\
\n\
    model.Pot_Ba_ch = Var(model.n, model.t, initialize=0)                         # Power in battery charge [MW]\n\
    model.Pot_Ba_dc = Var(model.n, model.t, initialize=0)                         # Power in battery discharge [MW]\n\
    model.u_ch = Var(model.n, model.t, within=Binary, initialize=0)               # Status of battery charge {Binary}\n\
    model.u_dc = Var(model.n, model.t, within=Binary, initialize=0)               # Status of battery discharge [Binary]\n\
    model.e_b = Var(model.n, model.t, domain=NonNegativeReals, initialize=0)      # Energy of battery [MWh]\n\
\n\
    model.PC_Power_imbalance = Var(model.t, bounds=(0,1e6))                       # Power imbalance post contingency [MW]\n\
\n\
    model.Total_inertia = Var(model.t, bounds=(0,1e6))                            # Total inertia of the system at time t\n\
\n\
    model.A_ch = Var(model.i, model.n, model.t, domain=NonNegativeReals)          # Charge Auxiliary \n\
    model.A_dc = Var(model.i, model.n, model.t, domain=NonNegativeReals)          # Discharge Auxiliary variables\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # OBJETIVE FUNCTION\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    def Cost_rule(model):\n\
        return sum(sum((model.costSU[i,t])+(model.FG_min[i]*model.status[i,t])+(sum(model.P_seg[i,t,k]*model.C_seg[i,k] for k in model.k)) + (model.C_Reserve[i]*model.Reserve[i,t]) for i in model.i) for t in model.t) + sum(sum(model.C_D_Ba[t]*(model.Pot_Ba_ch[n,t]) for n in model.n) for t in model.t) + sum(sum((model.C_DR[b,t]*model.Pot_DR[b,t]) for b in model.b) for t in model.t)\n\
    model.Objetivo = Objective(rule = Cost_rule, sense=minimize)\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # CONSTRAINS\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    ## Power generation SM limits constraints\n\
\n\
    def P_seg_base_max_rule1(model,i,t,k):\n\
        return model.P_seg[i,t,k]>=0\n\
    model.P_seg_lim1 = Constraint(model.i, model.t, model.k, rule=P_seg_base_max_rule1)\n\
\n\
    def P_seg_base_max_rule2(model,i,t,k):\n\
        return model.P_seg[i,t,k]<=model.P_seg_max[i,k]*model.status[i,t];\n\
    model.P_seg_lim2 = Constraint(model.i, model.t, model.k, rule=P_seg_base_max_rule2)\n\
\n\
    def P_sum_rule(model,i,t):\n\
        return model.P[i,t] == model.status[i,t]*model.PG_min[i] + sum(model.P_seg[i,t,k] for k in model.k)\n\
    model.P_sum = Constraint(model.i, model.t, rule=P_sum_rule)\n\
\n\
    def P_lim_max_rule(model,i,t):\n\
        return model.P[i,t] + model.Reserve[i,t] <= model.PG_max[i]*model.status[i,t]\n\
    model.P_max_lim = Constraint(model.i, model.t, rule=P_lim_max_rule)\n\
\n\
    def P_lim_min_rule(model,i,t):\n\
        return model.P[i,t] >= model.PG_min[i]*model.status[i,t]\n\
    model.P_min_lim = Constraint(model.i, model.t, rule=P_lim_min_rule)\n\
\n\
    def bin_cons1_rule(model,i,t):\n\
        if t == model.t.first():\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.onoff_t0[i]\n\
        else:\n\
            return model.SU[i,t] - model.SD[i,t] == model.status[i,t] - model.status[i,t-1]\n\
    model.bin_cons1 = Constraint(model.i, model.t, rule=bin_cons1_rule)\n\
\n\
    def bin_cons2_rule(model,i,t):\n\
        return model.SU[i,t] + model.SD[i,t] <= 1\n\
    model.bin_cons2 = Constraint(model.i, model.t, rule=bin_cons2_rule)\n\
\n\
    def CostSUfn_init(model,i,t):\n\
        return model.costSU[i,t] == df_SM_Unit.loc[i,'CSU']*model.SU[i,t]\n\
    model.CostSUfn = Constraint(model.i, model.t, rule=CostSUfn_init)\n\
\n\
    def ramp_up_fn_rule(model,i,t):\n\
        if t > 1:\n\
            return model.P[i,t] - model.P[i,t-1] <= df_SM_Unit.loc[i,'Ramp_Up']\n\
        else:\n\
            return Constraint.Skip\n\
    model.ramp_up_fn = Constraint(model.i, model.t, rule=ramp_up_fn_rule)\n\
\n\
    def ramp_dw_fn_rule(model,i,t):\n\
        if t > 1:\n\
            return model.P[i,t] - model.P[i,t-1] >= -df_SM_Unit.loc[i,'Ramp_Down']\n\
        else:\n\
            return Constraint.Skip\n\
    model.ramp_dw_fn = Constraint(model.i, model.t, rule=ramp_dw_fn_rule)\n\
\n\
    # def min_up_dn_time_1_rule(model,i,t):\n\
    #     if model.L_up_min[i] + model.L_down_min[i] > 0 and t < model.L_up_min[i] + model.L_down_min[i]:\n\
    #         return model.status[i,t] == model.onoff_t0[i]\n\
    #     else:\n\
    #         return Constraint.Skip\n\
    # model.min_up_dn_time_1 = Constraint(model.i, model.t, rule=min_up_dn_time_1_rule)\n\
\n\
    # def min_up_dn_time_2_rule(model,i,t):\n\
    #     return sum(model.SU[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_ON']+1 and tt <= t) <= model.status[i,t]\n\
    # model.min_up_dn_time_2 = Constraint(model.i, model.t, rule=min_up_dn_time_2_rule)\n\
\n\
    # def min_up_dn_time_3_rule(model,i,t):\n\
    #     return sum(model.SD[i,tt] for tt in model.tt if tt >= t-df_SM_Unit.loc[i,'Min_OFF']+1 and tt <= t) <= 1-model.status[i,t]\n\
    # model.min_up_dn_time_3 = Constraint(model.i, model.t, rule=min_up_dn_time_3_rule)\n\
\n\
    ## Renewable constraints\n\
\n\
    ## Power generation WIND limits constraints\n\
\n\
    def P_lim_max_rule_w(model,w,t):\n\
        return model.Pw[w,t] <= model.Pw_max[w,t]\n\
    model.P_max_lim_w = Constraint(model.w, model.t, rule=P_lim_max_rule_w)\n\
\n\
\n\
    ## Power generation SOLAR limits constraints\n\
\n\
    def P_lim_max_rule_s(model,s,t):\n\
        return model.Ppv[s,t] <= model.Ppv_max[s,t]\n\
    model.P_max_lim_s = Constraint(model.s, model.t, rule=P_lim_max_rule_s)\n\
\n\
\n\
    ## Demand Response constraints\n\
\n\
    def Demand_response_rule(model,b,t):\n\
        return model.Pot_DR[b,t] <= model.Pot_DR_Max[b,t]\n\
    model.Demand_response = Constraint(model.b, model.t, rule=Demand_response_rule)\n\
\n\
\n\
    ## ESS constraints\n\
\n\
    def power_c_max_rule(model,n,t):\n\
        return model.Pot_Ba_ch[n,t] <= model.C_Pot[n]*model.u_ch[n,t]\n\
    model.power_c_max = Constraint(model.n, model.t, rule=power_c_max_rule)\n\
\n\
    def power_c_max_2_rule(model,n,t):\n\
        return model.Pot_Ba_ch[n,t] >= 0\n\
    model.power_c_max_2 = Constraint(model.n, model.t, rule=power_c_max_2_rule)\n\
\n\
    def power_d_max_rule(model,n,t):\n\
        return model.Pot_Ba_dc[n,t] <= model.C_Pot[n]*model.u_dc[n,t]\n\
    model.power_d_max = Constraint(model.n, model.t, rule=power_d_max_rule)\n\
\n\
    def power_d_max_2_rule(model,n,t):\n\
        return model.Pot_Ba_dc[n,t] >= 0\n\
    model.power_d_max_2 = Constraint(model.n, model.t, rule=power_d_max_2_rule)\n\
\n\
    def sim_rule(model,n,t):\n\
        return model.u_ch[n,t] + model.u_dc[n,t] <= 1\n\
    model.sim = Constraint(model.n, model.t, rule=sim_rule)\n\
\n\
    # relation betwent energy status and power charging and discharging Constraint\n\
\n\
    def energy_rule(model,n,t):\n\
        if t == 1:\n\
            return model.e_b[n,t] == (model.E_max[n]*model.SOC_ini[n]) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])\n\
        else:\n\
            return model.e_b[n,t] == (model.e_b[n,t-1]*(1-model.s_dc[n])) + (model.n_ch[n]*model.Pot_Ba_ch[n,t]) - ((model.Pot_Ba_dc[n,t])/model.n_dc[n])\n\
    model.energy = Constraint(model.n, model.t, rule=energy_rule)\n\
\n\
    # Energy limits\n\
\n\
    def energy_limit_rule(model,n,t):\n\
        return model.e_b[n,t] <= model.E_max[n]\n\
    model.energy_limit = Constraint(model.n, model.t, rule=energy_limit_rule)\n\
\n\
    def energy_limit_min_rule(model,n,t):\n\
        return model.e_b[n,t] >= model.E_max[n]*model.SOC_min[n]\n\
    model.energy_limit_min = Constraint(model.n, model.t, rule=energy_limit_min_rule)\n\
\n\
    def energy_final_value_rule(model,n,t):\n\
        return model.e_b[n,model.t.first()] == model.e_b[n,model.t.last()]\n\
    model.energy_final_value = Constraint(model.n, model.t, rule=energy_final_value_rule)\n\
\n\
\n\
    ## Power balance constrains\n\
\n\
    def line_flow_rule(model, t, l):\n\
        return model.pf[l,t] == model.MVA_base*(1/df_branch.loc[l,'X'])*sum(model.theta[b,t]*df_line_map.loc[l,b] for b in model.b if df_line_map.loc[l,b] != 0)\n\
    model.line_flow = Constraint(model.t, model.l, rule=line_flow_rule)\n\
\n\
    def line_min_rule(model, t, l):\n\
        return model.pf[l,t] >= - df_branch.loc[l,'Flowlimit']\n\
    model.line_min = Constraint(model.t, model.l, rule=line_min_rule)\n\
\n\
    def line_max_rule(model, t, l):\n\
        return model.pf[l,t] <= df_branch.loc[l,'Flowlimit']\n\
    model.line_max = Constraint(model.t, model.l, rule=line_max_rule)\n\
\n\
    def Power_bus_rule(model, t, b):\n\
        return model.P_bus[b,t] == sum(model.pf[l,t]*df_line_map.loc[l,b] for l in model.l if df_line_map.loc[l,b] != 0)\n\
    model.Power_bus = Constraint(model.t, model.b, rule=Power_bus_rule)\n\
\n\
    def PowerG_bus_rule(model, t, b):\n\
        return model.PG_bus[b,t] == sum(model.P[i,t] for i in model.i if df_SM_map.loc[i,b])\n\
    model.PowerG_bus = Constraint(model.t, model.b, rule=PowerG_bus_rule)\n\
\n\
    def PowerW_bus_rule(model, t, b):\n\
        return model.PW_bus[b,t] == sum(model.Pw[w,t] for w in model.w if df_SM_map.loc[w,b])\n\
    model.PowerW_bus = Constraint(model.t, model.b, rule=PowerW_bus_rule)\n\
\n\
    def PowerPv_bus_rule(model, t, b):\n\
        return model.PPV_bus[b,t] == sum(model.Ppv[s,t] for s in model.s if df_SM_map.loc[s,b])\n\
    model.PowerPv_bus = Constraint(model.t, model.b, rule=PowerPv_bus_rule)\n\
\n\
    def PowerDC_bus_rule(model, t, b):\n\
        return model.PDC_bus[b,t] == sum(model.Pot_Ba_dc[n,t] for n in model.n if df_ESS_map.loc[n,b])\n\
    model.PowerDC_bus = Constraint(model.t, model.b, rule=PowerDC_bus_rule)\n\
\n\
    def PowerCH_bus_rule(model, t, b):\n\
        return model.PCH_bus[b,t] == sum(model.Pot_Ba_ch[n,t] for n in model.n if df_ESS_map.loc[n,b])\n\
    model.PowerCH_bus = Constraint(model.t, model.b, rule=PowerCH_bus_rule)\n\
\n\
    def power_balance_rule(model,t):\n\
        return sum(model.P[i,t] for i in model.i) + sum(model.Pw[w,t] for w in model.w) + sum(model.Ppv[s,t] for s in model.s) + sum(model.Pot_Ba_dc[n,t] for n in model.n) == sum(model.P_load[b,t]- model.Pot_DR[b,t] for b in model.b) + sum(model.Pot_Ba_ch[n,t] for n in model.n)\n\
    model.power_balance = Constraint(model.t, rule=power_balance_rule)\n\
\n\
    def power_balance_rule2(model, t, b):\n\
        return model.PG_bus[b,t] + model.PW_bus[b,t] + model.PPV_bus[b,t] + model.PDC_bus[b,t] - model.PCH_bus[b,t] - model.P_load[b,t] - model.Pot_DR[b,t] == model.P_bus[b,t]\n\
    model.power_balance2 = Constraint(model.t, model.b, rule=power_balance_rule2)\n\
\n\
\n\
    ## Primary Reserve limits\n\
\n\
    def Primary_reserve_rule1(model,i,t):\n\
        return model.Reserve[i,t]<= model.Reserve_Max[i]\n\
    model.Primary_reserve1 = Constraint(model.i, model.t, rule = Primary_reserve_rule1)\n\
\n\
    def Primary_reserve_rule2(model,i,t):\n\
        return  model.Reserve[i,t]>=0\n\
    model.Primary_reserve2 = Constraint(model.i, model.t, rule = Primary_reserve_rule2)\n\
\n\
    def Primary_reserve_rule3(model,t):\n\
        return sum(model.Reserve[i,t] for i in model.i) >= model.PC_Power_imbalance[t]\n\
    model.Primary_reserve3 = Constraint(model.t, rule = Primary_reserve_rule3)\n\
\n\
\n\
    ## Post Contingency Frequency Dynamics\n\
\n\
    # System power imbalance\n\
\n\
    def Power_imbalance_rule1(model,t):\n\
        return model.PC_Power_imbalance[t] == model.Power_imbalance[t] - sum(model.C_Pot[n] - model.Pot_Ba_dc[n,t] + model.Pot_Ba_ch[n,t] for n in model.n)\n\
    model.Power_Imbalance_1 = Constraint(model.t, rule = Power_imbalance_rule1)\n\
\n\
    # Energy of ESS in contingency\n\
\n\
    def contingency_energy_rule1(model,n,t):\n\
        return (model.e_b[n,t]*(1-model.s_dc[n])) - model.Delta_energy[n] >= model.E_max[n]*model.SOC_min[n]\n\
    model.contingency_energy_1 = Constraint(model.n, model.t, rule=contingency_energy_rule1)\n\
\n\
    def contingency_energy_rule2(model,n,t):\n\
        return (model.e_b[n,t]*(1-model.s_dc[n])) - model.Delta_energy[n] <= model.E_max[n]\n\
    model.contingency_energy_2 = Constraint(model.n, model.t, rule=contingency_energy_rule2)\n\
\n\
    # # Total inertia value\n\
\n\
    def Total_inertia_rule(model,t):\n\
        return model.Total_inertia[t] == sum(model.Inertia[i]*model.PG_max[i]*model.status[i,t] for i in model.i)/model.Nominal_Frequency\n\
    model.Total_Inertia_value = Constraint(model.t, rule = Total_inertia_rule)\n\
\n\
    # # Frequency desviation limit\n\
\n\
    def Frequency_desviation_rule(model,t):\n\
        return model.Total_inertia[t] >= model.PC_Power_imbalance[t]/(2*model.Max_Frequency_desviation)\n\
    Frequency_limits1 = Constraint(model.t, rule = Frequency_desviation_rule)\n\
\n\
    def Frequency_desviation_rule2(model,i,t):\n\
        return (model.Reserve[i,t]*model.Power_imbalance[t]) - sum(model.Reserve[i,t]*model.C_Pot[n] for n in model.n) + sum(model.A_dc[i,n,t] for n in model.n) - sum(model.A_ch[i,n,t] for n in model.n) <= 2*model.Ramp[i]*model.Total_inertia[t]*(model.Nominal_Frequency - model.Minimum_Frequency - model.DB_Frequency)\n\
    model.Frequency_limits2 = Constraint(model.i, model.t, rule = Frequency_desviation_rule2)\n\
\n\
    ## Reformulation - Linearization Technique (RLT)\n\
\n\
    def Reformulation1_rule(model,i,n,t):\n\
        return model.A_ch[i,n,t] - (model.Reserve_Max[i]*model.Pot_Ba_ch[n,t]) - (model.C_Pot[n]*model.Reserve[i,t]) + (model.C_Pot[n]*model.Reserve_Max[i]) >= 0\n\
    model.Reformulation1 = Constraint(model.i, model.n, model.t, rule = Reformulation1_rule)\n\
\n\
    def Reformulation2_rule(model,i,n,t):\n\
        return - model.A_ch[i,n,t] + (model.C_Pot[n]*model.Reserve[i,t])>= 0\n\
    model.Reformulation2 = Constraint(model.i, model.n, model.t, rule = Reformulation2_rule)\n\
\n\
    def Reformulation3_rule(model,i,n,t):\n\
        return - model.A_ch[i,n,t] + (model.Reserve_Max[i]*model.Pot_Ba_ch[n,t])>= 0\n\
    model.Reformulation3 = Constraint(model.i, model.n, model.t, rule = Reformulation3_rule)\n\
\n\
    def Reformulation4_rule(model,i,n,t):\n\
        return model.A_dc[i,n,t] - (model.Reserve_Max[i]*model.Pot_Ba_dc[n,t]) - (model.C_Pot[n]*model.Reserve[i,t]) + (model.C_Pot[n]*model.Reserve_Max[i]) >= 0\n\
    model.Reformulation4 = Constraint(model.i, model.n, model.t, rule = Reformulation4_rule)\n\
\n\
    def Reformulation5_rule(model,i,n,t):\n\
        return - model.A_dc[i,n,t] + (model.C_Pot[n]*model.Reserve[i,t])>= 0\n\
    model.Reformulation5 = Constraint(model.i, model.n, model.t, rule = Reformulation5_rule)\n\
\n\
    def Reformulation6_rule(model,i,n,t):\n\
        return - model.A_dc[i,n,t] + (model.Reserve_Max[i]*model.Pot_Ba_dc[n,t])>= 0\n\
    model.Reformulation6 = Constraint(model.i, model.n, model.t, rule = Reformulation6_rule)\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # SOLVER CONFIGURATION\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    def pyomo_postprocess(options=None, instance=None, results=None):\n\
        model.Objetivo.display()\n\
\n\
\n\
    if opt_option == 1:\n\
        opt = SolverManagerFactory('neos')\n\
        results = opt.solve(model, opt='cplex')\n\
        #sends results to stdout\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
    else:\n\
        opt = SolverFactory('glpk')\n\
        results = opt.solve(model)\n\
        results.write()\n\
        print('\u005cnDisplaying Solution\u005cn' + '-'*60)\n\
        pyomo_postprocess(None, model, results)\n\
\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
    # DATA\n\
    #-------------------------------------------------------------------------------------------------------------------------------------------\n\
\n\
    ############################################ DATA ####################################################################################\n\
\n\
    P = {}\n\
    Reserve = {}\n\
    Pw = {}\n\
    Ppv = {}\n\
    P_Ba_dc = {}\n\
    P_Ba_ch = {}\n\
    P_Ba = {}\n\
    Energy_Ba = {}\n\
    P_load = {}\n\
    RoCoF_Ini = []\n\
    RoCoF_Ba = []\n\
    RoCoF_lim = []\n\
    Delta_f_Ini = []\n\
    Delta_f_Ba = []\n\
    Delta_f_Max = []\n\
    f_nadir_Ini = []\n\
    f_nadir_Ba = []\n\
    f_min = []\n\
    Time = []\n\
    Reserve_obj = []\n\
    Reserve_BESS = {}\n\
    Generacion = {}\n\
\n\
    C_nadir = sum(model.Ramp[i]for i in model.i)\n\
\n\
    Data1 = []\n\
    Data2 = []\n\
    Data3 = []\n\
    Data4 = []\n\
    Data5 = []\n\
    Data6 = []\n\
    Data7 = []\n\
\n\
    for t in model.t:\n\
        RoCoF_Ini.append((model.Power_imbalance[t])/(2*model.Total_inertia[t].value))\n\
        RoCoF_Ba.append((model.PC_Power_imbalance[t].value)/(2*model.Total_inertia[t].value))\n\
        RoCoF_lim.append(model.Max_Frequency_desviation.value)\n\
        f_nadir_Ini.append(model.Nominal_Frequency.value - model.DB_Frequency.value - (((model.Power_imbalance[t])**2)/(2*C_nadir*model.Total_inertia[t].value)))\n\
        f_nadir_Ba.append(model.Nominal_Frequency.value - model.DB_Frequency.value - (((model.PC_Power_imbalance[t].value)**2)/(2*C_nadir*model.Total_inertia[t].value)))\n\
        f_min.append(model.Minimum_Frequency.value)\n\
        Delta_f_Ini.append(model.Nominal_Frequency.value - f_nadir_Ini[t-1])\n\
        Delta_f_Ba.append(model.Nominal_Frequency.value - f_nadir_Ba[t-1])\n\
        Delta_f_Max.append(model.Nominal_Frequency.value - model.Minimum_Frequency.value)\n\
        Reserve_obj.append(model.Power_imbalance[t])\n\
        Time.append(t)\n\
\n\
        P_H = 0\n\
        P_Gas = 0\n\
        P_Coal = 0\n\
        P_Oil = 0\n\
        P_Wind = 0\n\
        P_Solar = 0\n\
        P_BESS = 0\n\
\n\
\n\
        for i in model.i:\n\
            if df_SM_Unit.loc[i,'Fuel_Type'] == 'HYDRO':\n\
                P_H = model.P[i,t].value + P_H\n\
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'GAS':\n\
                P_Gas = model.P[i,t].value + P_Gas\n\
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'COAL':\n\
                P_Coal = model.P[i,t].value + P_Coal\n\
            elif df_SM_Unit.loc[i,'Fuel_Type'] == 'ACPM' or df_SM_Unit.loc[i,'Fuel_Type'] == 'DIESEL' or df_SM_Unit.loc[i,'Fuel_Type'] == 'COMBUSTOLEO':\n\
                P_Oil = model.P[i,t].value + P_Oil\n\
\n\
        Data1.append(P_H)\n\
        Data2.append(P_Gas)\n\
        Data3.append(P_Coal)\n\
        Data4.append(P_Oil)\n\
\n\
        for w in model.w:\n\
            P_Wind = df_Renewable.loc[t,w] + P_Wind\n\
\n\
        Data5.append(P_Wind)\n\
\n\
        for s in model.s:\n\
            P_Solar = df_Renewable.loc[t,s] + P_Solar\n\
\n\
        Data6.append(P_Solar)\n\
\n\
        for n in model.n:\n\
            P_BESS = model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value + P_BESS\n\
\n\
        Data7.append(P_BESS)\n\
\n\
    Generacion['Hydro'] = Data1\n\
    Generacion['Natural gas'] = Data2\n\
    Generacion['Coal'] = Data3\n\
    Generacion['Oil'] = Data4\n\
    Generacion['Wind'] = Data5\n\
    Generacion['Solar'] = Data6\n\
    Generacion['BESS'] = Data7\n\
\n\
\n\
    for i in model.i:\n\
        Data1 = []\n\
        Data2 = []\n\
        for t in model.t:\n\
            Data1.append(model.P[i,t].value)\n\
            Data2.append(model.Reserve[i,t].value)\n\
        P[i] = Data1\n\
        Reserve[i] = Data2\n\
\n\
    for w in model.w:\n\
        Data1 = []\n\
        for t in model.t:\n\
            Data1.append(df_Renewable.loc[t,w])\n\
        Pw[w] = Data1\n\
\n\
    for s in model.s:\n\
        Data1 = []\n\
        for t in model.t:\n\
            Data1.append(df_Renewable.loc[t,s])\n\
        Ppv[s] = Data1\n\
\n\
    for n in model.n:\n\
        Data1 = []\n\
        Data2 = []\n\
        Data3 = []\n\
        Data4 = []\n\
        Data5 = []\n\
        for t in model.t:\n\
            Data1.append(model.Pot_Ba_dc[n,t].value)\n\
            Data2.append(model.e_b[n,t].value)\n\
            Data3.append(model.Pot_Ba_ch[n,t].value)\n\
            Data4.append(model.Pot_Ba_dc[n,t].value - model.Pot_Ba_ch[n,t].value)\n\
            Data5.append(model.C_Pot[n] - model.Pot_Ba_dc[n,t].value + model.Pot_Ba_ch[n,t].value)\n\
        P_Ba_dc[n] = Data1\n\
        Energy_Ba[n] = Data2\n\
        P_Ba_ch[n] = Data3\n\
        P_Ba[n] = Data4\n\
        Reserve_BESS[n] = Data5\n\
\n\
\n\
    for b in model.b:\n\
        Data1 = []\n\
        for t in model.t:\n\
            Data1.append(model.P_load[b,t])\n\
        P_load[b] = Data1\n\
\n\
    ############################################ PLOTS ###################################################################################\n\
\n\
    ## Plot configuration:\n\
\n\
    title_font = {'fontname':'Arial', 'size':'25', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}\n\
    axis_font = {'fontname':'Arial', 'size':'14'}\n\
\n\
    P_acumulado = []\n\
    P_acumulado_labels = []\n\
\n\
    # Plot Power Dispatch\n\
\n\
    fig=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    for n in model.n:\n\
         if sum(P_Ba[n])!= 0:\n\
            P_acumulado.append(P_Ba[n])\n\
            ax.plot(Time,P_Ba[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))\n\
            P_acumulado_labels.append(n)\n\
    for i in model.i:\n\
        if sum(P[i])>0:\n\
            P_acumulado.append(P[i])\n\
            ax.plot(Time,P[i], linestyle = '-', lw = 2.5 ,label = '{}'.format(i))\n\
            P_acumulado_labels.append(i)\n\
    for w in model.w:\n\
        if sum(Pw[w])>0:\n\
            P_acumulado.append(Pw[w])\n\
            ax.plot(Time,Pw[w], linestyle = '-', lw = 2.5 ,label = '{}'.format(w))\n\
            P_acumulado_labels.append(w)\n\
    for s in model.s:\n\
        if sum(Ppv[s])>0:\n\
            P_acumulado.append(Ppv[s])\n\
            ax.plot(Time,Ppv[s], linestyle = '-', lw = 2.5 ,label = '{}'.format(s))\n\
            P_acumulado_labels.append(s)\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Power Dispatch', **title_font)\n\
\n\
    fig1=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.stackplot(Time,P_acumulado, labels = P_acumulado_labels)\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Power Dispatch', **title_font)\n\
\n\
    fig2=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.stackplot(Time,Generacion['Hydro'],Generacion['Natural gas'],Generacion['Coal'],Generacion['Oil'],Generacion['Wind'],Generacion['Solar'],Generacion['BESS'], labels = ['Hydro','Natural gas','Coal','Oil','Wind','Solar','BESS'])\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Power Dispatch', **title_font)\n\
\n\
    # Plot Power Demand\n\
\n\
    D_acumulado = []\n\
    D_acumulado_labels = []\n\
\n\
    plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    for b in model.b:\n\
        D_acumulado.append(P_load[b])\n\
        ax.plot(Time,P_load[b], linestyle = '-', lw = 2.5 ,label = '{}'.format(b))\n\
        D_acumulado_labels.append(b)\n\
    for n in model.n:\n\
        D_acumulado.append(P_Ba_ch[n])\n\
        ax.plot(Time,P_Ba_ch[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))\n\
        D_acumulado_labels.append(n)\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Power Demand', **title_font)\n\
\n\
\n\
    fig3=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.stackplot(Time,D_acumulado, labels = D_acumulado_labels)\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Power Demand', **title_font)\n\
\n\
    # Plot Reserve\n\
\n\
    R_acumulado = []\n\
    R_acumulado_labels = []\n\
\n\
    fig3=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    for i in model.i:\n\
        if sum(Reserve[i])>0:\n\
            R_acumulado.append(Reserve[i])\n\
            ax.plot(Time,Reserve[i], linestyle = '-', lw = 2.5 ,label = '{}'.format(i))\n\
            R_acumulado_labels.append(i)\n\
    for n in model.n:\n\
        if sum(Reserve_BESS[n])>0:\n\
            R_acumulado.append(Reserve_BESS[n])\n\
            ax.plot(Time,Reserve_BESS[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))\n\
            R_acumulado_labels.append(n)\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Primary Power Reserve', **title_font)\n\
\n\
    fig4=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.stackplot(Time,R_acumulado, labels = R_acumulado_labels)\n\
    ax.plot(Time,Reserve_obj, linestyle = '--', color='r', lw = 2.5 ,label = 'Power imbalance capacity')\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MW]', **axis_font)\n\
    plt.title('Primary Power Reserve', **title_font)\n\
\n\
    # Plot Energy in BESS\n\
\n\
    fig5=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    for n in model.n:\n\
        ax.plot(Time,Energy_Ba[n], linestyle = '-', lw = 2.5 ,label = '{}'.format(n))\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=10)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[MWh]', **axis_font)\n\
    plt.title('Energy in ESS', **title_font)\n\
\n\
    # Plot RoCoF\n\
\n\
    x = np.arange(len(Time))\n\
    fig6=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.bar((x+1) - 0.35/2,RoCoF_Ini,0.35,label = 'Without BESS')\n\
    ax.bar((x+1) + 0.35/2,RoCoF_Ba,0.35,label = 'With BESS')\n\
    ax.plot(Time,RoCoF_lim, linestyle = '--', color='r', lw = 2.5 ,label = 'RoCoF limit')\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)\n\
    ax.set_xticklabels(Time)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[Hz/s]', **axis_font)\n\
    plt.title('RoCoF', **title_font)\n\
\n\
    # # Plot fnadir\n\
\n\
    x = np.arange(len(Time))\n\
    fig7=plt.figure(figsize=(15,9))\n\
    ax = plt.subplot(111)\n\
    ax.bar((x+1) - 0.35/2,Delta_f_Ini,0.35,label = 'Without BESS')\n\
    ax.bar((x+1) + 0.35/2,Delta_f_Ba,0.35,label = 'With BESS')\n\
    ax.plot(Time,Delta_f_Max, linestyle = '--', color='r', lw = 2.5 ,label = 'Delta f limit')\n\
    box = ax.get_position()\n\
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])\n\
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)\n\
    ax.set_xticklabels(Time)\n\
    plt.xticks(Time,**axis_font)\n\
    plt.yticks(**axis_font)\n\
    plt.grid(True)\n\
    plt.xlabel('Hour', **axis_font)\n\
    plt.ylabel('[Hz]', **axis_font)\n\
    plt.title('Delta of frequency', **title_font)\n\
\n\
    Figuras=[fig,fig1,fig2,fig3,fig4,fig5,fig6,fig7]\n\
\n\
    return Figuras\n\
")