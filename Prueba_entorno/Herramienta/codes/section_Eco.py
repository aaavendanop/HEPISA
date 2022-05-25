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

def dashboard_Eco(data2,data3):

    st.markdown("<h1 style='text-align: center; color: black;'>Análisis financiero</h1>", unsafe_allow_html=True)
    # image = Image.open('arbitrage.jpeg')
    # st.image(image, caption='', use_column_width=True)
    st.markdown('En esta sección de la herramienta es posible realizar un análisis \
    financiero para evaluar la conveniencia económica de implementar proyectos con \
    uso de baterías para el almacenamiento de energía eléctrica en aplicaciones \
    como arbitraje, reducción de congestiones de red, entre otros. Así, se contemplan\
    los costos teóricos asociados al tipo de batería, y es necesario introducir \
    por parámetro la información relacionada a los ingresos esperados, la capacidad\
    de las baterías, el tiempo de operación del proyecto, la política de capital \
    de trabajo, el costo de oportunidad, etc. Esto con el fin de obtener los \
    indicadores financieros VPN y TIR, los cuales permitirán analizar la viabilidad\
    del proyecto, así como el comportamiento del Flujo de Caja Libre.')
    st.markdown('## Parámetros seleccionados para la simulación')
    st.sidebar.markdown('### Ingrese los parámetros de simulación')

# Ingresar tiempo de simulación
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [Años]:', min_value=1, max_value=1000000)
    st.write('El horizonte de simulación es de: '+str(time_sim)+' año(s)')

# Selección de tecnología de SAE
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data2.index,key='2')
    if technology=='New':
        st.markdown('Ingrese los costos de CAPEX asociados al SAEB que quiere simular:')
        costP=st.text_input('Costo por potencia [USD/MW]: ',key='5')
        costE=st.text_input('Costo por energía (Baterías) [USD/MWh]: ',key='6')
        costAC=st.text_input('Costo por equipos AC [USD/MW]: ',key='7')
        costMano=st.text_input('Costo de construcción y puesta en servicio [USD/MW]: ',key='8')
        costPredio=st.text_input('Costo del predio [USD]: ',key='10')
        costLic=st.text_input('Costo de licenciamiento [USD]: ',key='11')
        costCon=st.text_input('Tarifa de conexión [USD]: ',key='12')
        st.markdown('Ingrese los costos de OPEX asociados al SAEB que quiere simular:')
        OM_fix=st.text_input('Costo O&M fijo [USD/kW/año]: ',key='13')
        OM_var=st.text_input('Costo O&M variable [cUSD/kWh]: ',key='14')

    else:
        st.markdown('El SAE seleccionado tiene los siguientes costos de CAPEX asociados:')
        st.dataframe(data2.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))
        st.markdown('El SAE seleccionado tiene los siguientes costos de OPEX asociados:')
        st.dataframe(data3.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))
        costP=data2.iloc[data2.index.get_loc(technology),1]
        costE=data2.iloc[data2.index.get_loc(technology),0]
        costAC=data2.iloc[data2.index.get_loc(technology),2]
        costMano=data2.iloc[data2.index.get_loc(technology),3]
        costPredio=data2.iloc[data2.index.get_loc(technology),4]
        costLic=data2.iloc[data2.index.get_loc(technology),5]
        costCon=data2.iloc[data2.index.get_loc(technology),6]

        OM_fix=data3.iloc[data3.index.get_loc(technology),0]
        OM_var=data3.iloc[data3.index.get_loc(technology),1]

#Ingrese el tamaño del SAE
    st.sidebar.markdown('Ingrese el tamaño del SAE a simular:')
    E_max = st.sidebar.number_input('Energía [MWh]')
    Pot_max = st.sidebar.number_input('Potencia [MW]')

## Politica de Working Capital
    politica=st.sidebar.selectbox('Seleccione la política de capital de trabajo',['Meses','Rotación en días'],key='2')
    if politica== 'Meses':
        meses=st.sidebar.number_input('Ingrese el número de meses:', min_value=1, max_value=1000000)
    if politica== 'Rotación en días':
        dias=st.sidebar.number_input('Ingrese el número de días:', min_value=1, max_value=365)

###Costo de oportunidad

    Coportunidad=st.sidebar.number_input('Ingrese el costo de oportunidad efectivo anual en COP (%):', min_value=1, max_value=10000000000)
    Coportunidad=Coportunidad/100
# Seleción de archivo con proyección de TRM y IPP
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_TRM = st.sidebar.file_uploader('Seleccione el archivo con proyecciones de TRM:', type=['csv','xlsx'])

# Seleción de archivo con Ingresos
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_Ingresos = st.sidebar.file_uploader('Seleccione el archivo con los ingresos esperados del proyecto:', type=['csv','xlsx'])

#https://totoro.banrep.gov.co/analytics/saw.dll?Download&Format=excel2007&Extension=.xls&BypassCache=true&lang=es&path=%2Fshared%2FSeries%20Estad%C3%ADsticas_T%2F1.%20IPC%20base%202008%2F1.2.%20Por%20a%C3%B1o%2F1.2.2.IPC_Total%20nacional%20-%20IQY


    def run_VF(value,n,x):
        # OM_fix*(1+x)
        y=[0]*(n+1)
        for i in range(0,n+1):
            y[i]=float(value)*np.power(x+1,i)
        return y

    if st.sidebar.button('Simular'):
        TRM_proy = pd.read_excel(file_TRM, sheet_name='TRM', header=0,usecols=['Proyección TRM'])
        ingresos =pd.read_excel(file_Ingresos, sheet_name='Ingresos', header=0,usecols=['Ingresos [USD]'])
        y_OM_fix=run_VF(OM_fix,time_sim,0.05)
        y_OM_var=run_VF(OM_var,time_sim,0.05)
        y_costE=run_VF(costE,time_sim,0.05)
        st.title('Resultados:')
        cambio=[0]*len(y_OM_fix)
        if len(y_OM_fix)>10:
            cambio[10]=1
        flujos=[0]*len(y_OM_fix)
        flujos[0]=(y_costE[0]*E_max+costP*Pot_max \
        +costAC*Pot_max+costMano*E_max+costPredio+costLic+costCon*Pot_max)*-1

        egresos=[0]*len(y_OM_fix)
        for i in range (1,len(y_OM_fix)):
            egresos[i]=-(y_OM_fix[i]*Pot_max+y_OM_var[i]*E_max+y_costE[i]*E_max*cambio[i])
            flujos[i]=ingresos.loc[i,'Ingresos [USD]']+egresos[i]

        flujos_COP=[0]*len(y_OM_fix)
        # TRM_periodo=TRM_proy.values.tolist()

        ingresos_COP=[0]*len(y_OM_fix)
        for i in range (0,len(y_OM_fix)):
            flujos_COP[i]=flujos[i]*TRM_proy.loc[i,'Proyección TRM']/1000000000
            ingresos_COP[i]=ingresos.loc[i,'Ingresos [USD]']*TRM_proy.loc[i,'Proyección TRM']/1000000000

        egresos_COP=[0]*len(y_OM_fix)
        for i in range (1,len(y_OM_fix)):
            egresos_COP[i]=egresos[i]*TRM_proy.loc[i,'Proyección TRM']/1000000000
        egresos_COP[0]=flujos_COP[0]

# Valor presente neto
        VPN=np.npv(Coportunidad, flujos_COP)
        st.write('El VPN es: ' + '%i'%VPN + ' MM COP')
# TIR or TVR

        def Negativos(Flujo):
            count=0
            for i in range(0,len(Flujo)):
                if Flujo[i]<0:
                    count=count+1
            return count

        if Negativos(flujos_COP)==1:
            tir=np.irr(flujos_COP)
            st.write('La TIR es: ' + '%f'%round(tir*100,2) + ' %')
        else:
            st.write('La TIR no puede ser calculada debido a que hay más de un cambio de signo en los flujos.')
            tvr=np.mirr(flujos_COP,Coportunidad,Coportunidad)
            st.write('La TVR es: ' +'%f'%round(tvr*100,2)+ ' %')

        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        ax.bar(list(range(0,len(y_OM_fix))),(ingresos_COP),label='ingresos', width = 0.5, color='g')
        ax.bar(list(range(0,len(y_OM_fix))),(egresos_COP),label='egresos', width = 0.5,color='r')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)
        plt.xlabel('Time [Años]')
        plt.ylabel('[MM COP]')
        plt.ylim([-100, 80])
        st.pyplot(fig)

####
        file_TRM_hist='1.1.1.TCM_Serie histórica IQY.xlsx'
        TRM_hist = pd.read_excel(file_TRM_hist, sheet_name='TRM')
        TRM_hist = TRM_hist.dropna()
        # st.write(TRM_hist.describe())
        # st.dataframe(TRM_hist.TRM)
        TRM_hist.Fecha = pd.to_datetime(TRM_hist.Fecha,format='%d-%m-%Y')
        # set the column as the index
        TRM_hist.set_index('Fecha', inplace=True)

        fig1=plt.figure(figsize=(15,9))
        plt.plot(TRM_hist)
        st.pyplot(fig1)
        # st.write(TRM_hist)

###### Movimiento Browniano#####

        # Calculo de retornos logaritmicos
        log_returns = np.log(TRM_hist / TRM_hist.shift(periods=1))
        log_returns=log_returns.dropna()
        # st.dataframe(log_returns.tail(10)*100)
        # st.write()
        # Parametros para el movimiento Browniano
        ## Media diaria y media anual
        time_mb=10 #  Ventana de tiempo en años
        media_d=log_returns.iloc[-time_mb*365:].mean()
        # st.write(media_d*100)
        media_m=(1+media_d.iloc[0])**30-1
        media_a=(1+media_d)**365-1
        # st.write(media_a*100)
        ### desviación
        std_d=log_returns.iloc[-time_mb*365:].std()
        # st.write(std_d*100)
        std_m=std_d.iloc[0]*(30)**(1/2)
        std_a=std_d*(365)**(1/2)
        # st.write(std_a*100)
        ###
        def gen_paths(S0, r, sigma, T, M, I):
            ''' Generates Monte Carlo paths for geometric Brownian Motion.
            Parameters
            ==========
            S0 : float
            iniial stock/index value
            r : float
            constant short rate
            sigma : float
            constant volatility
            T : float
            final time horizon
            M : int
            number of time steps/intervals
            I : int
            number of paths to be simulated
            Returns
            =======
            paths : ndarray, shape (M + 1, I)
            simulated paths given the parameters
            '''
            dt= float(T) /M
            paths = np.zeros((M + 1, I), np.float64)
            paths[0] = S0
            for t in range(1, M + 1):
                rand=np.random.standard_normal(I)
                rand=(rand-rand.mean()) / rand.std()
                paths[t]=paths[t-1]* np.exp((r- 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
            return paths
        aaa=gen_paths(TRM_hist.iloc[-1], media_m, std_m, 2, 24, 500)

        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        ax.plot(aaa)
        plt.xlabel('Time [Meses]')
        plt.ylabel('[USD/COP]')
        # plt.ylim([0, 6000])
        st.pyplot(fig)

        aaa=aaa.mean(axis = 1)
        # st.write(aaa)
        p_1=aaa[1:13].mean(axis = 0)
        p_2=aaa[13:].mean(axis = 0)
        # st.write(p_1)
        # st.write(p_2)
        prediction=[0]*(time_sim+1);

        for i in range(2,time_sim+1):
            prediction[i]=p_2

        prediction[0]=TRM_hist.iloc[-1]
        prediction[1]=p_1
        # st.write(prediction)

        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        ax.step(range(0,time_sim+1),prediction)
        plt.xlabel('Time [Años]')
        plt.ylabel('[USD/COP]')
        # plt.ylim([0, 6000])
        st.pyplot(fig)

        # regresión a la media

        ## AJuste por ARIMA

        # fig2=plt.plot(aaa[:, :15]) #Graficamos solo 15 caminos, todos los datos renglones o time steps, 15 columnas
        # plt.grid(True)
        # plt.xlabel('time steps')
        # plt.ylabel('index level')
        # st.pyplot(fig2)

        ######
