#Librería de intersaz
import streamlit as st
# Librerías para manipular datos
from os import path
from os import remove
# Librearias para manejo de datos
import pandas as pd
import numpy as np
import numpy_financial as npf
# Librarías para graficas
import matplotlib.pyplot as plt
import pydeck as pdk
# Libraries for reading data form internet
import requests
from bs4 import BeautifulSoup
import re
# Importing other functions
from funciones.proyeccion import mov_brow

###Función valor futuro 
def run_VF(value,n,x):
    # OM_fix*(1+x)
    # x debe ser inflación US
    y=[0]*(n+1)
    for i in range(0,n+1):
        y[i]=float(value)*np.power(x+1,i)
    return y

#### Inflación USD (año anterior - varios años (ej: si 2018 necesito 2019 hasta año de inversión))  
Inflacion_USA=0.021

def dashboard_Eco(data2,data3):
    st.markdown("<h1 style='text-align: center; color: black;'>Análisis financiero</h1>", unsafe_allow_html=True)
    # image = Image.open('arbitrage.jpeg')
    # st.image(image, caption='', use_column_width=True)
    st.markdown("En esta sección de la herramienta es posible realizar un análisis \
     financiero para evaluar la conveniencia económica de implementar proyectos con \
     uso de baterías para el almacenamiento de energía eléctrica en aplicaciones \
     como arbitraje, reducción de congestiones de red, entre otros, mediante la construcción del flujo de caja libre del proyecto. Así, se contemplan\
      los costos teóricos asociados al tipo de batería, y es necesario introducir \
      por parámetro la información relacionada a los ingresos esperados, la capacidad\
       de las baterías, el tiempo de operación del proyecto, la política de capital \
       de trabajo, el costo de oportunidad, etc. Esto con el fin de obtener los \
       indicadores financieros VPN y TIR, los cuales permitirán analizar la viabilidad\
        del proyecto, así como el comportamiento del Flujo de Caja Libre.")
    st.markdown("## Parámetros seleccionados para la simulación")
    st.sidebar.header("Ingrese los parámetros de simulación:")
    st.sidebar.subheader("Parámetros generales")
# Ingrese el año en donde desea efectuar la inversión
    year_ini=st.sidebar.number_input('Ingrese el año en donde efectuará la inversión:', min_value=2021, step=1, max_value=2200)
    st.write("El año en donde se efectuará la inversión (año 0) es: " + str(year_ini))

# Ingresar tiempo de simulación
    time_sim=st.sidebar.number_input('Ingrese el horizonte de simulación [Años]:', min_value=1, value=15, max_value=1000000)
    st.write("El horizonte de simulación es de: "+str(time_sim)+" año(s)")
# Ingrese el delta de tiempo (ej. meses, años) de la simulación
    entry_delta_t=st.sidebar.selectbox('Ingrese el paso/delta de tiempo (ej. meses, años) de la simulación:',['Años','Meses'],key='eco_0')
    if entry_delta_t=='Años':
        delta_t=1
        delta_t_inv=12
    if entry_delta_t=='Meses':
        delta_t=12
        delta_t_inv=1

    time_sim=time_sim*delta_t
# Ingrese la moneda con la que desea realizar el análisis financiero
    entry_moneda=st.sidebar.selectbox('¿Con cuál moneda desea realizar el análisis financiero?',['COP','USD'],key='eco_1')
    if entry_moneda== 'COP':
        unidades='COP'
    # Definir TRM
        tipo_TRM=st.sidebar.selectbox('Como quiere ingresar la TRM al modelo',['Fija','Proyección propia','Proyección simulada'],key='eco_2')
        if tipo_TRM== 'Fija':
                # Ingresar TRM
            TRM_select=st.sidebar.selectbox('Seleccione la TRM para la simulación',['Hoy','Otra'],key='eco_3')
            if TRM_select=="Hoy":
                URL = 'https://www.dolar-colombia.com/'
                page = requests.get(URL)
                soup = BeautifulSoup(page.content, 'html.parser')
                rate = soup.find_all(class_="exchange-rate")
                TRM=str(rate[0])
                TRM= re.findall('\d+', TRM )
                TRM_final=TRM[0]+TRM[1]+"."+TRM[2]
                st.write("La TRM seleccionada para la simulación es de: "+TRM_final + " COP")
                TRM_proy=[float(TRM_final)]*(time_sim+1)
            else:
                TRM_final=st.sidebar.number_input("Ingrese la TRM para la simulación: ", min_value=0)
                TRM_proy=[float(TRM_final)]*(time_sim+1)
                st.write("La TRM seleccionada para la simulación es de: "+str(TRM_final) + " COP")
        elif tipo_TRM== 'Proyección propia':
            st.set_option('deprecation.showfileUploaderEncoding', False)
            file_TRM = st.sidebar.file_uploader("Seleccione el archivo con proyecciones de TRM:", type=["csv","xlsx"])
            if file_TRM != None:
                TRM_df = pd.read_excel(file_TRM, sheet_name='TRM', header=0,usecols=["Proyección TRM"])
                TRM_proy=[0]*(time_sim+1)
                # st.write(TRM_df)
                for i in range(0,time_sim+1):
                    TRM_proy[i]=TRM_df.iloc[i,0]
                # st.write(TRM_proy)
        elif tipo_TRM== 'Proyección simulada':
    # Seleción de archivo con proyección de TRM
            TRM_data=st.sidebar.selectbox('Seleccione el método para realizar proyección de parámetros (TRM)',['Movimiento Browniano Geométrico'],key='eco_4')
            if TRM_data== 'Movimiento Browniano Geométrico':
                st.set_option('deprecation.showfileUploaderEncoding', False)
                file_TRM_his = st.sidebar.file_uploader("Seleccione el archivo con los históricos de TRM:", type=["csv","xlsx"])
###FUNCION MOVIMIENTO Browniano
            TRM_proy = mov_brow(file_TRM_his,time_sim)
    else:
        unidades='USD'
        TRM_proy=[1]*(time_sim+1)

###Costo de oportunidad
    Coportunidad=st.sidebar.number_input('Ingrese el costo de oportunidad efectivo anual (%) en ' +str(entry_moneda) + ':', min_value=1, value=10, max_value=100)
    Coportunidad=Coportunidad/100
    if entry_delta_t=='Meses':
        Coportunidad=(1+Coportunidad)**(1/12)-1

### Información del SAE
    st.sidebar.subheader("Información del SAE a simular")

# Selección de tecnología de SAE
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnología de SAE',data2.index,key='eco_5')
    if technology=="Nuevo":
        st.markdown("Ingrese los costos de CAPEX asociados al SAEB que quiere simular (en USD " +str(year_ini)+ "):")
        costP=st.number_input("Costo por potencia [USD/MW]: ",value=209000, min_value=0, step=1)
        costE=st.number_input("Costo por energía (Baterías) [USD/MWh]: ",value=70000, min_value=0, step=1)
        costAC=st.number_input("Costo por equipos AC [USD/MW]: ",value=100000, min_value=0, step=1)
        costMano=st.number_input("Costo de construcción y puesta en servicio [USD/MW]: ",value=101000, min_value=0, step=1)
        costPredio=st.number_input("Costo del predio [USD]: ",value=250000, min_value=0, step=1)
        costLic=st.number_input("Costo de licenciamiento [USD]: ",value=50000, min_value=0, step=1)
        costCon=st.number_input("Tarifa de conexión [USD]: ",value=30000, min_value=0, step=1)
        st.markdown("Ingrese los costos de OPEX asociados al SAEB que quiere simular (en USD " +str(year_ini)+ "):")
        OM_fix=st.number_input("Costo O&M fijo [USD/kW/año]: ",value=10000, min_value=0, step=1)
        OM_var=st.number_input("Costo O&M variable [cUSD/kWh]: ",value=0.3, min_value=0.0, step=0.1)
    else:
        year_tabla=2018
        st.markdown("El SAE seleccionado tiene los siguientes costos de CAPEX asociados (valores USD 2018):")
        # st.dataframe(data2.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))
        st.dataframe(data2.loc[[technology]])
        st.markdown("El SAE seleccionado tiene los siguientes costos de OPEX asociados (valores USD 2018):")
        st.dataframe(data3.loc[[technology]].style.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ]))
        # st.dataframe(data3.loc[[technology]].T.style.format({'Degradación [pu]':'{:.3%}','Autodescarga [pu]':'{:.3%}'}))
        costP=float(data2.iloc[data2.index.get_loc(technology),1])
        costE=float(data2.iloc[data2.index.get_loc(technology),0])
        costAC=float(data2.iloc[data2.index.get_loc(technology),2])
        costMano=float(data2.iloc[data2.index.get_loc(technology),3])
        costPredio=float(data2.iloc[data2.index.get_loc(technology),4])
        costLic=float(data2.iloc[data2.index.get_loc(technology),5])
        costCon=float(data2.iloc[data2.index.get_loc(technology),6])
        OM_fix=float(data3.iloc[data3.index.get_loc(technology),0])
        OM_var=float(data3.iloc[data3.index.get_loc(technology),1])
        ### Valores  a año inicial 
        costP=costP*(1+Inflacion_USA)**(year_ini-year_tabla)
        costE= costE*(1+Inflacion_USA)**(year_ini-year_tabla)
        costAC=costAC*(1+Inflacion_USA)**(year_ini-year_tabla)
        costMano=costMano*(1+Inflacion_USA)**(year_ini-year_tabla)
        costPredio=costPredio*(1+Inflacion_USA)**(year_ini-year_tabla)
        costLic=costLic*(1+Inflacion_USA)**(year_ini-year_tabla)
        costCon=costCon*(1+Inflacion_USA)**(year_ini-year_tabla)
        OM_fix= OM_fix*(1+Inflacion_USA)**(year_ini-year_tabla)
        OM_var= OM_var*(1+Inflacion_USA)**(year_ini-year_tabla)
#Ingrese el tamaño del SAE

    E_max = st.sidebar.number_input('Energía [MWh]', min_value=0,value=50)
    Pot_max = st.sidebar.number_input('Potencia [MW]', min_value=0, value=50)

    st.sidebar.subheader("Información relacionada con los ingresos")
## Forma de ingresar los beneficios/ingresos
    entry_ingresos=st.sidebar.selectbox('¿Cómo desea introducir los ingresos?',['Archivo con ingresos','Simular en la herramieta'],key='eco_6')
    if entry_ingresos== 'Archivo con ingresos':
# Seleción de archivo con Ingresos
       st.set_option('deprecation.showfileUploaderEncoding', False)
       file_Ingresos = st.sidebar.file_uploader("Seleccione el archivo con los ingresos esperados del proyecto en " +str(entry_moneda)+":", type=["csv","xlsx"])
       vida_util_bat = st.sidebar.number_input("Ingrese la vida útil de la batería en años:",value=10,min_value=0,max_value=100)
       ener_disp_dia=1
    else: 
## Definir ingresos
        ingresos_opc=st.sidebar.selectbox('Seleccione los ingresos a considerar',['Arbitraje','AGC', 'Arbitraje+AGC'],key='eco_7')
        if ingresos_opc== 'Arbitraje':
            vida_util_bat=10
            ener_disp_dia=1
            


# valor de salvamento --  vender al final de la vida útil  - diferente para BESS, AC y CPS
    st.sidebar.subheader("Datos de depreciación de los activos")
    BESS_sal = st.sidebar.number_input('Valor de salvamento de las baterías ['+str(entry_moneda)+"]")
    CPS_sal = st.sidebar.number_input('Valor de salvamento de la electrónica de potencia ['+str(entry_moneda)+"]")
    AC_sal=  st.sidebar.number_input('Valor de salvamento de los equipos AC ['+str(entry_moneda)+"]")



## Politica de Working Capital
    st.sidebar.subheader("Datos de capital de trabajo")
    if entry_delta_t=='Años':
        # politica=st.sidebar.selectbox('Seleccione la política de capital de trabajo',['Rotación en días'],key='eco_81')
        dias=st.sidebar.number_input('Ingrese el número de días de rotación de cuentas por cobrar:', min_value=1, step=1, max_value=365)
    if entry_delta_t=='Meses':
        # politica=st.sidebar.selectbox('Seleccione la política de capital de trabajo',['Meses'],key='eco_82')
        meses=st.sidebar.number_input('Ingrese el número de meses de la política:', min_value=1, step=1, max_value=12)
 

#https://totoro.banrep.gov.co/analytics/saw.dll?Download&Format=excel2007&Extension=.xls&BypassCache=true&lang=es&path=%2Fshared%2FSeries%20Estad%C3%ADsticas_T%2F1.%20IPC%20base%202008%2F1.2.%20Por%20a%C3%B1o%2F1.2.2.IPC_Total%20nacional%20-%20IQY


    if st.sidebar.button('Simular'):

        #Lectura de Ingresos
        if entry_ingresos== 'Archivo con ingresos':
            ingresos =pd.read_excel(file_Ingresos, sheet_name='Ingresos', header=0,usecols=["Ingresos [USD]"])
        else:
            ingresos=[0]*(time_sim+1)

        ####Arreglos en horizonte simulación   (acá debería ser inflación USA proyectada)
        y_OM_fix=run_VF(OM_fix,time_sim,Inflacion_USA)
        y_OM_var=run_VF(OM_var,time_sim,Inflacion_USA)
        y_costE=run_VF(costE,time_sim,Inflacion_USA)

        # st.write(y_OM_fix)

        ###Mostrar resultados
        st.title("Resultados:")

        ### capital de trabajo 
        if entry_delta_t=='Años':
            capital_trabajo=[0]*(time_sim+1)
            capital_trabajo[time_sim]=0
            for i in range(1,time_sim):
                capital_trabajo[i]=(float(dias)*ingresos.loc[i,"Ingresos [USD]"])/365
            # st.write(capital_trabajo)

        else:
            capital_trabajo=[0]*(time_sim+1)
            capital_trabajo[time_sim]=0
            for i in range(1,time_sim):
                if i <= float(meses):
                    capital_trabajo[i]=float(ingresos.loc[i,"Ingresos [USD]"]+capital_trabajo[i-1])
                else:
                    capital_trabajo[i]=float(ingresos.loc[i,"Ingresos [USD]"]+capital_trabajo[i-1]-ingresos.loc[(i-float(meses)),"Ingresos [USD]"])
            # st.write(capital_trabajo)
        
        #Delta capital de trabajo
        delta_CT=[0]*(time_sim+1)
        for i in range(1,time_sim+1):
            delta_CT[i]=capital_trabajo[i]-capital_trabajo[i-1]
        # st.write(delta_CT)

        # Descarga de enrgía promedio por día -- OPEX y_OM_fix tiene valor anual
        OPEX=[0]*(time_sim+1)
        for i in range(1,time_sim+1):
            OPEX[i]=((y_OM_fix[i]*Pot_max)/delta_t+y_OM_var[i]*E_max*ener_disp_dia*30*delta_t_inv)*TRM_proy[i] 

        #CAPEX
        #cambio de baterias
        costo=[0]*(time_sim+1)
        costo[0]=y_costE[0]*TRM_proy[0]
        cambio=[0]*(time_sim+1)  # cambio de baterías
        vida_util_bat=vida_util_bat*delta_t
        vida=1
        while int(vida*vida_util_bat) < time_sim:
            cambio[int(vida*vida_util_bat)]=1
            vida=vida+1
        # st.write(cambio)

        for i in range(1,time_sim+1):
            if cambio[i-1]==0:
                costo[i]=costo[i-1]
            elif cambio[i-1]==1:
                costo[i]=y_costE[i-1]*TRM_proy[i-1]
        # st.write(costo)
                
        #arreglo CAPEX 
        CAPEX=[0]*(time_sim+1)
        CAPEX[0]=(y_costE[0]*E_max+costP*Pot_max+costAC*Pot_max+costMano*E_max+costPredio+costLic+costCon*Pot_max)*TRM_proy[0] 
        for i in range (1,(time_sim+1)):
            CAPEX[i]=(y_costE[i]*E_max*cambio[i])*TRM_proy[i] 
        # st.write(CAPEX)
        # st.write(OPEX)

###########IMPUESTOS##########################
### Depreciación
        # depreciacion con linea recta

        depreciacion=[0]*(time_sim+1)
        for i in range(1,time_sim+1):
            depreciacion[i]=(costo[i]*E_max-BESS_sal)/vida_util_bat + (costP*Pot_max*TRM_proy[0] -CPS_sal)/time_sim +(costAC*Pot_max*TRM_proy[0]-AC_sal)/time_sim
        
        # st.write(depreciacion)
        # impuestos operacionales
        imp_op=[0]*(time_sim+1)
        tasa_renta=0.31
        for i in range(1,time_sim+1):
            imp_op[i]=max(0,(ingresos.loc[i,"Ingresos [USD]"]-OPEX[i]-depreciacion[i])*tasa_renta)
        # st.write(imp_op)

        # FLUJOS  
        flujos=[0]*(time_sim+1)
        for i in range (0,time_sim+1):
            flujos[i]=ingresos.loc[i,"Ingresos [USD]"]-CAPEX[i]-OPEX[i]-delta_CT[i]-imp_op[i]


        # st.write(ingresos)
        ##### FLUJOS PARA GRAFICA
        OPEX_g=[0]*(time_sim+1)
        CAPEX_g=[0]*(time_sim+1)  
        ingresos_g =[0]*(time_sim+1)   
        delta_CT_g=[0]*(time_sim+1)  
        imp_op_g=[0]*(time_sim+1)   
        for i in range (0,time_sim+1):
            OPEX_g[i]=-OPEX[i]
            CAPEX_g[i]=-CAPEX[i]
            ingresos_g[i]=ingresos.loc[i,"Ingresos [USD]"]
            delta_CT_g[i]=-delta_CT[i]
            imp_op_g[i]=-imp_op[i]

# Valor presente neto
        VPN=npf.npv(Coportunidad, flujos)      
        st.write("El VPN es: " + "{:.3e}".format(VPN)+ " "+unidades)
        # st.write("El VPN es: " + "%i "%VPN + unidades)
# TIR or TVR
        def Negativos(Flujo):
            count=0
            for i in range(0,len(Flujo)):
                if Flujo[i]<0:
                    count=count+1
            return count

        if Negativos(flujos)==1:
            tir=npf.irr(flujos)
            st.write("La TIR es: " + "%f"%round(tir*100,2) + " %")
        else:
            st.write("La TIR no puede ser calculada debido a que hay más de un cambio de signo en los flujos.")
            tvr=npf.mirr(flujos,Coportunidad,Coportunidad)
            st.write("La TVR es: " +"%f"%round(tvr*100,2)+ " %")



##### Grafica de flujos discriminada

        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        
        ax.bar(list(range(0,len(y_OM_fix))),(ingresos_g),label="Ingresos", width = 0.5, color='g')
        ax.bar(list(range(0,len(y_OM_fix))),(CAPEX_g),label="CAPEX", width = 0.5, color='y')
        ax.bar(list(range(0,len(y_OM_fix))),(OPEX_g),label="OPEX", width = 0.5,color='r')
        ax.bar(list(range(0,len(y_OM_fix))),(delta_CT_g),label="delta_CT", width = 0.5,color='b')
        ax.bar(list(range(0,len(y_OM_fix))),(imp_op_g),label="imp_op", width = 0.5,color='m')
        ax.plot(flujos,color='k',label="Flujos Netos")
        
        # ax.bar(list(range(0,len(y_OM_fix))),(delta_CT_COP),label="Variación Capital de trabajo", width = 0.5,color='y',bottom=np.array(CT_COP_button))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)
        plt.xlabel('Time [Años]')
        plt.ylabel('[COP]')
        st.pyplot(fig)




