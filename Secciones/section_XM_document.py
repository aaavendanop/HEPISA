# coding: utf-8
## Librería de inferfaz
import streamlit as st
## Librerías para manipular datos
from os import path, remove
## Librerías para manejo de datos
import pandas as pd
import numpy as np
## Librerías for reading data form internet
from datetime import date, timedelta, datetime
## Importing optimization functions
from modelos.Ope_SAE.XM.despacho import *
from funciones.nuevo_bess import bat_param
## Other libraries
from math import floor

def text_formul_math_v1():
    return r"""

    ## **Función Objetivo**
    $$ \begin{aligned}
        \min{} \sum_{t} \sum_{R} Pofe_{[R]}\cdot V\_GenRec_{[R][t]} + \sum_t \sum_{RT} PAP_{[RT]}\cdot B\_Arr_{[RT][t]} \\
        + \sum_t CROest\cdot V\_Rac{[t]} + \sum_t \sum_s PC_{[s][t]}\cdot V\_PC_{[s][t]} \cdot \Delta t \\
        + \sum_{tpd} \sum_{s} CROest\cdot V\_DoC_{[s][tpd]}
    \end{aligned} $$

    donde $Pofe_{[R]}$ es un vector que representa el precio ofertado por cada unidad/planta de generación \$/MW; $V\_GenRec_{[R][t]}$ es un
    vector que representa la generación de cada unidad/planta de generación en MW; $PAP_{[RT]}$ es el precio de arranque/parada de cada
    unidad/planta de generación en \$; $B\_Arr_{[RT][t]}$ es una variable binaria que representa el estado de encedido o apagado de la
    unidad/planta de generación; $CROest$ es el Costo Incremental Operativo de Racionamiento de Energía en \$/MWh; $V\_Rac{[t]}$ es la energía
    racionada en el sistema en MWh; $PC_{[s][t]}$ es la valoración del costo de carga de cada SAE $s$ en \$/MWh; $V\_PC_{[s][t]}$ es la
    potencia de carga del SAE $s$ en el tiempo $t$ en MW; $\Delta t$ valor constante con valor de 1h; $V\_DoC_{[s][tpd]}$ es el estado de
    carga del SAE $s$ en MWh.

    ## **Restricciones**

    **Balance de generación demanda considerando almacenamiento**

    $$ \sum_R V\_GenRec_{[R][t]} + V\_Rac_{[t]} + \sum_s V\_PD_{[s][t]} = Dem_{[t]} + \sum_s V\_PC_{[s][t]} \hspace{2mm}
    \forall t $$

    donde $V\_PD_{[s][t]}$ es la potencia de descarga del SAE $s$ en el tiempo $t$ en MW; $Dem_{[t]}$ es el pronostico de demanda en cada
    instante de tiempo $t$.

    **Balance del almacenamiento**

    $$ V\_SoC_{[s][t]} = V\_SoC_{[s][t-1]}\cdot \left(1 - \eta_{SoC[s]} \right) + \eta_{c[s]}\cdot V\_PC_{[s][t]}
    - \dfrac{V\_PD_{[s][t]}}{\eta_{d[s]}} \hspace{2mm} \forall s, t $$

    donde $V\_SoC_{[s][t]}$ es el estado de carga del SAE $s$ en el tiempo $t$ en MWh; $\eta_{SoC[s]}$ es la autodescarga del SAE;
    $\eta_{c[s]}$ y $\eta_{d[s]}$ son las eficiencias de carga/descarga del SAE $s$ en pu.

    **Balance de Estado de Carga**

    $$ V\_DoC_{[s][t]} = SoCmax_{[s]}\cdot Cap_{[s]} - V\_SoC_{[s][t]} \hspace{2mm}
    \forall s, t $$

    donde $SoCmax_{[s]}$ es el estado máximo de carga del SAE $s$ en pu; $Cap_{[s]}$ es la capacidad máxima de energía del SAE $s$ en MWh.

    **Capacidad máxima de almacenamiento**

    $$ SoCmin_{[s]}\cdot Cap_{[s]} \leq V\_SoC_{[s][t]} \leq SoCmax_{[s]}\cdot Cap_{[s]} \hspace{2mm}
    \forall s, t $$

    donde $SoCmin_{[s]}$ es el estado mínimo de carga del SAE $s$ en pu.

    **Causalidad de la carga/descarga**

    $$ B\_PC_{[s][t]} + B\_PD_{[s][t]} \leq 1 \hspace{2mm}
    \forall s, t $$

    $$ 0 \leq V\_PC_{[s][t]} \leq PCmax_{[s][t]} \cdot B\_PC_{[s][t]} \hspace{2mm}
    \forall s, t $$

    $$ 0 \leq V\_PD_{[s][t]} \leq PDmax_{[s][t]} \cdot B\_PD_{[s][t]} \hspace{2mm}
    \forall s, t $$

    donde $B\_PC_{[s][t]}$ y $B\_PD_{[s][t]}$ son las variables binarias que representan cuando se está cargando/descargando el SAE $s$ en el
    tiempo $t$; $PCmax_{[s][t]}$ y $PDmax_{[s][t]}$ son los límites máximos de carga/descarga del SAE $s$ en el tiempo $t$ en MW.

    **Carga y descarga requerida**

    $$ V\_PD_{[s][td]} = PDdes_{[s][td]} \hspace{2mm}
    \forall s, td $$

    $$ V\_PC_{[s][td]} \geq PCdes_{[s][td]} \hspace{2mm}
    \forall s, td $$

    donde $PDdes_{[s][td]}$ y $PCdes_{[s][td]}$ son los valores requeridos de carga/descarga en el periodo $td$ (conjunto de periodos en donde
    aplique un nivel de carga/descarga requerido) del SAE $s$ en MW.

    """

def text_formul_math_v2():
    return r"""
    ## **Función Objetivo**
    $$ \begin{aligned}
        \min{} \sum_{t} \sum_{r} Pofe_{[R]}\cdot V\_GenRec_{[R][t]} +
        \sum_t \sum_{rt} PAP_{[rt]}\cdot B\_Arr_{[rt][t]} + \\
        \sum_t CROest\cdot V\_Rac{[t]} + \sum_t \sum_s PC_{[s][t]}\cdot V\_PC_{[s][t]}\cdot \Delta t + \\
        \sum_{tpd} \sum_{s} CROest\cdot Cap_{[s]} \cdot V\_SoD{[s][tpd]}
    \end{aligned} $$

    donde $Pofe_{[R]}$ es un vector que representa el precio ofertado por cada unidad/planta de generación \$/MW; $V\_GenRec_{[R][t]}$ es un
    vector que representa la generación de cada unidad/planta de generación en MW; $PAP_{[rt]}$ es el precio de arranque/parada de cada
    unidad/planta de generación en \$; $B\_Arr_{[rt][t]}$ es una variable binaria que representa el estado de encedido o apagado de la
    unidad/planta de generación; $CROest$ es el Costo Incremental Operativo de Racionamiento de Energía en \$/MWh; $V\_Rac{[t]}$ es la energía
    racionada en el sistema en MWh; $PC_{[s][t]}$ es la valoración del costo de carga de cada SAE $s$ en \$/MWh; $V\_PC_{[s][t]}$ es la
    potencia de carga del SAE $s$ en el tiempo $t$ en MW; $\Delta t$ valor constante con valor de 1h; $V\_SoD{[s][tpd]}$ es el estado de
    descarga del SAE $s$ en el periodo previo de descarga $tpd$ en pu; $Cap_{[s]}$ es la capacidad máxima de energía del SAE $s$ en MWh.

    ## **Restricciones**

    ### Restricciones del sistema

    **Balance de generación demanda considerando almacenamiento**

    $$ \sum_r V\_GenRec_{[R][t]} + V\_Rac{[t]} + \sum_s V\_PD{[s][t]} \cdot ECS_{[s][t]} =
    Dem_{[t]} \\ + \sum_s V\_PC_{[s][t]} \cdot ECS_{[s][t]} \hspace{2mm} \forall t $$

    donde $V\_PD_{[s][t]}$ es la potencia de descarga del SAE $s$ en el tiempo $t$ en MW; $Dem_{[t]}$ es la demanda del SIN en cada
    instante de tiempo $t$ en MW; $ECS_{[s][t]}$ es el estado de conexión del SAEB $n$ en cada instante de tiempo $t$ al SIN.

    ### Límites en Generación

    **Balance del almacenamiento**

    $$ V\_SoC_{[s][t]} = V\_SoC\_E{[s][t-1]} + \\ ECS_{[s][t]} \cdot \left(
    \eta_{c[s]}\cdot \dfrac{V\_PC_{[s][t]}}{Cap_{[s]}} - \dfrac{1}{\eta_{d[s]}} \cdot
    \dfrac{V\_PD{[s][t]}}{Cap_{[s]}} \right) \hspace{2mm} \forall s, t $$

    donde $V\_SoC_{[s][t]}$ es el estado de carga del SAE $s$ en el tiempo $t$ en MWh; $V\_SoC\_E{[s][t-1]}$ es el estado de carga del SAE $n$
    en la hora $t$, afectado por la eficiencia de almacenamiento en pu; $\eta_{c[s]}$ y $\eta_{d[s]}$ son las eficiencias de carga/descarga
    del SAE $s$ en pu.

    **Afectación del estado de carga por eficiencia de almacenamiento**

    $$ -(B\_PC_{[s][t]} + B\_PD_{[s][t]}) + V\_SoC_{[s][t]}\cdot (1 - \eta_{SoC[s]})
                    \leq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall s, t $$

    $$ V\_SoC_{[s][t]} \cdot (1 - \eta_{SoC[s]}) + (B\_PC_{[s][t]} + B\_PD_{[s][t]})
                    \geq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall s, t $$

    $$ -(1 - B\_PC_{[s][t]}) + V\_SoC_{[s][t]} \leq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall
                    s, t $$

    $$ V\_SoC_{[s][t]} + (1 - B\_PC_{[s][t]}) \geq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall
                    s, t $$

    $$ -(1 - B\_PD_{[s][t]}) + V_{Soc[s][t]} \leq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall
                    s, t $$

    $$ V\_SoC_{[s][t]} + (1 - B\_PD_{[s][t]}) \geq V\_SoC\_E{[s][t-1]} \hspace{2mm} \forall
                    s, t $$

    donde $B\_PC_{[s][t]}$ y $B\_PD_{[s][t]}$ son las variables binarias que representan cuando se está cargando/descargando el SAE $s$ en el
    tiempo $t$; $\eta_{SoC[s]}$ es la eficiencia de almacenamiento del SAE $s$.

    **Balance de Estado de Carga**

    $$ V\_SoD{[s][t]} = 1 - V\_SoC_{[s][t]} \hspace{2mm} \forall
                    s, t $$

    donde $V\_SoD{[s][t]}$ es el estado de descarga del SAEB $s$ en el tiempo $t$ en pu.

    **Mínimo y máximo Estado de Carga del almacenamiento**

    $$ SoC_{min[s]} \leq V\_SoC_{[s][t]} \leq SoC_{max[s]} \hspace{2mm} \forall
                    s, t $$

    donde $SoCmin_{[s]}$ y $SoCmax_{[s]}$ son los estados mínimos y máximos de carga operativos del SAE $s$ en pu.

    **Mínimo técnico del sistema de almacenamiento**

    $$ V_{SoC[s][t]} \geq SoC\_MT{[s]} \hspace{2mm} \forall
                    s, t $$

    donde $SoC\_MT{[s]}$ es el estado mínimo técnico del SAE $s$ en pu.

    **Causalidad de la carga/descarga**

    $$ B\_PC_{[s][t]} + B\_PD_{[s][t]} \leq 1 \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$

    $$ 0 \leq V\_PC_{[s][t]} \leq PCmax_{[s][t]} \cdot B\_PC_{[s][t]} \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$

    $$ 0 \leq V\_PD{[s][t]} \leq PDmax_{[s][t]} \cdot B\_PD_{[s][t]} \hspace{2mm} \forall
                    s, t, ECS_{[s][t]} = 1 $$

    donde $PCmax_{[s][t]}$ y $PDmax_{[s][t]}$ son los límites máximos de carga/descarga del SAE $s$ en el tiempo $t$ en MW.

    **Carga y descarga requerida**

    $$V\_PD{[s][tr]} = \begin{cases}
                    PDreq_{[s][tr]} & \forall s, tr, ECS_{[s][t]} = 1 \\
                    0 & \forall s, t \neq tr, ECS_{[s][t]} = 1
                    \end{cases}$$

    $$ V\_PC_{[s][tr]} \geq PCreq_{[s][tr]} \hspace{2mm} \forall
                    s, td, ECS_{[s][t]} = 1 $$

    donde $PDreq_{[s][tr]}$ y $PCreq_{[s][tr]}$ son los valores requeridos de carga/descarga en el periodo $tr$ (conjunto de periodos en donde
    aplique un nivel de carga/descarga requerido) del SAE $s$ en MW.

    """

def NOTA():
    return r"""
        Para que el modelo tenga sentido, en términos de formulación matemática, alguno de los siguientes parámetros debe ser 0:

        * Valor de carga requerido en la hora $tp$ ($tr$) del SAE $n$
        * Valor de descarga requerido en la hora $tp$ ($tr$) del SAE $n$

        Por otro lado, **y más importante que la consideración anterior**, para que el modelo de optimización del despacho con SAE tenga
        sentido, se sugiere que el valor del parámetro *Valor de carga requerido en la hora $tp$ ($tr$) del SAE $n$* sea 0.

    """

def dashboard_XMDoc(data1, s_study):

    if s_study == 'Versión 1':

        ## Formulación metemática

        formulacion = st.expander(label='Formulación Matemática', expanded=False)
        with formulacion:
            st.write(text_formul_math_v1())
            st.write("")

        nota_imp = st.expander(label='NOTA IMPORTANTE', expanded=False)
        with nota_imp:
            st.write(NOTA())
            st.write('')

        st.markdown("## Parámetros seleccionados para la simulación")
        st.sidebar.markdown("### Ingrese los parámetros de simulación")


    elif s_study == 'Versión 2':

        ## Formulación metemática

        formulacion = st.expander(label='Formulación Matemática', expanded=False)
        with formulacion:
            st.write(text_formul_math_v2())
            st.write("")

        nota_imp = st.expander(label='NOTA IMPORTANTE', expanded=False)
        with nota_imp:
            st.write(NOTA())
            st.write('')

        st.markdown("## Parámetros seleccionados para la simulación")
        st.sidebar.markdown("### Ingrese los parámetros de simulación")

    # tiempos de descarga previo y tiempos de descarga
    st.sidebar.write('### Tiempos previos de carga/descarga y tiempos de descarga')
    tdp_i = st.sidebar.number_input('Periodo inicial previo a un bloque de descarga', min_value=0, max_value=23)
    tdp_f = st.sidebar.number_input('Periodo final previo a un bloque de descarga', min_value=0, max_value=23)
    td_i = st.sidebar.number_input('Periodo inicial de un bloque de carga/descarga', min_value=0, max_value=23)
    td_f = st.sidebar.number_input('Periodo final de un bloque de carga/descarga', min_value=0, max_value=23)
    st.write('Conjunto de periodos previos a un bloque de descarga: ', '[', str(tdp_i), ',', str(tdp_f), ']')
    st.write('Conjunto de periodos de carga/descarga requerida: ', '[', str(td_i), ',', str(td_f), ']')

    nPeriodos = abs(td_f - td_i) + 1

    #Ingrese los parámetros del SAE
    st.sidebar.write('### Parámetros del(de los) SAE')

    SAE_num = st.sidebar.number_input('Número de SAE a considerar', min_value=1)

    SAE_num_list = ['SAE {}'.format(x+1) for x in range(SAE_num)]

    if s_study == 'Versión 1':

        df_SAEInfo = pd.DataFrame(index=['Potencia máxima de carga [MW]','Potencia máxima de descarga [MW]',
                                    'Valor de carga requerido [MW]','Valor de descarga requerido [MW]','Energía [MWh]',
                                    'Eficiencia (round-trip) [pu]','Autodescarga [%/h]','Estado de carga mínimo [pu]',
                                    'Estado de carga máximo [pu]'],
                                    columns=SAE_num_list).fillna(0)

    else:

        df_SAEInfo = pd.DataFrame(index=['Potencia máxima de carga [MW]','Potencia máxima de descarga [MW]',
                                    'Valor de carga requerido [MW]','Valor de descarga requerido [MW]','Energía [MWh]',
                                    'Eficiencia (round-trip) [pu]','Autodescarga [%/h]','Estado de carga mínimo [pu]',
                                    'Estado de carga máximo [pu]','Estado de carga mínimo técnico [pu]'],
                                    columns=SAE_num_list).fillna(0)

    for i in range(SAE_num):
        PC_max = st.sidebar.number_input('Potencia máxima de carga del SAE {}'.format(i+1), min_value=1, help='En MW', key='A{}'.format(i))
        df_SAEInfo.loc['Potencia máxima de carga [MW]','SAE {}'.format(i+1)] = PC_max

    for i in range(SAE_num):
        PD_max = st.sidebar.number_input('Potencia máxima de descarga del SAE {}'.format(i+1), min_value=1, help='En MW', key='B{}'.format(i))
        df_SAEInfo.loc['Potencia máxima de descarga [MW]','SAE {}'.format(i+1)] = PD_max

    for i in range(SAE_num):
        E_max = st.sidebar.number_input('Energía del SAE {}'.format(i+1), min_value=1, help='En MWh', key='C{}'.format(i))
        df_SAEInfo.loc['Energía [MWh]','SAE {}'.format(i+1)] = E_max

    ## Selección de tecnología de SAE

    techAutoD = {'Li-Ion':0.0000625, 'PbA':0.0042, 'Sodio-sulfuro':0.0042, 'VRFB':0.0021, 'Celda de Combustible':0.042,
                'Central de Bombeo':0.042, 'Volante':2.1, 'CAES':0.042, 'SMES':0.625, 'SCES':0.42, 'Thermal':0.042, 'Nuevo':0}

    for i in range(SAE_num):
        autoD_key = st.sidebar.selectbox('Seleccione el tipo de tecnologí­a del SAE {}'.format(i+1), (techAutoD.keys()))

        if autoD_key == 'Nuevo':
            autoD_val = st.sidebar.number_input('Ingrese el valor de autodescarga del SAE {}'.format(i+1), min_value=0., value=0.000063, key='D{}'.format(i))
        else:
            autoD_val = techAutoD[autoD_key]

        df_SAEInfo.loc['Autodescarga [%/h]','SAE {}'.format(i+1)] = autoD_val

    for i in range(SAE_num):
        effR = st.sidebar.number_input('Eficiencia global del SAE {}'.format(i+1), min_value=0., max_value=1., value=0.9, key='E{}'.format(i))
        df_SAEInfo.loc['Eficiencia (round-trip) [pu]','SAE {}'.format(i+1)] = effR

    for i in range(SAE_num):
        socMIN = st.sidebar.number_input('Estado de carga mínimo del SAE {}'.format(i+1), min_value=0.1, step=0.1, max_value=1., key='F{}'.format(i))
        df_SAEInfo.loc['Estado de carga mínimo [pu]','SAE {}'.format(i+1)] = socMIN

    for i in range(SAE_num):
        socMAX = st.sidebar.number_input('Estado de carga máximo del SAE {}'.format(i+1), min_value=0.1, step=0.1, max_value=1., key='G{}'.format(i))
        df_SAEInfo.loc['Estado de carga máximo [pu]','SAE {}'.format(i+1)] = socMAX

    if s_study == 'Versión 2':
        for i in range(SAE_num):
            socMT = st.sidebar.number_input('Estado de carga mínimo técnico del SAE {}'.format(i+1), min_value=0., step=0.01, max_value=1., key='H{}'.format(i))
            df_SAEInfo.loc['Estado de carga mínimo técnico [pu]','SAE {}'.format(i+1)] = socMT

    PC_des_max = []
    PD_des_max = []

    for i in df_SAEInfo.columns:
        PC_des_ = int(floor(df_SAEInfo.loc['Energía [MWh]',i] * (1 - df_SAEInfo.loc['Estado de carga mínimo [pu]',i]) / round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]',i],0.5),4)))

        if PC_des_ > df_SAEInfo.loc['Potencia máxima de carga [MW]',i]:
            PC_des_ = int(df_SAEInfo.loc['Potencia máxima de carga [MW]',i])

        PD_des_ = int(floor(df_SAEInfo.loc['Energía [MWh]',i] * ((1 - df_SAEInfo.loc['Estado de carga mínimo [pu]',i]) - (1 - round(pow(df_SAEInfo.loc['Eficiencia (round-trip) [pu]',i],0.5),4))) / nPeriodos))

        if PD_des_ > df_SAEInfo.loc['Potencia máxima de descarga [MW]',i]:
            PD_des_ = int(df_SAEInfo.loc['Potencia máxima de descarga [MW]',i])

        PC_des_max.append(PC_des_)
        PD_des_max.append(PD_des_)

    for i in range(SAE_num):
        PC_des = st.sidebar.number_input('Valor de carga requerido en la hora {} del SAE {}'.format(td_i, i+1), min_value=0, max_value=PC_des_max[i], help='En MW', key='I{}'.format(i))
        df_SAEInfo.loc['Valor de carga requerido [MW]','SAE {}'.format(i+1)] = PC_des

    for i in range(SAE_num):
        PD_des = st.sidebar.number_input('Valor de descarga requerido en la hora {} del SAE {}'.format(td_i, i+1), min_value=0, max_value=PD_des_max[i], help='En MW', key='J{}'.format(i))
        df_SAEInfo.loc['Valor de descarga requerido [MW]','SAE {}'.format(i+1)] = PD_des

    df_SAEInfo_style = df_SAEInfo.style.format("{:.2f}")
    df_SAEInfo_style = df_SAEInfo_style.format(formatter="{:.1E}", subset=pd.IndexSlice[['Autodescarga [%/h]'], :])

    if SAE_num == 1:
        st.write('Los parámetros del SAE son:')
    else:
        st.write('Los parámetros de los SAE son:')

    st.dataframe(df_SAEInfo_style)

    ## Simulación

    st.sidebar.write('### Simulación')

    # Ingresar tiempo de simulación
    time_sim = 24
    st.write("El horizonte de simulación es de: " + str(time_sim) + "h")

    # Selección fecha
    fecha = st.sidebar.date_input('Día de simulación', value = date.today() - timedelta(days=30), max_value = date.today() - timedelta(days=30))
    st.write('La fecha de simulación seleccionada fue: ' + str(fecha))

    # Seleccionar solver
    solver=st.sidebar.selectbox('Seleccione el tipo de solucionador/solver',['CPLEX','GLPK'])
    if solver=='CPLEX':
        st.write("El solucionador seleccionado es: "+solver)
    else:
        st.write("El solucionador seleccionado es: "+solver)

    ## Correr función de optimización

    if s_study == 'Versión 1':

        def run_despacho(fecha):

            ## Descarga y lectura de archivos

            with st.spinner('Descargando información de precios de oferta, disponibilidad de unidades de generación y demanda de energía...'):

                agents, load, MPO, AGC, ReadingTime, fechaD, fecha, cro = read_files(fecha.strftime('%d-%m-%Y'))

                (D, P, CONF, C, PAPUSD, PAP, MO, AGCP, AGCU, PRU, CNA, df_PRON_DEM,
                df_AGC, df_MPO, df_demandaSIN, OrganizeTime) = organize_file_agents(agents, load, AGC, MPO, fechaD, fecha)

                df_disp, df_minop, df_ofe, df_DispAGC, DPT = data_process_planta(D, P, PAP, MO, AGCP, PAPUSD, fecha)

            with st.spinner('Ejecutando simulación ...'):

                opt_results = opt_despacho(df_disp, df_minop, df_ofe, df_MPO, df_PRON_DEM, df_SAEInfo, SAE_num, tdp_i, tdp_f, td_i, td_f, solver)

            Output_data, tiempo = opt_results

            #Imprimir resultados
            st.write("## Resultados:")
            st.write('Tiempo de simulación: ' + str(tiempo))

            graph_results_opeXM(Output_data)

    elif s_study == 'Versión 2':

        def run_despacho(fecha):

            with st.spinner('Descargando información de precios de oferta, disponibilidad de unidades de generación y demanda de energía...'):

                agents, load, MPO, AGC, ReadingTime, fechaD, fecha, cro = read_files(fecha.strftime('%d-%m-%Y'))

                (D, P, CONF, C, PAPUSD, PAP, MO, AGCP, AGCU, PRU, CNA, df_PRON_DEM,
                df_AGC, df_MPO, df_demandaSIN, OrganizeTime) = organize_file_agents(agents, load, AGC, MPO, fechaD, fecha)

                df_disp, df_minop, df_ofe, df_DispAGC, DPT = data_process_planta(D, P, PAP, MO, AGCP, PAPUSD, fecha)

            with st.spinner('Ejecutando simulación ...'):

                opt_results = opt_despacho_2(df_disp, df_minop, df_ofe, df_MPO, df_PRON_DEM, df_SAEInfo, SAE_num, tdp_i, tdp_f, td_i, td_f, solver)

            #Imprimir resultados
            st.title("## Resultados:")
            Output_data, tiempo = opt_results
            st.write('Tiempo de simulación: ' + str(tiempo))

            graph_results_opeXM(Output_data)

#############Simulation button###############################################
    button_sent = st.sidebar.button("Simular")
    if button_sent:
        # if path.exists('Resultados/resultados_size_loc.xlsx'):
        #     remove('Resultados/resultados_size_loc.xlsx')
        run_despacho(fecha)