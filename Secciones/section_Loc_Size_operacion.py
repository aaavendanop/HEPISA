# coding: utf-8
## Librería de inferfaz
import streamlit as st
## Librerías para manipular datos
from os import path, remove
## Librerías para manejo de datos
import pandas as pd
import numpy as np
## Librerías para gráficos
import matplotlib.pyplot as plt
import pydeck as pdk
## Librerías for reading data form internet
import requests
from bs4 import BeautifulSoup
import re
import time
## Importing optimization functions
from modelos.LocSize.Operacion.Loc_dim_OpC import opt_dim, graph_results_ope
from funciones.nuevo_bess import bat_param

def text_formul_math_comp():
    return r"""
    ## **Función Objetivo**
    $$ \begin{aligned}
        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left( p_{i,b,t}^{th}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
        C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} \\
        + \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} \\
        + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}
    \end{aligned}$$

    donde $p_{i,b,t}^{th}$ es la potencia en MW entregada por el generador térmico $i$ en el nodo $b$ en el instante de tiempo $t$;
    $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW; $C_{i}^{dn}$ y $C_{i}^{up}$ son,
    respectivamente, los costos de parada y de arranque de cada unidad de generación en \$; $SD_{i,t}$ y $SU_{i,t}$ representan,
    respectivamente, los estados de encendido y apagado de cada unidad de generación; $p_{j,b,t}^{hyd}$ es la potencia en MW entregada
    por el generador $i$ en el nodo $b$ en el instante de tiempo $t$; $C_{j}^{hyd}$ es un vector que representa los costos de cada unidad
    de generación hidráulica en \$/MW; $P_{n,b}^{SAEB}$ es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el
    costo del equipo de conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh. Finalmente,
    $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh.

    ## **Restricciones**

    ### Restricciones del sistema

    **Balance de Potencia**

    $$ \begin{aligned}
        \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}} p_{j,b,t}^{hyd} +
        \sum_{w \in \mathcal{B}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}
        \left(p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf} \right) \\ + \sum_{n \in \mathcal{N}_{b}}
        \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{f} \hspace{2mm} \forall
        b \in \mathcal{B}, t \in \mathcal{T}
    \end{aligned}$$

    donde $p_{w,b,t}^{ren}$ es la potencia en MW entregada por el generador renovable $w$ conectado al nodo $b$ en el instante de tiempo $t$;
    $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
    $q_{b,r,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
    $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia
    de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{f}$ es la demanda del sistema del nodo $b$ en el
    instante de tiempo $t$.

    ### Límites en Generación

    **Límites en la capacidad de generación térmica**

    $$ P_{i}^{min} \leq p_{i,b,t}^{th} \leq P_{i}^{max} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$

    donde $P_{i}^{min}$ y $P_{i}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.

    **Límites en la capacidad de generación hidráulica**

    $$ P_{j}^{min} \leq p_{j,b,t}^{hyd} \leq P_{j}^{max} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$

    donde $P_{j}^{min}$ y $P_{j}^{max}$ son los límites mínimos y máximos de la capacidad de generación hidráulica, respectivamente.

    **Límites de generación en unidades renovables**

    $$ p_{w,b,t}^{ren} \leq P_{w,t}^{f} \hspace{2mm} \forall
    w \in \mathcal{W}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $P_{w,t}^{f}$ es el límite máximo de la capacidad de generación de unidades renovables.

    ### Generadores Térmicos

    **Rampas de generadores térmicos**

    $$ p_{i,t+1}^{th} - p_{i,t}^{th} \leq R_{i}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$

    $$ p_{i,t}^{th} - p_{i,t+1}^{th} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$

    donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica, respectivamente;
    $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente; $x_{i,t}$ representa el
    estado programado para las unidades de generación térmica.

    **Operación de unidades térmicas**

    $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

    $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

    **Tiempos mínimos de encendido/apagado**

    $$ x_{i,t} = g_{i}^{on/off} \quad \forall t \in \left(L_{i}^{up,min}+L_{i}^{dn,min}\right),
    i \in \mathcal{I} $$

    $$ \sum_{tt=t-g_{i}^{up}+1} SU_{i,tt} \leq x_{i,tt} \quad \forall t
    \geq L_{i}^{up,min} $$

    $$ \sum_{tt=t-g_{i}^{dn}+1} SD_{i,tt} \leq 1-x_{i,tt} \quad \forall t
    \geq L_{i}^{dn,min} $$

    donde $g_{i}^{on/off}$ es el estado inicial de las unidades de generación térmica; $L_{i}^{up,min}$ y $L_{i}^{dn,min}$ son los tiempos
    mínimos de encendido y apagado de las unidades de generación térmica.

    ### Unidades hidráulicas de generación

    $$ Q_{j}^{min} \leq q_{j,t} \leq Q_{j}^{max} \hspace{2mm} \forall
    j \in \mathcal{J}, t \in \mathcal{T}$$

    $$ V_{j}^{min} \leq v_{j,t} \leq V_{j}^{max} \hspace{2mm} \forall
    j \in \mathcal{J}, t \in \mathcal{T} $$

    $$ 0 \leq s_{j,t} \leq Q_{j,t} \hspace{2mm} \forall
    j \in \mathcal{J}, t \in \mathcal{T} $$

    $$ v_{j,t} = v_{j,t-1} + 3600 \Delta t \left(I_{t} - \sum_{j \in \mathcal{J}} q_{j,t} -
    s_{j,t} \right) \hspace{2mm} j \in \mathcal{J}, t \in \mathcal{T} $$

    $$ P_{j,t}^{hyd} = H_{j} \cdot q_{j,t} \hspace{2mm} \forall
    j \in \mathcal{J}, t \in \mathcal{T} $$

    donde $Q_{j}^{min}$ y $Q_{j}^{max}$ son los límites mínimos y máximos del flujo de agua; $q_{j,t}$ es el flujo de agua de la unidad de
    generación $j$ en el instante de tiempo $t$; $V_{j}^{min}$ y $V_{j}^{max}$ son los límites mínimos y máximos del volumen de agua en el
    embalse asociado a la unidad de generación $j$; $v_{j,t}$ es el volumen de agua del embalse asociado a la unidad de generación $j$ en
    cada instante de tiempo $t$; $s_{j,t}$ es el limite en los vertimientos de agua del embalse asociado a la unidad de generación $j$ en
    cada instante de tiempo $t$; $H_{j}$ es el factor de conversión de cada unidad de generación hidráulica $j$.

    ### Flujo de potencia DC y pérdidas
    **Cálculo del flujo de potencia por cada línea**

    $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
    (b,r) \in \mathcal{L}, t \in \mathcal{T} $$

    donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$ representan el
    valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.

    **Cálculo de las pérdidas eléctricas de cada línea**

    $$ q_{b,r,t}^{pf} = G_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right)^2 \hspace{2mm}
    \forall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$

    $$ \delta_{b,r}^{+} + \delta_{b,r}^{-} = \sum_{k=1}^{K} \delta_{b,r}(k)
    \hspace{2mm} k = 1,...,K $$

    $$ \alpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}
    k = 1, ... , K $$

    $$ q_{b,r,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \alpha_{b,r}(k)\cdot \delta_{b,r}(k)
    \hspace{2mm} \forall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$

    donde $G_{b,r}$ es la conductancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,r}^{+}$ y $\delta_{b,r}^{-}$ son
    variables utilizadas para representar el cálculo lineal del valor absoluto dentro del modelo matemático; $\alpha_{b,r}(k)$ y $\delta_{b,r}(k)$
    representan la pendiente y el valor del bloque de linealización de la diferencia angular $\left(\delta_{b} - \delta_{r} \right)$,
    respectivamente; $\Delta \delta_{b,r}$ representa el valor máximo que puede tomar la diferencia angular $\left(\delta_{b} - \delta_{r} \right)$.

    **Límites en el flujo de potencia en las líneas**

    $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} + \frac{1}{2} \cdot q_{b,r,t}^{pf} \leq P_{b,r}^{max}
    \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$

    donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
    que conecta los nodos $b$ y $r$.

    ### Restricciones sistemas de almacenamiento de energía basados en baterías
    **Variables binarias de estado de los SAEB**

    $$ u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \forall n \in \mathcal{N}, t
    \in \mathcal{T} $$

    donde $u_{n,t}^{ch}$ y $u_{n,t}^{dc}$ son las variables binarias que dan las señales de carga y descarga de los SAEB, respectivamente.

    **Relación entre la potencia y energía de los SAEB**

    $$ e_{n,b,t} = e_{n,b,t-1}\cdot \left(1 - \eta_{n}^{SoC} \right) + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -
    \frac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
    b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$

    donde $e_{n,b,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,t-1}$ es la cantidad de energía en el periodo anterior del SAEB;
    $p_{n,b,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,t}^{dc}$ es la potencia de descarga
    del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.

    **Límite de energía de los SAEB**

    $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño del SAEB
    a instalar en términos de energía.

    **Límite de potencia de los SAEB**

    $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,t}^{ch} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{ch} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq Z \cdot u_{n,t}^{dc} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $Z$ es un valor constante muy grande y $P_{n,b}$ es la potencia nominal del SAEB, la cual constituye una variable de decisión que
    determina el tamaño óptimo del SAEB en términos de potencia.

    """

def text_formul_math_simp():
    return r"""

    ## **Función Objetivo**

    $$ \begin{aligned}
        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left( p_{i,b,t}\cdot C_{i}^{gen} \right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +
        \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \\
        \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}
    \end{aligned}$$

    donde $p_{i,b,t}^{th}$ es la potencia en MW entregada por el generador térmico $i$ en el nodo $b$ en el instante de tiempo $t$;
    $p_{j,b,t}^{hyd}$ es la potencia en MW entregada por el generador $i$ en el nodo $b$ en el instante de tiempo $t$; $C_{j}^{hyd}$
    es un vector que representa los costos de cada unidad de generación hidráulica en \$/MW; $P_{n,b}^{SAEB}$ es la capacidad de potencia
    en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la
    capacidad de energía del SAEB en MWh. Finalmente, $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh.

    ## **Restricciones**

    ### Restricciones del sistema

    **Balance de potencia**

    $$ \begin{aligned}
        \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}} p_{j,b,t}^{hyd} +
        \sum_{w \in \mathcal{B}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}
        p_{b,r,t}^{pf} \\ + \sum_{n \in \mathcal{N}_{b}}
        \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{f} \hspace{2mm} \forall
        b \in \mathcal{B}, t \in \mathcal{T}
    \end{aligned}$$

    donde $p_{w,b,t}^{ren}$ es la potencia en MW entregada por el generador renovable $w$ conectado al nodo $b$ en el instante de tiempo $t$;
    $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
    $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia
    de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{f}$ es la demanda del sistema del nodo $b$ en el
    instante de tiempo $t$.

    ### Límites en Generación

    **Límites en la capacidad de generación térmica**

    $$ P_{i}^{min} \leq p_{i,b,t}^{th} \leq P_{i}^{max} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$

    donde $P_{i}^{min}$ y $P_{i}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.

    **Límites en la capacidad de generación hidráulica**

    $$ P_{j}^{min} \leq p_{j,b,t}^{hyd} \leq P_{j}^{max} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$

    donde $P_{j}^{min}$ y $P_{j}^{max}$ son los límites mínimos y máximos de la capacidad de generación hidráulica, respectivamente.

    **Límites de generación en unidades renovables**

    $$ p_{w,b,t}^{ren} \leq P_{w,t}^{f} \hspace{2mm} \forall
    w \in \mathcal{W}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $P_{w,t}^{f}$ es el límite máximo de la capacidad de generación de unidades renovables.

    ### Flujo de potencia DC

    **Cálculo del flujo de potencia por cada línea**

    $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
    (b,r) \in \mathcal{L}, t \in \mathcal{T} $$

    donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$ representan el
    valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.

    **Límites en el flujo de potencia de las líneas**

    $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} \leq P_{b,r}^{max}
    \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$

    donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
    que conecta los nodos $b$ y $r$.

    ### Restricciones sistemas de almacenamiento de energía basados en baterías

    **Variables binarias de estado de los SAEB**

    $$ u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \forall n \in \mathcal{N},
    t \in \mathcal{T} $$

    donde $u_{n,t}^{ch}$ y $u_{n,t}^{dc}$ son las variables binarias que dan las señales de carga y descarga de los SAEB, respectivamente.

    **Relación entre la potencia y energía de los SAEB**

    $$ e_{n,b,t} = e_{n,b,t-1}\cdot \eta_{n}^{SoC} + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -
    \frac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
    b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$

    donde $e_{n,b,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,t-1}$ es la cantidad de energía en el periodo anterior del SAEB;
    $p_{n,b,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,t}^{dc}$ es la potencia de descarga
    del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.

    **Límite de energía de los SAEB**

    $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño del SAEB
    a instalar en términos de energía.

    **Límite de potencia de los SAEB**

    $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,t}^{ch} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    $$ p_{n,b,t}^{ch} \leq P_{n,b} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq Z \cdot u_{n,t}^{dc} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq P_{n,b} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    donde $Z$ es un valor constante muy grande y $P_{n,b}$ es la potencia nominal del SAEB, la cual constituye una variable de decisión que
    determina el tamaño óptimo del SAEB en términos de potencia.

    """

def text_formul_math_sens(info_):

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in info_:

        st.write("## **Función Objetivo**")
        st.write(r"""
                        $$ \begin{aligned}
                            \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                            \left( p_{i,b,t}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
                            C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} \\
                            + \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                            \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} \\
                            + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                            E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}
                        \end{aligned}$$
                    """)
        st.write(r"""
                donde $p_{i,b,t}^{th}$ es la potencia en MW entregada por el generador térmico $i$ en el nodo $b$ en el instante de tiempo $t$;
                $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW; $C_{i}^{dn}$ y $C_{i}^{up}$ son,
                respectivamente, los costos de parada y de arranque de cada unidad de generación en \$; $SD_{i,t}$ y $SU_{i,t}$ representan,
                respectivamente, los estados de encendido y apagado de cada unidad de generación; $p_{j,b,t}^{hyd}$ es la potencia en MW entregada
                por el generador $i$ en el nodo $b$ en el instante de tiempo $t$; $C_{j}^{hyd}$ es un vector que representa los costos de cada unidad
                de generación hidráulica en \$/MW; $P_{n,b}^{SAEB}$ es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el
                costo del equipo de conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh. Finalmente,
                $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh.
                """)

    else:

        st.write("## **Función Objetivo**")
        st.write(r"""
                $$ \begin{aligned}
                    \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left( p_{i,b,t}\cdot C_{i}^{gen} \right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Térmicos} +
                    \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left( p_{j,b,t}^{hyd} \cdot C_{j}^{hyd}\right)}_{Costos\hspace{1mm}por\hspace{1mm}generadores\hspace{1mm}Hidráulicos} + \\
                    \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                    E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB}
                \end{aligned}$$
                """)

        st.write(r"""
                donde $p_{i,b,t}^{th}$ es la potencia en MW entregada por el generador térmico $i$ en el nodo $b$ en el instante de tiempo $t$;
                $p_{j,b,t}^{hyd}$ es la potencia en MW entregada por el generador $i$ en el nodo $b$ en el instante de tiempo $t$; $C_{j}^{hyd}$
                es un vector que representa los costos de cada unidad de generación hidráulica en \$/MW; $P_{n,b}^{SAEB}$ es la capacidad de potencia
                en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la
                capacidad de energía del SAEB en MWh. Finalmente, $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh.
                """)

    #### Restricciones del sistema
    st.write("## **Restricciones**")

    st.write('### Restricciones del sistema')
    st.write('**Balance de potencia**')

    if 'Pérdidas' in info_:

        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}} p_{j,b,t}^{hyd} +
                    \sum_{w \in \mathcal{B}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}
                    \left(p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf} \right) \\ + \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{f} \hspace{2mm} \forall
                    b \in \mathcal{B}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""donde $p_{w,b,t}^{ren}$ es la potencia en MW entregada por el generador renovable $w$ conectado al nodo $b$ en el
                instante de tiempo $t$; $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en
                cada instante de tiempo $t$; $q_{b,r,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos
                $b$ y $r$ en cada instante de tiempo $t$; $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el
                instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia de descarga del SAEB $n$ conectado al nodo $b$ en el instante de
                tiempo $t$; $D_{b,t}^{f}$ es la demanda del sistema del nodo $b$ en el instante de tiempo $t$.
                """)
    else:

        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{th} + \sum_{j \in \mathcal{J}_{b}} p_{j,b,t}^{hyd} +
                    \sum_{w \in \mathcal{B}_{b}} p_{w,b,t}^{ren} - \sum_{(b,r) \in \mathcal{L}}
                    p_{b,r,t}^{pf} \\ + \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{f} \hspace{2mm} \forall
                    b \in \mathcal{B}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{w,b,t}^{ren}$ es la potencia en MW entregada por el generador renovable $w$ conectado al nodo $b$ en el instante de
                tiempo $t$; $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante
                de tiempo $t$; $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$;
                $p_{n,b,t}^{ch}$ es la potencia de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{f}$ es
                la demanda del sistema del nodo $b$ en el instante de tiempo $t$.
                """)

    st.write('### Límites en Generación')
    st.write('**Límites en la capacidad de generación térmica**')
    st.write(r"""
            $$ P_{i}^{min} \leq p_{i,b,t}^{th} \leq P_{i}^{max} \hspace{2mm} \forall
            t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$
            """)
    st.write(r"""
            donde $P_{i}^{min}$ y $P_{i}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.
            """)
    st.write('**Límites en la capacidad de generación hidráulica**')
    st.write(r"""
                $$ P_{j}^{min} \leq p_{j,b,t}^{hyd} \leq P_{j}^{max} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$
            """)
    st.write(r"""
            donde $P_{j}^{min}$ y $P_{j}^{max}$ son los límites mínimos y máximos de la capacidad de generación hidráulica, respectivamente.
            """)
    st.write('**Límites de generación en unidades renovables**')
    st.write(r"""
                $$ p_{w,b,t}^{ren} \leq P_{w,t}^{f} \hspace{2mm} \forall
                w \in \mathcal{W}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $P_{w,t}^{f}$ es el límite máximo de la capacidad de generación de unidades renovables.
            """)

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in info_:

        st.write('### Generadores Térmicos')

        st.write('**Rampas de generadores térmicos**')
        st.write(r"""
                $$ p_{i,t+1}^{th} - p_{i,t}^{th} \leq R_{i}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ p_{i,t}^{th} - p_{i,t+1}^{th} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica,
                respectivamente; $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente;
                $x_{i,t}$ representa el estado programado para las unidades de generación térmica.
                """)

        st.write('**Operación de unidades térmicas**')
        st.write(r"""
                $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)

        st.write('**Tiempos mínimos de encendido/apagado**')
        st.write(r"""
                $$ x_{i,t} = g_{i}^{on/off} \quad \forall t \in \left(L_{i}^{up,min}+L_{i}^{dn,min}\right),
                i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ \sum_{tt=t-g_{i}^{up}+1} SU_{i,tt} \leq x_{i,tt} \quad \forall t
                \geq L_{i}^{up,min} $$
                """)
        st.write(r"""
                $$ \sum_{tt=t-g_{i}^{dn}+1} SD_{i,tt} \leq 1-x_{i,tt} \quad \forall t
                \geq L_{i}^{dn,min} $$
                """)
        st.write(r"""
                donde $g_{i}^{on/off}$ es el estado inicial de las unidades de generación térmica; $L_{i}^{up,min}$ y $L_{i}^{dn,min}$ son
                los tiempos mínimos de encendido y apagado de las unidades de generación térmica.
                """)

    if 'Embalses' in info_:

        st.write('#### Unidades hidráulicas de generación')
        st.write(r"""
                $$ Q_{j}^{min} \leq q_{j,t} \leq Q_{j}^{max} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T}$$
                """)
        st.write(r"""
                $$ V_{j}^{min} \leq v_{j,t} \leq V_{j}^{max} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                $$ 0 \leq s_{j,t} \leq Q_{j,t} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                $$ v_{j,t} = v_{j,t-1} + 3600 \Delta t \left(I_{t} - \sum_{j \in \mathcal{J}} q_{j,t} -
                s_{j,t} \right) \hspace{2mm} j \in \mathcal{J}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                $$ P_{j,t}^{hyd} = H_{j} \cdot q_{j,t} \hspace{2mm} \forall
                j \in \mathcal{J}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $Q_{j}^{min}$ y $Q_{j}^{max}$ son los límites mínimos y máximos del flujo de agua; $q_{j,t}$ es el flujo de agua de la
                unidad de generación $j$ en el instante de tiempo $t$; $V_{j}^{min}$ y $V_{j}^{max}$ son los límites mínimos y máximos del
                volumen de agua en el embalse asociado a la unidad de generación $j$; $v_{j,t}$ es el volumen de agua del embalse asociado a
                la unidad de generación $j$ en cada instante de tiempo $t$; $s_{j,t}$ es el limite en los vertimientos de agua del embalse
                asociado a la unidad de generación $j$ en cada instante de tiempo $t$; $H_{j}$ es el factor de conversión de cada unidad de
                generación hidráulica $j$.
                """)

    if "Pérdidas" in info_:

        st.write('### Flujo de potencia DC y pérdidas')
        st.write('**Cálculo del flujo de potencia por cada línea**')
        st.write(r"""
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$
                representan el valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.
                """)
        st.write('**Cálculo de las pérdidas eléctricas de cada línea**')
        st.write(r"""
                $$ q_{b,r,t}^{pf} = G_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right)^2 \hspace{2mm}
                \forall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                $$ \delta_{b,r}^{+} + \delta_{b,r}^{-} = \sum_{k=1}^{K} \delta_{b,r}(k)
                \hspace{2mm} k = 1,...,K $$
                """)
        st.write(r"""
                $$ \alpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}
                k = 1, ... , K $$
                """)
        st.write(r"""
                $$ q_{b,r,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \alpha_{b,r}(k)\cdot \delta_{b,r}(k)
                \hspace{2mm} \forall (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $G_{b,r}$ es la conductancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,r}^{+}$ y $\delta_{b,r}^{-}$
                son variables utilizadas para representar el cálculo lineal del valor absoluto dentro del modelo matemático;
                $\alpha_{b,r}(k)$ y $\delta_{b,r}(k)$ representan la pendiente y el valor del bloque de linealización de la diferencia angular
                $\left(\delta_{b} - \delta_{r} \right)$, respectivamente; $\Delta \delta_{b,r}$ representa el valor máximo que puede tomar la
                diferencia angular $\left(\delta_{b} - \delta_{r} \right)$.
                """)
        st.write('**Límites en el flujo de potencia en las líneas**')
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} + \frac{1}{2} \cdot q_{b,r,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa
                de la línea que conecta los nodos $b$ y $r$.
                """)

    else:

        st.write('### Flujo de potencia DC')
        st.write('**Cálculo del flujo de potencia por cada línea**')
        st.write(r"""
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$
                representan el valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.
                """)
        st.write('**Límites en el flujo de potencia de las líneas**')
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa
                de la línea que conecta los nodos $b$ y $r$.
                """)

    st.write('### Restricciones sistemas de almacenamiento de energía basados en baterías')
    st.write('**Variables binarias de estado de los SAEB**')
    st.write(r"""
            $$ u_{n,t}^{ch} + u_{n,t}^{dc} \leq 1 \hspace{2mm} \forall n \in \mathcal{N}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $u_{n,t}^{ch}$ y $u_{n,t}^{dc}$ son las variables binarias que dan las señales de carga y descarga de los SAEB, respectivamente.
            """)
    st.write('**Relación entre la potencia y energía de los SAEB**')
    st.write(r"""
            $$ e_{n,b,t} = e_{n,b,t-1}\cdot \left(1 - \eta_{n}^{SoC} \right) + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -
            \frac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
            b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$
            """)
    st.write(r"""
            donde $e_{n,b,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,t-1}$ es la cantidad de energía en el periodo anterior del
            SAEB; $p_{n,b,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,t}^{dc}$ es
            la potencia de descarga del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.
            """)
    st.write('**Límite de energía de los SAEB**')
    st.write(r"""
            $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño
            del SAEB a instalar en términos de energía.
            """)
    st.write('**Límite de potencia de los SAEB**')
    st.write(r"""
            $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,t}^{ch} \hspace{2mm} \forall n \in \mathcal{N}, b\in \mathcal{B}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{ch} \leq P_{n,b} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{dc} \leq Z \cdot u_{n,t}^{dc} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{dc} \leq P_{n,b} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B}, t
            \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $Z$ es un valor constante muy grande y $P_{n,b}$ es la potencia nominal del SAEB, la cual constituye una variable de
            decisión que determina el tamaño óptimo del SAEB en términos de potencia.
            """)
    return ""

def dashboard_DLOp(data1):

    ##

    st.markdown("## Parámetros seleccionados para la simulación")

    ## Selección de archivo del sistema

    st.sidebar.markdown("### Ingrese los parámetros de simulación ")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_system = st.sidebar.file_uploader("Seleccione el archivo con el sistema a simular", type=['csv','xlsx'], help='Se admiten archivos .csv o .xlsx')

    ## Selección de tecnología de SAE
    Eff, degra, autoD, DoD, costP, costE, Aa, Bb, ciclos = bat_param(data1,0)

    ## Vida útil del SAEB

    vida_util = st.sidebar.number_input("Ingrese el tiempo estimado de operación del proyecto [año(s)]", min_value=1, max_value=1000000, help='Valores típicos: 10, 15 o 20 años')
    st.write("La vida útil del SAEB es: " + str(vida_util) + ' año(s)')

    ## Horizonte de simulación

    time_sim = st.sidebar.number_input("Ingrese el horizonte de simulación [h]", min_value=1, max_value=1000000)
    st.write("El horizonte de simulación es: " + str(time_sim) + ' hora(s)')

    ##

    st.sidebar.write('### Ingrese características del modelo a aplicar')

    sensitivity = st.sidebar.selectbox('Añadir/Quitar restricciones al modelo base', ('Sí', 'No'))

    if sensitivity == 'No':

        ss_study = st.sidebar.selectbox('Tipo de modelo', ('Completo', 'Simplificado'))

    else:

        restr_list_compl = ['Límites de generación', 'Flujos de potencia DC', 'Límites en el Flujo por las líneas',
                            'Balance de potencia', 'Sistemas de Almacenamiento de Energía',
                            'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)', 'Pérdidas', 'Embalses']
        restr_list_simp = restr_list_compl[0:5]
        st.sidebar.write('**Restricciones base**')
        st.sidebar.write('Límites de generación\n\n', 'Flujos de potencia DC\n\n', 'Límites en el Flujo por las líneas\n\n', 'Balance de potencia\n\n', 'Sistemas de Almacenamiento de Energía')
        st.sidebar.write('**Restricciones adicionales**')
        restr_list = st.sidebar.multiselect(label='', options=sorted(restr_list_compl[5:8]), default=restr_list_compl[5:8])
        ss_study = restr_list

    st.write('### Formulación matemática usada')

    ## Formulación matemática

    if ss_study == 'Completo':

        formulacion = st.expander(label='Formulación Matemática', expanded=False)
        with formulacion:
            st.write(text_formul_math_comp())
            st.write("")

    elif ss_study == 'Simplificado':

        formulacion = st.expander(label='Formulación Matemática', expanded=False)
        with formulacion:
            st.write(text_formul_math_simp())
            st.write("")

    else:

        formulacion = st.expander(label='Formulación Matemática', expanded=False)
        with formulacion:
            st.write(text_formul_math_sens(ss_study))

    ## Tipo de modelo

    if ss_study == 'Simplificado':

        tipo = 'Simplificado'

    elif ss_study == 'Completo':

        tipo = 'Completo'

    else:

        tipo = 'Personalizado'

    st.write('El tipo de modelo seleccionado es: ', tipo)

    ## Solver

    st.sidebar.write('### Simulación')
    solver = st.sidebar.selectbox("Seleccione el tipo de Solver", ['CPLEX','GLPK'])

    if solver == 'CPLEX':

        st.write("El solucionador seleccionado fue: " + solver)

    else:

        st.write("El solucionador seleccionado fue: " + solver)

    ## Correr función de optimización

    def run_dim_size_OpC():

        with st.spinner('Ejecutando simulación ...'):

            Power, Energy, name_file, tiempo, map_data_2, map_data_1, cost_ope, Output_data = opt_dim(file_system, Eff, DoD, 0.2, time_sim, vida_util, costP,
                                                                                        costE, autoD, solver, ss_study)

        ## Impresión de resultados

        st.write("Resultados:")
        st.write("Tiempo de simulación: " + str(tiempo))

        st.write("Nodos en donde se instala SAEB: ")
        dict_result_loc = {}

        for i in Power.index:

            result_loc = []

            if Power.loc[i] != 0:

                result_loc.append(Power.loc[i])
                result_loc.append(Energy.loc[i])
                dict_result_loc[i] = result_loc

        df_dict_result_loc = pd.DataFrame.from_dict(dict_result_loc, orient='index', columns=['Potencia [MW]', 'Energía [MWh]'])
        st.dataframe(df_dict_result_loc.style.format(thousands=',', precision=1, decimal='.'))

        cost_ope = cost_ope.rename(columns={'1': 'Valor función objetivo [$USD]'})
        st.write("Valor función objetivo")
        st.dataframe(cost_ope.style.format(thousands=',', precision=1, decimal='.'))

        graph_results_ope(Output_data, ss_study)

        point_size = [0] * Power.shape[0]
        point_colors = [0] * Power.shape[0]

        for i in range(Power.shape[0]):

            if Power.loc[Power.index[i]] > 0:

                point_size[i] = 40000
                point_colors[i] = [10, 230, 10]

            else:

                point_size[i] = 15000
                point_colors[i] = [230, 158, 10]

        st.write('#### Mapa de localización de SAE en Colombia')
        st.write('')
        st.write('')
        st.write('')

        map_data_1['exits_radius'] = point_size
        map_data_1['color'] = point_colors
        midpoint = (np.average(map_data_1['lat']), np.average(map_data_1['lon']))
        st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=4),
            layers=[pdk.Layer('ScatterplotLayer', data=map_data_1, get_position='[lon, lat]', get_color='color',
            get_radius='exits_radius'),],))

    ## Simulatión button

    button_sent = st.sidebar.button("Simular")

    if button_sent:

        if path.exists("Resultados/resultados_size_loc.xlsx"):

            remove("Resultados/resultados_size_loc.xlsx")

        run_dim_size_OpC()