# coding: utf-8
## Librería de inferfaz
import streamlit as st
## Librerías para manipular datos
from os import path
from os import remove
## Librerías para manejo de datos
import pandas as pd
import numpy as np
## Librerías para gráficos
import matplotlib.pyplot as plt
import pydeck as pdk
## Librerías for reading data form internet
from bs4 import BeautifulSoup
import re
from datetime import date, timedelta, datetime
## Importing read functions
from funciones.read_download_files import *
## Importing optimization functions
from modelos.LocSize.Restricciones.Loc_dim_ReS import DISC, DPRLNSC, DPRLNSC_H,graph_results_res
## Other functions
from funciones.nuevo_bess import bat_param
import funciones.pydataxm as pxm

def text_formul_math_deter():
    return r"""
    ## **Función Objetivo**
    $$ \begin{aligned}
        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left(p_{i,b,t}^{re}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
        C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}de\hspace{1mm}generación} \\
        + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
        + \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left(\left(p_{i,b,t}^{re} - p_{i,d,t}^{id} \right)\cdot C_{t}^{MPO} \right)}_{Penalización} \\
    \end{aligned}$$

    donde $p_{i,b,t}^{re}$ es la potencia de la simulación del despacho real, en MW, entregada por el generador $i$ en el nodo $b$ en el
    instante de tiempo $t$; $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW;
    $C_{i}^{dn}\cdot SD_{i,t}$ y $C_{i}^{up}\cdot SU_{i,t}$ representan, respectivamente, los costos de arranque y parada de las unidades de
    generación; $P_{n,b}^{SAEB}$ es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de
    conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de
    almacenamiento en \$/MWh; $p_{i,d,t}^{id}$ es la potencia que resulta del despacho ideal, en MW, entregada por el generador $i$ en el
    nodo $b$ en el instante de tiempo $t$. Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.
    ## **Restricciones**

    ### Restricciones del sistema

    **Balance de Potencia**

    $$ \begin{aligned}
        \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{re}  - \sum_{(b,r) \in \mathcal{L}}
        \left(p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf} \right) + \sum_{n \in \mathcal{N}_{b}}
        \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{r} \hspace{2mm} \forall
        b \in \mathcal{B}, t \in \mathcal{T}
    \end{aligned}$$

    donde $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
    $q_{b,r,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
    $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia
    de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{r}$ la demanda real del sistema del día estudiado.

    ### Límites en Generación

    $$ P_{i}^{min} \leq p_{i,b,t}^{re} \leq P_{i}^{max} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$

    donde $P_{i}^{min}$ y $P_{i}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.

    ### Generadores Térmicos

    **Rampas de generadores térmicos**

    $$ p_{i,t+1}^{re} - p_{i,t}^{re} \leq R_{i}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$

    $$ p_{i,t}^{re} - p_{i,t+1}^{re} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$

    donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica, respectivamente;
    $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente; $x_{i,t}$ representa el
    estado programado para las unidades de generación térmica.

    **Variables binarias de operación de unidades térmicas**

    $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

    $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

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

    $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,b,t}^{sta} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{ch} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq Z \cdot \left(1 - u_{n,b,t}^{sta}\right) - Z \cdot \left(1 - u_{n,b}^{ins} \right) \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,t}^{dc} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $Z$ es un valor constante muy grande; $u_{n,b,t}^{sta}$ es una variable binaria que modela el comportamiento del SAEB $n$
    instalado en el nodo $b$ en cada instante de tiempo $t$; $u_{n,b}^{ins}$ es una variable binaria que determina si el SAEB $n$ es instalado
    en el nodo $b$.

    **Limite en el número de SAEB que se instalan**

    $$ u_{n,b,t}^{sta} \leq u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    $$ E_{n,b}^{SAEB} \leq Z \cdot u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N},
    b \in \mathcal{B}$$

    $$ \sum_{\mathcal{N}} u_{n,b}^{ins} \leq N_{max} \hspace{2mm} \forall n \in \mathcal{N},
    b \in \mathcal{B} $$

    donde $N_{max}$ es el número máximo de SAEB que se pueden instalar.

    """

def text_formul_math_escen():
    return r"""
    ## **Función Objetivo**
    $$ \begin{aligned}
        \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left(C_{i}^{dn}\cdot SD_{i,t} + C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}de\hspace{1mm}arranque/parada} \\
        + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
        E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
        + \underbrace{\sum_{s \in \mathcal{S}} \pi_{s} \sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
        \left(p_{i,b,s,t}^{re}\cdot C_{i,s}^{gen} + \left(p_{i,b,s,t}^{re} - p_{i,d,s,t}^{id} \right)\cdot C_{t,s}^{MPO} \right)}_{Penalización} \\
    \end{aligned}$$

    donde $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW; $C_{i}^{dn}\cdot SD_{i,t}$ y
    $C_{i}^{up}\cdot SU_{i,t}$ representan, respectivamente, los costos de arranque y parada de las unidades de generación; $P_{n,b}^{SAEB}$
    es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de conversión de potencia en \$/MW;
    $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh;
    $\pi_{s}$ es la probabilidad de ocurrencia de cada escenario; $p_{i,b,s,t}^{re}$ es la potencia de la simulación del despacho real,
    en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$; $p_{i,s,d,t}^{id}$ es la potencia
    que resulta del despacho ideal, en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$.
    Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.

    ## **Restricciones**

    ### Restricciones del sistema

    **Balance de Potencia**

    $$ \begin{aligned}
        \sum_{i \in \mathcal{I}_{b}} p_{i,b,s,t}^{re}  - \sum_{(b,r) \in \mathcal{L}}
        \left(p_{b,r,s,t}^{pf} + \frac{1}{2} q_{b,r,s,t}^{pf} \right) + \\ \sum_{n \in \mathcal{N}_{b}}
        \left(p_{n,b,s,t}^{dc} - p_{n,b,s,t}^{ch} \right) = D_{b,s,t}^{f} \hspace{2mm} \forall
        b \in \mathcal{B}, s \in \mathcal{S}, t \in \mathcal{T}
    \end{aligned}$$

    donde $p_{b,r,s,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ para el escenario $s$ en cada
    instante de tiempo $t$; $q_{b,r,s,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos $b$ y $r$ para
    el escenario $s$ en cada instante de tiempo $t$; $p_{n,b,s,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ para el
    escenario $s$ en el instante de tiempo $t$; $p_{n,b,s,t}^{ch}$ es la potencia de descarga del SAEB $n$ conectado al nodo $b$ para el
    escenario $s$ en el instante de tiempo $t$; $D_{b,s,t}^{r}$ la demanda real del sistema del día estudiado.

    ### Límites en Generación

    $$ P_{i,s}^{min} \leq p_{i,b,s,t}^{re} \leq P_{i,s}^{max} \hspace{2mm} \forall
    i \in \mathcal{I}, b \in \mathcal{B}, s \in \mathcal{S}, t \in \mathcal{T} $$

    donde $P_{i,s}^{min}$ y $P_{i,s}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.

    ### Generadores Térmicos

    **Rampas de generadores térmicos**

    $$ p_{i,s,t+1}^{re} - p_{i,s,t}^{re} \leq R_{i,s}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i,s}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, s \in \mathcal{S}, i \in \mathcal{I} $$

    $$ p_{i,s,t}^{re} - p_{i,s,t+1}^{re} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i,s}^{min} \hspace{2mm}
    \forall t, t + 1 \in \mathcal{T}, s \in \mathcal{S}, i \in \mathcal{I} $$

    donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica, respectivamente;
    $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente; $x_{i,t}$ representa el
    estado programado para las unidades de generación térmica.

    **Variables binarias de operación de unidades térmicas**

    $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

    $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
    t \in \mathcal{T}, i \in \mathcal{I} $$

    ### Flujo de potencia DC y pérdidas
    **Cálculo del flujo de potencia por cada línea**

    $$ p_{b,r,s,t}^{pf} = B_{b,r} \cdot \left(\delta_{b,s} - \delta_{r,s} \right) \hspace{2mm} \forall
    (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$

    donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,s}$ y $\delta_{r,s}$ representan el
    valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada para el escenario $s$, respectivamente.

    **Cálculo de las pérdidas eléctricas de cada línea**

    $$ q_{b,r,s,t}^{pf} = G_{b,r} \cdot \left(\delta_{b,s} - \delta_{r,s} \right)^2 \hspace{2mm}
    \forall (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$

    $$ \delta_{b,r,s}^{+} + \delta_{b,r,s}^{-} = \sum_{k=1}^{K} \delta_{b,r,s}(k)
    \hspace{2mm} k = 1,...,K $$

    $$ \alpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}
    k = 1, ... , K $$

    $$ q_{b,r,s,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \alpha_{b,r}(k)\cdot \delta_{b,r,s}(k)
    \hspace{2mm} \forall (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$

    donde $G_{b,r}$ es la conductancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,r,s}^{+}$ y $\delta_{b,r,s}^{-}$ son
    variables utilizadas para representar el cálculo lineal del valor absoluto dentro del modelo matemático; $\alpha_{b,r}(k)$ y $\delta_{b,r,s}(k)$
    representan la pendiente y el valor del bloque de linealización de la diferencia angular $\left(\delta_{b,s} - \delta_{r,s} \right)$,
    respectivamente; $\Delta \delta_{b,r}$ representa el valor máximo que puede tomar la diferencia angular $\left(\delta_{b,s} - \delta_{r,s} \right)$.

    **Límites en el flujo de potencia en las líneas**

    $$ -P_{b,r}^{max} \leq p_{b,r,s,t}^{pf} + \frac{1}{2} \cdot q_{b,r,s,t}^{pf} \leq P_{b,r}^{max}
    \hspace{2mm} \forall l \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$

    donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
    que conecta los nodos $b$ y $r$.

    ### Restricciones sistemas de almacenamiento de energía basados en baterías

    **Relación entre la potencia y energía de los SAEB**

    $$ e_{n,b,s,t} = e_{n,b,s,t-1}\cdot \left(1 - \eta_{n}^{SoC} \right) + \left( \eta^{ch}_{n} \cdot p_{n,b,s,t}^{ch} -
    \frac{P_{n,b,s,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
    b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$

    donde $e_{n,b,s,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,s,t-1}$ es la cantidad de energía en el periodo anterior del SAEB;
    $p_{n,b,s,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,s,t}^{dc}$ es la potencia de descarga
    del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.

    **Límite de energía de los SAEB**

    $$ e_{n,b,s,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño del SAEB
    a instalar en términos de energía.

    **Límite de potencia de los SAEB**

    $$ p_{n,b,s,t}^{ch} \leq Z \cdot u_{n,b,t}^{sta} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,s,t}^{ch} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,s,t}^{dc} \leq Z \cdot \left(1 - u_{n,b,t}^{sta}\right) - Z \cdot \left(1 - u_{n,b}^{ins} \right) \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    $$ p_{n,b,s,t}^{dc} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
    n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$

    donde $Z$ es un valor constante muy grande; $u_{n,b,t}^{sta}$ es una variable binaria que modela el comportamiento del SAEB $n$
    instalado en el nodo $b$ en cada instante de tiempo $t$; $u_{n,b}^{ins}$ es una variable binaria que determina si el SAEB $n$ es instalado
    en el nodo $b$.

    **Limite en el número de SAEB que se instalan**

    $$ u_{n,b,t}^{sta} \leq u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
    t \in \mathcal{T} $$

    $$ E_{n,b}^{SAEB} \leq Z \cdot u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N},
    b \in \mathcal{B}$$

    $$ \sum_{\mathcal{N}} u_{n,b}^{ins} \leq N_{max} \hspace{2mm} \forall n \in \mathcal{N},
    b \in \mathcal{B} $$

    donde $N_{max}$ es el número máximo de SAEB que se pueden instalar.

    """

def text_formul_math_deter_sen(res_list):

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in res_list:

        st.write("## **Función Objetivo**")
        st.write(r"""
                $$ \begin{aligned}
                    \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(p_{i,b,t}^{re}\cdot C_{i}^{gen} + C_{i}^{dn}\cdot SD_{i,t} +
                    C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}de\hspace{1mm}generación} \\
                    + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                    E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
                    + \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(\left(p_{i,b,t}^{re} - p_{i,d,t}^{id} \right)\cdot C_{t}^{MPO} \right)}_{Penalización} \\
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{i,b,t}^{re}$ es la potencia de la simulación del despacho real, en MW, entregada por el generador $i$ en el nodo $b$ en el
                instante de tiempo $t$; $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW;
                $C_{i}^{dn}\cdot SD_{i,t}$ y $C_{i}^{up}\cdot SU_{i,t}$ representan, respectivamente, los costos de arranque y parada de las unidades de
                generación; $P_{n,b}^{SAEB}$ es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de
                conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de
                almacenamiento en \$/MWh; $p_{i,d,t}^{id}$ es la potencia que resulta del despacho ideal, en MW, entregada por el generador $i$ en el
                nodo $b$ en el instante de tiempo $t$. Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.
                """)

    else:

        st.write("## **Función Objetivo**")
        st.write(r"""
                $$ \begin{aligned}
                    \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(p_{i,b,t}^{re}\cdot C_{i}^{gen}\right)}_{Costos\hspace{1mm}de\hspace{1mm}generación} \\
                    + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                    E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
                    + \underbrace{\sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(\left(p_{i,b,t}^{re} - p_{i,d,t}^{id} \right)\cdot C_{t}^{MPO} \right)}_{Penalización} \\
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{i,b,t}^{re}$ es la potencia de la simulación del despacho real, en MW, entregada por el generador $i$ en el nodo $b$ en el
                instante de tiempo $t$; $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW;
                $P_{n,b}^{SAEB}$ es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de
                conversión de potencia en \$/MW; $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de
                almacenamiento en \$/MWh; $p_{i,d,t}^{id}$ es la potencia que resulta del despacho ideal, en MW, entregada por el generador $i$ en el
                nodo $b$ en el instante de tiempo $t$. Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.
                """)

    #### Restricciones del sistema

    st.write("## **Restricciones**")

    st.write('### Restricciones del sistema')
    st.write('**Balance de potencia**')

    if 'Pérdidas' in res_list:
        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{re}  - \sum_{(b,r) \in \mathcal{L}}
                    \left(p_{b,r,t}^{pf} + \frac{1}{2} q_{b,r,t}^{pf} \right) + \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{r} \hspace{2mm} \forall
                    b \in \mathcal{B}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
                $q_{b,r,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
                $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia
                de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{r}$ la demanda real del sistema del día estudiado.
                """)
    else:
        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,t}^{re}  - \sum_{(b,r) \in \mathcal{L}}
                    p_{b,r,t}^{pf} + \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,t}^{dc} - p_{n,b,t}^{ch} \right) = D_{b,t}^{r} \hspace{2mm} \forall
                    b \in \mathcal{B}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{b,r,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ en cada instante de tiempo $t$;
                $p_{n,b,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $p_{n,b,t}^{ch}$ es la potencia
                de descarga del SAEB $n$ conectado al nodo $b$ en el instante de tiempo $t$; $D_{b,t}^{r}$ la demanda real del sistema del día estudiado.
                """)

    st.write("### Límites en Generación")
    st.write(r"""
            $$ P_{i}^{min} \leq p_{i,b,t}^{re} \leq P_{i}^{max} \hspace{2mm} \forall
            t \in \mathcal{T}, i \in \mathcal{I}, b \in \mathcal{B} $$
            """)
    st.write(r"""
            donde $P_{i}^{min}$ y $P_{i}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.
            """)

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in res_list:

        st.write("### Operación de Generadores Térmicos")
        st.write("**Rampas de generadores térmicos**")
        st.write(r"""
                $$ p_{i,t+1}^{re} - p_{i,t}^{re} \leq R_{i}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ p_{i,t}^{re} - p_{i,t+1}^{re} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica, respectivamente;
                $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente; $x_{i,t}$ representa el
                estado programado para las unidades de generación térmica.
                """)

        st.write("**Variables binarias de operación de unidades térmicas**")
        st.write(r"""
                $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)

    if 'Pérdidas' in res_list:
        st.write("### Flujo de potencia DC y pérdidas")
        st.write("**Cálculo del flujo de potencia por cada línea**")
        st.write(r"""
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$ representan el
                valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.
                """)

        st.write("**Cálculo de las pérdidas eléctricas de cada línea**")
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
                donde $G_{b,r}$ es la conductancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,r}^{+}$ y $\delta_{b,r}^{-}$ son
                variables utilizadas para representar el cálculo lineal del valor absoluto dentro del modelo matemático; $\alpha_{b,r}(k)$ y $\delta_{b,r}(k)$
                representan la pendiente y el valor del bloque de linealización de la diferencia angular $\left(\delta_{b} - \delta_{r} \right)$,
                respectivamente; $\Delta \delta_{b,r}$ representa el valor máximo que puede tomar la diferencia angular $\left(\delta_{b} - \delta_{r} \right)$.
                """)

        st.write("**Límites en el flujo de potencia en las líneas**")
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} + \frac{1}{2} \cdot q_{b,r,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
                que conecta los nodos $b$ y $r$.
                """)
    else:
        st.write("### Flujo de potencia DC")
        st.write("**Cálculo del flujo de potencia por cada línea**")
        st.write(r"""
                $$ p_{b,r,t}^{pf} = B_{b,r} \cdot \left(\delta_{b} - \delta_{r} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b}$ y $\delta_{r}$ representan el
                valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada, respectivamente.
                """)

        st.write("**Límites en el flujo de potencia en las líneas**")
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
                que conecta los nodos $b$ y $r$.
                """)

    st.write("### Restricciones sistemas de almacenamiento de energía basados en baterías")
    st.write("**Relación entre la potencia y energía de los SAEB**")
    st.write(r"""
            $$ e_{n,b,t} = e_{n,b,t-1}\cdot \left(1 - \eta_{n}^{SoC} \right) + \left( \eta^{ch}_{n} \cdot p_{n,b,t}^{ch} -
            \frac{P_{n,b,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
            b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$
            """)
    st.write(r"""
            donde $e_{n,b,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,t-1}$ es la cantidad de energía en el periodo anterior del SAEB;
            $p_{n,b,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,t}^{dc}$ es la potencia de descarga
            del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.
            """)

    st.write("**Límite de energía de los SAEB**")
    st.write(r"""
            $$ e_{n,b,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño del SAEB
            a instalar en términos de energía.
            """)

    st.write("**Límite de potencia de los SAEB**")
    st.write(r"""
            $$ p_{n,b,t}^{ch} \leq Z \cdot u_{n,b,t}^{sta} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{ch} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{dc} \leq Z \cdot \left(1 - u_{n,b,t}^{sta}\right) - Z \cdot \left(1 - u_{n,b}^{ins} \right) \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,t}^{dc} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $Z$ es un valor constante muy grande; $u_{n,b,t}^{sta}$ es una variable binaria que modela el comportamiento del SAEB $n$
            instalado en el nodo $b$ en cada instante de tiempo $t$; $u_{n,b}^{ins}$ es una variable binaria que determina si el SAEB $n$ es instalado
            en el nodo $b$.
            """)

    st.write("**Limite en el número de SAEB que se instalan**")
    st.write(r"""
            $$ u_{n,b,t}^{sta} \leq u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
            t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ E_{n,b}^{SAEB} \leq Z \cdot u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N},
            b \in \mathcal{B}$$
            """)
    st.write(r"""
            $$ \sum_{\mathcal{N}} u_{n,b}^{ins} \leq N_{max} \hspace{2mm} \forall n \in \mathcal{N},
            b \in \mathcal{B} $$
            """)
    st.write(r"""
            donde $N_{max}$ es el número máximo de SAEB que se pueden instalar.
            """)
    return ""

def text_formul_math_escen_sen(res_list):

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in res_list:
        st.write("## **Función Objetivo**")
        st.write(r"""
                $$ \begin{aligned}
                    \min \underbrace{\sum_{i \in \mathcal{I}}\sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(C_{i}^{dn}\cdot SD_{i,t} + C_{i}^{up}\cdot SU_{i,t}\right)}_{Costos\hspace{1mm}de\hspace{1mm}arranque/parada} \\
                    + \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                    E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
                    + \underbrace{\sum_{s \in \mathcal{S}} \pi_{s} \sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(p_{i,b,s,t}^{re}\cdot C_{i,s}^{gen} + \left(p_{i,b,s,t}^{re} - p_{i,d,s,t}^{id} \right)\cdot C_{t,s}^{MPO} \right)}_{Penalización} \\
                \end{aligned}$$
                """)
        st.write(r"""
                donde $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW; $C_{i}^{dn}\cdot SD_{i,t}$ y
                $C_{i}^{up}\cdot SU_{i,t}$ representan, respectivamente, los costos de arranque y parada de las unidades de generación; $P_{n,b}^{SAEB}$
                es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de conversión de potencia en \$/MW;
                $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh;
                $\pi_{s}$ es la probabilidad de ocurrencia de cada escenario; $p_{i,b,s,t}^{re}$ es la potencia de la simulación del despacho real,
                en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$; $p_{i,s,d,t}^{id}$ es la potencia
                que resulta del despacho ideal, en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$.
                Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.
                """)
    else:
        st.write("## **Función Objetivo**")
        st.write(r"""
                $$ \begin{aligned}
                    \min \underbrace{\sum_{n \in \mathcal{N}} \sum_{b \in \mathcal{B}} \left( P_{n,b}^{SAEB}\cdot C_{n}^{pot} +
                    E_{n}^{SAEB}\cdot C_{n,b}^{ene}\right)}_{Costos\hspace{1mm}por\hspace{1mm}SAEB} \\
                    + \underbrace{\sum_{s \in \mathcal{S}} \pi_{s} \sum_{j \in \mathcal{J}} \sum_{b \in \mathcal{B}}\sum_{t \in \mathcal{T}}
                    \left(p_{i,b,s,t}^{re}\cdot C_{i,s}^{gen} + \left(p_{i,b,s,t}^{re} - p_{i,d,s,t}^{id} \right)\cdot C_{t,s}^{MPO} \right)}_{Penalización} \\
                \end{aligned}$$
                """)
        st.write(r"""
                donde $C_{i}^{gen}$ es un vector que representa los costos de cada unidad de generación en \$/MW; $P_{n,b}^{SAEB}$
                es la capacidad de potencia en MW del SAEB $n$ en el nodo $b$; $C_{n}^{pot}$ es el costo del equipo de conversión de potencia en \$/MW;
                $E_{n}^{SAEB}$ es la capacidad de energía del SAEB en MWh; $C_{n,b}^{ene}$ es el costo de los equipos de almacenamiento en \$/MWh;
                $\pi_{s}$ es la probabilidad de ocurrencia de cada escenario; $p_{i,b,s,t}^{re}$ es la potencia de la simulación del despacho real,
                en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$; $p_{i,s,d,t}^{id}$ es la potencia
                que resulta del despacho ideal, en MW, entregada por el generador $i$ en el nodo $b$ para el escenario $s$ en el instante de tiempo $t$.
                Finalmente, $C_{t}^{MPO}$ es el máximo precio de oferta en el instante de tiempo $t$.
                """)

    #### Restricciones del sistema
    st.write("## **Restricciones**")

    st.write("### Restricciones del sistema")
    st.write("**Balance de Potencia**")

    if 'Pérdidas' in res_list:
        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,s,t}^{re}  - \sum_{(b,r) \in \mathcal{L}}
                    \left(p_{b,r,s,t}^{pf} + \frac{1}{2} q_{b,r,s,t}^{pf} \right) + \\ \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,s,t}^{dc} - p_{n,b,s,t}^{ch} \right) = D_{b,s,t}^{f} \hspace{2mm} \forall
                    b \in \mathcal{B}, s \in \mathcal{S}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{b,r,s,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ para el escenario $s$ en cada
                instante de tiempo $t$; $q_{b,r,s,t}^{pf}$ son las pérdidas de potencia asociadas a cada línea conectada entre los nodos $b$ y $r$ para
                el escenario $s$ en cada instante de tiempo $t$; $p_{n,b,s,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ para el
                escenario $s$ en el instante de tiempo $t$; $p_{n,b,s,t}^{ch}$ es la potencia de descarga del SAEB $n$ conectado al nodo $b$ para el
                escenario $s$ en el instante de tiempo $t$; $D_{b,s,t}^{r}$ la demanda real del sistema del día estudiado.
                """)
    else:
        st.write(r"""
                $$ \begin{aligned}
                    \sum_{i \in \mathcal{I}_{b}} p_{i,b,s,t}^{re} - \sum_{(b,r) \in \mathcal{L}}
                    p_{b,r,s,t}^{pf} + \sum_{n \in \mathcal{N}_{b}}
                    \left(p_{n,b,s,t}^{dc} - p_{n,b,s,t}^{ch} \right) = D_{b,s,t}^{f} \\ \hspace{2mm} \forall
                    b \in \mathcal{B}, s \in \mathcal{S}, t \in \mathcal{T}
                \end{aligned}$$
                """)
        st.write(r"""
                donde $p_{b,r,s,t}^{pf}$ es el flujo de potencia activa en cada línea conectada entre los nodos $b$ y $r$ para el escenario $s$ en cada
                instante de tiempo $t$; $p_{n,b,s,t}^{dc}$ es la potencia de carga del SAEB $n$ conectado al nodo $b$ para el
                escenario $s$ en el instante de tiempo $t$; $p_{n,b,s,t}^{ch}$ es la potencia de descarga del SAEB $n$ conectado al nodo $b$ para el
                escenario $s$ en el instante de tiempo $t$; $D_{b,s,t}^{r}$ la demanda real del sistema del día estudiado.
                """)

    st.write("### Límites en Generación")
    st.write(r"""
            $$ P_{i,s}^{min} \leq p_{i,b,s,t}^{re} \leq P_{i,s}^{max} \hspace{2mm} \forall
            i \in \mathcal{I}, b \in \mathcal{B}, s \in \mathcal{S}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $P_{i,s}^{min}$ y $P_{i,s}^{max}$ son los límites mínimos y máximos de la capacidad de generación térmica, respectivamente.
            """)

    if 'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)' in res_list:

        st.write("### Operación de Generadores Térmicos")

        st.write("**Rampas de generadores térmicos**")
        st.write(r"""
                $$ p_{i,s,t+1}^{re} - p_{i,s,t}^{re} \leq R_{i,s}^{up} \cdot x_{i,t} + SU_{i,t+1} \cdot P_{i,s}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, s \in \mathcal{S}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ p_{i,s,t}^{re} - p_{i,s,t+1}^{re} \leq R_{i}^{dn} \cdot x_{i,t} + SD_{i,t+1} \cdot P_{i,s}^{min} \hspace{2mm}
                \forall t, t + 1 \in \mathcal{T}, s \in \mathcal{S}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                donde $R_{i}^{up}$ y $R_{i}^{dn}$ son las rampas de potencia de subida y bajada de las unidades de generación térmica, respectivamente;
                $SU_{i,t+1}$ y $SD_{i,t+1}$ son las señales de encendido y apagado de las unidades térmicas, respectivamente; $x_{i,t}$ representa el
                estado programado para las unidades de generación térmica.
                """)

        st.write("**Variables binarias de operación de unidades térmicas**")
        st.write(r"""
                $$ SU_{i,t} - SD_{i,t} = x_{i,t} - x_{i,t-1} \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)
        st.write(r"""
                $$ SU_{i,t} + SD_{i,t} \leq 1 \hspace{2mm} \forall
                t \in \mathcal{T}, i \in \mathcal{I} $$
                """)

    if 'Pérdidas' in res_list:
        st.write("### Flujo de potencia DC y pérdidas")
        st.write("**Cálculo del flujo de potencia por cada línea**")
        st.write(r"""
                $$ p_{b,r,s,t}^{pf} = B_{b,r} \cdot \left(\delta_{b,s} - \delta_{r,s} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,s}$ y $\delta_{r,s}$ representan el
                valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada para el escenario $s$, respectivamente.
                """)

        st.write("**Cálculo de las pérdidas eléctricas de cada línea**")
        st.write(r"""
                $$ q_{b,r,s,t}^{pf} = G_{b,r} \cdot \left(\delta_{b,s} - \delta_{r,s} \right)^2 \hspace{2mm}
                \forall (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                $$ \delta_{b,r,s}^{+} + \delta_{b,r,s}^{-} = \sum_{k=1}^{K} \delta_{b,r,s}(k)
                \hspace{2mm} k = 1,...,K $$
                """)
        st.write(r"""
                $$ \alpha_{b,r}(k) = (2k-1)\cdot \Delta \delta_{b,r} \hspace{2mm}
                k = 1, ... , K $$
                """)
        st.write(r"""
                $$ q_{b,r,s,t}^{pf} = G_{b,r}\cdot \sum_{k=1}^{K} \alpha_{b,r}(k)\cdot \delta_{b,r,s}(k)
                \hspace{2mm} \forall (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $G_{b,r}$ es la conductancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,r,s}^{+}$ y $\delta_{b,r,s}^{-}$ son
                variables utilizadas para representar el cálculo lineal del valor absoluto dentro del modelo matemático; $\alpha_{b,r}(k)$ y $\delta_{b,r,s}(k)$
                representan la pendiente y el valor del bloque de linealización de la diferencia angular $\left(\delta_{b,s} - \delta_{r,s} \right)$,
                respectivamente; $\Delta \delta_{b,r}$ representa el valor máximo que puede tomar la diferencia angular $\left(\delta_{b,s} - \delta_{r,s} \right)$.
                """)

        st.write("**Límites en el flujo de potencia en las líneas**")
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,s,t}^{pf} + \frac{1}{2} \cdot q_{b,r,s,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
                que conecta los nodos $b$ y $r$.
                """)
    else:
        st.write("### Flujo de potencia DC")
        st.write("**Cálculo del flujo de potencia por cada línea**")
        st.write(r"""
                $$ p_{b,r,s,t}^{pf} = B_{b,r} \cdot \left(\delta_{b,s} - \delta_{r,s} \right) \hspace{2mm} \forall
                (b,r) \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $B_{b,r}$ es la susceptancia de la línea que se conecta entre los nodos $b$ y $r$; $\delta_{b,s}$ y $\delta_{r,s}$ representan el
                valor del ángulo de los nodos que son conectados por la línea, ángulo de salida y de llegada para el escenario $s$, respectivamente.
                """)

        st.write("**Límites en el flujo de potencia en las líneas**")
        st.write(r"""
                $$ -P_{b,r}^{max} \leq p_{b,r,s,t}^{pf} \leq P_{b,r}^{max}
                \hspace{2mm} \forall l \in \mathcal{L}, s \in \mathcal{S}, t \in \mathcal{T} $$
                """)
        st.write(r"""
                donde $-P_{b,r}^{max}$ y $P_{b,r}^{max}$ son los límites mínimos y máximos de flujo de potencia en el flujo de potencia activa de la línea
                que conecta los nodos $b$ y $r$.
                """)

    st.write("### Restricciones sistemas de almacenamiento de energía basados en baterías")
    st.write("**Relación entre la potencia y energía de los SAEB**")
    st.write(r"""
            $$ e_{n,b,s,t} = e_{n,b,s,t-1}\cdot \left(1 - \eta_{n}^{SoC} \right) + \left( \eta^{ch}_{n} \cdot p_{n,b,s,t}^{ch} -
            \frac{P_{n,b,s,t}^{dc}}{\eta^{dc}_{n}} \right)\cdot \Delta t \hspace{2mm} \forall
            b \in \mathcal{B}, n \in \mathcal{N}, t \in \mathcal{T} \hspace{10mm} $$
            """)
    st.write(r"""
            donde $e_{n,b,s,t}$ es la cantidad de energía actual del SAEB; $e_{n,b,s,t-1}$ es la cantidad de energía en el periodo anterior del SAEB;
            $p_{n,b,s,t}^{ch}$ es la potencia de carga del SAEB; $\eta^{ch}_{n}$ es la eficiencia de carga del SAEB; $P_{n,b,s,t}^{dc}$ es la potencia de descarga
            del SAEB; $\eta^{dc}_{n}$ es la eficiencia de carga del SAEB.
            """)

    st.write("**Límite de energía de los SAEB**")
    st.write(r"""
            $$ e_{n,b,s,t} \leq E_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $E_{n,b}^{SAEB}$ es la capacidad energética nominal del SAEB y constituye una variable de decisión que determina el tamaño del SAEB
            a instalar en términos de energía.
            """)

    st.write("**Límite de potencia de los SAEB**")
    st.write(r"""
            $$ p_{n,b,s,t}^{ch} \leq Z \cdot u_{n,b,t}^{sta} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,s,t}^{ch} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,s,t}^{dc} \leq Z \cdot \left(1 - u_{n,b,t}^{sta}\right) - Z \cdot \left(1 - u_{n,b}^{ins} \right) \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ p_{n,b,s,t}^{dc} \leq P_{n,b}^{SAEB} \hspace{2mm} \forall
            n \in \mathcal{N}, b \in \mathcal{B}, t \in \mathcal{T} $$
            """)
    st.write(r"""
            donde $Z$ es un valor constante muy grande; $u_{n,b,t}^{sta}$ es una variable binaria que modela el comportamiento del SAEB $n$
            instalado en el nodo $b$ en cada instante de tiempo $t$; $u_{n,b}^{ins}$ es una variable binaria que determina si el SAEB $n$ es instalado
            en el nodo $b$.
            """)

    st.write("**Limite en el número de SAEB que se instalan**")
    st.write(r"""
            $$ u_{n,b,t}^{sta} \leq u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N}, b \in \mathcal{B},
            t \in \mathcal{T} $$
            """)
    st.write(r"""
            $$ E_{n,b}^{SAEB} \leq Z \cdot u_{n,b}^{ins} \hspace{2mm} \forall n \in \mathcal{N},
            b \in \mathcal{B}$$
            """)
    st.write(r"""
            $$ \sum_{\mathcal{N}} u_{n,b}^{ins} \leq N_{max} \hspace{2mm} \forall n \in \mathcal{N},
            b \in \mathcal{B} $$
            """)
    st.write(r"""
            donde $N_{max}$ es el número máximo de SAEB que se pueden instalar.
            """)
    return ""

def get_system_data(dates):

    dispCome_list = []
    demada_list = []
    PrecioOfe_list = []
    minop_list = []
    MPO_list = []
    df_AGC_all_ = []

    consult = pxm.ReadDB()

    k = 0

    for fecha in dates:

        #### Disponibilidad Comercial

        df_dispCome = consult.request_data("DispoCome", 0, fecha, fecha)
        df_Recursos = consult.request_data("ListadoRecursos", 0, fecha, fecha)
        df_Recursos = df_Recursos.rename(columns={'Values_Code': 'Values_code'})
        df_dispCome_simp = pd.merge(left=df_dispCome, right=df_Recursos[['Values_code','Values_Name']], left_on='Values_code', right_on='Values_code').rename(columns={'Values_Name':'Plantas'})
        df_dispCome_simp['sce'] = k + 1
        df_dispCome_simp = df_dispCome_simp.set_index(['Plantas','sce']).drop(['Id','Values_code','Date'], axis=1).astype(float).div(1e3)
        df_dispCome_simp.columns = [x for x in range(1,25)]

        #### Demanda por hora

        df_demanda = consult.request_data("DemaCome", 0, fecha, fecha)
        df_demanda = df_demanda.drop(['Id','Values_code','Date'],axis=1).astype(float).div(1e3)
        df_demanda.columns = [x for x in range(1,25)]
        df_demanda = df_demanda.squeeze()

        #### Máximo Precio de Oferta Nacional
        df_MPO = consult.request_data("MaxPrecOferNal", 0, fecha, fecha)
        df_MPO = df_MPO.drop(['Id','Values_code','Date'], axis=1).astype(float).mul(1e3)
        df_MPO.columns = [x for x in range(1,25)]

        #### Precios de oferta y arranque/parada

        ## Precios de oferta

        df_PrecioOfe = consult.request_data("PrecOferDesp", 0, fecha, fecha)
        df_PrecioOfe_simp = pd.merge(left=df_PrecioOfe, right=df_Recursos[['Values_code','Values_Name']], left_on='Values_code', right_on='Values_code').rename(columns={'Values_Name':'Plantas'})
        df_PrecioOfe_simp['sce'] = k + 1
        df_PrecioOfe_simp = df_PrecioOfe_simp.set_index(['Plantas','sce']).drop(['Id','Values_code','Date'], axis=1).astype(float)
        df_PrecioOfe_simp = df_PrecioOfe_simp.append(df_dispCome_simp)
        df_PrecioOfe_simp = df_PrecioOfe_simp.groupby(df_PrecioOfe_simp.index).first()
        df_PrecioOfe_simp.index = pd.MultiIndex.from_tuples(df_PrecioOfe_simp.index)
        df_PrecioOfe_simp.index.names = ['Plantas', 'sce']
        df_PrecioOfe_simp = df_PrecioOfe_simp[['Values_Hour01']].rename(columns={'Values_Hour01':'Precio'}).fillna(0).mul(1e3)

        ## Precios de arranque /parada

        fecha = fecha.strftime('%d-%m-%Y')

        agents_all_, AGC_all_ = read_files_PAP([fecha])

        PAPUSD_all, PAP_all, MO_all, df_AGC_, OrganizeTime = organize_file_agents_PAP_MO(agents_all_, AGC_all_)

        for i,l in df_PrecioOfe_simp.index:
            for j in PAP_all[0].keys():
                if i.replace(' ','') == j:
                    df_PrecioOfe_simp.loc[i,'PAPUSD'] = PAPUSD_all[0][i.replace(' ','')][0]
                    df_PrecioOfe_simp.loc[i,'PAP'] = PAP_all[0][i.replace(' ','')][0]
        df_PrecioOfe_simp = df_PrecioOfe_simp.fillna(0)

        #### Mínimos operativos
        df_minop = pd.DataFrame(index=df_dispCome_simp.index, columns=df_dispCome_simp.columns).fillna(0)

        for i,l in df_minop.index:
            for j in MO_all[0].keys():
                for t in df_dispCome_simp.columns:
                    if i.replace(' ','') == j:
                        df_minop.loc[(i,l),t] = MO_all[0][i.replace(' ','')][t-1]

        dispCome_list.append(df_dispCome_simp)
        demada_list.append(df_demanda)
        PrecioOfe_list.append(df_PrecioOfe_simp)
        minop_list.append(df_minop)
        MPO_list.append(df_MPO)
        df_AGC_all_.append(df_AGC_)

        k += 1

    return dispCome_list, minop_list, demada_list, PrecioOfe_list, MPO_list, df_AGC_all_

def dashboard_DLOr(data1):

    ##

    st.markdown("## Parámetros seleccionados para la simulación")

    ## Selección de archivo del sistema

    st.sidebar.markdown("### Ingrese los parámetros de simulación ")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    file_system = st.sidebar.file_uploader("Seleccione el archivo con el sistema a simular:", type=['csv','xlsx'], help='Se admiten archivos .csv o .xlsx')

    ## Selección de tecnología de SAE

    Eff, degra, autoD, DoD, costP, costE, Aa, Bb, ciclos = bat_param(data1,0)

    ## Vida útil del SAEB

    vida_util = st.sidebar.number_input("Ingrese el tiempo estimado de operación del proyecto [año(s)]", min_value=1, max_value=1000000, help='Valores típicos: 10, 15 o 20 años')
    st.write("La vida útil del SAEB es: " + str(vida_util) + ' años')

    ## Horizonte de simulación

    time_sim = 24
    st.write("El horizonte de simulación es: " + str(time_sim) + ' horas')

    ##

    st.sidebar.write('### Ingrese características del modelo a aplicar')

    ## Formulación Matemática

    study_type = st.sidebar.selectbox('Modelo:', ('Determinista', 'Escenarios'))

    sensitivity = st.sidebar.selectbox('Añadir/Quitar restricciones:', ('Sí', 'No'))

    if sensitivity == 'No':
        model_sen = 'No'
    else:
        restr_list_compl = ['Límites de generación', 'Flujos de potencia DC', 'Límites en el Flujo por las líneas',
                            'Balance de potencia', 'Sistemas de Almacenamiento de Energía',
                            'Operación de unidades térmicas detallada (rampas, Tiempos mínimos de encendido/apagado)', 'Pérdidas']
        restr_list_simp = restr_list_compl[0:5]
        st.sidebar.write('**Restricciones base**')
        st.sidebar.write('Límites de generación\n\n', 'Flujos de potencia DC\n\n', 'Límites en el Flujo por las líneas\n\n', 'Balance de potencia\n\n', 'Sistemas de Almacenamiento de Energía')
        st.sidebar.write('**Restricciones adicionales**')
        restr_list = st.sidebar.multiselect(label='', options=sorted(restr_list_compl[5:7]), default=restr_list_compl[5:7])
        model_sen = restr_list

    if study_type == 'Determinista':

        st.write('### Formulación matemática usada')

        formulacion = st.expander(label='Formulación Matemática', expanded=False)

        if sensitivity == 'No':
            with formulacion:
                st.write(text_formul_math_deter())
                st.write("")
        else:
            with formulacion:
                st.write(text_formul_math_deter_sen(model_sen))
                st.write("")

        BESS_limit = st.sidebar.selectbox("Limitar número de SAEB", ('Sí','No'))

        if BESS_limit == 'Sí':
            BESS_number = st.sidebar.number_input("Número máximo de SAEB", min_value=1, max_value=1000000)
        else:
            BESS_number = 50

    elif study_type == 'Escenarios':

        st.write('### Modelo de formulación matemática usado')

        formulacion = st.expander(label='Formulación Matemática', expanded=False)

        if sensitivity == 'No':
            with formulacion:
                st.write(text_formul_math_escen())
                st.write("")
        else:
            with formulacion:
                st.write(text_formul_math_escen_sen(model_sen))
                st.write("")

        scenarios_number = st.sidebar.number_input('Ingrese el número de escenarios', min_value=2, max_value=10)

        BESS_limit = st.sidebar.selectbox("Limitar número de SAEB", ('Sí','No'))

        if BESS_limit == 'Sí':
            BESS_number = st.sidebar.number_input("Número máximo de SAEB", min_value=1, max_value=1000000)
        else:
            BESS_number = 50

    ## Tipo de modelo

    st.write('El tipo de modelo seleccionado es: ', study_type)

    ##

    st.sidebar.write('### Simulación')

    if study_type == 'Determinista':

        dates = []
        date_ = st.sidebar.date_input("Ingrese el día que vaya a simular", value = date.today() - timedelta(days=30), max_value = date.today() - timedelta(days=30))
        dates = [date_]

        data_from = st.sidebar.selectbox("Seleccione la fuente de información de las variables del mercado de energía:", ("XM (Colombia)","Otro"))

    else:

        dates = []

        for i in range(1,scenarios_number+1):
            date_ = st.sidebar.date_input("Ingrese la fecha para obtener los datos del escenario número {}".format(i), key=str(i), value = date.today() - timedelta(days=30), max_value = date.today() - timedelta(days=30))
            dates.append(date_)

        data_from = st.sidebar.selectbox("Seleccione la fuente de información de las variables del mercado de energía:", ("XM (Colombia)","Otro"))

    ## Solver

    solver = st.sidebar.selectbox("Seleccione el tipo de Solver", ['CPLEX','GLPK'])

    if solver == 'CPLEX':

        st.write("El solucionador seleccionado fue: " + solver)

    else:

        st.write("El solucionador seleccionado fue: " + solver)

    ## Correr función de optimización

    def run_dim_size_ReS():

        with st.spinner('Descargando información de precios de oferta, disponibilidad de unidades de generación y demanda de energía...'):
        #     df_disp, df_minop, df_demandaSIN_all, df_ofe, df_MPO_all, df_AGC_all  = get_system_data(dates)

            actual_path = os.getcwd()
        #     db_files = os.path.join(actual_path, 'Casos_estudio/loc_size')

        #     despacho_tipo = 'Acorde a disponibilidad Real'

        # with st.spinner('Ejecutando simulación del despacho ideal...'):
        #     df_PI, tiempo_DI, name_file_DI  = DISC(df_demandaSIN_all, df_disp, df_minop, df_ofe, 1532530, solver)

        db_files = os.path.join(actual_path, 'Casos_estudio/gen_db.xlsx')

        # with st.spinner('Ejecutando simulación del despacho real...'):
        #     df_P_size, df_E_size, name_file_DR, tiempo_DR, map_data_2, map_data_1, df_cost_DR, Output_data = DPRLNSC_H(file_system, db_files, df_demandaSIN_all, df_disp, df_minop, despacho_tipo, df_ofe, dates, df_PI, df_MPO_all,
        #                                                                     Eff, DoD, 0.2, time_sim, vida_util, costP, costE, autoD, BESS_number, BESS_limit, model_sen, solver)

        ## Impresión de resultados

        st.write("Resultados:")
        # st.write("Tiempo de simulación: " + str(tiempo_DI+tiempo_DR))

        st.write("Nodos en donde se instala SAEB: ")
        dict_result_loc = {}

        # for i in df_P_size.index:

        #     result_loc = []

        #     if df_P_size.loc[i] != 0:

        #         result_loc.append(round(df_P_size.loc[i],1))
        #         result_loc.append(round(df_E_size.loc[i],1))
        #         dict_result_loc[i] = result_loc

        ## ----- eliminar -----*

        file_name = 'Resultados/results_David.xlsx'

        df_P_size = pd.read_excel(file_name, sheet_name='P_size', header=0, index_col=0)
        df_E_size = pd.read_excel(file_name, sheet_name='E_size', header=0, index_col=0)
        map_data_2 = pd.read_excel(file_name, sheet_name='Branch', header=0, index_col=0)
        map_data_1 = pd.read_excel(file_name, sheet_name='Bus', header=0, index_col=0)
        df_cost_DR = pd.read_excel(file_name, sheet_name='cost', header=0, index_col=0)

        ## ----- eliminar -----*

        df_dict_result_loc = pd.DataFrame.from_dict(dict_result_loc, orient='index', columns=['Potencia [MW]', 'Energía [MWh]'])
        st.dataframe(df_dict_result_loc.style.format(thousands=',', precision=1, decimal='.'))

        cost_ope = df_cost_DR.rename(columns={'1': 'Valor función objetivo [$COP]'})
        st.write("Valor función objetivo")
        st.dataframe(cost_ope.style.format(thousands=',', precision=1, decimal='.'))

        # graph_results_res(Output_data, model_sen)

        point_size = [0] * df_P_size.shape[0]
        point_colors = [0] * df_P_size.shape[0]


        # for i in range(df_P_size.shape[0]):

        #     if df_P_size.loc[df_P_size.index[i]] > 0:

        #         point_size[i] = 40000
        #         point_colors[i] = [10, 230, 10]

        #     else:

        #         point_size[i] = 15000
        #         point_colors[i] = [230, 158, 10]

        ## ----- eliminar -----*

        for i in range(df_P_size.shape[0]):

                if df_P_size.loc[df_P_size.index[i],0] > 0:

                        point_size[i] = 40000
                        point_colors[i] = [10, 230, 10]

                else:

                        point_size[i] = 15000
                        point_colors[i] = [230, 158, 10]

        ## ----- eliminar -----*

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

        if path.exists("Resultados/resultados_size_loc_res.xlsx"):

            remove("Resultados/resultados_size_loc_res.xlsx")

        run_dim_size_ReS()