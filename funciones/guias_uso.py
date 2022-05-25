import streamlit as st
import base64
from io import BytesIO
import os

def text_ope():
    return """

    #### Opciones barra lateral:
    * **Seleccione el archivo con el sistema a simular:** El usuario debe seleccionar un archivo en formato .xlsx o .csv que contenga \
    la información del sistema a simular.
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación \
    cargará valores predefinidos para cada tecnología. También se da la opción de ingresar parámetros manualmente.
    * **Tiempo estimado de operación del proyecto:** El usuario debe ingresar la vida útil del activo en años, con el fin de calcular el valor equivalente \
    del activo en el horizonte de simulación seleccionado.
    * **Ingrese el horizonte de simulación:** El usuario debe ingresar el horizonte de simulación en horas.
    * **Añadir/Quitar restricciones al modelo base:** El usuario puede añadir o quitar diferentes restricciones para determinar el impacto de cada una de \
        ellas en el modelo matemático. Por defecto, y para garantizar el correcto funcionamiento del modelo matemático, algunas restricciones \
        están siempre activas. Estas restricciones son denominadas, en la barra lateral, como *Restricciones Base*. El usuario tiene las siguientes opciones.
        * **Sí:** Se pueden añadir/quitar restricciones al modelo matemático.
        * **No:** Se despliegan opciones con modelos predefinidos.
    * **Restricciones adicionales (solo aplica si se selecciona la opción *"Sí"* en *Añadir/Quitar restricciones al modelo base*):** El usuario puede añadir \
    restricciones adicionales al modelo de optimización para determinar el impacto de cada una de ellas.
    * **Tipo de Modelo (Solo aplica si se selecciona la opción *"No"* en *Añadir/Quitar restricciones*):** El usuario puede seleccionar entre \
    los siguientes modelos predefinidos.
        * **Completo:** Modelo de optimización que contiene todas las restricciones del Despacho Económico, Unit Commitment, Generación \
            hidráulica, pérdidas, etc. Para más información de las restricciones consulte el menú llamado *Formulación Matemática*.
        * **Simplificado:** Modelo de optimización en el que no se tiene en cuenta algunas restricciones para hacer que el modelo sea \
            resuelto más rápidamente, sin perjudicar su validez. Para más información de las restricciones consulte el menú \
            llamado *Formulación Matemática*.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).
    """

def text_rest():
    return """

    #### Opciones barra lateral:
    * **Seleccione el archivo con el sistema a simular:** El usuario debe seleccionar un archivo en formato .xlsx o .csv que contenga \
    la información del sistema a simular.
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación \
    cargará valores predefinidos para cada tecnología. También se da la opción de ingresar parámetros manualmente.
    * **Tiempo estimado de operación del proyecto:** El usuario debe ingresar la vida útil del activo en años, con el fin de calcular el valor equivalente \
    del activo en el horizonte de simulación seleccionado.
    * **Tipo de Modelo:** El usuario puede seleccionar entre un modelo determinista o uno basado en escenarios.
        * **Determinista:** Modelo de optimización en el cual unicamente se determina el tamaño y ubicación del SAEB teniendo un único \
        día de estudio.
        * **Escenarios:** Modelo de optimización en el cual se selecionan diferentes días de estudio (escenarios) para determinar el tamaño y ubicación\
        del SAEB, con el fin de considerar el impacto en los resultados de aspectos como la hidrología y la demanda en diferentes epocas del año.
    * **Añadir/Quitar restricciones al modelo base:** El usuario puede añadir o quitar diferentes restricciones para determinar el impacto de cada una de \
        ellas en el modelo matemático. Por defecto, y para garantizar el correcto funcionamiento del modelo matemático, algunas restricciones \
        están siempre activas. Estas restricciones son denominadas, en la barra lateral, como *Restricciones Base*. El usuario tiene las siguientes opciones.
        * **Sí:** Se pueden añadir/quitar restricciones al modelo matemático.
        * **No:** Se despliegan opciones con modelos predefinidos.
    * **Restricciones adicionales (Solo aplica si se selecciona la opción *"Sí"* en *Añadir/Quitar restricciones al modelo base*):** El usuario puede añadir \
    restricciones adicionales al modelo de optimización para determinar el impacto de cada una de ellas.
    * **Ingrese el(los) día(s) que vaya a simular:** El usuario debe ingresar el día o los días que se vayan a simular (el número de días\
    depende del tipo de modelo seleccionado y de la cantidades de escenarios seleccionados).
    * **Descargar valores de XM (Solo aplica si se selecciona la opción *Determinista* en *Tipo de Modelo*):** El usuario debe seleccionar si se descargan los datos de la página de XM o se utilizan archivos locales.
        * **Sí:** Los valores descargados corresponden a las ofertas (disponibilidad, precio oferta, precio arranque/parada) realizadas el \
        día anterior, para el día seleccionado.
        * **No:** Los valores son obtenidos de una base de datos local. Estos valores corresponden a las disponibilidades reales, precios de \
        arranque/parada y precios de oferta publicados luego del día seleccionado.
    * **Limitar número de SAEB:** El usuario debe seleccionar si se limita o no el número de SAEB que se puedan instalar.
    * **Número máximo de SAEB (Solo aplica si se selecciona la opción *Sí* en *Limitar número de SAEB*):** El usuario debe indicar la cantidad\
    máxima de SAEB que pueden ser instalados.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).

    **NOTA: Por defecto el horizonte de simulación, para cada escenario, en estos modelos de optimización (Determinista y Escenarios) es de \
    24 horas**.
    """

def text_arb_loc():
    return """

    ## Opciones barra lateral:
    ### Parámetros del SAE:
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación \
    cargará valores predefinidos para cada tecnología. Si se selecciona la opción "Nuevo" se podrán ingresar los parámetros manualmente, con \
    información personalizada.
    * **Seleccione el tipo limitación para el dimensionamiento:** El usuario puede seleccionar el tipo de limitación que desea tener en cuenta para\
    el dimensionamiento óptimo del SAE
        * **Sin limitación:** No se consideran restricciones de tamaño o presupuesto para el dimensionamiento óptimo del SAE
        * **Potencia/Energía:** El usuario debe suministrar los límites máximos de capacidad potencia y energía para el dimensionamiento del SAE.\
        Estas limitaciones de potencia y energía pueden estar sujetas a la capacidad máxima de transporte de energía que tiene la red eléctrica en el nodo\
        que se desea instalar el SAE.
        * **Presupuesto:** El usuario debe suministrar el monto máximo de inversión capital que se espera para el proyecto. La cantidad  suministrada debe estar en USD.
    * **Tiempo estimado de operación del proyecto:** El usuario debe ingresar la vida útil del activo en años con el fin de calcular el valor equivalente \
    del activo en el horizonte de simulación seleccionado.
    ### Parámetros de la simulación:
    * **Seleccione la fuente de información de las variables del mercado de energía:** El usuario debe seleccionar la fuente de información de los\
        precios de energía eléctrica.
        * **XM (Colombia):** Se extraen las variables del mercado de energía directamente de la base de datos del operador del sistema eléctrico colombiano (XM).
        * **Otro:** El usuario deberá cargar el archivo (csv, xlsx) con los datos del precio de la energía eléctrica para el horizonte de simulación deseado.\
        El formato base para el archivo se encuentra al final de la guía de uso.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).
    ### Formato base:
    """
    
    
    

def text_arb_op():
    return """

    ## Opciones barra lateral:
    ### Parámetros del SAE:
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación \
    cargará valores predefinidos para cada tecnología. Si se selecciona la opción "Nuevo" se podrán ingresar los parámetros manualmente, con \
    información personalizada.
    * **Ingrese el tamaño del PCS [MW]:** El usuario debe seleccionar el tamaño en capacidad de potencia del SAE que desea simular.
    * **Ingrese el tamaño del Almacenamiento [MWh] :** El usuario debe seleccionar el tamaño en capacidad de almacenamiento de energía del SAE que desea simular.
    ### Parámetros de la simulación:
    * **Seleccione la fuente de información de las variables del mercado de energía:** El usuario debe seleccionar la fuente de información de los\
        precios de energía eléctrica.
        * **XM (Colombia):** Se extraen las variables del mercado de energía directamente de la base de datos del operador del sistema eléctrico colombiano (XM).
        * **Otro:** El usuario deberá cargar el archivo (csv, xlsx) con los datos del precio de la energía eléctrica para el horizonte de simulación deseado.\
        El formato base para el archivo se encuentra al final de la guía de uso.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).
    ### Formato base:
    """

def text_MEM():
    return """

    ## Opciones barra lateral:
    ### Modelo del Mercado de Energía Mayorista (MEM):
    * **Seleccione el modelo de MEM que desea simular:** El usuario puede seleccionar el tipo de Mercado de Energía Mayorista (MEM) que desea tener en cuenta para\
    la simulación de la operación óptima del SAE.
        * **Despacho AGC:** Se realizará el despacho óptimo uninodal para suplir la demanda de reserva secundaria de energía necesarias (Holgura) para asegurar la operación segura del sistema.\
                            Únicamente se tendrán en cuenta las plantas u/o unidades habilitadas para suministrar dicho servicio.
        * **Despacho Ideal:** Se realizará el despacho óptimo uninodal de los diferentes recursos de generación disponibles en el sistema para suplir la demanda de energía total del sistema.\
                            Dentro de este despacho óptimo se tienen en cuenta restricciones propias de las plantas de generación como mínimos y máximos técnicos, rampas, tiempos de encendido y apagado, costos de operación y costos de arranque y parada.
        * **Despacho AGC + Ideal:** Se realizará el despacho óptimo uninodal de dos etapas, en donde la primera etapa corresponde al despacho AGC y una segunda etapa será el despacho ideal de los diferentes recursos de generación disponibles en el sistema.\
                            Este modelo se asemeja al esquema actual del despacho de energía en el sistema eléctrico colombiano.
        * **Despacho programado:** Se realizará el despacho óptimo de dos etapas, en donde la primera etapa corresponderá al despacho AGC y una segunda etapa será el despacho ideal de los diferentes recursos de generación disponibles en el sistema.\
                            Para este modelo se tienen en cuenta las restricciones de red (límites de flujo de potencia) por medio de un flujo de potencia DC.
        Estas limitaciones de potencia y energía pueden estar sujetas a la capacidad máxima de transporte de energía que tiene la red eléctrica en el nodo\
        que se desea instalar el SAE.
        * **Co-optimización:** Se realizará el despacho óptimo del mercado de energía y del mercado de reserva secundaria de manera simultánea. Para este caso el usuario tendrá la opción de ajustar las restricciones del modelo, en donde podrá agregar o eliminar restricciones\
                            relacionadas con los servicios complementarios (regulación primaria y secundaria de frecuencia), degradación del SAE y flujo de potencia DC.
    
    ### Parámetros del sistema:
    * **Seleccione la fuente de información de las variables del mercado de energía:** El usuario debe seleccionar la fuente de información de los\
        precios de energía eléctrica.
        * **XM (Colombia):** Se extraen las variables del mercado de energía directamente de la base de datos del operador del sistema eléctrico colombiano (XM).\
                            Para esta opción se toma el modelo base los elementos del sistema eléctrico colombiano (2020) con una tensión de operación nominal mayor o igual a 220 kV.
        * **Otro:** El usuario deberá cargar el archivo (csv, xlsx) con los parámetros del sistema eléctrico para el horizonte de simulación deseado.\
        El formato base para el archivo se encuentra al final de la guía de uso.
    * **¿Desea modificar las restricciones del modelo?:** El usuario podrá modificar las restricciones que componen el modelo.
        * **Sí:** El usuario podrá agregar o eliminar las siguientes restricciones:
            * **RPF generadores:** Se agrega la restricción de obligación que tiene los generadores despachados centralmente de suministrar una reserva primaria de frecuencia (RPF) igual o mayor al 3% de la energía despachada en el mercado de energía:
            * **Degradación SAE:** Se tiene en cuenta la curva de degradación (# ciclos vs DoD) que tienen los sistemas de almacenamiento de energía 
            * **Flujos de potencia DC:** Se consideran dentro del modelo los límites de flujo de potencia que tiene el sistema de potencia que se desea analizar. Estos límites se determinan por medio de un flujo de potencia DC.
            * **RoCoF sistema:** Se limita la tasa de cambio de frecuencia en el tiempo ante una contingencia igual a holgura solicitada por el sistema. Esta restricción se incluye con el fin de evitar grandes esfuerzos de los generadores síncronos.
            * **Fnadir sistema:** Se limita la frecuencia mínima alcanzada por el sistema ante una contingencia igual a holgura solicitada por el sistema. Esta restricción se incluye con el objetivo de evitar la activación de los esquemas de desconexión automática de carga.
            * **Fqss sistema:** Se limita la frecuencia mínima alcanzada durante el estado Quasi-estable de la regulación primaria de frecuencia
        * **No:** En el modelo se tendrá en cuenta todas las restricciones planteadas anteriormente.
    
    ### Parámetros del SAE:
    * **¿Desea agregar un SAE al modelo?:** El usuario podrá agregar un SAE al sistema y modelo que va a simular.
        * **Sí:** El usuario podrá agregar un SAE al modelo a simular:
            * **Nombre del SAE:** Se agrega el nombre del SAE que desea incluir en el modelo.
            * **Ingrese el nodo de ubicación de SAE:** El usuario debe seleccionar la ubicación del SAE dentro del sistema de potencia cargado en el modelo.
            * **Flujos de potencia DC:** Se consideran dentro del modelo los límites de flujo de potencia que tiene el sistema de potencia que se desea analizar. Estos límites se determinan por medio de un flujo de potencia DC.
            * **Ingrese el tamaño en potencia [MW]:** El usuario debe seleccionar el tamaño en capacidad de potencia del SAE que desea simular.
            * **Ingrese el tamaño en energía [MWh]:** El usuario debe seleccionar el tamaño en capacidad de almacenamiento de energía del SAE que desea simular.
            * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación
                cargará valores predefinidos para cada tecnología. Si se selecciona la opción "Nuevo" se podrán ingresar los parámetros manualmente, con
                información personalizada.
        * **No:** No se agregan SAE al modelo.
    
    ### Parámetros de la simulación:
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).    
    ### Formato base:
    """

def text_opeXM_v1():
    return r"""

    ## Opciones barra lateral:
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación
    cargará valores para la autodescarga predefinidos para cada tecnología. También se da la opción de ingresar parámetros manualmente. Por
    la naturaleza de este problema en especifico, los demás parámetros del SAE, como los estados de carga mínimo y máximo o la eficiencia,
    se deben cargar manualmente a la aplicación.
    * **Periodo inicial previo a un bloque de descarga:** El usuario debe ingresar la hora inicial del conjunto previo a un bloque de
    carga/descarga del SAE.
    * **Periodo final previo a un bloque de descarga:** El usuario debe ingresar la hora final del conjunto previo a un bloque de
    carga/descarga del SAE.
    * **Periodo inicial de un bloque de carga/descarga:** El usuario debe ingresar la hora inicial del periodo de carga/descarga del SAE.
    * **Periodo final de un bloque de carga/descarga:** El usuario debe ingresar la hora final del periodo de carga/descarga del SAE.
    * **Número de SAE a considerar:** El usuario debe ingresar el número de SAE que se van a considerar en la simulación. Por defecto, este
    valor es 1.
    * **Potencia máxima de carga del SAE n:** El usuario debe ingresar la potencia máxima de carga de cada SAE $n$ que vaya a ser considerado
    en la simulación.
    * **Potencia máxima de descarga del SAE n:** El usuario debe ingresar la potencia máxima de descarga de cada SAE $n$ que vaya a ser
    considerado en la simulación.
    * **Energía del SAE n:** El usuario debe ingresar la energía máxima de cada SAE que vaya a ser considerado en la simulación.
    * **Seleccione el tipo de tecnologí­a del SAE:**
    * **Eficiencia global del SAE n:** El usuario debe ingresar la eficiencia global (round trip efficiency) de cada SAE que vaya a ser
    considerado en la simulación.
    * **Estado de carga mínimo del SAE:** El usuario debe ingresar el estado de carga mínimo de cada SAE que vaya a ser considerado en la
    simulación.
    * **Estado de carga máximo del SAE:** El usuario debe ingresar el estado de carga máximo de cada SAE que vaya a ser considerado en la
    simulación.
    * **Valor de carga requerido del SAE n:** El usuario debe ingresar el valor requerido de carga en los periodos de carga/descarga de cada
    SAE. Consultar el menú desplegable *NOTA IMPORTANTE* para seleccionar adecuadamente el valor de este parámetro.
    * **Valor de descarga requerido del SAE n:** El usuario debe ingresar el valor requerido de descarga en los periodos de carga/descarga de cada
    SAE. Consultar el menú desplegable *NOTA IMPORTANTE* para seleccionar adecuadamente el valor de este parámetro.
    * **Día de simulación:** El usuario debe ingresar el día que se vaya a simular.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).

    **NOTAS:**
    * **Por defecto el horizonte de simulación es de 24 horas**

    * **La formulación descrita en esta sección difiere en algunos aspectos con la mostrada en el documento base de XM, estos cambios
    se realizaron para ajustar el modelo original, ya que presentaba fallos en las unidades de algunos parámetros.**
    """

def text_opeXM_v2():
    return r"""

    ## Opciones barra lateral:
    * **Seleccione el tipo de tecnología de SAE:** El usuario debe seleccionar el tipo de tecnología a utilizar en el estudio, la aplicación
    cargará valores para la autodescarga predefinidos para cada tecnología. También se da la opción de ingresar parámetros manualmente. Por
    la naturaleza de este problema en especifico, los demás parámetros del SAE, como los estados de carga mínimo y máximo o la eficiencia,
    se deben cargar manualmente a la aplicación.
    * **Periodo inicial previo a un bloque de descarga:** El usuario debe ingresar la hora inicial del conjunto previo a un bloque de
    carga/descarga del SAE.
    * **Periodo final previo a un bloque de descarga:** El usuario debe ingresar la hora final del conjunto previo a un bloque de
    carga/descarga del SAE.
    * **Periodo inicial de un bloque de carga/descarga:** El usuario debe ingresar la hora inicial del periodo de carga/descarga del SAE.
    * **Periodo final de un bloque de carga/descarga:** El usuario debe ingresar la hora final del periodo de carga/descarga del SAE.
    * **Número de SAE a considerar:** El usuario debe ingresar el número de SAE que se van a considerar en la simulación. Por defecto, este
    valor es 1.
    * **Potencia máxima de carga del SAE n:** El usuario debe ingresar la potencia máxima de carga de cada SAE $n$ que vaya a ser considerado
    en la simulación.
    * **Potencia máxima de descarga del SAE n:** El usuario debe ingresar la potencia máxima de descarga de cada SAE $n$ que vaya a ser
    considerado en la simulación.
    * **Energía del SAE n:** El usuario debe ingresar la energía máxima de cada SAE que vaya a ser considerado en la simulación.
    * **Eficiencia global del SAE n:** El usuario debe ingresar la eficiencia global (round trip efficiency) de cada SAE que vaya a ser
    considerado en la simulación.
    * **Estado de carga mínimo del SAE:** El usuario debe ingresar el estado de carga mínimo de cada SAE que vaya a ser considerado en la
    simulación. En esta simulación el estado de carga mínimo hace referencia al valor operativo mínimo que puede tener el SAE, este valor
    es escogido por el CND.
    * **Estado de carga máximo del SAE:** El usuario debe ingresar el estado de carga máximo de cada SAE que vaya a ser considerado en la
    simulación. En esta simulación el estado de carga máximo hace referencia al valor operativo máximo que puede tener el SAE, este valor
    es escogido por el CND.
    * **Estado de carga mínimo técnico del SAE:** El usuario debe ingresar el estado de carga mínimo técnico de cada SAE que vaya a ser
    considerado en la simulación. En esta simulación el estado de carga mínimo técnico hace referencia al valor recomendado por el fabricante/proveedor del
    SAE, como el mínimo valor de estado de carga que el SAE puede tener sin comprometer su vida útil.
    * **Valor de carga requerido del SAE n:** El usuario debe ingresar el valor requerido de carga en los periodos de carga/descarga de cada
    SAE. Consultar el menú desplegable *NOTA IMPORTANTE* para seleccionar adecuadamente el valor de este parámetro.
    * **Valor de descarga requerido del SAE n:** El usuario debe ingresar el valor requerido de descarga en los periodos de carga/descarga de cada
    SAE. Consultar el menú desplegable *NOTA IMPORTANTE* para seleccionar adecuadamente el valor de este parámetro.
    * **Día de simulación:** El usuario debe ingresar el día que se vaya a simular.
    * **Seleccione el tipo de solucionador/solver:** El usuario debe seleccionar el tipo de solucionador/solver que se vaya a utilizar.
        * **GLPK:** Se utilizan recursos propios del equipo en donde se esté corriendo el problema de optimización. Se recomienda para equipos \
        con alto rendimiento o para simulaciones con horizontes de simulación cortos o sistemas pequeños.
        * **CPLEX:** Se utiliza el servidor remoto NEOS que cuenta con una licencia de este solucionador. Tiene limitaciones de tiempo de\
        simulación (hasta 8 horas) y tamaño del problema (menor a 3Gb en memoria RAM).

    **NOTAS:**
    * **Por defecto el horizonte de simulación es de 24 horas.**

    """
    
def get_table_download_link(file,):
    val = file.read()
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Formato_base.xlsx">Formato base para la simulación (xlsx)</a>'

def guias(study, name):
    if study == 'Dimensionamiento y Localización':
        if name == 'Arbitraje':
            inst_ope = st.expander(label="Guía de uso", expanded=False)
            file = open('{}/Casos_estudio/Plantillas/Plantilla_Dim_Arb.xlsx'.format(os.getcwd()),'rb')
            with inst_ope:
                st.write(text_arb_loc())
                st.write("")
                st.write(get_table_download_link(file), unsafe_allow_html=True)
        elif name == 'Minimizar costo de operación':
            inst_ope = st.expander(label="Guía de uso", expanded=False)
            with inst_ope:
                st.write(text_ope())
                st.write("")
        else:
            inst_ope = st.expander(label="Guía de uso", expanded=False)
            with inst_ope:
                st.write(text_rest())
                st.write("")
    if study == 'Operación del SAE (MEM-SC, XM)':
        if name == 'Arbitraje':
            inst_ope = st.expander(label="Guía de uso", expanded=False)
            file = open('{}/Casos_estudio/Plantillas/Plantilla_Dim_Arb.xlsx'.format(os.getcwd()),'rb')
            with inst_ope:
                st.write(text_arb_op())
                st.write("")
                st.write(get_table_download_link(file), unsafe_allow_html=True)
        elif name == 'Mercado Energía Mayorista':
            inst_ope = st.expander(label="Guía de uso", expanded=False)
            file = open('{}/Casos_estudio/Plantillas/Plantilla_MEM.xlsx'.format(os.getcwd()),'rb')
            with inst_ope:
                st.write(text_MEM())
                st.write("")
                st.write(get_table_download_link(file), unsafe_allow_html=True)
        else:
            if name == 'Versión 1':
                inst_opeXM = st.expander(label='Guía de uso', expanded=False)
                with inst_opeXM:
                    st.write(text_opeXM_v1())
                    st.write('')
            else:
                inst_opeXM = st.expander(label="Guía de uso", expanded=False)
                with inst_opeXM:
                    st.write(text_opeXM_v2())
                    st.write('')