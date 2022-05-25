import pandas as pd
from pandas import ExcelWriter
from pyomo.environ import *
from pyomo import environ as pym
from pyomo import kernel as pmo
from pyomo.opt import SolverFactory
import numpy as np
import os

# para poder corer GLPK desde una API
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
###

def opt(file_prices,txt_eff,txt_deg,txt_auto,txt_SOC_ini,txt_SOC_min,txt_cost_P,txt_cost_E,txt_inv,txt_trm,txt_time_sim,combo):

    # Carga de archivos de precio

    df_Precios = pd.read_excel(file_prices, sheet_name = 'Precios', header = 0, index_col = 0)
    # Caracteristicas del sistema de almacenamiento de energia

    Big_number = 1e20
    Eficiencia = float (txt_eff)      # Eficiencia del sistema de almacenamiento de energia [%]
    Degradacion = float (txt_deg)     # [%Capacidad/ciclo]
    Autodescarga = float (txt_auto)   # [%Capacidad/hora]
    SOC_ini = 1-float (txt_SOC_ini)     # Estado de carga inicial del sistema de almacenamiento de energia [%]
    SOC_min = 0.2                    # Estado de carga minimo del sistema de almacenamiento de energia [%]
    Costo_Potencia = float (txt_cost_P)  # Costo del sistema de potencia del sistema de almacenamiento de energia [$USD/MW]
    Costo_Energia = float (txt_cost_E)   # Costo del sistema de Energia (Baterias) del sistema de almacenamiento de energia [$USD/MWh]
    Inv_limit=float (txt_inv)         # límite de inversión
    TRM = float (txt_trm)             # Tasa de cambio COP/USD
    # ---------------------------------------------------------------------------

    model = ConcreteModel()

    # ---------------------------------------------------------------------------
    # Sets:
    N_horas = int (txt_time_sim)
    model.t = RangeSet(1,N_horas)

    # ---------------------------------------------------------------------------
    # Parameters from interface:

    def P_init(model,t):
        return df_Precios.loc[t,'Precio']
    model.P = Param(model.t, initialize=P_init)

    model.Eficiencia = Param(initialize = Eficiencia)
    model.Degradacion = Param(initialize = Degradacion)
    model.SOCmin = Param(initialize = SOC_min)
    model.Autodescarga = Param(initialize = Autodescarga)
    model.CostoEnergia = Param(initialize = Costo_Energia)

    # ---------------------------------------------------------------------------
    # Variables:

    #model.Degra= Var(model.t,domain = NonNegativeReals)

    model.Carga = Var(model.t, domain = Binary)
    model.Descarga = Var(model.t, domain = Binary)

    model.EnergiaComprada = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)
    model.EnergiaVendida = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)

    model.CapacidadEnergia = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)
    model.NivelEnergia = Var(model.t,domain = NonNegativeReals, bounds=(0,1e6), initialize=0)

    model.E_max = Var(domain = NonNegativeReals, bounds=(0,1e6), initialize=0)
    model.C_Pot= Var(domain = NonNegativeReals, bounds=(0,1e6), initialize=0)

    model.EnergiaAcumulada = Var(model.t,domain = NonNegativeReals, initialize=0)

    # ---------------------------------------------------------------------------
    # Funcion Objetivo:

    def Objetivo_rule(model):
        return  sum(model.P[t]*1000*(model.EnergiaVendida[t]-model.EnergiaComprada[t]) \
                for t in model.t) - model.C_Pot*Costo_Potencia*TRM - Costo_Energia*model.E_max*TRM -\
                1e-4*sum(model.Carga[t]+model.Descarga[t] for t in model.t)
    model.Objetivo = Objective(rule = Objetivo_rule, sense = maximize)

    # ---------------------------------------------------------------------------
    # Restricciones:

    ##### Investment limit
    def Investment_limit_rule(model):
        return model.C_Pot*Costo_Potencia + Costo_Energia*model.E_max <= Inv_limit
    model.IL = Constraint(model.t, rule = Investment_limit_rule)

    def Limite_energia_comprada_rule(model,t):
        if t == 1:
            return model.EnergiaComprada[t] <= model.E_max*(1 - SOC_ini)/model.Eficiencia
        else:
            return model.EnergiaComprada[t] <= (model.CapacidadEnergia[t-1] - model.NivelEnergia[t-1])/model.Eficiencia

    model.LEC = Constraint(model.t, rule = Limite_energia_comprada_rule)

    def Limite_energia_vendida_rule(model,t):
        if t == 1:
            return model.EnergiaVendida[t] <= model.E_max*(SOC_ini-SOC_min)*model.Eficiencia
        else:
            return model.EnergiaVendida[t] <= model.NivelEnergia[t-1]*model.Eficiencia

    model.LEV = Constraint(model.t, rule = Limite_energia_vendida_rule)

    def Limite_Nivel_Energia_rule(model,t):
        return model.NivelEnergia[t]<=model.E_max
    model.LNE = Constraint(model.t, rule = Limite_Nivel_Energia_rule)

    def Limite_Capacidad_Energia_rule(model,t):
        return model.CapacidadEnergia[t]<=model.E_max
    model.LCE = Constraint(model.t, rule = Limite_Capacidad_Energia_rule)

    ####### Constraints for variables Carga, Descarga
    def Capacidad_maxima_inversor_c_rule(model,t):
        return model.EnergiaComprada[t] <= Big_number * model.Carga[t]
    model.CMIc = Constraint(model.t, rule = Capacidad_maxima_inversor_c_rule)

    def Capacidad_maxima_inversor_v_rule(model,t):
        return model.EnergiaVendida[t] <= Big_number * model.Descarga[t]
    model.CMIv = Constraint(model.t, rule = Capacidad_maxima_inversor_v_rule)

    def Capacidad_maxima_inversor_c_2_rule(model,t):
        return model.EnergiaComprada[t] <= model.C_Pot
    model.CMIc2 = Constraint(model.t, rule = Capacidad_maxima_inversor_c_2_rule)

    def Capacidad_maxima_inversor_v_2_rule(model,t):
        return model.EnergiaVendida[t] <= model.C_Pot
    model.CMIv2 = Constraint(model.t, rule = Capacidad_maxima_inversor_v_2_rule)

    def Simultaneidad_rule(model,t):
        return model.Carga[t]+model.Descarga[t]<=1
    model.Simultaneidad = Constraint(model.t, rule = Simultaneidad_rule)

    def Nivel_de_carga_rule(model,t):
        if t == 1:
            return model.NivelEnergia[t] == model.E_max*SOC_ini
        else:
            return model.NivelEnergia[t] == (1-model.Autodescarga)*model.NivelEnergia[t-1] + (model.EnergiaComprada[t]*model.Eficiencia) - ((model.EnergiaVendida[t])/model.Eficiencia)
    model.NDC = Constraint(model.t, rule = Nivel_de_carga_rule)

    ####### Degradacion del sistema de almacenamiento de energia:
    ####### basado en informacion presentada en: Impact of battery degradation on energy arbitrage revenue of grid-level energy storage

    def Energia_Vendida_Acumulada_rule(model,t):
        if t == 1:
            return model.EnergiaAcumulada[t] == model.EnergiaVendida[t]
        else:
            return model.EnergiaAcumulada[t] == model.EnergiaAcumulada[t-1] + model.EnergiaVendida[t]
    model.EVA = Constraint(model.t, rule = Energia_Vendida_Acumulada_rule)

    def Degradacion_rule(model,t):
        if t == 1:
            return model.CapacidadEnergia[t] == model.E_max
        else:
            return model.CapacidadEnergia[t] == model.E_max - model.Degradacion*model.EnergiaAcumulada[t-1]
    model.DR = Constraint(model.t, rule = Degradacion_rule)

    #### --------------------------------------------------------------------------------------------------------------------------------

    def SOC_minimo_rule(model,t):
        return model.NivelEnergia[t] >= model.E_max* model.SOCmin
    model.SOCm = Constraint(model.t, rule = SOC_minimo_rule)

    def pyomo_postprocess(options=None, instance=None, results=None):
        model.Objetivo.display()
        model.E_max.display()
        model.C_Pot.display()
    # ---------------------------------------------------------------------------
    # Configuracion:
    solver_selected=combo
    if solver_selected== 'CPLEX':
        opt = SolverManagerFactory('neos')
        results = opt.solve(model, opt='cplex')
        #sends results to stdout
        results.write()
        print('\nDisplaying Solution\n' + '-'*60)
        pyomo_postprocess(None, model, results)
    else:
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        results.write()
        print('\nDisplaying Solution\n' + '-'*60)
        pyomo_postprocess(None, model, results)

    #################################################################################
    #######################Creación de Archivo Excel#################################
    #################################################################################

    V_Pot_Ba_ch = np.ones(len(model.t))
    V_Pot_Ba_dc = np.ones(len(model.t))
    V_e_b = np.ones(len(model.t))
    V_cost = model.Objetivo.expr
    V_E_size = model.E_max.value
    V_P_size = model.C_Pot.value
    V_capacity=np.ones(len(model.t))
    V_E_acum=np.ones(len(model.t))

    for t in model.t:
        V_Pot_Ba_ch[t-1] = model.EnergiaComprada[t].value

    for t in model.t:
        V_Pot_Ba_dc[t-1] = model.EnergiaVendida[t].value

    for t in model.t:
        V_e_b[t-1] = model.NivelEnergia[t].value

    for t in model.t:
        V_capacity[t-1] = model.CapacidadEnergia[t].value

    for t in model.t:
        V_E_acum[t-1] = model.EnergiaAcumulada[t].value

    df_Pot_Ba_ch = pd.DataFrame(V_Pot_Ba_ch)
    df_Pot_Ba_dc = pd.DataFrame(V_Pot_Ba_dc)
    df_e_b = pd.DataFrame(V_e_b)
    df_capacity=pd.DataFrame(V_capacity)
    df_E_acum=pd.DataFrame(V_E_acum)

    df_E_size = pd.DataFrame(V_E_size, index=['1'], columns=['Energía [MWh]'])
    # df_E_size  = df_E_size.drop(['2'], axis=0)
    df_P_size = pd.DataFrame(V_P_size, index=['1'], columns=['Potencia [MW]'])
    # df_P_size  = df_P_size.drop(['2'], axis=0)
    df_cost = pd.DataFrame(V_cost, index=['1','2'], columns=['Cost'])
    df_cost  = df_cost.drop(['2'], axis=0)

    mydir = os.getcwd()
    name_file = 'Resultados/resultados_size_loc_arbitraje.xlsx'

    path = os.path.join(mydir, name_file)

    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')

    df_Pot_Ba_ch.to_excel(writer, sheet_name='BESS_Ch_Power', index=True)
    df_Pot_Ba_dc.to_excel(writer, sheet_name='BESS_Dc_Power', index=True)
    df_e_b.to_excel(writer, sheet_name='BESS_Energy', index=True)
    df_capacity.to_excel(writer, sheet_name='Capacity', index=True)
    df_E_acum.to_excel(writer, sheet_name='E_acumulada', index=True)

    df_E_size.to_excel(writer, sheet_name='Energy_size', index=True)
    df_P_size.to_excel(writer, sheet_name='Power_size', index=True)
    df_cost.to_excel(writer, sheet_name='cost', index=True)

    mapping = {0 : 'Precio'}
    df_Pot_Ba_ch=df_Pot_Ba_ch.rename(columns=mapping)
    df_Pot_Ba_dc=df_Pot_Ba_dc.rename(columns=mapping)
    Precios=df_Precios.reset_index()
    df_Ingresos= Precios* (df_Pot_Ba_dc-df_Pot_Ba_ch)/1000
    df_Ingresos=df_Ingresos.drop(['index'], axis=1)
    df_Ingresos=df_Ingresos.dropna()

    df_Ingresos.to_excel(writer, sheet_name='Ingresos', index=True)
    # y=df_Ingresos.sum(axis = 0, skipna = True)

    writer.save()
    writer.close()

    ##########################################################################

    return model.E_max.value, model.C_Pot.value
