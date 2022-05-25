 
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from traitlets.traitlets import default
import pandas as pd

def degradation_curve(a,b,DoD):
        return a*(DoD**(-b))

def degradation_curve_figure(a,b,size):
    
        DoD = np.linspace(0,1,10)
        
        fig_Deg_curve = go.Figure()
        fig_Deg_curve.add_trace(go.Scatter(x=DoD, y=degradation_curve(a,b,DoD), name='Degradacion',line_shape='linear'))
        fig_Deg_curve.update_layout(autosize=True,width=700,height=500,plot_bgcolor='rgba(0,0,0,0)')
        fig_Deg_curve.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        fig_Deg_curve.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E4E2E2', linecolor='black', mirror=True)
        if size == 0:
            fig_Deg_curve.update_layout(autosize=False, width=670, height=300, margin=dict(l=10, r=10, b=10, t=15), font={'size': 10},xaxis_title='DoD [pu]',yaxis_title='# Ciclos')
        else:
            fig_Deg_curve.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=8),title='Curva de degradacion',xaxis_title='DoD [p.u.]',yaxis_title='# Ciclos')
        
        return fig_Deg_curve

def bat_param(data1,bin_deg):   
    
    technology=st.sidebar.selectbox('Seleccione el tipo de tecnologí­a de SAE',data1.index)
    if technology=="Nuevo":
        st.markdown("### Ingrese las caracterí­sticas del SAEB a simular:")
        Eff=st.number_input("Ingrese la eficiencia global del SAEB (round-trip efficiency) [pu]: ", value=0.95, min_value=0.0, step=0.01, max_value=1.0)
        DoD=st.number_input("Ingrese la profundidad de descarga (DoD) máxima [pu]: ", value=0.8, min_value=0.0, step=0.1, max_value=1.0)
        ciclos=st.number_input("Ingrese la vida útil en ciclos a la DOD máxima: ", value=5000, min_value=0, step=1)
        autoD=st.number_input("Ingrese el valor de autodescarga por hora [pu/h]: ", value=0.000063, min_value=0.0, step=0.000001, format='%f')
        costP=st.number_input("Ingrese el costo por potencia [USD/MW]: ", value=50000, min_value=0)
        costE=st.number_input("Ingrese el costo por energí­a [USD/MWh]: ", value=176000,min_value=0)
        
        data_idx = data1.columns.tolist()
        data_idx.pop()
        data_idx.pop()
        df2 = pd.DataFrame()
        df2[''] = data_idx
        df2['Nuevo'] = [Eff,DoD,ciclos,autoD,costP,costE]
        df2.set_index('',inplace=True)
        # data1.iloc[df.index.get_loc(technology),0] = Eff
        # data1.iloc[df.index.get_loc(technology),1] = DoD
        # data1.iloc[df.index.get_loc(technology),2] = ciclos
        # data1.iloc[df.index.get_loc(technology),3] = autoD
        # data1.iloc[df.index.get_loc(technology),4] = costP
        # data1.iloc[df.index.get_loc(technology),5] = costE
        st.markdown("Los datos del SAEB a simular son los siguientes:")
        x = df2
        x = x.style.format("{:.2f}")
        x = x.format(formatter="{:.1E}", subset=pd.IndexSlice[['Autodescarga [pu/h]'], :])
        x = x.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
        st.table(x)
        if bin_deg:
            degradacion=st.selectbox('¿Desea ingresar la curva de degradación?', ['Si','No'],index=1,help="Para mayor precisión en el cálculo de la degradación")
            if degradacion=='Si':
                st.write('Función de degradación de los SAEB')
                st.write(r"""
                        $$ NC(DoD) = a \cdot DoD^{-b}
                        $$ """)
                a=st.number_input("Ingrese el valor de (a): ",value=3015.9)
                b=st.number_input("Ingrese el valor de (b): ",value=1.463)
                size = 1
                fig_Deg_curve = degradation_curve_figure(a,b,size)
                st.write(fig_Deg_curve)
            else:
                a = 3015.9
                b = 1.463
                size = 1
                fig_Deg_curve = degradation_curve_figure(a,b,size)
        else:
            a = 0
            b = 0

    else:
        col1, col2 = st.columns(2)
        st.write("* El SAE seleccionado tiene las siguientes características:")
        data_idx = data1.columns.tolist()
        data_idx.pop()
        data_idx.pop()
        df2 = data1[data_idx]
        x = df2.loc[[technology]].T
        x = x.style.format("{:.2f}")
        x = x.format(formatter="{:.1E}", subset=pd.IndexSlice[['Autodescarga [pu/h]'], :])
        x = x.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
        st.table(x)
        Eff=data1.iloc[data1.index.get_loc(technology),0]
        DoD=data1.iloc[data1.index.get_loc(technology),1]
        ciclos=data1.iloc[data1.index.get_loc(technology),2]
        autoD=data1.iloc[data1.index.get_loc(technology),3]
        costP=data1.iloc[data1.index.get_loc(technology),4]
        costE=data1.iloc[data1.index.get_loc(technology),5]
        if bin_deg:
            a = data1.iloc[data1.index.get_loc(technology),6]
            b = data1.iloc[data1.index.get_loc(technology),7]
            size = 0
            fig_Deg_curve = degradation_curve_figure(a,b,size)
            st.write('* Curva de degradación por defecto:')
            st.write(fig_Deg_curve)
        else:
            a = 0
            b = 0

    degra=0.2/ciclos
    return Eff,degra,autoD,DoD,costP,costE,a,b,ciclos



        