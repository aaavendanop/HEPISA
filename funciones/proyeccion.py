import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def mov_brow(file_TRM_his,time_sim):
    if file_TRM_his != None:
        TRM_hist = pd.read_excel(file_TRM_his, sheet_name='TRM')
        TRM_hist = TRM_hist.dropna()
        # st.write(TRM_hist.describe())
        # st.dataframe(TRM_hist.TRM)
        TRM_hist.Fecha = pd.to_datetime(TRM_hist.Fecha,format='%d-%m-%Y')
        # set the column as the index
        TRM_hist.set_index('Fecha', inplace=True)
        st.subheader("El histórico de TRM usado para la proyección es:")
        fig1=plt.figure(figsize=(15,9))
        plt.plot(TRM_hist)
        st.pyplot(fig1)
        # st.write(TRM_hist)
    ###### Movimiento Browniano#####
        # Cálculo de retornos logaritmicos
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
        caminos=gen_paths(TRM_hist.iloc[-1], media_m, std_m, 2, 24, 500)
        st.subheader("Caminos obtenidos con movimiento browniano para los próximos 24 meses:")
        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        ax.plot(caminos)
        plt.xlabel('Time [Meses]')
        plt.ylabel('[USD/COP]')
        # plt.ylim([0, 6000])
        st.pyplot(fig)

        caminos=caminos.mean(axis = 1)
        # st.write(caminos)

        p_1=caminos[1:13].mean(axis = 0)
        p_2=caminos[13:].mean(axis = 0)
        # st.write(float(TRM_hist.iloc[-1]))
        # st.write(p_1)
        # st.write(p_2)
        prediction=[0]*(time_sim+1)

        for i in range(2,time_sim+1):
            prediction[i]=p_2

        prediction[0]=float(TRM_hist.iloc[-1])
        prediction[1]=p_1
        # st.write(prediction)
        st.subheader("Comportamiento asumido para TRM:")
        fig=plt.figure(figsize=(15,9))
        ax = fig.add_axes([0,0,1,1])
        ax.step(range(0,time_sim+1),prediction)
        plt.xlabel('Time [Años]')
        plt.ylabel('[USD/COP]')
        # plt.ylim([0, 6000])
        st.pyplot(fig)
        return prediction

        ##### regresión a la media

        ##### AJuste por ARIMA

        # fig2=plt.plot(aaa[:, :15]) #Graficamos solo 15 caminos, todos los datos renglones o time steps, 15 columnas
        # plt.grid(True)
        # plt.xlabel('time steps')
        # plt.ylabel('index level')
        # st.pyplot(fig2)

        ######


#         ####
#         # file_TRM_hist="1.1.1.TCM_Serie histórica IQY.xlsx"
#         # TRM_hist = pd.read_excel(file_TRM_hist, sheet_name='TRM')
#         TRM_hist = file_TRM_hist
#         TRM_hist = TRM_hist.dropna()
#         # st.write(TRM_hist.describe())
#         # st.dataframe(TRM_hist.TRM)
#         TRM_hist.Fecha = pd.to_datetime(TRM_hist.Fecha,format='%d-%m-%Y')
#         # set the column as the index
#         TRM_hist.set_index('Fecha', inplace=True)

#         fig1=plt.figure(figsize=(15,9))
#         plt.plot(TRM_hist)
#         st.pyplot(fig1)
#         # st.write(TRM_hist)

# ###### Movimiento Browniano#####

#         # Calculo de retornos logaritmicos
#         log_returns = np.log(TRM_hist / TRM_hist.shift(periods=1))
#         log_returns=log_returns.dropna()
#         # st.dataframe(log_returns.tail(10)*100)
#         # st.write()
#         # Parametros para el movimiento Browniano
#         ## Media diaria y media anual
#         time_mb=10 #  Ventana de tiempo en años
#         media_d=log_returns.iloc[-time_mb*365:].mean()
#         # st.write(media_d*100)
#         media_m=(1+media_d.iloc[0])**30-1
#         media_a=(1+media_d)**365-1
#         # st.write(media_a*100)
#         ### desviación
#         std_d=log_returns.iloc[-time_mb*365:].std()
#         # st.write(std_d*100)
#         std_m=std_d.iloc[0]*(30)**(1/2)
#         std_a=std_d*(365)**(1/2)
#         # st.write(std_a*100)
#         ###
#         def gen_paths(S0, r, sigma, T, M, I):
#             ''' Generates Monte Carlo paths for geometric Brownian Motion.
#             Parameters
#             ==========
#             S0 : float
#             iniial stock/index value
#             r : float
#             constant short rate
#             sigma : float
#             constant volatility
#             T : float
#             final time horizon
#             M : int
#             number of time steps/intervals
#             I : int
#             number of paths to be simulated
#             Returns
#             =======
#             paths : ndarray, shape (M + 1, I)
#             simulated paths given the parameters
#             '''
#             dt= float(T) /M
#             paths = np.zeros((M + 1, I), np.float64)
#             paths[0] = S0
#             for t in range(1, M + 1):
#                 rand=np.random.standard_normal(I)
#                 rand=(rand-rand.mean()) / rand.std()
#                 paths[t]=paths[t-1]* np.exp((r- 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
#             return paths
#         aaa=gen_paths(TRM_hist.iloc[-1], media_m, std_m, 2, 24, 500)

#         fig=plt.figure(figsize=(15,9))
#         ax = fig.add_axes([0,0,1,1])
#         ax.plot(aaa)
#         plt.xlabel('Time [Meses]')
#         plt.ylabel('[USD/COP]')
#         # plt.ylim([0, 6000])
#         st.pyplot(fig)

#         aaa=aaa.mean(axis = 1)
#         # st.write(aaa)
#         p_1=aaa[1:13].mean(axis = 0)
#         p_2=aaa[13:].mean(axis = 0)
#         # st.write(p_1)
#         # st.write(p_2)
#         prediction=[0]*(time_sim+1);

#         for i in range(2,time_sim+1):
#             prediction[i]=p_2

#         prediction[0]=TRM_hist.iloc[-1]
#         prediction[1]=p_1
#         # st.write(prediction)

#         fig=plt.figure(figsize=(15,9))
#         ax = fig.add_axes([0,0,1,1])
#         ax.step(range(0,time_sim+1),prediction)
#         plt.xlabel('Time [Años]')
#         plt.ylabel('[USD/COP]')
#         # plt.ylim([0, 6000])
#         st.pyplot(fig)