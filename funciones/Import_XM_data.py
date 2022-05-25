# -*- coding: utf-8 -*-
"""
@author: Andres Felipe Penaranda Bayona
"""

from funciones.pydataxm import*
from datetime import*
import urllib.request
import re

# In[Functions]

def date_data(date1,date2):
    numdays = abs((date2 - date1).days) + 1
    dates = [date1 + timedelta(days=x) for x in range(numdays)]
    date_list = [date_obj.strftime('%Y-%m-%d') for date_obj in dates]
    Hours = list(range(0,24))
    return date_list,Hours

def Procces_hourly_data(df1,date1,date2,type_id):
    consult = ReadDB()
    date_list,Hours = date_data(date1,date2)
    df1.columns = Hours
    if type_id == 'Rec':
        df = consult.request_data("ListadoRecursos", 0, date1, date2)
        df2 = df[['Values_Code']]
    elif type_id == 'Sis':
        df2 = pd.DataFrame()
        df2['Values_Code'] = ['Sistema']
    hours_count = 0
    for d in date_list:
        for h in Hours:
            Values = []
            for g in df2['Values_Code']:
                try:   
                    Values.append(df1.loc[(g,d),h])
                except:
                    Values.append(0)
            df2[hours_count] = Values
            hours_count = hours_count + 1
    df2 = df2.set_index('Values_Code')
    return df2

def Dispo_Comercial(date1,date2):
    consult = ReadDB()
    df1 = consult.request_data("DispoCome", 0, date1, date2)
    df1 = df1.set_index(['Values_code', 'Date'])
    df1 = df1.drop(['Id'], axis=1)
    df1 = df1.astype(float)
    df1 = df1.div(1000)
    df2 = Procces_hourly_data(df1,date1,date2,'Rec')
    return df2

def Precio_Oferta(date1,date2):
    consult = ReadDB()
    df1 = consult.request_data("PrecOferDesp", 0, date1, date2)
    df1 = df1.set_index(['Values_code', 'Date'])
    df1 = df1.drop(['Id'], axis=1)
    df1 = df1.astype(float)
    df1 = df1.mul(1000)
    df2 = Procces_hourly_data(df1,date1,date2,'Rec')
    return df2

def Demanda(date1,date2):
    consult = ReadDB()
    df1 = consult.request_data("DemaCome", 0, date1, date2)
    df1 = df1.set_index(['Values_code', 'Date'])
    df1 = df1.drop(['Id'], axis=1)
    df1 = df1.astype(float)
    df1 = df1.div(1000)
    df2 = Procces_hourly_data(df1,date1,date2,'Sis')
    df2 = df2.set_index(pd.Index(['Total']))
    return df2

def extract_str(element, agents_glb):
    idx = ', ' + str(element) + ', '
    dicc = {}
    for key, value in agents_glb.items():
        if value.find(idx) >= 0:
            dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = value[value.find(idx)+len(idx)::].split(',')
    return dicc

def extract_num(element, agents_glb):
    idx = ', ' + str(element) + ', '
    dicc = {}
    for key, value in agents_glb.items():
        if value.find(idx) >= 0:
            dicc[''.join(reversed(value[value.find(idx)-2::-1]))] = [float(x) for x in value[value.find(idx)+len(idx)::].split(',')]
    return dicc

def extract_offer_data(data_type,date):
    
    year = date.year
    month = date.month
    day = date.day

    if date.month < 10:
        month = '0{}'.format(month)
    if date.day < 10:
        day = '0{}'.format(day)
    
    if data_type == 'HolAGC':
        
        url_oferta = 'http://www.xm.com.co/despachoprogramado/{}-{}/dAGC{}{}.TXT'.format(year,month,month,day)
        response_oferta = urllib.request.urlopen(url_oferta)
        data_oferta = response_oferta.read()
        agents_offer = data_oferta.decode("utf-8")
        df_AGC = pd.DataFrame([x.split(',') for x in agents_offer.split('\n')])
        DATA = df_AGC.set_index(0)
    
    else:

        url_oferta = 'http://www.xm.com.co/ofertainicial/{}-{}/OFEI{}{}.txt'.format(year,month,month,day)
        response_oferta = urllib.request.urlopen(url_oferta)
        data_oferta = response_oferta.read()
        agents_offer = data_oferta.decode("utf-8")
        df_OFEI = pd.DataFrame([x.split(';') for x in agents_offer.split('\n')])
        dic_OFEI = df_OFEI.to_dict('dict')
        none_val, agents_glb = list(dic_OFEI.items())[0]
        
        nul_val = []
        
        for key, value in agents_glb.items():
            if value == str('\r') or str(''):
                nul_val.append(key)
                
        for i in nul_val:
            del(agents_glb[i])
        
        if data_type == 'AGCP':
            DATA = extract_num('AGCP', agents_glb)
        elif data_type == 'MO':
            DATA = extract_num('MO', agents_glb)
        elif data_type == 'PAP':
            CONF = extract_num('CONF', agents_glb)
            df_CONF = pd.DataFrame(CONF)
            df_CONF
            list_PAP = ['{} , PAP{}'.format(i, int(df_CONF.loc[0,i])) for i in df_CONF.columns]   
            DATA = {}
        
            for idx in list_PAP:
                for key, value in agents_glb.items():
                    if value.find(idx) >= 0:
                        if len(re.findall('\\b' + idx + '\\b', value)) != 0:
                            DATA[idx.split(' , ')[0]] = [int(value[value.find(idx)+len(idx)::].split(',')[1])]
    
    return DATA
        
    
def Plant_offer_data(data_type,date1,date2):
    
    numdays = abs((date2 - date1).days) + 1
    dates = [date1 + timedelta(days=x) for x in range(numdays)]
    consult = ReadDB()
    df = consult.request_data("ListadoRecursos", 0, date1, date2)
    df2 = df[['Values_Code','Values_Name']]
    hours_count = 0
    for d in dates:
        data = extract_offer_data(data_type,d)
        for h in list(range(0,24)):
            Values = []
            for p in df2['Values_Name']:
                p_name = p.replace(' ','')
                if p_name in data.keys():
                    if data_type == 'PAP':
                        Values.append(data[p_name][0])
                    else:
                        Values.append(data[p_name][h])
                else:
                    Values.append(0)
            df2[hours_count] = Values
            hours_count = hours_count + 1
    df2 = df2.set_index('Values_Code')
    df2 = df2.drop(['Values_Name'], axis=1)
    return df2


def Holgura_data(date1,date2):
    
    numdays = abs((date2 - date1).days) + 1
    dates = [date1 + timedelta(days=x) for x in range(numdays)]
    data_type = 'HolAGC'
    hours_count = 0
    
    df2 = pd.DataFrame()
    # df2['Values_Code'] = ['Sistema']
    
    for d in dates:
        data = extract_offer_data(data_type,d)
        for h in list(range(0,24)):
            H = float(data.loc['"Total"', h+1])
            df2[hours_count] = [H]
            hours_count = hours_count + 1
    df2 = df2.set_index(pd.Index(['Total']))
    return df2


# In[Main]

if __name__ == "__main__":
    
    date1 = dt.date(2021, 7, 1)
    date2 = dt.date(2021, 7, 6)
    DisComercial = Dispo_Comercial(date1,date2)
    PrecOferta = Precio_Oferta(date1,date2)
    Demanda_sis = Demanda(date1,date2)
    AGCOferta = Plant_offer_data('AGCP',date1,date2)
    MOferta = Plant_offer_data('MO',date1,date2)
    PAPferta = Plant_offer_data('PAP',date1,date2)
    Holgura = Holgura_data(date1,date2)