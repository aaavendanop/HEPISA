import pandas as pd
from pyomo.environ import value
import os
from openpyxl import load_workbook
from datetime import timedelta, datetime

def pyomo_df(element,*fecha):
    if len(element) != 0:
        data = value(element)
        df_data = pd.DataFrame(data, index=['1','2'], columns=['1'])
        df_data = df_data.drop(['2'], axis=0)
    else:
        df_data = 0

    return df_data

def pyomo1_df(element,*fecha):
    if len(element) != 0:
        data = {i: value(v) for i, v in element.items()}
        keys = data.items()
        idx = pd.MultiIndex.from_tuples(keys)
        df_data = pd.DataFrame(data, index=[0]).stack().loc[0]
    else:
        df_data = 0
    return df_data

def pyomo2_df(element,*fecha):

    if len(element) != 0:
        data = {(i, j): value(v) for (i, j), v in element.items()}
        keys, values = zip(*data.items())
        idx = pd.MultiIndex.from_tuples(keys)
        df_data = pd.DataFrame(data, index=[0]).stack().loc[0]
    else:
        df_data = 0
    return df_data

def pyomo2_df_mod(element):
    if len(element) != 0:
        data = {(i, j): value(v) for (i, j), v in element.items()}
        keys, values = zip(*data.items())
        df_data = pd.DataFrame(data, index=[0]).stack().loc[0]
        df_data.loc[0] = df_data.loc[1]
        df_data = df_data.sort_index()
    else:
        df_data = 0

    return df_data

def pyomo3_df(element,*fecha):

    if len(element) != 0:
        data = {(i,j,k): value(v) for (i,j,k), v in element.items()}
        keys, values = zip(*data.items())
        idx = pd.MultiIndex.from_tuples(keys)
        df_data = pd.DataFrame(data, index=[0]).stack().loc[0]
    else:
        df_data = 0
    return df_data

def save_excel(dfs,name_file,sheets,option):
    mydir = os.getcwd()
    if option == 'N':
        name_bat_file = str(name_file) + '.xlsx'
        path = os.path.join(mydir, name_bat_file)
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        for i in range(len(dfs)):
            dfs[i].to_excel(writer, sheet_name=sheets[i], index=True)
        writer.save()
        writer.close()
    if option == 'A':
        name_bat_file = str(name_file) + '.xlsx'
        path = os.path.join(mydir, name_bat_file)
        book = load_workbook(path)
        writer = pd.ExcelWriter(path, engine='openpyxl')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        for i in range(len(dfs)):
            col = pd.read_excel(path, sheet_name=sheets[i]).columns
            dfs[i].to_excel(writer, sheet_name=sheets[i], startcol=len(col), index=False)
        writer.save()
        writer.close()
