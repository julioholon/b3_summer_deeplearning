# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:34:16 2017

@author: arosa
"""
import pandas as pd
import os
import datetime
import numpy as np

header_list='SESSION_DATE','INSTRUMENT','TRADE_NO','TRADE_PRICE','TRADE_QTY','TRADE_TIME','TRADER_IND','BUY_ORDER_DATE','BUY_ORDER_SEQ_NO','BUY_ORDER_SEC_ID','BUY_ORDER_IND','SELL_ORDER_DATE','SELL_ORDER_SEQ_NO','SELL_ORDER_SEC_ID','SELL_ORDER_IND','TRADE_CROSS_IND','BUY_MEMBER','SELL_MEMBER'
headers_to_keep='INSTRUMENT','TRADE_PRICE','TRADE_TIME'
instrument_list='PETR4','BVMF3','ITUB4'
data_dir = 'C:/tmp/deep-mind/201705'
time_lag = '10Min'


def load_data(data_dir):
    df = pd.DataFrame(columns=headers_to_keep)
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.zip'):
            zipfilepath = os.path.join(data_dir, filename)
            print("loading file:", zipfilepath)
            dfi = pd.read_csv(zipfilepath
                    , compression='zip'
                    , skiprows=1 # Remove original header
                    , skipfooter=1 # Remove original trailer
                    , names=header_list
                    , engine='python' # Necessary to use skiprows
                    , sep=';'
                    , converters={
                            'INSTRUMENT':strip
                            , 'TRADE_TIME':str_to_time}
                    , parse_dates=[3]
                    , dtype={'TRADE_PRICE':'float'})
            dfi = filter_columns(dfi)
            dfi = filter_instruments(dfi)
            dfi = filter_conflated_prices(dfi)
            df = df.append(dfi, ignore_index=True)
    return df

def filter_columns(df):
    for column_name in df.columns:
        if column_name not in headers_to_keep:
            df = df.drop(column_name, 1) #1 for y axis, 0 for x axis
    return df


def filter_instruments(df):
    return df.query('INSTRUMENT in @instrument_list')

def filter_conflated_prices(df):
    ndf = pd.DataFrame() # creates a new df
    for instrument in instrument_list:
        dfi = df.query('INSTRUMENT == @instrument').drop_duplicates(['TRADE_TIME']).set_index("TRADE_TIME")
        tsi = pd.Series(dfi.query('INSTRUMENT == @instrument').iloc[:,1])
        #XXX make "15Min" a param
        tsi = tsi.resample(time_lag, closed='right').ffill()
        ndfi = pd.DataFrame()
        ndfi = ndfi.assign(INSTRUMENT=np.full(tsi.size, instrument), TRADE_TIME=tsi.index,
                           TRADE_PRICE=tsi.values)
        ndf = ndf.append(ndfi)
    return ndf

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

def str_to_time(text):
    try:
        return datetime.datetime.strptime(text, '%H:%M:%S.%f')
    except AttributeError:
        return text

def generate_csv_per_instrument(df):
    for instrument in instrument_list:
        filename = "{}/lag{}-{}.csv".format(data_dir, time_lag, instrument)
        df.query('INSTRUMENT == @instrument').to_csv(filename)

df = load_data(data_dir)
generate_csv_per_instrument(df)