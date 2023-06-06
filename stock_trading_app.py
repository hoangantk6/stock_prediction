import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os

import tensorflow as tf
from keras.layers import Conv1D, Dense, LSTM, Dropout
from datetime import datetime

window_size = 30
delta_T = 7

vn_path = '../DL4AI-200011-project/data/data-vn-20230228/'
nasdaq_path = '../DL4AI-200011-project/data/data-nasdaq/'

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, 1))
        self.lstm1 = LSTM(window_size, return_sequences=True)
        self.lstm2 = LSTM(window_size)
        self.dropout = Dropout(0.2)
        self.dense1 = Dense(window_size, activation='tanh')
        self.dense2 = Dense(delta_T, activation='tanh')
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.lstm1(x)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

def max_profit(prices, threshold):
    n = len(prices)
    if n < 2:
        return 0, [], []

    selling_date = 0
    buying_date = 0
    total_profit = 0
    buy_times = []
    sell_times = []

    for i in range(1, len(prices)):
        if prices[i] >= prices[i - 1]:
            selling_date += 1
        else:
            profit = prices[selling_date] - prices[buying_date]
            if profit > threshold:
                total_profit += profit
                buy_times.append(buying_date)
                sell_times.append(selling_date)
            selling_date = buying_date = i

    profit = prices[selling_date] - prices[buying_date]
    if profit > threshold:
        total_profit += profit
        buy_times.append(buying_date)
        sell_times.append(selling_date)

    return total_profit, buy_times, sell_times

def trading(market,trade_date, path, threshold, window_size = 30, delta_T = 10):
    if market == 'NASDAQ':
        date = pd.read_csv(path+'csv/AAPL.csv')
        date['Date'] = pd.to_datetime(date['Date'], format='%d-%m-%Y')
        date = date[date['Date'] > trade_date]['Date'][:delta_T]
        
    if market == 'VIETNAM':
        date = pd.read_csv(path+'stock-historical-data/A32-UpcomIndex-History.csv')
        date['TradingDate'] = pd.to_datetime(date['TradingDate'], format='%Y-%m-%d')
        date = date[date['TradingDate'] > trade_date]['TradingDate'][:delta_T]
        vn_csv = glob.glob(os.path.join(path+'stock-historical-data/', "*.csv"))
    
    #keep date and month only
    date = [str(i.month)+'/'+str(i.day) for i in date]
    trade_table = pd.DataFrame(columns = ['Company', 'Sector','Profit']+ date)
    
    with open(path+'symbols.pkl', 'rb') as fp:
        symbols = pickle.load(fp)
    for sec in symbols:
        model = Model()
        model.load_weights(path+'model/'+sec+'/model.ckpt')
        for company in symbols[sec]:
            if market == 'NASDAQ':
                data = pd.read_csv(path+'csv/'+company+'.csv')
                data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
            if market == 'VIETNAM':
                for i in vn_csv:
                    if company in i:
                        data = pd.read_csv(i)
                        data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'TradingDate']]
                        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
                        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
                        break
            
            x_data  = data[data['Date'] <= trade_date][-window_size-1:]
            print(x_data)
            
            today_price = x_data.iloc[-1]['Close']
            x_data  = x_data['Close'].values

            #normalize data
            x_data = (x_data - today_price)/today_price
            x_data = x_data.reshape(1, x_data.shape[0], 1)

            pred = model.predict(x_data)[0]
            pred = pred*today_price + today_price
            
            profit, buy_times, sell_times = max_profit(pred, threshold)
            if profit == 0:
                continue
            pred_table = {}
            pred_table['Company'] = company
            pred_table['Sector'] = sec
            pred_table['Profit'] = profit
            
            for i in range(len(pred)):
                if i in buy_times:
                    pred_table[date[i]] = 'buy'
                elif i in sell_times:
                    pred_table[date[i]] = 'sell'
                else:
                    pred_table[date[i]] = ' '
            trade_table = pd.concat([trade_table, pd.DataFrame(pred_table, index = [0], columns=['Company', 'Sector','Profit']+ date)])
            sort_table = trade_table.sort_values(by=['Profit'], ascending=False)
    return sort_table

st.set_page_config(layout="wide")
st.title('Swing Trading Decision Support System')
st.text('Author: Nguyen Hoang An. Student ID: 200011')
st.markdown('---')

path = nasdaq_path
col1, col2 = st.columns(2)
with col2:
    market = st.selectbox('Select Market', ('NASDAQ', 'VIETNAM'))
    if market == 'NASDAQ':
        path = nasdaq_path
    if market == 'VIETNAM':
        path = vn_path

with col1:
    trade_date = st.date_input("Pick a trading date", value=pd.to_datetime('01-01-2022'))
    if market == 'NASDAQ':
        trade_date = str(trade_date).split('-')
        trade_date = datetime(int(trade_date[0]), int(trade_date[1]), int(trade_date[2]))
        trade_table = trading('NASDAQ',trade_date, path, threshold = 0.01)
    if market == 'VIETNAM':
        trade_date = str(trade_date).split('-')
        trade_date = datetime(int(trade_date[0]), int(trade_date[1]), int(trade_date[2]))
        trade_table = trading('VIETNAM', trade_date, path, threshold = 100)

st.markdown('---')
col3, col4 = st.columns(2)
with col3:
    st.header('Potential trading opportunities')
    st.table(trade_table.head(10))
with col4:
    st.header('Unconsidered trading opportunities')
    st.table(trade_table.tail(10))
st.markdown('---')