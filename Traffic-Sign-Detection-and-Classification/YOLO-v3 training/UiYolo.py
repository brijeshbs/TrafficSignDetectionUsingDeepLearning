import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime as dt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from numpy import array
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

global stock_dataset

st.set_page_config(page_title="Quantitative Algorithms Stock", page_icon="💹", layout="wide")

with st.container():
    st.title("Stock Price Detection using Quantitative Algorithms")

header = st.container()
database = st.container()
initial = st.container()
body = st.container()
svr = st.container()
random_forest = st.container()
knn = st.container()
lstm = st.container()
gru = st.container()
hybrid = st.container()
final = st.container()


with header:
    st.subheader("DataSet Upload")
    data_file=st.file_uploader("Upload Stock Dataset in CSV",type=["csv"])
    def fileCheck():
        if data_file is not None:
            stock_dataset=pd.read_csv(data_file)
            return stock_dataset


stock_dataset = fileCheck()
if(stock_dataset is None):
    time.sleep(1000)

with database:
        st.dataframe(stock_dataset)
global bist100
bist100 = stock_dataset
bist100.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)
# Checking null value
bist100.isnull().sum()
# Convert date field from string to Date format and make it index
# Sort Values by Date
bist100['date'] = pd.to_datetime(bist100.date)
bist100.sort_values(by='date', inplace=True)
df2 = bist100[bist100.date>='2015-01-01']
df2 = df2[df2.date<'2020-01-01']
bist100=df2

with initial:
    st.write("Starting date: ",bist100.iloc[0][0])
    st.write("Ending date: ", bist100.iloc[-1][0])
    st.write("Duration: ", bist100.iloc[-1][0]-bist100.iloc[0][0])

closedf = bist100[['date','close']]

with body:  
    fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date Time Range','close':'Close Stock Price (in Rs)'})
    fig.update_traces(marker_line_width=2, opacity=1)
    fig.update_layout(title_text='Stock Closing Price Chart', plot_bgcolor='white', font_size=15, font_color='black')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


close_stock = closedf.copy()
del closedf['date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

def create_dataset(dataset, time_step=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


#graph = [['Algorithms','Main Tuning Parameters (Hyper parameters)','RMSE','MSE','MAE','Variance Regressio Score','R**2 Score',
#            'Mean Gamma Deviance','Mean Poisson Deviance']]
#st.write(graph)
