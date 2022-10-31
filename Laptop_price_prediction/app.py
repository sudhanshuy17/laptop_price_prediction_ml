import streamlit as st

import pickle
import numpy as np

# let's import the model

model = pickle.load(open('model.pkl', 'rb'))

df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
typeName = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop in Kg')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# Ips
ips = st.selectbox('Ips Display', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
                                                '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
                                                '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024, 2048])

# gpu
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

# os
os = st.selectbox('OS', df['Os'].unique())

if st.button('Make Prediction'):
    #  pass
    # ppi = None
    
    if touchscreen == 'Yes':
        touchscreen = 1
    else :
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else :
        ips = 0    
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    feat_list = np.array([company,typeName,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os,])
    # query = np.array([company,typeName,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os,])
    # query = query.reshape(1,12)  # cause there is 1 row 12 columns
    feat_list = feat_list.reshape(1,12)
    
    st.title("The predicted price of this configuration Laptop is: " + str(int(np.exp(model.predict(feat_list)[0]))))
