import streamlit as st
import pandas as pd               
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
%matplotlib inline     
sns.set(color_codes=True)
# from sklearn.neighbors import KNeighborsClassifier


st.title("Aplikasi DATA MINING")

st.write("AHMAD FANANI/200411100143/PENAMBANGAN DATA/A")

dataSource, preProcessing, modelling, implementation = st.tabs(["Data Source", "Preprocessing", "Modelling", "Implementation"])

with dataSource:
   st.header("DATA SOURCE")
   st.write("Pilih dataset")
#    st.write("Masukkan URL")
#    inpDataset = st.text_input()
   inpDataset = st.radio(
    "",
    ('Pima Indians Datasets'))
    df = pd.read_csv('diabetes.csv')








with preProcessing:
   st.header("PREPROCESSING")
with modelling:
   st.header("MODELLING")
with implementation:
   st.header("IMPLEMENTATION")


# quality = st.number_input('Insert a quality')
# prescreen = st.number_input('Insert a prescreen')
# ma2 = st.number_input('Insert a ma2')
# ma3 = st.number_input('Insert a ma3')
# ma4 = st.number_input('Insert a ma4')
# ma5 = st.number_input('Insert a ma5')
# ma6 = st.number_input('Insert a ma6')
# ma7 = st.number_input('Insert a ma7')
# exudate8 = st.number_input('Insert a exudate8')
# exudate9 = st.number_input('Insert a exudate9')
# exudate10 = st.number_input('Insert a exudate10')
# exudate11 = st.number_input('Insert a exudate11')
# exudate12 = st.number_input('Insert a exudate12')
# exudate13 = st.number_input('Insert a exudate13')
# exudate14 = st.number_input('Insert a exudate14')
# exudate15 = st.number_input('Insert a exudate15')
# euDist = st.number_input('Insert a euDist')
# diameter = st.number_input('Insert a diameter')
# amfm_class = st.number_input('Insert a amfm_class')

# data_baru = [[quality, prescreen, ma2, ma3, ma4, ma5, ma6, ma7, 
#                 exudate8, exudate9, exudate10, exudate11, exudate12, exudate13,exudate14,exudate15,
#                 euDist, diameter, amfm_class]]



# if st.button("Tentukan"):
#     knn = joblib.load("modelKnnNormalisasi.pkl")
#     inp_pred = knn.predict(data_baru)
#     st.write(inp_pred)

