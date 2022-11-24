import streamlit as st
import pandas as pd               
import numpy as np

# from sklearn.neighbors import KNeighborsClassifier


st.title("Aplikasi DATA MINING")

st.write("AHMAD FANANI/200411100143/PENAMBANGAN DATA/A")

dataSource, preProcessing, modelling, implementation = st.tabs(["Data Source", "Preprocessing", "Modelling", "Implementation"])

with dataSource:
   st.header("DATA SOURCE")
   st.write("Dataset Pima Indians Diabetes")
   df = pd.read_csv('diabetes.csv')
   st.table(df.head(10))
   st.table(df.tail(5))  
   st.write(df.dtypes)
   st.write(df.shape)








with preProcessing:
   st.header("PREPROCESSING")
   st.title("Duplikasi Data")
   duplicate_rows_df = df[df.duplicated()]
   st.write("Number of duplicate row", duplicate_rows_df)

   st.write("Baris sebelum di hapus data yang duplikat")
   st.write(df.count())

   df = df.drop_duplicates()

   st.write("Baris setelah di hapus data yang duplikat")
   st.write(df.count())

   st.title("Missing Value")
   st.write("Data yang missing value")
   st.write(df.isnull().sum())
    #drop mising value
   df = df.dropna() 
   st.write("Data setelah dihapus missing value ")
   st.write(df.count())


   st.title("Normalisasi")



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

