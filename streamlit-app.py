import streamlit as st
import pandas as pd               
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# from sklearn.neighbors import KNeighborsClassifier


st.title("Aplikasi DATA MINING")

st.write("AHMAD FANANI/200411100143/PENAMBANGAN DATA/A")

dataSource, preProcessing, modelling, implementation = st.tabs(["Data Source", "Preprocessing", "Modelling", "Implementation"])

with dataSource:
   st.title("DATA SOURCE")
   st.write("Dataset Pima Indians Diabetes")
   df = pd.read_csv('diabetes.csv')
   st.write("Menampilkan 10 baris paling atas dari dataset")
   st.table(df.head(10))
   st.write("Menampilkan 10 baris paling bawah dari dataset")
   st.table(df.tail(10))  
   st.write("Menampilkan nama fitur dan tipe data")
   st.write(df.dtypes)
   st.write("Menampilkan jumlah baris dan kolom")
   st.write(df.shape)








with preProcessing:
   st.title("PREPROCESSING")
   st.header("Duplikasi Data")
   duplicate_rows_df = df[df.duplicated()]
   st.write("Number of duplicate row", duplicate_rows_df)

   st.write("Baris sebelum di hapus data yang duplikat")
   st.write(df.count())

   df = df.drop_duplicates()

   st.write("Baris setelah di hapus data yang duplikat")
   st.write(df.count())

   st.header("Missing Value")
   st.write("Data yang missing value")
   st.write(df.isnull().sum())
    #drop mising value
   df = df.dropna() 
   st.write("Data setelah dihapus missing value ")
   st.write(df.count())


   st.header("Normalisasi")

   X = df.drop(columns=['Outcome'])
   y = df['Outcome'].values

   st.write("Menampilkan semua fitur")
   st.table(X)
   st.write("Menampilkan semua label")
   st.table(y)

   normalisasi = st.radio(
    "Silahkan memilih jenis normalisasi",
    ('MinMax Scaler', 'Standard Scaler'))
   if normalisasi == "MinMax Scaler":
        scaler1 = MinMaxScaler()
        scaled = scaler1.fit_transform(X)
        features_names = X.columns.copy()
        scaledMinMax_features = pd.DataFrame(scaled, columns=features_names)
        st.write("Menampilkan semua fitur yang telah dinormalisasi dengan MinMax Scaler")
        st.write(scaledMinMax_features.head(10))
   elif normalisasi == "Standard Scaler":
        scaler2 = StandardScaler()
        scaledStandard = scaler2.fit_transform(X)
        st.write("Menampilkan semua fitur yang telah dinormalisasi dengan Standar Scaler")
        st.write(scaledStandard.head(10))



with modelling:
   st.title("MODELLING")
with implementation:
   st.title("IMPLEMENTATION")


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

