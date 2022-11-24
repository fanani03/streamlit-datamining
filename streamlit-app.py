import streamlit as st
import pandas as pd               
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

   st.write("Menampilkan 10 baris fitur")
   st.table(X.head(10))
   st.write("Menampilkan 10 baris label")
   st.table(y[0:11])

   normalisasi = st.radio(
    "Silahkan memilih jenis normalisasi",
    ('Tanpa Normalisasi','MinMax Scaler', 'Standard Scaler'))
   if normalisasi == "MinMax Scaler":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        scaledMinMax_features = pd.DataFrame(scaled, columns=features_names)
        st.write("Menampilkan semua fitur yang telah dinormalisasi dengan MinMax Scaler")
        st.write(scaledMinMax_features.head(10))
        X = scaledMinMax_features
   elif normalisasi == "Standard Scaler":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        scaledStandard_features = pd.DataFrame(scaled, columns=features_names)
        st.write("Menampilkan semua fitur yang telah dinormalisasi dengan Standar Scaler")
        st.write(scaledStandard_features.head(10))
        X = scaledStandard_features
   else:
        st.write("Menampilkan semua fitur tanpa normalisasi skala")
        st.write(X.head(10))
        X = df.drop(columns=['Outcome'])




with modelling:
   st.title("MODELLING")
   model = st.radio("Silahkan memilih jenis Model",('KNN','Gaussian Naive Bayes','Decision Tree'))

   if model == "KNN":
        st.header("KNN")
        xtrain,xtest,ytrain,ytest=train_test_split(X,y, test_size=0.2, random_state=0)
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics
        #Membuat k 1 sampai 25
        k_range = range(1,26)
        scores = {}
        scores_list = []
        for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(xtrain,ytrain)
                y_pred=knn.predict(xtest)
                scores[k] = metrics.accuracy_score(ytest,y_pred)
                scores_list.append(metrics.accuracy_score(ytest,y_pred))
        
        st.write("Hasil Pengujian K=1 sampai K=25")
        st.line_chart(pd.DataFrame(scores_list))
        akurasi = accuracy_score(ytest,y_pred)
        for i in k_range:
            if akurasi == scores_list[i]:
                k=i
        st.success("Hasil akurasi tertinggi = " + str(akurasi*100) + " Pada Nilai K = " + str(k))
   elif model == "Gaussian Naive Bayes":
        st.header("Gaussian Naive Bayes")
   elif model == "Decision Tree":
        st.header("Decision Tree")





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

