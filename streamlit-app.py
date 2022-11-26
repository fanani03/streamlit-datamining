import streamlit as st
import pandas as pd               
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

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

    # Mengambil semua fitur kecuali label
   X = df.iloc[:,:-1]
   y = df.iloc[:,-1]

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
        X = scaledMinMax_features
        st.write(X.head(10))
        
   elif normalisasi == "Standard Scaler":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        scaledStandard_features = pd.DataFrame(scaled, columns=features_names)
        st.write("Menampilkan semua fitur yang telah dinormalisasi dengan Standar Scaler")
        X = scaledStandard_features
        st.write(X.head(10))
        
   else:
        st.write("Menampilkan semua fitur tanpa normalisasi skala")
        X = df.drop(columns=['Outcome'])
        st.write(X.head(10))
        


xtrain,xtest,ytrain,ytest=train_test_split(X,y, test_size=0.2, random_state=0)


with modelling:
   knn = KNeighborsClassifier()
   clf = GaussianNB()
   d3 = DecisionTreeClassifier()
   model = 0

   st.title("MODELLING")
   model = st.radio("Silahkan memilih jenis Model",('KNN','Gaussian Naive Bayes','Decision Tree'))

   if model == "KNN":
        st.header("KNN")
        model = 0
        
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
        akurasi1 = accuracy_score(ytest,y_pred)
        k=0
        for i in range(1,25):
            if akurasi1 == scores_list[i]:
               k=i
        st.success("Hasil akurasi tertinggi = " + str(akurasi1*100) + " Pada Nilai K = " + str(k))


   elif model == "Gaussian Naive Bayes":
        model = 1
        st.header("Gaussian Naive Bayes")
        # GaussianNB
        clf = GaussianNB()
        # set training data
        clf.fit(xtrain,ytrain)
        #data uji
        y_predNaive = clf.predict(xtest)
        # y_pred
        akurasi2 = accuracy_score(ytest,y_predNaive)
        st.success("Hasil akurasi = " + str(akurasi2))

   elif model == "Decision Tree":
        model = 2
        st.header("Decision Tree")
        d3 = DecisionTreeClassifier()

        #Memasukkan dataset untuk proses training
        d3.fit(xtrain, ytrain)
        y_predic = d3.predict(xtest)
        akurasi3 = accuracy_score(ytest, y_predic)
        st.success("Hasil akurasi = " + str(akurasi3))


with implementation:
     st.title("IMPLEMENTATION")
     st.header("Masukkan Data Baru")
     pregnancies = st.slider('Pregnancies', 0, 17, 5)
     glucose = st.slider('glucose', 0, 199, 5)
     bloodPressure = st.slider('blood Pressure', 0, 122, 5)
     SkinThickness = st.slider('SkinThickness', 0, 100, 5)
     insulin = st.slider('insulin', 0, 846, 5)
     bmi = st.slider('bmi', 0, 68, 5)
     diabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.1)
     age = st.slider('Pregnancies', 21, 81, 5)
     st.header("Hasil Prediksi")

     dataBaru = [[pregnancies, glucose, bloodPressure, SkinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
     dataLabel = ['Sehat', 'Gejala Diabetes']
     impModel = st.radio("Silahkan memilih jenis Model",('KNN','Gaussian Naive Bayes','Decision Tree'),key = "<uniquevalueofsomesort>")
     if impModel == 'KNN':
          if model == 0:
               knn = KNeighborsClassifier(n_neighbors=k)
               knn.fit(X,y)
               y_predict = knn.predict(dataBaru)
               st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])
          else:
               st.error("Pilih KNN pada tab Modelling")
     
     elif impModel == 'Gaussian Naive Bayes':
          if model == 1:
               y_predict = clf.predict(dataBaru)
               st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])
          else:
               st.error("Pilih Gaussian Naive Bayes pada tab Modelling")
     elif impModel == 'Decision Tree':
          if model == 2:
               y_predict = d3.predict(dataBaru)
               st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])
          else:
               st.error("Pilih Decision Tree pada tab Modelling")
     

          
     
     





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

