import streamlit as st
import pandas as pd               
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler

st.title("Aplikasi DATA MINING")

st.write("AHMAD FANANI/200411100143/PENAMBANGAN DATA/A")

dataSource, preProcessing, modelling, implementation = st.tabs(["Data Source", "Preprocessing", "Modelling", "Implementation"])

with dataSource:
   st.title("DATA SOURCE")
   st.write("Dataset Pima Indians Diabetes")
   st.write("Dataset publik diambil dari https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
   st.write("Kumpulan data ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan dari kumpulan data ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes atau tidak, berdasarkan pengukuran diagnostik tertentu yang termasuk dalam kumpulan data. Beberapa batasan ditempatkan pada pemilihan instance ini dari database yang lebih besar. Secara khusus, semua pasien di sini adalah perempuan berusia minimal 21 tahun keturunan India Pima.")
   st.write("dataset terdiri dar 8 fitur yaitu kehamilan, glukosa, tekanan darah, ketebalan kulit, insulin, BMI dan fungsi silsilah diabetes. Sedangkan 1 fitur untuk hasil gejala")


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

     X = df.drop(columns=['Outcome'])
     y = df['Outcome'].values
     duplikasiData, missingValue, normalisasi = st.tabs(["Duplikasi Data", "Missing Value", "Normalisasi Data"])

     with duplikasiData:
     
          st.header("Duplikasi Data")
          duplicate_rows_df = df[df.duplicated()]
          st.write("Number of duplicate row", duplicate_rows_df)

          st.write("Baris sebelum di hapus data yang duplikat")
          st.write(df.count())

          df = df.drop_duplicates()

          st.write("Baris setelah di hapus data yang duplikat")
          st.write(df.count())

     with missingValue:

          st.header("Missing Value")
          st.write("Data yang missing value")
          st.write(df.isnull().sum())
          #drop mising value
          df = df.dropna() 
          st.write("Data setelah dihapus missing value ")
          st.write(df.count())

     with normalisasi:
               
          st.header("Normalisasi Dengan MinMax Scaller")

          # Mengambil semua fitur kecuali label


          st.write("Menampilkan 10 baris fitur")
          st.table(X.head(10))
          st.write("Menampilkan 10 baris label")
          st.table(y[0:11])



          #   ################### Normalisasi####################
          scaler = MinMaxScaler()
          scaled = scaler.fit_transform(X)
          features_names = X.columns.copy()
          scaledMinMax_features = pd.DataFrame(scaled, columns=features_names)
          st.write("Menampilkan semua fitur yang telah dinormalisasi dengan MinMax Scaler")
          X = scaledMinMax_features
          st.table(X.head(10))

          # SAVE NORMALISASI

          filename = "normalisasiMinMax.sav"
          joblib.dump(scaler, filename) 

     
#==================SPLIT DATA ====================#     
xtrain,xtest,ytrain,ytest=train_test_split(X,y, test_size=0.2, random_state=0)


with modelling:
     st.title("MODELLING")

     # KNN
     knn = KNeighborsClassifier()
     st.header("KNN")
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
     akurasiKNN = accuracy_score(ytest,y_pred)
     k=0

     #SAVE MODEL

     filenameModel = 'modelKnn.pkl'
     joblib.dump(knn, filenameModel)

     for i in range(1,25):
          if akurasiKNN == scores_list[i]:
               k=i
     st.success("Hasil akurasi tertinggi = " + str(round(akurasiKNN,4)*100) + "%" + " Pada Nilai K = " + str(k))


     #GaussianNB
     clf = GaussianNB()
     st.header("Gaussian Naive Bayes")
     # GaussianNB
     clf = GaussianNB()
     # set training data
     clf.fit(xtrain,ytrain)
     #data uji
     y_predNaive = clf.predict(xtest)
     # y_pred
     akurasiGaus = accuracy_score(ytest,y_predNaive)
     st.success("Hasil akurasi = " + str(round(akurasiGaus,4)*100) + "%")

     #SAVE MODEL

     filenameModel = 'modelGaussianNB.pkl'
     joblib.dump(knn, filenameModel)

     #SVM
     st.header("SVM")
     from sklearn import svm
     svm = svm.SVC()
     svm.fit(xtrain,ytrain)
     y_predSVM = svm.predict(xtest)
     akurasiSVM = accuracy_score(ytest,y_predSVM)
     st.success("Hasil akurasi = " + str(round(akurasiSVM,4)*100) + "%")

     #SAVE MODEL

     filenameModel = 'modelSVM.pkl'
     joblib.dump(knn, filenameModel)

     #D3     
     st.header("Decision Tree")
     d3 = DecisionTreeClassifier()
     d3.fit(xtrain, ytrain)
     y_predic = d3.predict(xtest)
     akurasiD3 = accuracy_score(ytest, y_predic)
     st.success("Hasil akurasi = " + str(round(akurasiD3,4)*100) + "%")

     #SAVE MODEL

     filenameModel = 'modelD3.pkl'
     joblib.dump(knn, filenameModel)

     
     all_model = [
               ["KNN", akurasiKNN],
               ["Gaussian Naive Bayes", akurasiGaus],
               ["Support Vector Machine", akurasiSVM],
               ["Decision Tree", akurasiD3]
               ]
     ind_akurasi = 1
     model = max(akurasiKNN,akurasiGaus, akurasiSVM,akurasiD3)
     for i in range(4):
          if model == all_model[i][ind_akurasi]:
               model_fit = all_model[i]

     st.write("Hasil akurasi tertinggi dimiliki oleh " + str(model_fit[0]) + " dengan nilai akurasi " + str(round(model,4)*100) + "%")


with implementation:
     st.title("IMPLEMENTATION")
     st.header("Masukkan Data Baru")
     pregnancies = st.slider('Pregnancies', 0, 12, 0)
     glucose = st.slider('Glukosa', 50, 150, 0)
     bloodPressure = st.slider('Blood Pressure', 50, 90, 0)
     SkinThickness = st.slider('Skin Thickness', 0, 50, 0)
     insulin = st.slider('Insulin', 0, 150, 0)
     bmi = st.slider('BMI', 20, 50, 0)
     diabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 1.5, 0.0)
     age = st.slider('Age', 20, 70, 0)


     if st.button('TENTUKAN'):
     
          dataInput = [pregnancies, glucose, bloodPressure, SkinThickness, insulin, bmi, diabetesPedigreeFunction, age]
          dataLabel = ['Sehat', 'Gejala Diabetes']

          dataMentah = df.drop(columns=['Outcome'])
          new = pd.Series(dataInput, index=dataMentah.columns)
          dataBaru = dataMentah.append(new,ignore_index=True)

          #Load normalisasi

          scaler = joblib.load('normalisasiMinMax.sav')
          # scaler = MinMaxScaler()
          dataNorm = scaler.fit_transform(dataBaru)
          hasilNorminput = pd.DataFrame(dataNorm)

          final = hasilNorminput.tail(1)

          st.write("Data masukan")
          st.table(dataBaru.tail(1))
          st.write("Proses Normalisasi dengan menggunakan MinMaxScaller")
          st.table(hasilNorminput.tail(1))
          
          #KNN
          st.header("KNN")
          knn = joblib.load('modelKnn.pkl')
          knn.fit(xtrain,ytrain)
          y_predict = knn.predict(final)
          st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])

          #GaussianNB
          st.header("Gaussian Naive Bayes")
          clf = joblib.load('modelGaussianNB.pkl')
          y_predict = clf.predict(final)
          st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])

          #SVM
          st.header("SVM")
          svm = joblib.load('modelSVM.pkl')
          y_predict = svm.predict(final)
          st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])

          #D3
          st.header("Decision Tree")
          d3 = joblib.load('modelD3.pkl')
          y_predict = d3.predict(final)
          st.success("Hasil Prediksi adalah = " + dataLabel[y_predict[0]])

