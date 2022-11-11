import streamlit as st

st.title("Aplikasi Penentuan Diabetes dengan KNN")

quality = st.number_input('Insert a quality')
prescreen = st.number_input('Insert a prescreen')
ma2 = st.number_input('Insert a ma2')
ma3 = st.number_input('Insert a ma3')
ma4 = st.number_input('Insert a ma4')
ma5 = st.number_input('Insert a ma5')
ma6 = st.number_input('Insert a ma6')
ma7 = st.number_input('Insert a ma7')
exudate8 = st.number_input('Insert a exudate8')
exudate9 = st.number_input('Insert a exudate9')
exudate10 = st.number_input('Insert a exudate10')
exudate11 = st.number_input('Insert a exudate11')
exudate12 = st.number_input('Insert a exudate12')
exudate13 = st.number_input('Insert a exudate13')
exudate14 = st.number_input('Insert a exudate14')
exudate15 = st.number_input('Insert a exudate15')
euDist = st.number_input('Insert a euDist')
diameter = st.number_input('Insert a diameter')
amfm_class = st.number_input('Insert a amfm_class')

data_baru = [[quality, prescreen, ma2, ma3, ma4, ma5, ma6, ma7, 
                exudate8, exudate9, exudate10, exudate11, exudate12, exudate13,exudate14,exudate15,
                euDist, diameter, amfm_class]]


st.write(inp_pred)


 if st.button("Tentukan"):
    knn = joblib.load("modelKnnNormalisasi.pkl")
    inp_pred = knn.predict(data_baru)
    st.write(inp_pred)

