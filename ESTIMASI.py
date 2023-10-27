import pickle
import streamlit as st

model = pickle.load(open('estimasi_harga_mobil_merc.sav','rb'))

st.title('ESTIMASI HARGA MOBIL MERC')

year = st.number_input('MASUKAN TAHUN MERC')
mileage = st.number_input('MASUKAN KM MERC')
tax = st.number_input('MASUKAN PAJAK')
mpg = st.number_input('MASUKAN BBM MERC')
engineSize = st.number_input('MASUKAN UKURAN MESIN')


predict = ''

if st.button('ESTIMASI'):
    predict = model.predict(
        [[year, mileage, tax, mpg, engineSize]]
    )

    st.write ('ESTIMASI MOBIL MERC :', predict*20000)