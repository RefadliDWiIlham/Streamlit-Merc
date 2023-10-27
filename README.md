# Laporan Proyek Machine Learning
### Nama : Refadli Dwi Ilham
### Nim : 211351121
### Kelas : Pagi B

## Domain Proyek

Estimasi harga mobil merc ini boleh digunakan sebagai patokan bagi semua orang yang ingin membeli atau menjual kendaraan produk dari merc
## Business Understanding

Lebih menghemat waktu agar tidak perlu menanyakan harga yang cocok untuk menjual atau membeli mobil merek merc

Bagian laporan ini mencakup:

### Problem Statements

- Tidak mungkin seseorang yang ingin menjual atau membeli mobil merek merc harus menanyakan kepada setiap orang yang memiliki mobil merc agar tau harga yang pas

### Goals

- mencari solusi untuk memudahkan orang-orang yang mencari harga yang cocok untuk menjual atau membeli mobil merek merc


    ### Solution statements
    - Pengembangan Platform Pencarian Harga yang cocok untuk membeli atau menjual mobil merc Berbasis Web, Solusi pertama adalah mengembangkan platform pencarian Harga yang cocok untuk membeli atau menjual mobil merc mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses cepat dan mudah ke informasi tentang estimasi Harga yang cocok untuk membeli atau menjual mobil merc
    - Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi Harga yang cocok untuk membeli atau menjual mobil merc.Dataset ini mengandung 13120 baris dan lebih dari 9 columns.

kaggle datasets download -d adityadesai13/used-car-dataset-ford-and-mercedes  

### Variabel-variabel sebagai berikut:
- year : Tahun mobil merc dibuat
- price : Harga mobil merc 
- mileage : Jumlah KM mobil berjalan
- tax : pajak mobil setiap 1 tahun sekali
- mpg : mil per galon konsumsi bahan bakar mobil

## Data Preparation

DESKRIPSI LIBRARY
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
MEMANGGIL DATASET
```python
df= pd.read_csv('/content/drive/MyDrive/ml1/merc.csv')
```
DESKRIPSI DATASET
```python
df.head()
```
```python
df.info()
```
```python
sns.heatmap(df.isnull())
```
```python
df.describe()
```
VISUALISASI DATA
```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
```
JUMLAH
```python
models = df.groupby('model').count()[['tax']].sort_values(by='tax',ascending=True).reset_index()
models = models.rename(columns={'tax':'numberOfCars'})
```
```python
fig = plt.figure(figsize=(16,5))
sns.barplot(x=models['model'],y=models['numberOfCars'], color="green")
plt.xticks(rotation=50)
```
```python
df['model'].value_counts().plot(kind='bar')
```
UKURAN MESIN
```python
engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax').reset_index()
engine = engine.rename(columns={'tax':'count'})
```
```python
plt.figure(figsize=(15,5))
sns.barplot(x=engine['engineSize'],y=engine['count'], color='green')
```
DISTRIBUSI MILEAGE
```python
plt.figure(figsize=(15,5))
sns.distplot(df['mileage'])
```
DISTRIBUSI HARGA
```python
plt.figure(figsize=(15,5))
sns.distplot(df['price'])
```
SELEKSI FITUR
- Menentukan Label dan Attribute
```python
attribute = ['year','mileage','tax','mpg','engineSize']
x = df[attribute]
y = df['price']
x.shape, y.shape
```
SPILIT DATA TRAINING & DATA TESTING
```python
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```
## Modeling

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
```python
score = lr.score(X_test, y_test)
print('Akurasi Regresi Linear = ', score)
```
Akurasi Regresi Linear =  0.689668140479531

- MENCOBA MELAKUKAN INPUTAN
```python
input_data = np.array([[2019,5000,145,30.2,2]])
prediction = lr.predict(input_data)

print('Estimasi Harga Mobil',prediction)
```
Estimasi Harga Mobil [33650.20023991]

dan keluar hasil estimasi harga yang cocok untuk menjual atau membeli mobil merc

- selanjutnya kita rubah modelnya menjadi bentuk sav
```python
iimport pickle

filename = 'estimasi_harga_mobil_merc.sav'
pickle.dump(lr,open(filename,'wb'))
```
## Evaluation
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
```python
score = lr.score(X_test, y_test)
print('Akurasi Regresi Linear = ', score)
```
Akurasi Regresi Linear =  0.689668140479531

metode statistik yang digunakan untuk menganalisis hubungan antara satu atau lebih variabel independen dan variabel dependen biner, yang digunakan untuk klasifikasi.
## Deployment
https://app-merc-vvjh2hdmdapfdc7qnxx8wm.streamlit.app/
