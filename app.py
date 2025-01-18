import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import streamlit as st

file_path = 'setelah_outlier.xlsx'
data_insurance = pd.read_excel(file_path)

# Fitur dan target
X = data_insurance[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = np.log1p(data_insurance['charges'])  # Log-transformasi variabel target

# Standardisasi fitur-fitur numerik
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(X)

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Menggunakan Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Tampilan Streamlit
st.title('Aplikasi Prediksi Biaya Asuransi')
st.write('Gunakan formulir di bawah untuk memasukkan fitur dan memprediksi biaya asuransi.')

# Formulir input untuk fitur
age = st.number_input('Usia', min_value=0, value=25)
sex = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
children = st.number_input('Jumlah Anak', min_value=0, value=0)
smoker = st.selectbox('Perokok', ['Ya', 'Tidak'])
region = st.selectbox(
    'Wilayah', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

# Mengonversi input menjadi format numerik
sex_value = 1 if sex == 'Perempuan' else 0
smoker_value = 1 if smoker == 'Ya' else 0
region_mapping = {'Northeast': 0, 'Northwest': 3,
                  'Southeast': 1, 'Southwest': 2}
region_value = region_mapping[region]

# region_mapping = {
#     'northeast': 0,
#     'southeast': 1,
#     'southwest': 2,
#     'northwest': 3,
# }
# df['region'] = df['region'].map(region_mapping)

# sex_mapping = {'male': 0, 'female': 1}
# df['sex'] = df['sex'].map(sex_mapping)

# smoker_mapping = {'yes': 1, 'no': 0}
# df['smoker'] = df['smoker'].map(smoker_mapping)


# Membentuk fitur input
fitur_input = np.array(
    [[age, sex_value, bmi, children, smoker_value, region_value]])
fitur_input = scaler.transform(fitur_input)
fitur_input_poly = poly.transform(fitur_input)

# Prediksi biaya
prediksi_biaya = np.expm1(model.predict(fitur_input_poly)[
                          0])  # Transformasi invers log

# Menampilkan hasil prediksi
st.subheader('Hasil Prediksi Biaya Asuransi')
st.write(f'Biaya yang diprediksi: $ {prediksi_biaya:,.0f}')

# Menampilkan metrik evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader('Evaluasi Model')
st.write(f'Mean Squared Error (MSE): {mse:.2f}')
st.write(f'R-squared (R²): {r2:.2f}')
