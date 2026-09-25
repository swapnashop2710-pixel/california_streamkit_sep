import numpy as np
import joblib
import streamlit as st

# Load saved model
obj = joblib.load('california.joblib')

model = obj['model']
cols = obj['columns']

In = []

st.title('California House Price Prediction')

for i in cols:
    v = st.number_input(f'Enter {i} value:')
    In.append(v)

if st.button('Predict'):
    out = model.predict([In])
    st.success(f'The Median House Value is: {out[0]:.2f}')