import streamlit as st
import numpy as np
import pickle


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

        return data


data = load_model()

regressor_loaded = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    st.title("Software Developer Compensation Prediction")
    st.write("We need some information to predict the salary")

    countries = ("Australia", "Brazil", "Canada", "France", "Germany", "India", "Italy", "Netherlands", "Poland", "Russian Federation", "Spain", "Sweden", "United Kingdom of Great Britain and Northern Ireland", "United States of America")

    education = ("Bachelor's degree", "Post grad", "Less than a Bachelors'", "Master's degree")

    country = st.selectbox("Country", countries)
    educationLevel = st.selectbox("Education Level", education)
    expereicne = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        x = np.array([[country, educationLevel, expereicne]])
        x[:,0] = le_country.transform(x[:,0])
        x[:,1] = le_education.transform(x[:,1])
        x = x.astype(float)

        compensation = regressor_loaded.predict(x)
        st.subheader(f"The estimated compensation is ${compensation[0]:.2f}")