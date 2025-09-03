import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.preprocessing import OrdinalEncoder
from streamlit_lottie import st_lottie
import requests

model = joblib.load("C:\\Users\\Asus\\Sallary_model.pkl")

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hr = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tutvdkg0.json")

st.set_page_config(page_title="HR Salary Predictor", page_icon="💼", layout="centered")

st.title("Salary Prediction App")
st_lottie(lottie_hr, height=250, key="hr")

st.write("Fill in the employee details below to predict salary:")

education = st.selectbox("Education", ["Master", "PhD", "Bachelor"])
jobrole = st.selectbox("Job Role", ["Director", "Sales Executive", "Product Manager", "HR Manager"])
department = st.selectbox("Department", ["IT", "Sales", "HR", "Finance"])
experience = st.number_input("Years of Experience", 0, 40, 2)
performance = st.slider("Performance Rating", 1, 5, 3)
citytier = st.radio("City Tier", [1, 2, 3])

if st.button("Predict Salary"):
    with st.spinner("Analyzing employee profile..."):
        time.sleep(2)

        new_employee = pd.DataFrame({
            "Education": [education],
            "JobRole": [jobrole],
            "Department": [department],
            "ExperienceYears": [experience],
            "PerformanceRating": [performance],
            "CityTier": [citytier]
        })

        oe = OrdinalEncoder(categories=[["Master", "PhD", "Bachelor"]])
        new_employee["Education"] = oe.fit_transform(new_employee[['Education']]).astype(int)

        new_employee = pd.get_dummies(new_employee, columns=["JobRole", "Department"], drop_first=True)

        model_features = model.feature_names_in_
        for col in model_features:
            if col not in new_employee.columns:
                new_employee[col] = 0
        new_employee = new_employee[model_features]

        salary = model.predict(new_employee)[0]

        st.success(f"Predicted Salary: ₹ {int(salary):,}")
        st.balloons()
