import streamlit as st
import joblib
import numpy as np


model = joblib.load("RF_model.pkl")

st.title("🚢 Titanic Survival Prediction App")


pclass = st.selectbox("Pclass ", [1, 2, 3])
sex = st.selectbox("Sex ", ["male", "female"])
age = st.number_input("Age ", min_value=0, max_value=100, value=30)
sibsp = st.number_input("SibSp ", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch ", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare ", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Embarked ", ["C", "Q", "S"])


sex_val = 1 if sex == "female" else 0   
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_map[embarked]


if st.button("Predict"):
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("❌ Prediction: Not Survived")
