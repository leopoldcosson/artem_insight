import pandas as pd
import streamlit as st
import joblib
from datetime import time


# Convert datetime.time to float
def time_to_float(t):
    return t.hour + t.minute / 60 + t.second / 3600


# Convert float to datetime.time
def float_to_time(time_float):
    hours = int(time_float)
    minutes = int((time_float - hours) * 60)
    seconds = int(((time_float - hours) * 60 - minutes) * 60)
    return time(hour=hours, minute=minutes, second=seconds)


# Return expected waiting time
def return_waiting_time(DoctorName, BookedTime):
    # Load the model from the file
    loaded_model = joblib.load('decision_tree_model.joblib')

    # Load the encoder from the file
    loaded_label_encoder = joblib.load('label_encoder.joblib')

    # Format
    df = pd.DataFrame(index=[0], data={'DoctorName': DoctorName, 'BookedTime': BookedTime})
    df['BookedTime'] = df['BookedTime'].apply(time_to_float)
    df['DoctorName'] = loaded_label_encoder.transform(df['DoctorName'])

    # Now you can use the loaded model to make predictions
    prediction = loaded_model.predict(df)[0]

    return float_to_time(prediction)


# Création de l'interface Streamlit
st.title("Rendez-vous à l'hôpital")

with st.form("form"):
    doctor = st.selectbox("Choisissez votre docteur :", ['ANCHOR', 'LOCUM', 'FLOATING'])
    hour = st.time_input("Choisissez votre heure de rendez-vous")

    submit_button = st.form_submit_button("Soumettre")

    if submit_button:
        ExpectedWaitingTime = return_waiting_time(doctor, hour)
        st.write(f"Votre temps d'attente est estimé à : {ExpectedWaitingTime} min")
