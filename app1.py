
import streamlit as st
import pandas as pd
from sklearn import datasets
import pickle
from sklearn.ensemble import RandomForestClassifier
import csv
st.write("""
# **Heart Disease Prediction App**
This app predicts the possibility of  heart disease in a patient 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex = st.sidebar.selectbox('Sex',('1','0')) 
    CP = st.sidebar.slider('chest pain', 0, 4, 3)
    BP = st.sidebar.slider('blood pressure', 80, 250, 100)
    CH = st.sidebar.slider('cholestrol', 100, 600, 120)
    BS = st.sidebar.slider('blood sugar', 0, 1, 1)
    EKG = st.sidebar.slider('electro cardio graphy', 0, 2, 1)
    HR = st.sidebar.slider('max heart rate', 60, 220, 120)
    EA = st.sidebar.slider('Exercise angina', 0, 1, 0)
    ST = st.sidebar.slider('ST depression', 0, 10, 1)
    slope = st.sidebar.slider('Slope of ST', 0, 3, 2)
    vessel= st.sidebar.slider('no of vessels', 0, 3, 0)
    thallium = st.sidebar.slider('thallium', 0, 10, 3)
    data = {'age': age,
            'Sex': sex,
            'cp': CP,
            'trestbps': BP,
            'chol':CH,
            'fbs':BS,
            'electro cardio graphy':EKG,
            'max heart rate':HR,
            'Exercise angina':EA,
            'ST depression':ST,
            'slope of ST':slope,
            'no of vessel':vessel,
            'thallium':thallium}
    features = pd.DataFrame(data, index=[0])
    return features
##
'''input_df = user_input_features()
heart = pd.read_csv('heart-2.csv')
X =  heart.drop(columns=['target'])
Y = heart.target


df = pd.concat([input_df,heart],axis=0)


st.subheader('User Input parameters')
st.write(df)
     


clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

target_names = ["Absence","Presence"]
st.subheader('Class labels and their corresponding index number')
st.write(target_names)

st.subheader('Prediction')
st.write(target_names[prediction[0]])


st.subheader('Prediction Probability')
st.write(prediction_proba)'''