import streamlit as st
import pandas as pd
import pickle


# Load the model
with open('GB.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the pipeline
with open('pipe.pkl', 'rb') as f:
    pipeline = pickle.load(f)

st.header('Agriculture Yield Prediction Model')

df = pd.read_csv('C:\Agri_Yield_Streamlit\datafile (1).csv')



crop_name = st.text_input("Enter Crop Name:")
state = st.text_input("Enter State:")
a2fl = st.number_input("Enter Cost of Cultivation (`/Hectare) A2+FL")
c2 = st.number_input("Enter Cost of Cultivation (`/Hectare) C2")
cost_of_production = st.number_input("Enter Cost of Production (`/Quintal) C2")

# Centering the button
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col3:
    if st.button("Submit"):
            answer = model.predict(pipeline.transform([[crop_name.upper(), state.capitalize(), a2fl, c2, cost_of_production]]))
            st.write(f'Predicted Yield(Quintal/Hectare) for Crop {crop_name}  : ', answer)
