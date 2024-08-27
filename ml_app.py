import streamlit as st
import pandas as pd
from joblib import load

# Load your trained model
model = load('bbc_gradient_boosting_model.pkl')

# Define a function for user input
def user_input_features():
    data = {}
    data['cnt_children'] = st.number_input('Number of Children', min_value=0, max_value=20, value=0)
    data['amt_income_total'] = st.number_input('Total Income', min_value=10000, max_value=1000000, value=50000)
    data['region_population_relative'] = st.number_input('Region Population Relative', min_value=0.0, max_value=1.0, value=0.01, format="%.5f")
    data['flag_work_phone'] = st.selectbox('Work Phone', [0, 1])
    data['flag_phone'] = st.selectbox('Phone Availability', [0, 1])
    data['flag_email'] = st.selectbox('Email Availability', [0, 1])
    data['region_rating_client'] = st.number_input('Region Rating Client', min_value=1, max_value=3, value=2)
    data['hour_appr_process_start'] = st.number_input('Hour of Process Start', min_value=0, max_value=23, value=12)
    data['reg_region_not_live_region'] = st.selectbox('Registered Region Not Live Region', [0, 1])
    data['reg_city_not_live_city'] = st.selectbox('Registered City Not Live City', [0, 1])
    data['reg_city_not_work_city'] = st.selectbox('Registered City Not Work City', [0, 1])
    data['ext_source_2'] = st.number_input('External Source 2', min_value=0.0, max_value=1.0, value=0.5, format="%.5f")
    data['ext_source_3'] = st.number_input('External Source 3', min_value=0.0, max_value=1.0, value=0.5, format="%.5f")
    data['obs_60_cnt_social_circle'] = st.number_input('Observed 60 Count Social Circle', min_value=0, max_value=50, value=0)
    data['def_60_cnt_social_circle'] = st.number_input('Default 60 Count Social Circle', min_value=0, max_value=50, value=0)
    data['amt_req_credit_bureau_mon'] = st.number_input('Amount Requested Credit Bureau (Monthly)', min_value=0, max_value=100, value=0)
    data['amt_req_credit_bureau_year'] = st.number_input('Amount Requested Credit Bureau (Yearly)', min_value=0, max_value=100, value=0)
    data['age'] = st.number_input('Age', min_value=18, max_value=100, value=30)
    data['years_employed'] = st.number_input('Years Employed', min_value=0, max_value=50, value=5)
    data['id_change_years'] = st.number_input('ID Change Years', min_value=0, max_value=50, value=0)
    data['last_phone_change_years'] = st.number_input('Last Phone Change Years', min_value=0, max_value=50, value=0)
    data['accompanied_with_Unaccompanied'] = st.selectbox('Accompanied With Unaccompanied', [0, 1])
    data['income_type_State servant'] = st.selectbox('Income Type: State Servant', [0, 1])
    data['income_type_Working'] = st.selectbox('Income Type: Working', [0, 1])
    data['education_type_Lower secondary / Incomplete higher'] = st.selectbox('Education Type: Lower Secondary / Incomplete Higher', [0, 1])
    data['education_type_Secondary / secondary special'] = st.selectbox('Education Type: Secondary / Secondary Special', [0, 1])
    data['family_status_Married'] = st.selectbox('Family Status: Married', [0, 1])
    data['family_status_Separated / Widow'] = st.selectbox('Family Status: Separated / Widow', [0, 1])
    data['gender'] = st.selectbox('Gender', [0, 1])
    data['own_car'] = st.selectbox('Owns a Car', [0, 1])

    return pd.DataFrame(data, index=[0])

# Title of the app
st.title('Machine Learning Model Prediction App')

# Collect user input
input_df = user_input_features()

# Predict using the model
if st.button('Predict'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0, 1]
    
    st.subheader('Prediction Result:')
    st.write(f'Probability of being positive: {probability * 100:.2f}%')
    st.write('Predicted Class: ', 'Positive' if prediction[0] == 1 else 'Negative')
