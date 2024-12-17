# import all libraries 
import numpy as np 
import pickle 
import streamlit as st 
# venv\Scripts\activate.bat
# Loading the saved model 
loaded_model = pickle.load(open('diabetes_model.sav', 'rb')) 

# Creating a function for prediction 

def diabetes_prediction(input_data): 

	
	input_data_as_nparray = np.asarray(input_data) 
	input_data_reshaped = input_data_as_nparray.reshape(1, -1) 

	prediction = loaded_model.predict(input_data_reshaped) 

	if prediction == 0: 
		return 'Non Diabetic'
	else: 
		return 'Diabetic'

def main(): 

	 
	st.title('Diabetes Prediction Web App') 

	 
	Pregnancies = st.text_input('No. of Pregnancies:') 
	Glucose = st.text_input('Glucose level:') 
	BloodPressure = st.text_input('Blood Pressure value:') 
	SkinThickness = st.text_input('Skin thickness value:') 
	Insulin = st.text_input('Insulin level:') 
	BMI = st.text_input('BMI value:') 
	DiabetesPedigreeFunction = st.text_input( 
		'Diabetes pedigree function value:') 
	Age = st.text_input('Age:') 

	 
	diagnosis = '' 

	 
	if st.button('Predict'): 
		diagnosis = diabetes_prediction( 
			[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]) 

	st.success(diagnosis) 

if __name__ == '__main__': 
	main()

#python -m streamlit run diabetes.py
#pip install scikit-learn
