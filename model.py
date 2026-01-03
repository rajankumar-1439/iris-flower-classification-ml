import streamlit as st
import numpy as np
import pickle

model_path="iris_prediction.pkl"

with open(model_path,"rb") as f:
    model=pickle.load(f)

st.title("Iris Flower Prediction")    

speal_length=st.slider("Sepal Length (cm)",4.0,8.0)
speal_width=st.slider("Sepal Width (cm)",2.0,5.0)
petal_length=st.slider("Petal Length (cm)",1.0,7.0) 
petal_width=st.slider("Petal Width (cm)",0.1,2.5)

if st.button("Predict"):
    input_data=np.array([[speal_length,speal_width,petal_length,petal_width]])
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The predicted Iris species is: {species[prediction[0]]}")   