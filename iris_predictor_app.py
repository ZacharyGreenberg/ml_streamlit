import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#1
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

#2
df, target_names = load_data()
X = df.iloc[:, :-1]
y = df["species"]

model = RandomForestClassifier()
model.fit(X, y)

#3
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)',
        'predicted species'
    ])

st.title("Iris Flower Species Predictor")
st.subheader("Input flower measurements below:")

with st.form("add_row_form"):
    sepal_length = st.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
    sepal_width = st.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
    petal_length = st.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
    petal_width = st.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))
    
    submitted = st.form_submit_button("Predict")


if submitted:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_species = target_names[prediction[0]]
    
    new_row = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width],
        'predicted species': [predicted_species]
    })
    
    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    st.success(f"Predicted species: **{predicted_species}**")


st.subheader("Prediction History")
st.dataframe(st.session_state.df)
