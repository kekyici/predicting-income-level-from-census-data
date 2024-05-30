
import streamlit as st
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from constant import *
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


url1 = "https://predicting-income-level-from-census-data-1model.streamlit.app/"
url2 = "https://predicting-income-level-from-census-app.streamlit.app/"

# Create a button in the sidebar
if st.sidebar.button("Go to Income Prediction"):
    st.markdown(f'<meta http-equiv="refresh" content="0; url={url1}">', unsafe_allow_html=True)
elif st.sidebar.button("Go to Data Exploration"):
    st.markdown(f'<meta http-equiv="refresh" content="0; url={url2}">', unsafe_allow_html=True)

def load_and_process_data():
    processor = DataProcessor(DATA_URL, COLUMN_NAMES, Z_SCORE_THRESHOLD)
    data = processor.data
    numerical_variables = ['age', 'education_num', 'hours_per_week']
    data = processor.remove_outliers(numerical_variables)
    data = processor.transform_data()
    return data

data = load_and_process_data()   
# Title and description
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])


st.title("Census Income Prediction Web App")
st.header("Income Prediction")

# Sidebar for income prediction
model = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"])
selected_features = st.multiselect("Select Features for Prediction", data.columns[:-1])
test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

# Income prediction
if selected_features and st.sidebar.button("Train and Predict"):
    # Data preprocessing
    X = data[selected_features]
    y = data["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model == "Random Forest":
        clf = RandomForestClassifier()
    elif model == "Decision Tree":
        clf = DecisionTreeClassifier()
    elif model == "SVM":
        clf = SVC()
    elif model == "Naive Bayes":
        clf = GaussianNB()
    elif model == "Logistic Regression":
        clf = LogisticRegression()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Display results
    st.subheader("Model Evaluation")
    st.write(f"Selected Model: {model}")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

# GitHub link
st.sidebar.header("GitHub Repository")
st.sidebar.write("[GitHub Repository](https://github.com/kekyici/predictingincomelevelfromcensusdata)")

# Footer
st.sidebar.header("About")
st.sidebar.write("This Streamlit app is for data exploration and income prediction using the Census Income dataset.")

# Data source
st.sidebar.header("Data Source")
st.sidebar.write("[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)")
