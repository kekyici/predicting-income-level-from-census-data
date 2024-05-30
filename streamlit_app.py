import streamlit as st
import numpy as np
import pandas as pd
from constant import *
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from data_processor import DataProcessor
from sklearn.metrics import accuracy_score, classification_report

explore_button = st.sidebar.button("Go to Data Exploration")
predict_button = st.sidebar.button("Go to Income Prediction")
#sidebar --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.success("Select a page above.")

# Load the dataset
@st.cache_data
def load_and_process_data():
    processor = DataProcessor(DATA_URL, COLUMN_NAMES, Z_SCORE_THRESHOLD)
    data = processor.data
    numerical_variables = ['age', 'education_num', 'hours_per_week']
    data = processor.remove_outliers(numerical_variables)
    data = processor.transform_data()
    return data

data = load_and_process_data()                        

# Title and description
st.title("Census Income Prediction Web App")
st.write("Explore the dataset and predict income levels.")


# Data Exploration Page
# Sidebar for data exploration options

st.header("Data Exploration ")
st.subheader("Summary Statistics")
st.write(data.describe())

st.subheader("Data Visualization")
selected_feature = st.selectbox("Select Feature for Visualization", data.columns)
selected_chart = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot"])
filtered_data = data[data["income"] == 1] if st.checkbox("Filter High Income") else data



# Data visualization
if selected_chart == "Histogram":
    st.bar_chart(filtered_data[selected_feature].value_counts())
elif selected_chart == "Bar Chart":
    st.bar_chart(filtered_data[selected_feature].value_counts())
else:
    st.scatter_chart(filtered_data.sample(100), x="age", y="hours_per_week")


le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])



st.header("Income Prediction")

# Sidebar for income prediction
model = st.selectbox("Select Model", ["Random Forest", "Decision Tree", "SVM", "Naive Bayes", "Logistic Regression"])
selected_features = st.multiselect("Select Features for Prediction", data.columns[:-1])
test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

# Income prediction
if selected_features and st.button("Train and Predict"):
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
