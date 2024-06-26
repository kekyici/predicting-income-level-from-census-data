import streamlit as st
from constant import *
from data_processor import DataProcessor
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

# default data
def load_and_process_data():
    processor = DataProcessor(DATA_URL, COLUMN_NAMES, Z_SCORE_THRESHOLD)
    data = processor.data
    numerical_variables = ['age', 'education_num', 'hours_per_week']
    data = processor.remove_outliers(numerical_variables)
    data = processor.transform_data()
    return data

data = load_and_process_data()   

st.set_page_config(
    page_title="SMART",
    page_icon="memo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## A 'Student Performance and Placement Prediction Tool' by Abhijit Mandal & Divyanshi"
    }
)

#remove all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title = "Census Income Prediction",
        options = ["Home", "Analysis", "Predictions", "Model"],
        icons = ["house", "upload", "graph-up", "magic", "speedometer", "file-earmark-arrow-down", "gear", "people"],
        menu_icon= "robot",
        default_index = 0,
    )

# ========= HOME TAB =========
if selected == "Home":
    st.title('Census Income Prediction')
    st.divider()
    st.header("Project Background :memo:")
    st.markdown('''
    ####
    Income inequality is a significant socioeconomic challenge with profound implications for individuals and society. Understanding the factors that contribute to inequality in income is critical for developing effective policy measures and addressing socioeconomic inequalities. The availability of census data with a wide range of demographic and employment-related attributes provides an opportunity to investigate these factors and develop predictive models for income levels. The goal of this data science project is to collect the dataset, as well as cleaning and analyzing the data.
    
    The target organizations and users of this project are government agencies, nonprofit organizations, research institutions, and businesses. Government agencies responsible for economic policy formulation, social welfare programs, and resource allocation can benefit from insights into income disparities and predictive models to target interventions more effectively. Nonprofit organizations focused on poverty alleviation, social justice, and community development can use the predictive model to identify vulnerable populations and tailor assistance programs accordingly. Academic institutions and research organizations studying income inequality, social mobility, and labor market dynamics can utilize the predictive model to enhance their understanding of socioeconomic trends and inform their research agenda. Businesses interested in market segmentation, consumer behavior analysis, and targeted marketing can leverage the predictive model to identify potential customers with specific income levels and tailor their products and services accordingly.
    
    There are some potential benefits of this project. Firstly, policy design can be improved. The predictive model can inform the design and implementation of targeted policy interventions aimed at reducing income inequality and promoting economic mobility. Next, it enhanced resource allocation. Government agencies and social welfare organizations can allocate resources more efficiently by targeting interventions toward individuals at risk of low income. Furthermore, this project contributes to addressing income inequality by providing insights that support efforts to uplift marginalized communities and reduce socioeconomic disparities.

    ''')
    
    st.markdown("#### `Get Started Now!`")



# ========= ANALYSIS TAB =========
if selected == "Analysis":
   
        st.balloons()
        st.title("Data Analysis 📊")
        st.header("Explore the dataset and predict income levels.")
    
        st.subheader("Summary Statistics")
        st.write(data.describe())

        st.subheader("Data Visualization")
        selected_feature = st.selectbox("Select Feature for Visualization", data.columns)
        selected_chart = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot"])
        filtered_data = data[data["income"] == '">50K"'] if st.checkbox("Filter High Income") else data
        predict_button = st.button("Show Visualization⚡")

        if predict_button:
            # Data visualization
            if selected_chart == "Histogram":
                st.bar_chart(filtered_data[selected_feature].value_counts())
            elif selected_chart == "Bar Chart":
                st.bar_chart(filtered_data[selected_feature].value_counts())
            else:
                st.scatter_chart(filtered_data.sample(100), x="age", y="hours_per_week")

# ========= PREDICTION TAB =======
def transform_input(age,  education, marital_status, occupation, relationship, sex, hours_per_week):
    # Define transformation mappings

    education_mapping = {
        'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5,
        'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12,
        'Doctorate': 13, '5th-6th': 14, 'Preschool': 15
    }
    marital_status_mapping = {
        'Single': 0, 'Married': 1
    }
    occupation_mapping = {
        'Blue_collar': 0, 'White_collar': 1, 'Brown_collar/Protective_service': 2, 'Pink_collar/Service_and_sales': 3
    }
    relationship_mapping = {
        'Married': 0, 'Single': 1, 'Separated': 2
    }

    sex_mapping = {
        'Male': 0, 'Female': 1
    }
    


    # Apply transformations

    education_encoded = education_mapping.get(education, -1)
    marital_status_encoded = marital_status_mapping.get(marital_status, -1)
    occupation_encoded = occupation_mapping.get(occupation, -1)
    relationship_encoded = relationship_mapping.get(relationship, -1)
    sex_encoded = sex_mapping.get(sex, -1)
    return [age, education_encoded, marital_status_encoded, occupation_encoded, relationship_encoded, sex_encoded,hours_per_week]

if selected == "Predictions":
    
    st.title("Income Level Prediction ⚡")
    st.subheader("Provide the inputs below 👇🏻")
    st.divider()
    st.markdown("##### _Here you will choose to use <span style='color:red'> Naive Bayes, Logistic Regression 🤖</span> Machine Learning Algorithm to create our Model to predict the Income Level of Individuals_.", unsafe_allow_html=True)
    st.markdown("##### _You just need to provide the individual's data to get started and predict their income level using our <span style='color:red'>well trained Model right here</span>_.", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Get user input for Age
        age = st.slider('Enter the Age 👇🏻', min_value=18, max_value=90, step=1)
       # Get user input for hours_per_week
        hours_per_week = st.slider('Enter the Hours Per Week 👇🏻', min_value=1, max_value=100, step=1)
        occupation = st.selectbox('Choose Occupation 💼', ['Blue_collar', 'White_collar', 'Brown_collar/Protective_service', 'Pink_collar/Service_and_sales'])

        model_choice = st.selectbox("Select Model", [ "Naive Bayes", "Logistic Regression"])
        predict_button = st.button("Predict the Income Level ⚡")

    with col2:
         # Get user input for Education
        education = st.selectbox('Choose Education Level 🎓', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        # Get user input for Marital Status
        marital_status = st.selectbox('Choose Marital Status 💍', ['Single', 'Married'])

        relationship = st.selectbox('Choose Relationship Status 💑', ['Married', 'Single', 'Separated'])
        # Get user input for Sex
        sex = st.selectbox('Choose Sex 🧑🏻‍🦱👧🏻', ['Male', 'Female'])
        # Get user input for Native Country
       
    # Check if the Predict Income button is clicked
    if predict_button:
        st.balloons()
        
        # Transform the input features
        user_data = transform_input(age, education, marital_status, occupation, relationship, sex, hours_per_week)
     # Prepare the user input as a dataframe
        user_df = pd.DataFrame([user_data], columns=[
            'age','education', 'marital_status',
            'occupation', 'relationship', 'sex',  'hours_per_week'
        ])
        
        st.divider()
        st.markdown("* ## Input Dataframe ⬇️")
        st.write(user_df)
       
        X = data.drop(['workclass','education','fnlwgt','capital_gain','capital_loss','native_country','race','income'], axis=1)
        y = data["income"]

        # Label Encoding
        label_encoders = {}
        for column in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Naive Bayes":
            clf = GaussianNB()
        elif model_choice == "Logistic Regression":
            clf = LogisticRegression()

        clf.fit(X_train, y_train)
        
            # Predict on user input
        prediction = clf.predict([user_data])
        prediction_proba = clf.predict_proba([user_data])

        # Display the prediction result
        st.markdown("* ## Prediction Result ✅")
        if prediction == 1:
            st.markdown("### <span style='color:lightgreen'>Income >50K 🎉</span>", unsafe_allow_html=True)
        else:
            st.markdown("### <span style='color:red'>Income <=50K 😢</span>", unsafe_allow_html=True)

# ========= PERFORMANCE TAB ======
if selected == "Model":
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])


    st.title("Census Income Prediction Web App")
    st.header("Income Prediction")
    data1 = data.drop(['fnlwgt','capital_gain','capital_loss','native_country','race','income','workclass','education'], axis=1)
    # Sidebar for income prediction
    model = st.selectbox("Select Model", ["SVM", "Naive Bayes", "Logistic Regression"])
    selected_features = st.multiselect("Select Features for Prediction", data1.columns)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    # Income prediction
    if selected_features and st.button("Train and Predict"):
        # Data preprocessing
        X = data.drop(['fnlwgt','capital_gain','capital_loss','native_country','race','income','workclass','education'], axis=1)
        y = data["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


        
        if model == "SVM":
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

