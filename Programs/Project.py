import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set up the Streamlit app
st.title("Parkinson's Disease Classification")
st.write("This app uses a machine learning model to classify Parkinson's Disease based on voice measurements.")

# Specify the local path to the dataset
file_path = r"C:\Users\karth\Downloads\parkinsons.csv"

# Load dataset directly from local path if it exists
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Check for missing values
    if df.isna().sum().sum() > 0:
        st.warning("Dataset contains missing values. Please handle them before proceeding.")
    else:
        st.write("No missing values detected.")

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number")
    correlations = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=False, square=True, cmap="coolwarm")
    st.pyplot(plt)

    # Selecting features and target variable
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                'MDVP:Jitter(Abs)', 'MDVP:PPQ', 'MDVP:Shimmer', 'NHR']
    
    # Ensure required columns are in the dataset
    if all(col in df.columns for col in features) and 'status' in df.columns:
        X = df[features]
        y = df['status']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        rf = RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Model predictions and accuracy
        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.subheader("Model Accuracy")
        st.write(f"The accuracy of the Random Forest model is: {accuracy_rf * 100:.2f}%")

        # Display Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred_rf))

        # Option to save the model
        if st.button("Save Model"):
            with open("parkinsons_rf_model.pkl", "wb") as f:
                pickle.dump(rf, f)
            st.success("Model saved as 'parkinsons_rf_model.pkl'.")

        # User input for prediction
        st.subheader("Predict Parkinson's Disease")
        st.write("Enter the values for prediction:")

        input_data = [st.number_input(f"Enter {col}", value=float(df[col].mean()), format="%.10f") for col in features]

        if st.button("Predict"):
            result = rf.predict([input_data])
            diagnosis = "Parkinson's Disease" if result[0] == 1 else "No Parkinson's Disease"
            st.write("Prediction:", diagnosis)
    else:
        st.error("The required columns are not present in the dataset. Please check the dataset.")
else:
    st.error("The specified file path does not exist. Please check the path and try again.")
