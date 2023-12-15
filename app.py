import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to load the model
def load_custom_model():
    model = load_model('darknet_model.h5')
    return model

# Initialize LabelEncoder and StandardScaler
le = LabelEncoder()
# Manually fit the LabelEncoder with the known labels
le.fit(['Normal', 'Malicious'])
scaler = StandardScaler()

# Function to preprocess the file uploaded
def preprocess(df, label_col):
    df['Label'] = le.transform(df['Label'].replace(['Non-Tor', 'NonVPN'], 'Normal').replace(['Tor', 'VPN'], 'Malicious'))
    labels = df[label_col]
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_features].drop(['Src Port', 'Dst Port', 'Protocol'], axis=1)
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_numeric.fillna(df_numeric.mean(), inplace=True)
    scaler.fit(df_numeric)  # Fit the scaler with numeric features
    df_scaled = scaler.transform(df_numeric)  # Apply scaling
    return df_scaled, labels

# Function to map numeric predictions to labels
def map_predictions(predictions):
    return ['Malicious' if pred > 0.5 else 'Safe' for pred in predictions[:, 0]]

# Function to map numeric labels to text labels
def map_actual_labels(labels):
    return ['Malicious' if label == 1 else 'Safe' for label in labels]

# Initialize the model
model = load_custom_model()

# Streamlit application layout
st.title('Darknet Traffic Prediction')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file (max size 10 MB)", type='csv', accept_multiple_files=False)

# Check if the file size is less than 10 MB
if uploaded_file is not None:
    if uploaded_file.size <= 10 * 1024 * 1024:  # 10 MB size
        try:
            # Convert the uploaded file to a dataframe
            input_df = pd.read_csv(uploaded_file)
            label_col = 'Label'
            # Preprocess the dataframe
            preprocessed_df_scaled, actual_labels = preprocess(input_df, label_col)
            # Predict
            predictions = model.predict(preprocessed_df_scaled)
            # Map predictions to labels
            mapped_predictions = map_predictions(predictions)
            # Map actual labels to text labels
            mapped_actual_labels = map_actual_labels(actual_labels)
            # Display the actual vs predicted
            comparison_df = pd.DataFrame({'Actual': mapped_actual_labels, 'Predicted': mapped_predictions})
            st.write("Actual vs Predicted:")
            st.table(comparison_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("File size exceeds 10 MB.")
else:
    st.markdown("""
This app is a simple prototype to predict Darknet traffic as either 'Malicious' or 'Safe'.
""")
