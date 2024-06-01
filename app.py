# Import library
import streamlit as st
import joblib

# Load your trained model and scalers
model = joblib.load('stacking_classifier_model.pkl')
scalers = joblib.load('scalers.pkl')

# Function to predict diabetes
def predict_diabetes(features):
    # Define which features need to be scaled and their order
    features_to_scale = ['bmi', 'age', 'HbA1c_level', 'blood_glucose_level']
    scaled_features = []

    # Iterate through the features and apply scaling where needed
    for i, feature in enumerate(features):
        if i >= 4:  # First 4 features are not scaled
            feature_name = features_to_scale[i - 4]
            scaled_feature = scalers[feature_name].transform([[feature]])[0][0]
            scaled_features.append(scaled_feature)
        else:
            scaled_features.append(feature)

    prediction = model.predict([scaled_features])
    return prediction[0]

# Streamlit application
def main():
    st.title("Diabetes Prediction System")
    st.write("Enter your details to predict diabetes")

    # User inputs
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: 'Smoker' if x == 1 else 'Non-Smoker')
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)
    hba1c = st.number_input("HbA1c Level", min_value=0.0, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)

    # Convert categorical data to numeric
    gender = 1 if gender == 'Female' else 0

    # Collect all features in the correct order
    features = [gender, hypertension, heart_disease, smoking, bmi, age, hba1c, glucose]

    if st.button("Predict"):
        result = predict_diabetes(features)
        st.write(f"Predicted Result: {'Diabetes' if result == 1 else 'No Diabetes'}")

if __name__ == "__main__":
    main()
