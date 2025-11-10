# --------------------------------------------------------------
# 🏥 Diabetes Prediction App (Number Inputs + CSV/Excel + PDF + Recommendations)
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------
# CSS Styling
# --------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2e86ab;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.prediction-positive {
    background-color: #ff6b6b;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
.prediction-negative {
    background-color: #51cf66;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Diabetes Prediction App
# --------------------------------------------------------------
class DiabetesPredictionApp:
    def __init__(self):
        self.load_models()
    
    # ----------------------------------------------------------
    # Load model and scaler
    # ----------------------------------------------------------
    def load_models(self):
        try:
            with open('best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            st.sidebar.success("✅ Models loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Model or scaler files not found.")
            st.stop()
    
    # ----------------------------------------------------------
    # Make prediction
    # ----------------------------------------------------------
    def predict_diabetes(self, input_data, threshold=0.50):
        try:
            input_scaled = self.scaler.transform([input_data])
            prob = self.model.predict_proba(input_scaled)[0, 1]
            prediction = 1 if prob >= threshold else 0
            return prediction, prob
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None, None
    
    # ----------------------------------------------------------
    # Manual input form
    # ----------------------------------------------------------
    def create_input_form(self):
        st.header("🔍 Patient Information (Manual Entry)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1, step=1)
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120, step=1)
        with col2:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 70, step=1)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20, step=1)
        with col3:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 1000, 80, step=1)
            bmi = st.number_input("BMI", 0, 70, 25, step=1)
        with col4:
            pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
            age = st.number_input("Age", 0, 120, 30, step=1)
        
        return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
    
    # ----------------------------------------------------------
    # Display prediction result with probability bar
    # ----------------------------------------------------------
    def display_prediction(self, input_data, prediction, probability, threshold, 
                           preventive_measures, foods_to_eat, foods_to_avoid, doctor_notes):
        st.markdown("---")
        st.header("🎯 Prediction Result")
        probability_pct = round(probability * 100)
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-positive">
                    HIGH RISK DETECTED<br>
                    Probability: {probability_pct}%<br>
                    Threshold used: {threshold}
                </div>
            """, unsafe_allow_html=True)
            st.info("Consult a healthcare professional immediately.")
        else:
            st.markdown(f"""
                <div class="prediction-negative">
                    LOW RISK<br>
                    Probability: {probability_pct}%<br>
                    Threshold used: {threshold}
                </div>
            """, unsafe_allow_html=True)
            st.success("Maintain a healthy lifestyle.")
        
        # Probability bar
        fig, ax = plt.subplots(figsize=(8,2))
        ax.barh([0], [probability], color='#ff6b6b' if prediction==1 else '#51cf66', height=0.5)
        ax.set_xlim(0,1)
        ax.set_xlabel("Probability")
        ax.set_title("Diabetes Risk Probability")
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        ax.legend()
        st.pyplot(fig)
        
        # Display recommendations
        st.subheader("📋 Suggested Preventive Measures")
        st.write(preventive_measures)
        st.subheader("🍎 Foods to Eat")
        st.write(foods_to_eat)
        st.subheader("🚫 Foods to Avoid")
        st.write(foods_to_avoid)
        st.subheader("🩺 Doctor Notes")
        st.write(doctor_notes)
        
        # Show input summary
        st.subheader("📝 Input Summary")
        feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        df_input = pd.DataFrame({'Feature': feature_names, 'Value': input_data})
        st.dataframe(df_input)
    
    # ----------------------------------------------------------
    # Upload CSV/Excel
    # ----------------------------------------------------------
    def upload_file_section(self):
        st.header("📁 Upload CSV or Excel for Batch Prediction")
        uploaded_file = st.file_uploader("Choose CSV or Excel", type=['csv','xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"File loaded: {uploaded_file.name}")
                st.dataframe(df.head())
                return df
            except Exception as e:
                st.error(f"Error loading file: {e}")
        return None
    
    # ----------------------------------------------------------
    # Generate PDF with recommendations & doctor notes
    # ----------------------------------------------------------
    def export_pdf(self, input_data, prediction, probability, preventive_measures, foods_to_eat, foods_to_avoid, doctor_notes):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Diabetes Prediction Result", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", "", 12)
        features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        for f, v in zip(features, input_data):
            pdf.cell(0,8,f"{f}: {v}", ln=True)
        pdf.ln(5)
        pdf.cell(0,8,f"Prediction: {'HIGH RISK' if prediction==1 else 'LOW RISK'}", ln=True)
        pdf.cell(0,8,f"Probability: {round(probability*100)}%", ln=True)
        pdf.ln(5)
        pdf.cell(0,8,"Suggested Preventive Measures:", ln=True)
        pdf.multi_cell(0,8, preventive_measures or "No suggestions available.")
        pdf.ln(2)
        pdf.cell(0,8,"Foods to Eat:", ln=True)
        pdf.multi_cell(0,8, foods_to_eat or "No suggestions available.")
        pdf.ln(2)
        pdf.cell(0,8,"Foods to Avoid:", ln=True)
        pdf.multi_cell(0,8, foods_to_avoid or "No suggestions available.")
        pdf.ln(2)
        pdf.cell(0,8,"Doctor Notes:", ln=True)
        pdf.multi_cell(0,8, doctor_notes or "Consult a healthcare professional.")
        
        pdf_buffer = BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        
        st.download_button(
            label="📄 Download PDF Result",
            data=pdf_buffer,
            file_name="diabetes_prediction.pdf",
            mime="application/pdf"
        )
    
    # ----------------------------------------------------------
    # Display feature descriptions in sidebar
    # ----------------------------------------------------------
    def display_feature_descriptions(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Feature Descriptions")
        feature_info = {
            "Pregnancies": "Number of times pregnant",
            "Glucose": "Plasma glucose concentration (mg/dL)",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (kg/m²)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function",
            "Age": "Age in years"
        }
        for f, desc in feature_info.items():
            st.sidebar.write(f"**{f}**: {desc}")
    
    # ----------------------------------------------------------
    # Display model info
    # ----------------------------------------------------------
    def display_model_info(self):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Model Information")
        st.sidebar.write("**Algorithm**: XGBoost Classifier")
        st.sidebar.write("**Training Data**: Pima Indians Diabetes Dataset")
        st.sidebar.write("**Samples**: 768 patients")
        st.sidebar.write("**Features**: 8 medical parameters")
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Model Performance")
        st.sidebar.metric("Accuracy","71%")
        st.sidebar.metric("Recall","93%")
        st.sidebar.metric("Precision","55%")
        st.sidebar.metric("F1-Score","69%")
    
    # ----------------------------------------------------------
    # Display feature importance
    # ----------------------------------------------------------
    def display_feature_importance(self):
        st.header("📊 Feature Importance")
        features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigree', 'Pregnancies','Insulin','SkinThickness','BloodPressure']
        importance = [0.25,0.18,0.15,0.12,0.10,0.08,0.07,0.05]
        fig, ax = plt.subplots(figsize=(10,6))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance in Diabetes Prediction")
        plt.tight_layout()
        st.pyplot(fig)
    
    # ----------------------------------------------------------
    # Display sample data
    # ----------------------------------------------------------
    def display_sample_data(self):
        st.header("📋 Sample Data Overview")
        sample_data = {
            'Feature': ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Pregnancies'],
            'Normal Range': ['70-100 mg/dL', '18.5-24.9', '< 45 years', '< 120/80 mmHg', '0-2'],
            'High Risk Range': ['> 126 mg/dL', '> 30', '> 45 years', '> 140/90 mmHg', '> 4']
        }
        st.table(pd.DataFrame(sample_data))
    
    # ----------------------------------------------------------
    # Run prediction section
    # ----------------------------------------------------------
    def run_prediction_section(self):
        st.subheader("Manual Entry Prediction")
        input_data = self.create_input_form()
        threshold = st.number_input("Prediction Threshold", 0.1, 0.9, 0.5, step=0.05)
        
        # Generate automated preventive measures and recommendations
        preventive_measures = "Maintain regular exercise, monitor blood sugar, control weight."
        foods_to_eat = "Vegetables, whole grains, lean protein, legumes, fruits in moderation."
        foods_to_avoid = "Sugary drinks, processed foods, high-fat dairy, excessive sweets."
        
        # Allow doctors to add their own notes
        doctor_notes = st.text_area("🩺 Doctor Notes / Recommendations / Medications", 
                                    value="You can add your own recommendations here.")
        
        if st.button("🔍 Predict Diabetes Risk"):
            pred, prob = self.predict_diabetes(input_data, threshold)
            if pred is not None:
                self.display_prediction(input_data, pred, prob, threshold,
                                        preventive_measures, foods_to_eat, foods_to_avoid,
                                        doctor_notes)
                self.export_pdf(input_data, pred, prob, preventive_measures, foods_to_eat, foods_to_avoid, doctor_notes)
    
    # ----------------------------------------------------------
    # Run batch prediction section
    # ----------------------------------------------------------
    def run_batch_prediction_section(self):
        st.subheader("Batch Prediction")
        df = self.upload_file_section()
        if df is not None and st.button("🔍 Predict All Rows"):
            predictions = []
            probabilities = []
            for _, row in df.iterrows():
                input_data = row[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].tolist()
                pred, prob = self.predict_diabetes(input_data)
                predictions.append(pred)
                probabilities.append(prob)
            df['Prediction'] = ['HIGH RISK' if p==1 else 'LOW RISK' for p in predictions]
            df['Probability (%)'] = [round(p*100) for p in probabilities]
            st.dataframe(df)
            st.download_button("📥 Download Results as CSV", df.to_csv(index=False), "predictions.csv")
    
    # ----------------------------------------------------------
    # Main run
    # ----------------------------------------------------------
    def run(self):
        st.markdown('<div class="main-header">🏥 Diabetes Prediction System</div>', unsafe_allow_html=True)
        self.display_model_info()
        self.display_feature_descriptions()
        
        st.sidebar.title("Navigation")
        section = st.sidebar.radio("Go to", ["Prediction", "Feature Importance", "Data Overview"])
        
        if section == "Prediction":
            self.run_prediction_section()
            self.run_batch_prediction_section()
        elif section == "Feature Importance":
            self.display_feature_importance()
        elif section == "Data Overview":
            self.display_sample_data()

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    app = DiabetesPredictionApp()
    app.run()

if __name__ == "__main__":
    main()
