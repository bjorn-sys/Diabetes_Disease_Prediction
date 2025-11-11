# --------------------------------------------------------------------
# 🩺 Breast Lump Classification App - SQLite Database Version
# --------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import uuid
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import HorizontalBarChart
import sqlite3
import json
import os

# =============================================================================
# DATABASE SETUP
# =============================================================================
DB_FILE = 'patients_database.db'

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Patients table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            medical_id TEXT,
            notes TEXT,
            created_date TEXT,
            updated_date TEXT
        )
    ''')
    
    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp TEXT,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            threshold REAL,
            features TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    ''')
    
    # App settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS app_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_notes_text TEXT,
            foods_eat_text TEXT,
            foods_avoid_text TEXT,
            created_date TEXT
        )
    ''')
    
    # Create default settings if none exist
    c.execute('SELECT COUNT(*) FROM app_settings')
    if c.fetchone()[0] == 0:
        default_notes = "Schedule biopsy\nFollow up in 2 weeks\nMonitor symptoms"
        default_eat = "Leafy greens\nFruits\nWhole grains\nLean proteins"
        default_avoid = "Processed meats\nAlcohol\nSugary foods\nHigh-fat dairy"
        
        c.execute('''
            INSERT INTO app_settings (doctor_notes_text, foods_eat_text, foods_avoid_text, created_date)
            VALUES (?, ?, ?, ?)
        ''', (default_notes, default_eat, default_avoid, datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
def save_patient(patient_data):
    """Save or update patient in database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if patient exists
    c.execute('SELECT COUNT(*) FROM patients WHERE patient_id = ?', (patient_data['patient_id'],))
    exists = c.fetchone()[0] > 0
    
    if exists:
        # Update existing patient
        c.execute('''
            UPDATE patients 
            SET name=?, age=?, gender=?, contact=?, medical_id=?, notes=?, updated_date=?
            WHERE patient_id=?
        ''', (
            patient_data['name'], patient_data['age'], patient_data['gender'],
            patient_data['contact'], patient_data['medical_id'], patient_data['notes'],
            datetime.now().strftime("%Y-%m-%d %H:%M"), patient_data['patient_id']
        ))
    else:
        # Insert new patient
        c.execute('''
            INSERT INTO patients 
            (patient_id, name, age, gender, contact, medical_id, notes, created_date, updated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['patient_id'], patient_data['name'], patient_data['age'],
            patient_data['gender'], patient_data['contact'], patient_data['medical_id'],
            patient_data['notes'], patient_data['created_date'], datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
    
    conn.commit()
    conn.close()

def save_prediction(prediction_data):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO predictions 
        (patient_id, timestamp, prediction, probability, risk_level, threshold, features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_data['patient_id'],
        prediction_data['timestamp'],
        prediction_data['prediction'],
        prediction_data['probability'],
        prediction_data['risk_level'],
        prediction_data['threshold'],
        json.dumps(prediction_data['features'])
    ))
    
    conn.commit()
    conn.close()

def get_patient(patient_id):
    """Get single patient by ID"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
    row = c.fetchone()
    
    if row:
        patient = {
            'patient_id': row[0],
            'name': row[1],
            'age': row[2],
            'gender': row[3],
            'contact': row[4],
            'medical_id': row[5],
            'notes': row[6],
            'created_date': row[7],
            'updated_date': row[8]
        }
        conn.close()
        return patient
    conn.close()
    return None

def search_patients(search_term="", limit=50, offset=0):
    """Search patients with pagination"""
    conn = sqlite3.connect(DB_FILE)
    
    if search_term:
        query = '''
            SELECT * FROM patients 
            WHERE name LIKE ? OR patient_id LIKE ? OR medical_id LIKE ?
            ORDER BY updated_date DESC 
            LIMIT ? OFFSET ?
        '''
        search_pattern = f'%{search_term}%'
        patients_df = pd.read_sql_query(query, conn, params=(search_pattern, search_pattern, search_pattern, limit, offset))
    else:
        query = 'SELECT * FROM patients ORDER BY updated_date DESC LIMIT ? OFFSET ?'
        patients_df = pd.read_sql_query(query, conn, params=(limit, offset))
    
    conn.close()
    return patients_df

def get_patient_count():
    """Get total number of patients"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM patients')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_patient_predictions(patient_id, limit=5):
    """Get recent predictions for a patient"""
    conn = sqlite3.connect(DB_FILE)
    
    query = '''
        SELECT * FROM predictions 
        WHERE patient_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    '''
    predictions_df = pd.read_sql_query(query, conn, params=(patient_id, limit))
    
    # Parse features JSON
    if not predictions_df.empty:
        predictions_df['features'] = predictions_df['features'].apply(lambda x: json.loads(x) if x else {})
    
    conn.close()
    return predictions_df

def get_app_settings():
    """Get application settings"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT doctor_notes_text, foods_eat_text, foods_avoid_text FROM app_settings ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    
    conn.close()
    
    if row:
        return {
            'doctor_notes_text': row[0],
            'foods_eat_text': row[1],
            'foods_avoid_text': row[2]
        }
    return None

def save_app_settings(settings):
    """Save application settings"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO app_settings (doctor_notes_text, foods_eat_text, foods_avoid_text, created_date)
        VALUES (?, ?, ?, ?)
    ''', (
        settings['doctor_notes_text'],
        settings['foods_eat_text'],
        settings['foods_avoid_text'],
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    
    conn.commit()
    conn.close()

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Breast Lump Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session_state():
    """Initialize session state with database values"""
    defaults = {
        'current_patient_id': None,
        'single_pred': None,
        'batch_pred': None,
        'show_tutorial': True,
        'show_new_patient_form': False,
        'inputs': {
            'concave points_mean': 0.08, 
            'radius_worst': 14.5, 
            'perimeter_mean': 90.0,
            'area_worst': 900.0, 
            'area_mean': 500.0, 
            'radius_mean': 8.5,
            'perimeter_worst': 95.0, 
            'concavity_mean': 0.12, 
            'concavity_worst': 0.18,
            'concave points_worst': 0.12
        },
        'current_page': 0,
        'patients_per_page': 20,
        'search_term': ""
    }
    
    # Set defaults
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load app settings from database
    settings = get_app_settings()
    if settings:
        for key, value in settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

initialize_session_state()

# =============================================================================
# PREDICTION LOGIC
# =============================================================================
class BreastCancerPredictor:
    def __init__(self):
        self.features = [
            'concave points_worst', 'concave points_mean', 'radius_worst',
            'perimeter_mean', 'area_worst', 'area_mean', 'radius_mean',
            'perimeter_worst', 'concavity_mean', 'concavity_worst'
        ]

    def predict(self, df, threshold=0.5):
        """
        Simple rule-based prediction logic for demonstration.
        """
        try:
            # Ensure all required features are present
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0.0

            probs = []
            for _, row in df.iterrows():
                # Simple scoring system based on feature values
                score = 0
                if row['radius_worst'] > 15: score += 1
                if row['concave points_worst'] > 0.1: score += 1
                if row['area_worst'] > 800: score += 1
                if row['concavity_worst'] > 0.2: score += 1
                if row['perimeter_worst'] > 100: score += 1

                # Convert score to probability (capped at 0.8 for demonstration)
                prob_malignant = min(0.8, score * 0.2)
                prob_benign = 1 - prob_malignant
                probs.append([prob_benign, prob_malignant])

            probs = np.array(probs)
            preds = (probs[:, 1] >= threshold).astype(int)

            # Assign risk levels based on probability
            risks = []
            for prob in probs[:, 1]:
                if prob < 0.2: 
                    risks.append(("Low Risk", prob))
                elif prob < 0.4: 
                    risks.append(("Mild Risk", prob))
                elif prob < 0.61: 
                    risks.append(("Moderate Risk", prob))
                else: 
                    risks.append(("High Risk", prob))

            return preds, probs, risks

        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Return safe default values
            n = len(df)
            return (
                np.zeros(n), 
                np.array([[0.8, 0.2]] * n), 
                [("Low Risk", 0.2)] * n
            )

# =============================================================================
# PDF REPORT GENERATION (Same as before)
# =============================================================================
def generate_pdf_report(patient_data, prediction_data, input_data, doctor_notes, foods_eat, foods_avoid):
    """Generate a PDF report for the patient analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#2E86AB')
    )
    story.append(Paragraph("BREAST LUMP ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Section
    story.append(Paragraph("PATIENT INFORMATION", styles['Heading2']))
    patient_info = [
        ["Patient Name:", patient_data['name']],
        ["Patient ID:", patient_data['patient_id']],
        ["Age:", str(patient_data['age'])],
        ["Gender:", patient_data['gender']],
        ["Contact:", patient_data.get('contact', 'N/A')],
        ["Medical ID:", patient_data.get('medical_id', 'N/A')],
        ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    
    patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8F9FA')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Prediction Results Section
    story.append(Paragraph("ANALYSIS RESULTS", styles['Heading2']))
    
    preds, probs, risks = prediction_data['pred'], prediction_data['probs'], prediction_data['risks']
    prediction_label = "Malignant" if preds[0] == 1 else "Benign"
    risk_level, probability = risks[0][0], probs[0][1]
    
    results_info = [
        ["Prediction:", f"<b>{prediction_label}</b>"],
        ["Risk Level:", f"<b>{risk_level}</b>"],
        ["Malignant Probability:", f"<b>{probability:.1%}</b>"],
        ["Confidence Score:", f"<b>{max(probs[0])*100:.1f}%</b>"],
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    
    results_table = Table(results_info, colWidths=[2*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Clinical Features Section
    story.append(Paragraph("CLINICAL FEATURES", styles['Heading2']))
    features_data = [["Feature", "Value"]]
    for feature, value in input_data.items():
        display_name = feature.replace('_', ' ').title()
        features_data.append([display_name, f"{value:.3f}"])
    
    features_table = Table(features_data, colWidths=[3*inch, 2*inch])
    features_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(features_table)
    story.append(Spacer(1, 20))
    
    # Medical Notes Section
    if doctor_notes and doctor_notes.strip():
        story.append(Paragraph("CLINICAL NOTES & RECOMMENDATIONS", styles['Heading2']))
        notes_style = ParagraphStyle(
            'NotesStyle',
            parent=styles['Normal'],
            backColor=colors.HexColor('#FFF3CD'),
            borderPadding=10,
            spaceAfter=12
        )
        story.append(Paragraph(doctor_notes.replace('\n', '<br/>'), notes_style))
        story.append(Spacer(1, 15))
    
    # Lifestyle Recommendations
    if (foods_eat and foods_eat.strip()) or (foods_avoid and foods_avoid.strip()):
        story.append(Paragraph("LIFESTYLE RECOMMENDATIONS", styles['Heading2']))
        
        recommendations_content = []
        
        if foods_eat and foods_eat.strip():
            recommendations_content.append(Paragraph("<b>Foods to Include:</b>", styles['Normal']))
            recommendations_content.append(Paragraph(foods_eat.replace('\n', '<br/>'), styles['Normal']))
            recommendations_content.append(Spacer(1, 10))
        
        if foods_avoid and foods_avoid.strip():
            recommendations_content.append(Paragraph("<b>Foods to Avoid:</b>", styles['Normal']))
            recommendations_content.append(Paragraph(foods_avoid.replace('\n', '<br/>'), styles['Normal']))
        
        for item in recommendations_content:
            story.append(item)
        
        story.append(Spacer(1, 20))
    
    # Risk Level Explanation
    story.append(Paragraph("RISK LEVEL INTERPRETATION", styles['Heading2']))
    risk_explanation = """
    <b>Low Risk (0-20%):</b> Routine screening recommended. Maintain healthy lifestyle.<br/>
    <b>Mild Risk (20-40%):</b> Increased monitoring advised. Follow-up in 3-6 months.<br/>
    <b>Moderate Risk (40-60%):</b> Further investigation recommended. Consider additional imaging.<br/>
    <b>High Risk (60-100%):</b> Immediate specialist consultation and biopsy recommended.
    """
    story.append(Paragraph(risk_explanation, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=1
    )
    disclaimer = """
    <i>This report is generated for educational and demonstration purposes only. 
    This is not a medical diagnosis. All clinical decisions should be made by qualified healthcare professionals. 
    Consult with your healthcare provider for proper medical advice and treatment.</i>
    """
    story.append(Paragraph(disclaimer, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_download_link(pdf_buffer, filename):
    """Create a download link for the PDF file."""
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="\
        background-color: #4CAF50;\
        color: white;\
        padding: 12px 24px;\
        text-align: center;\
        text-decoration: none;\
        display: inline-block;\
        border-radius: 4px;\
        font-weight: bold;\
        border: none;\
        cursor: pointer;">\
        📄 Download PDF Report</a>'
    return href

# =============================================================================
# UI HELPER FUNCTIONS (Same as before)
# =============================================================================
def create_risk_gauge(probability):
    """Create a Plotly risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Malignant Probability", 
            'font': {'size': 20, 'color': 'darkblue'}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "darkblue"
            },
            'bar': {'color': "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'lightgreen'},
                {'range': [20, 40], 'color': 'yellow'},
                {'range': [40, 60], 'color': 'orange'},
                {'range': [60, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_risk_barchart(probability, risk_level):
    """Create a horizontal bar chart showing risk level comparison."""
    risk_levels = ['Low Risk', 'Mild Risk', 'Moderate Risk', 'High Risk']
    risk_ranges = ['0-20%', '20-40%', '40-60%', '60-100%']
    risk_colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    values = [20, 20, 20, 20]
    current_risk_index = risk_levels.index(risk_level)
    
    fig = go.Figure()
    
    for i, (level, range_val, color, value) in enumerate(zip(risk_levels, risk_ranges, risk_colors, values)):
        fig.add_trace(go.Bar(
            y=[f"{level}\n{range_val}"],
            x=[value],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='black', width=1 if i != current_risk_index else 3)
            ),
            name=level,
            hovertemplate=f"<b>{level}</b><br>Range: {range_val}<extra></extra>"
        ))
    
    fig.add_trace(go.Scatter(
        y=[f"{risk_levels[current_risk_index]}\n{risk_ranges[current_risk_index]}"],
        x=[10],
        mode='markers+text',
        marker=dict(
            size=20,
            color='black',
            symbol='diamond'
        ),
        text=["📍 CURRENT"],
        textposition="middle right",
        textfont=dict(color='black', size=12, weight='bold'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="🩺 Risk Level Comparison",
            font=dict(size=16, color='darkblue', weight='bold')
        ),
        xaxis=dict(
            title="Risk Scale",
            range=[0, 25],
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            title="",
            showgrid=False
        ),
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white'
    )
    
    return fig

def create_feature_importance_chart(input_data, probability):
    """Create a bar chart showing feature contributions to the risk score."""
    features = list(input_data.keys())
    values = list(input_data.values())
    
    scores = []
    for feature, value in input_data.items():
        if 'worst' in feature and value > np.percentile(list(input_data.values()), 70):
            scores.append(min(3, (value / max(1, np.max(list(input_data.values()))) * 3)))
        else:
            scores.append(min(2, (value / max(1, np.max(list(input_data.values()))) * 2)))
    
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in features],
        'Importance': scores,
        'Value': values
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='📊 Feature Contribution to Risk Assessment',
        color='Importance',
        color_continuous_scale=['lightgreen', 'yellow', 'orange', 'red'],
        hover_data={'Value': ':.3f'}
    )
    
    fig.update_layout(
        height=300,
        xaxis_title="Contribution Score",
        yaxis_title="",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def show_tutorial():
    """Show the tutorial expander if it hasn't been hidden by the user."""
    if st.session_state.show_tutorial:
        with st.expander("🎓 Quick Start Guide - Click to Expand", expanded=True):
            st.markdown("""
            ### Welcome to the Breast Lump Classification App! 🩺

            **Getting Started:**
            
            1.  **Create a New Patient**: Use the 'New Patient' button
            2.  **Enter Clinical Data**: Fill in the feature values on the 'Single Prediction' tab
            3.  **Analyze**: Click 'Analyze Patient' to get the risk assessment
            4.  **Review**: Check the prediction, risk level, and recommendations
            5.  **Customize**: Edit the clinical notes and food recommendations as needed
            6.  **Export**: Download PDF reports for patient records
            7.  **Manage**: View all patient records and their history in the 'Patient History' tab

            **💾 Database Backend**: This app uses SQLite database and can handle 10,000+ patients efficiently.

            ---
            """)
            
            if st.button(
                "✅ Got it! Hide this guide", 
                key="hide_tutorial_button",
                use_container_width=True
            ):
                st.session_state.show_tutorial = False
                st.rerun()

def create_new_patient_form(unique_suffix=""):
    """Displays a form to create a new patient record with unique key."""
    
    if not st.session_state.show_new_patient_form:
        return
    
    st.subheader("📝 Create New Patient Record")
    
    form_key = f"new_patient_form_{unique_suffix}"
    
    with st.form(form_key, clear_on_submit=True):
        name = st.text_input(
            "Full Name*", 
            placeholder="Enter patient's full name",
            help="Required field"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age", 
                min_value=1, 
                max_value=120, 
                value=45,
                help="Patient's age in years"
            )
            contact = st.text_input(
                "Contact Info", 
                placeholder="Phone or email",
                help="Primary contact information"
            )
        
        with col2:
            gender = st.selectbox(
                "Gender", 
                ["Female", "Male", "Other"],
                help="Patient's gender"
            )
            medical_id = st.text_input(
                "Medical Record Number", 
                placeholder="Optional",
                help="Hospital or clinic record number"
            )
        
        notes = st.text_area(
            "Medical Notes", 
            placeholder="Relevant medical history, family history, etc.",
            help="Additional clinical notes and observations"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit_btn = st.form_submit_button(
                "✅ Create Patient", 
                use_container_width=True
            )
        with col2:
            cancel_btn = st.form_submit_button(
                "❌ Cancel", 
                use_container_width=True
            )
        
        if submit_btn:
            if name.strip():
                patient_id = str(uuid.uuid4())[:8].upper()
                patient_data = {
                    'patient_id': patient_id, 
                    'name': name.strip(), 
                    'age': age,
                    'gender': gender, 
                    'contact': contact, 
                    'medical_id': medical_id,
                    'notes': notes, 
                    'created_date': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Save to database
                save_patient(patient_data)
                
                st.session_state.current_patient_id = patient_id
                st.session_state.show_new_patient_form = False
                
                st.success(f"✅ Patient '{name}' created successfully!")
                st.rerun()
            else:
                st.error("❌ Patient name is required.")
        
        if cancel_btn:
            st.session_state.show_new_patient_form = False
            st.rerun()

def patient_selection_section():
    """UI section for selecting an existing patient with search and pagination."""
    st.subheader("👥 Patient Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search functionality
        search_term = st.text_input(
            "🔍 Search patients by name, ID, or medical ID...",
            value=st.session_state.search_term,
            key="patient_search"
        )
        
        if search_term != st.session_state.search_term:
            st.session_state.search_term = search_term
            st.session_state.current_page = 0  # Reset to first page when searching
        
        # Get patients with pagination
        offset = st.session_state.current_page * st.session_state.patients_per_page
        patients_df = search_patients(
            search_term=st.session_state.search_term,
            limit=st.session_state.patients_per_page,
            offset=offset
        )
        
        # Create patient options for dropdown
        if not patients_df.empty:
            patient_options = {
                f"{row['name']} (ID: {row['patient_id']})": row['patient_id'] 
                for _, row in patients_df.iterrows()
            }
            
            # Find current patient for default selection
            current_patient_display = next(
                (k for k, v in patient_options.items() 
                 if v == st.session_state.current_patient_id), 
                None
            )

            selected_display = st.selectbox(
                "Choose Patient",
                options=["Select a patient..."] + list(patient_options.keys()),
                index=(
                    list(patient_options.keys()).index(current_patient_display) + 1 
                    if current_patient_display else 0
                ),
                key="patient_selector"
            )
            
            if selected_display != "Select a patient...":
                st.session_state.current_patient_id = patient_options[selected_display]
            else:
                st.session_state.current_patient_id = None
            
            # Pagination controls
            total_patients = get_patient_count()
            total_pages = max(1, (total_patients + st.session_state.patients_per_page - 1) // st.session_state.patients_per_page)
            
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                if st.session_state.current_page > 0:
                    if st.button("⬅️ Previous", key="prev_page"):
                        st.session_state.current_page -= 1
                        st.rerun()
            with col2:
                st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
            with col3:
                st.write(f"Total patients: {total_patients}")
            with col4:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Next ➡️", key="next_page"):
                        st.session_state.current_page += 1
                        st.rerun()
        else:
            st.info("No patients found. Create a new patient to get started.")
            st.session_state.current_patient_id = None
            
    with col2:
        if st.button(
            "➕ New Patient", 
            key="new_patient_button", 
            use_container_width=True
        ):
            st.session_state.show_new_patient_form = True
            st.rerun()
    
    if st.session_state.current_patient_id:
        show_current_patient_info()

def show_current_patient_info():
    """Display a summary card for the currently selected patient."""
    patient = get_patient(st.session_state.current_patient_id)
    if not patient:
        st.session_state.current_patient_id = None
        return

    with st.container(border=True):
        st.markdown(f"### 👤 Current Patient: {patient['name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Patient ID:** {patient['patient_id']}")
            st.write(f"**Age:** {patient['age']}")
        
        with col2:
            st.write(f"**Gender:** {patient['gender']}")
            st.write(f"**Contact:** {patient.get('contact', 'N/A')}")
        
        with col3:
            st.write(f"**Medical ID:** {patient.get('medical_id', 'N/A')}")
            # Get prediction count from database
            pred_count = len(get_patient_predictions(patient['patient_id']))
            st.metric("Total Predictions", pred_count)

# =============================================================================
# TAB FUNCTIONS
# =============================================================================
def single_prediction_tab(predictor, threshold):
    """Content for the 'Single Patient Analysis' tab."""
    st.header("🔍 Single Patient Analysis")
    
    patient_selection_section()
    
    create_new_patient_form("single_prediction")
    
    if st.session_state.current_patient_id:
        st.subheader("🔢 Clinical Feature Input")
        
        with st.form("feature_input_form_single", clear_on_submit=False):
            cols = st.columns(3)
            input_data = {}
            features = list(st.session_state.inputs.keys())
            
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    display_name = feature.replace('_', ' ').title()
                    value = st.number_input(
                        display_name, 
                        value=float(st.session_state.inputs.get(feature, 0.0)),
                        key=f"input_{feature}_single", 
                        step=0.01, 
                        format="%.3f",
                        help=f"Enter value for {display_name}"
                    )
                    input_data[feature] = value
            
            analyze_button = st.form_submit_button(
                "🎯 Analyze Patient", 
                type="primary", 
                use_container_width=True
            )

        if analyze_button:
            st.session_state.inputs.update(input_data)
            input_df = pd.DataFrame([input_data])
            
            with st.spinner("🔬 Analyzing clinical features..."):
                preds, probs, risks = predictor.predict(input_df, threshold)
                
                st.session_state.single_pred = {
                    'pred': preds, 
                    'probs': probs, 
                    'risks': risks,
                    'input_data': input_data.copy()
                }
                
                # Save prediction to database
                prediction_record = {
                    'patient_id': st.session_state.current_patient_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'prediction': "Malignant" if preds[0] == 1 else "Benign",
                    'probability': float(probs[0][1] * 100),
                    'risk_level': risks[0][0],
                    'threshold': threshold,
                    'features': {k: float(v) for k, v in input_data.items()}
                }
                save_prediction(prediction_record)

        if st.session_state.single_pred:
            pred_data = st.session_state.single_pred
            preds, probs, risks = pred_data['pred'], pred_data['probs'], pred_data['risks']
            prediction_label = "Malignant" if preds[0] == 1 else "Benign"
            risk_level, probability = risks[0][0], probs[0][1]
            
            # Display prediction result with PDF download button
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if preds[0] == 1:
                    st.error(
                        f"🚨 **PREDICTION:** {prediction_label} | "
                        f"**RISK LEVEL:** {risk_level} | "
                        f"**PROBABILITY:** {probability:.1%}"
                    )
                else:
                    st.success(
                        f"✅ **PREDICTION:** {prediction_label} | "
                        f"**RISK LEVEL:** {risk_level} | "
                        f"**PROBABILITY:** {probability:.1%}"
                    )
            
            with col2:
                st.metric("Confidence Score", f"{max(probs[0])*100:.1f}%")
            
            with col3:
                # Generate and display PDF download link
                patient_data = get_patient(st.session_state.current_patient_id)
                pdf_buffer = generate_pdf_report(
                    patient_data,
                    pred_data,
                    pred_data['input_data'],
                    st.session_state.doctor_notes_text,
                    st.session_state.foods_eat_text,
                    st.session_state.foods_avoid_text
                )
                
                filename = f"Breast_Analysis_Report_{patient_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.markdown(create_download_link(pdf_buffer, filename), unsafe_allow_html=True)

            # Risk visualization section
            st.subheader("📊 Risk Assessment Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.plotly_chart(
                    create_risk_gauge(probability), 
                    use_container_width=True
                )
            
            with viz_col2:
                st.plotly_chart(
                    create_risk_barchart(probability, risk_level),
                    use_container_width=True
                )
            
            st.plotly_chart(
                create_feature_importance_chart(pred_data['input_data'], probability),
                use_container_width=True
            )
            
            # Clinical notes and instructions
            st.subheader("🩺 Clinical Notes & Instructions")
            
            updated_doctor_notes = st.text_area(
                "Doctor Notes & Recommendations", 
                value=st.session_state.doctor_notes_text, 
                key="doctor_notes_area", 
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                updated_foods_eat = st.text_area(
                    "Foods to Eat", 
                    value=st.session_state.foods_eat_text, 
                    key="foods_eat_area", 
                    height=120
                )
            with col2:
                updated_foods_avoid = st.text_area(
                    "Foods to Avoid", 
                    value=st.session_state.foods_avoid_text, 
                    key="foods_avoid_area", 
                    height=120
                )
            
            # Save settings when changed
            if (updated_doctor_notes != st.session_state.doctor_notes_text or
                updated_foods_eat != st.session_state.foods_eat_text or
                updated_foods_avoid != st.session_state.foods_avoid_text):
                
                st.session_state.doctor_notes_text = updated_doctor_notes
                st.session_state.foods_eat_text = updated_foods_eat
                st.session_state.foods_avoid_text = updated_foods_avoid
                
                # Save to database
                save_app_settings({
                    'doctor_notes_text': updated_doctor_notes,
                    'foods_eat_text': updated_foods_eat,
                    'foods_avoid_text': updated_foods_avoid
                })
            
            # Additional recommendations
            st.subheader("💡 Recommended Actions")
            if risk_level == "High Risk":
                st.warning("""
                **Immediate Actions Recommended:**
                - Schedule urgent biopsy
                - Consult with oncology specialist
                - Consider advanced imaging (MRI)
                - Discuss treatment options
                """)
            elif risk_level == "Moderate Risk":
                st.info("""
                **Follow-up Actions:**
                - Schedule follow-up in 2-4 weeks
                - Consider ultrasound
                - Monitor for changes
                - Discuss risk factors
                """)
            elif risk_level == "Mild Risk":
                st.info("""
                **Monitoring Recommendations:**
                - Regular self-exams
                - Follow-up in 3-6 months
                - Maintain healthy lifestyle
                - Annual screening
                """)
            else:
                st.success("""
                **Routine Care:**
                - Continue regular screenings
                - Maintain healthy habits
                - Annual clinical exams
                - Self-awareness of changes
                """)
                
            # Refresh PDF button
            st.info("💡 **Note**: If you update the clinical notes or food recommendations, click the button below to refresh the PDF report.")
            if st.button("🔄 Refresh PDF Report with Updated Notes", key="refresh_pdf_button"):
                # Regenerate PDF with updated notes
                pdf_buffer = generate_pdf_report(
                    patient_data,
                    pred_data,
                    pred_data['input_data'],
                    st.session_state.doctor_notes_text,
                    st.session_state.foods_eat_text,
                    st.session_state.foods_avoid_text
                )
                
                filename = f"Breast_Analysis_Report_{patient_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.markdown(create_download_link(pdf_buffer, filename), unsafe_allow_html=True)
                st.success("PDF report updated with latest notes!")
    else:
        st.info("👆 Please select or create a patient to begin analysis.")

def batch_prediction_tab(predictor, threshold):
    """Content for the 'Batch Patient Analysis' tab."""
    st.header("📁 Batch Patient Analysis")
    
    st.info(
        "Upload a CSV or Excel file with patient data. "
        "The file must contain columns matching the required clinical features."
    )
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx'], 
        key="batch_file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head())
            
            if st.button(
                "🔍 Analyze All Patients", 
                type="primary", 
                key="batch_analyze_button"
            ):
                with st.spinner(f"Analyzing {len(df)} patient records..."):
                    preds, probs, risks = predictor.predict(df, threshold)
                    
                    results_df = df.copy()
                    results_df['Prediction'] = [
                        'Malignant' if p == 1 else 'Benign' for p in preds
                    ]
                    results_df['Risk_Level'] = [r[0] for r in risks]
                    results_df['Probability_Malignant'] = [p[1] * 100 for p in probs]
                    
                    st.session_state.batch_pred = {'df': results_df}

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    if st.session_state.batch_pred:
        results_df = st.session_state.batch_pred['df']
        
        st.subheader("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(results_df))
        with col2:
            benign_count = (results_df['Prediction'] == 'Benign').sum()
            st.metric("Benign Cases", benign_count)
        with col3:
            st.metric("Malignant Cases", len(results_df) - benign_count)
        with col4:
            high_risk_count = (results_df['Risk_Level'] == 'High Risk').sum()
            st.metric("High Risk Cases", high_risk_count)
        
        st.subheader("📋 Detailed Results")
        st.dataframe(results_df)
        
        st.subheader("📊 Risk Distribution")
        risk_counts = results_df['Risk_Level'].value_counts()
        
        fig = Figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        ax1.pie(
            risk_counts.values, 
            labels=risk_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=['lightgreen', 'yellow', 'orange', 'red']
        )
        ax1.set_title('Risk Level Distribution', fontweight='bold')
        
        risk_counts.sort_index().plot(
            kind='bar', 
            ax=ax2, 
            color=['lightgreen', 'yellow', 'orange', 'red']
        )
        ax2.set_title('Risk Level Counts', fontweight='bold')
        ax2.set_ylabel('Number of Patients')
        ax2.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        
        st.pyplot(fig)

def patient_history_tab():
    """Content for the 'Patient Management & History' tab."""
    st.header("📋 Patient Management & History")
    
    if st.button(
        "➕ Create New Patient", 
        key="history_new_patient_button", 
        use_container_width=True
    ):
        st.session_state.show_new_patient_form = True
        st.rerun()
    
    create_new_patient_form("patient_history")
    st.divider()

    # Patient search and pagination
    st.subheader("🔍 Patient Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search by name, ID, or medical ID...",
            key="history_search"
        )
    with col2:
        patients_per_page = st.selectbox(
            "Patients per page",
            [10, 20, 50, 100],
            index=1,
            key="history_patients_per_page"
        )
    
    # Get patients with pagination
    offset = st.session_state.current_page * patients_per_page
    patients_df = search_patients(
        search_term=search_term,
        limit=patients_per_page,
        offset=offset
    )
    
    total_patients = get_patient_count()
    
    if patients_df.empty:
        st.info("📝 No patient records found. Create a patient to get started.")
        return
    
    st.subheader(f"👥 Patient Records ({total_patients} total)")
    
    # Display patients
    for _, patient_row in patients_df.iterrows():
        with st.expander(f"**{patient_row['name']}** (ID: {patient_row['patient_id']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age", patient_row['age'])
            with col2:
                st.metric("Gender", patient_row['gender'])
            with col3:
                pred_count = len(get_patient_predictions(patient_row['patient_id']))
                st.metric("Total Predictions", pred_count)
            
            if patient_row['notes']:
                st.write("**Medical Notes:**")
                st.info(patient_row['notes'])
                
            # Get prediction history
            predictions = get_patient_predictions(patient_row['patient_id'], limit=5)
            if not predictions.empty:
                st.write("**Prediction History (Latest 5):**")
                st.dataframe(
                    predictions[['timestamp', 'prediction', 'risk_level', 'probability']],
                    use_container_width=True
                )
            else:
                st.info("No predictions recorded for this patient yet.")
    
    # Pagination controls
    total_pages = max(1, (total_patients + patients_per_page - 1) // patients_per_page)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_page > 0:
            if st.button("⬅️ Previous", key="history_prev"):
                st.session_state.current_page -= 1
                st.rerun()
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
    with col3:
        if st.session_state.current_page < total_pages - 1:
            if st.button("Next ➡️", key="history_next"):
                st.session_state.current_page += 1
                st.rerun()

def about_tab():
    """Content for the 'About' tab."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Breast Lump Classification App 🩺

    This application is a demonstration tool designed to assist healthcare professionals 
    in analyzing breast lump characteristics and assessing cancer risk based on clinical features.

    ### 🎯 Purpose
    - Provide a simulated risk assessment for breast lumps based on clinical measurements
    - Support clinical decision-making with illustrative recommendations  
    - Maintain a robust system for patient records and prediction history
    - Generate comprehensive PDF reports for patient documentation

    ### 🔬 Key Features
    - **Single Patient Analysis**: Individual risk assessment with PDF export
    - **Batch Processing**: Analyze multiple patients from files
    - **Patient Management**: Comprehensive record keeping with SQLite database
    - **Visual Analytics**: Interactive risk gauges and charts
    - **Report Generation**: Professional PDF reports with customizable notes
    - **Scalable Database**: Supports 10,000+ patient records efficiently

    ### ⚠️ Medical Disclaimer
    **Important**: This tool is for educational and illustrative purposes only. 
    It is **not a medical device** and should not be used for actual diagnosis or treatment. 
    All clinical decisions must be made by qualified healthcare professionals.
    
    ---
    
    *Built with Streamlit for healthcare education and demonstration purposes.*
    """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application function to run the Streamlit app."""
    
    st.title("🩺 Breast Lump Classification System")
    st.markdown(
        "Analyze clinical features to assess breast cancer risk and support clinical decision-making."
    )
    
    show_tutorial()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        threshold = st.slider(
            "Classification Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Adjust the sensitivity for malignant classification. Higher values are more strict.",
            key="classification_threshold"
        )
        
        st.header("📊 Risk Categories")
        st.info("""
        **Risk Levels:**
        - 🟢 **Low Risk**: 0-20%
        - 🟡 **Mild Risk**: 20-40%  
        - 🟠 **Moderate Risk**: 40-60%
        - 🔴 **High Risk**: 60-100%
        """)
        
        st.header("🚀 Quick Actions")
        
        st.warning("The action below will reset all session data (but keeps database records).")
        if st.button("🔄 Reset Session Data", key="reset_session_button"):
            keys_to_reset = [
                'current_patient_id', 'single_pred', 'batch_pred', 
                'show_new_patient_form', 'current_page', 'search_term'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            initialize_session_state()
            st.success("Session data reset!")
            st.rerun()

        if st.button("📖 Show Tutorial", key="sidebar_show_tutorial_button"):
            st.session_state.show_tutorial = True
            st.rerun()
        
        st.header("📈 Application Status")
        patient_count = get_patient_count()
        st.metric("Patients in Database", patient_count)
        
        st.header("💾 Database Status")
        if os.path.exists(DB_FILE):
            file_size = os.path.getsize(DB_FILE)
            st.success(f"✅ SQLite Database")
            st.info(f"File size: {file_size:,} bytes")
            
            # Show database info
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            c.execute('SELECT COUNT(*) FROM predictions')
            pred_count = c.fetchone()[0]
            
            c.execute('SELECT COUNT(*) FROM app_settings')
            settings_count = c.fetchone()[0]
            
            conn.close()
            
            st.write(f"Predictions: {pred_count}")
            st.write(f"Settings: {settings_count}")
        else:
            st.warning("❌ Database file not found")
        
        if st.session_state.current_patient_id:
            current_patient = get_patient(st.session_state.current_patient_id)
            if current_patient:
                st.success(f"Selected: {current_patient['name']}")
        else:
            st.warning("No patient selected")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Prediction", 
        "📁 Batch Analysis", 
        "📋 Patient History", 
        "ℹ️ About"
    ])
    
    with tab1:
        single_prediction_tab(BreastCancerPredictor(), threshold)
    with tab2:
        batch_prediction_tab(BreastCancerPredictor(), threshold)
    with tab3:
        patient_history_tab()
    with tab4:
        about_tab()

if __name__ == "__main__":
    main()