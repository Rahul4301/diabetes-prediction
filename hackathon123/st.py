import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer

# Set page configuration
st.set_page_config(
    page_title="Advanced Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for modern UI
# Add Tailwind CSS for modern UI
def add_tailwind_css():
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
    .main {
        padding: 2rem;
        background-color: oklch(0.205 0 0);
    }
    .block-container {
        padding-top: 1.5rem;
    }
    h1, h2, h3 {
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: green;
        color: white;
        font-weight: bold;
        border: none !important; /* Removes border */
        outline: none !important; /* Removes outline */
        box-shadow: none !important; /* Removes any shadow */
    }
    .stProgress > div > div {
        background-color: oklch(0.205 0 0);
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: oklch(0.205 0 0);
        border-left: 5px solid #4CAF50;
    }
    .high-risk {
        border-left: 5px solid #f44336;
    }
    .moderate-risk {
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        border-left: 5px solid #4CAF50;
    }
    .feature-importance {
        background-color: oklch(0.268 0.007 34.298);
        padding: 15px;
        border-radius: 10px;
    }
    .explanation {
        background-color: oklch(0.205 0 0);
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

add_tailwind_css()

# Define the neural network model class
class DiabetesNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(DiabetesNN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        
        # Hidden layers with Batch Normalization and Dropout
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            ])
        
        # Output layer
        layers.extend([
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to create derived features
def create_derived_features(user_data_df):
    df_derived = user_data_df.copy()
    
    # Create BMI categories
    df_derived['BMI_Category'] = pd.cut(df_derived['BMI'], 
                                       bins=[0, 18.5, 25, 30, 100],
                                       labels=[0, 1, 2, 3])
    
    # Create Age groups
    df_derived['Age_Group'] = pd.cut(df_derived['Age'],
                                    bins=[20, 30, 40, 50, 60, 100],
                                    labels=[0, 1, 2, 3, 4])
    
    # Calculate Glucose to Insulin ratio (with safe division)
    df_derived['Glucose_Insulin_Ratio'] = df_derived['Glucose'] / df_derived['Insulin'].replace(0, 1)
    
    # Interaction between Age and BMI
    df_derived['Age_BMI_Interaction'] = df_derived['Age'] * df_derived['BMI'] / 100
    
    # Interaction between Pregnancies and Age
    df_derived['Preg_Age_Interaction'] = df_derived['Pregnancies'] * df_derived['Age'] / 10
    
    # Convert categorical columns to int (handle NaN values that might be generated for edge cases)
    df_derived['BMI_Category'] = df_derived['BMI_Category'].fillna(0).astype(int)
    df_derived['Age_Group'] = df_derived['Age_Group'].fillna(0).astype(int)
    
    return df_derived

# Data Preprocessing Class (must match the one from training script)
class DiabetesPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        
    def fit(self, X):
        # Replace zeros with NaN for specific columns where zero doesn't make sense
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        X_processed = X.copy()
        for col in cols_with_zeros:
            X_processed.loc[X_processed[col] == 0, col] = np.nan
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X_processed)
        
        # Apply power transform to make data more Gaussian-like
        X_transformed = self.power_transformer.fit_transform(X_imputed)
        
        # Scale the features
        self.scaler.fit(X_transformed)
        
        return X_processed
        
    def transform(self, X):
        # Replace zeros with NaN for specific columns
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        X_processed = X.copy()
        for col in cols_with_zeros:
            X_processed.loc[X_processed[col] == 0, col] = np.nan
        
        # Apply the same preprocessing as during fit
        X_imputed = self.imputer.transform(X_processed)
        X_transformed = self.power_transformer.transform(X_imputed)
        X_scaled = self.scaler.transform(X_transformed)
        
        return X_scaled
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Class for model wrapper (to be consistent with training script)
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
            return (probs >= 0.5).astype(int)
            
    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
            return np.hstack([1-probs, probs])  # Return probabilities for both classes

# Function to load the model and preprocessor
@st.cache_resource
def load_model_and_preprocessor(model_dir="model"):
    try:
        # Load model architecture
        architecture = torch.load(os.path.join(model_dir, 'model_architecture.pt'), 
                                map_location=torch.device('cpu'))
        input_dim = architecture['input_dim']
        hidden_dims = architecture['hidden_dims']
        
        # Initialize the model
        model = DiabetesNN(input_dim=input_dim, hidden_dims=hidden_dims)
        
        # Load the model weights
        model.load_state_dict(torch.load(os.path.join(model_dir, 'diabetes_model.pt'), 
                                         map_location=torch.device('cpu')))
        model.eval()
        
        # Create model wrapper
        wrapped_model = ModelWrapper(model)
        
        # Load preprocessor
        preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        
        # Load feature names
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        
        return wrapped_model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Function to make predictions
def predict_diabetes(model, preprocessor, feature_names, user_data):
    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Ensure DataFrame has the correct columns
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0  # Default value for missing features
    
    # Create derived features
    user_df_derived = create_derived_features(user_df)
    
    # Ensure the order of columns matches what the model expects
    user_df_derived = user_df_derived[feature_names]
    
    # Preprocess the data
    user_data_processed = preprocessor.transform(user_df_derived)
    
    # Make prediction
    prob = model.predict_proba(user_data_processed)[0, 1]  # Get probability of class 1
    
    return prob

# Function to load and display images
@st.cache_data
def display_image(image_path, caption=None):
    try:
        image = Image.open(image_path)
        return image
    except:
        st.warning(f"Could not load image: {image_path}")
        return None

# Function to get risk factors from user data
def get_risk_factors(user_data):
    risk_factors = []
    
    if user_data['Age'] > 45:
        risk_factors.append(("Age above 45", "Diabetes risk increases with age, particularly after 45."))
    
    if user_data['BMI'] >= 30:
        risk_factors.append(("Obesity (BMI ≥ 30)", "Obesity significantly increases insulin resistance."))
    elif user_data['BMI'] >= 25:
        risk_factors.append(("Overweight (BMI 25-29.9)", "Being overweight increases your risk of developing type 2 diabetes."))
    
    if user_data['Glucose'] >= 140:
        risk_factors.append(("High blood glucose", "Elevated blood sugar levels may indicate prediabetes or diabetes."))
    elif user_data['Glucose'] >= 100:
        risk_factors.append(("Elevated blood glucose", "Blood sugar levels between 100-140 mg/dL may indicate prediabetes."))
    
    if user_data['BloodPressure'] >= 90:
        risk_factors.append(("High blood pressure", "Hypertension is associated with increased diabetes risk."))
    elif user_data['BloodPressure'] >= 80:
        risk_factors.append(("Elevated blood pressure", "Blood pressure in the prehypertension range."))
    
    if user_data['DiabetesPedigreeFunction'] > 0.5:
        risk_factors.append(("Family history of diabetes", "Genetic factors play a significant role in diabetes risk."))
    
    if user_data.get('Insulin', 0) > 140:
        risk_factors.append(("High insulin levels", "May indicate insulin resistance."))
    
    if user_data['SkinThickness'] > 30:
        risk_factors.append(("Elevated skin fold thickness", "May be associated with insulin resistance."))
        
    if user_data['Pregnancies'] >= 4:
        risk_factors.append(("Multiple pregnancies", "Having had multiple pregnancies may increase diabetes risk."))
    
    return risk_factors

# Function to get recommendations based on risk
def get_recommendations(risk_level, risk_factors):
    # Common recommendations for all risk levels
    common_recs = [
        "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins",
        "Engage in regular physical activity (aim for at least 150 minutes per week)",
        "Maintain a healthy weight or work towards weight loss if overweight",
        "Monitor your health with regular check-ups"
    ]
    
    # Risk-specific recommendations
    if risk_level == "High Risk":
        specific_recs = [
            "Consult with a healthcare provider promptly for proper screening and evaluation",
            "Consider getting a fasting plasma glucose test or HbA1c test",
            "If recommended by your doctor, consider a continuous glucose monitor",
            "Learn about diabetes symptoms to watch for (increased thirst, frequent urination, fatigue)",
            "Consider meeting with a dietitian to develop a personalized nutrition plan"
        ]
    elif risk_level == "Moderate Risk":
        specific_recs = [
            "Schedule a check-up with your healthcare provider",
            "Consider getting a fasting plasma glucose test",
            "Focus on reducing refined carbohydrates and sugars in your diet",
            "Increase daily physical activity, particularly after meals",
            "Monitor your weight and work on maintaining it within a healthy range"
        ]
    else:  # Low Risk
        specific_recs = [
            "Continue your healthy habits",
            "Stay informed about diabetes prevention strategies",
            "Consider annual health screenings to monitor for changes",
            "Maintain awareness of family health history"
        ]
    
    # Add targeted recommendations based on specific risk factors
    targeted_recs = []
    risk_factor_names = [rf[0] for rf in risk_factors]
    
    if "Obesity (BMI ≥ 30)" in risk_factor_names or "Overweight (BMI 25-29.9)" in risk_factor_names:
        targeted_recs.append("Focus on gradual, sustainable weight loss of 5-10% of your current weight")
    
    if "High blood glucose" in risk_factor_names or "Elevated blood glucose" in risk_factor_names:
        targeted_recs.append("Limit intake of refined carbohydrates and added sugars")
        targeted_recs.append("Consider eating smaller, more frequent meals throughout the day")
    
    if "High blood pressure" in risk_factor_names or "Elevated blood pressure" in risk_factor_names:
        targeted_recs.append("Reduce sodium intake and consider following the DASH diet")
        targeted_recs.append("Limit alcohol consumption and avoid smoking")
    
    if "Family history of diabetes" in risk_factor_names:
        targeted_recs.append("Be particularly vigilant about regular screenings due to your genetic predisposition")
        
    if "Multiple pregnancies" in risk_factor_names:
        targeted_recs.append("Continue regular screenings post-pregnancy as gestational diabetes risk may persist")
    
    return common_recs, specific_recs, targeted_recs

# Main function
def main():
    # Load model and preprocessor
    model, preprocessor, feature_names = load_model_and_preprocessor()
    if model is None:
        st.warning("Please make sure you've trained the model first. Run the training script before using this app.")
        st.info("If you've already run the training script, check that the model files exist in the 'model' directory.")
        return
    
    # App header with color scheme
    st.markdown("""
    
    <div style="padding: 1.5rem; border-radius: 10px; background-color: oklch(0.274 0.006 286.033); margin-bottom: 20px;">
        <h1 style="margin:0; color: oklch(0.869 0.005 56.366);">Sugar-Sense: Advanced Diabetes Risk Assessment</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Deep learning model for Pima Indians diabetes prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About expander
    with st.expander("ℹ️ About this assessment tool"):
        st.markdown("""
        This advanced diabetes risk prediction tool uses a neural network trained on the Pima Indians Diabetes Dataset. 
        
        ### Data Source
        The dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases and includes diagnostic 
        measurements from females at least 21 years old of Pima Indian heritage.
        
        ### Model Details
        - **Architecture**: Multi-layer neural network with batch normalization and dropout
        - **Preprocessing**: Handling of missing values, feature scaling, and feature engineering
        - **Performance**: Trained with cross-validation and early stopping to prevent overfitting
        
        ### Privacy Notice
        All data entered is processed locally in your browser and is not stored or transmitted.
        
        ### Disclaimer
        This tool provides an assessment of diabetes risk based on statistical patterns and should not replace 
        professional medical advice. Always consult with a healthcare provider for proper diagnosis and guidance.
        """)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Risk Assessment", "Data Insights"])
    
    with tab1:
        st.subheader("Enter Your Information")
        
        # Create three columns for input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=21, max_value=120, value=35, 
                                help="Enter your age in years (must be at least 21)")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=110, 
                                    help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
            insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=100, 
                                    help="2-Hour serum insulin level")
            
        with col2:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2, 
                                        help="Number of times pregnant")
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=40, max_value=200, value=72, 
                                           help="Diastolic blood pressure")
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.3, 
                                              help="A function that scores the likelihood of diabetes based on family history")
            
        with col3:
            height = st.number_input("Height (cm)", min_value=20, max_value=220, value=165, 
                                   help="Your height in centimeters")
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=68, 
                                   help="Your weight in kilograms")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23, 
                                          help="Triceps skin fold thickness in millimeters")
            
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Display BMI with color coding
        bmi_col1, bmi_col2 = st.columns([1, 2])
        with bmi_col1:
            st.metric("BMI", f"{bmi:.1f}", help="Body Mass Index (weight in kg / height in m²)")
        
        with bmi_col2:
            if bmi < 18.5:
                st.info("BMI Category: Underweight")
            elif bmi < 25:
                st.success("BMI Category: Normal weight")
            elif bmi < 30:
                st.warning("BMI Category: Overweight")
            else:
                st.error("BMI Category: Obese")
        
        # Button to make prediction
        if st.button("Calculate My Risk", type="primary"):
            # Show spinner while processing
            with st.spinner("Analyzing your data..."):
                # Prepare user data
                user_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': diabetes_pedigree,
                    'Age': age
                }
                
                # Make prediction
                risk_probability = float(predict_diabetes(model, preprocessor, feature_names, user_data))
                risk_percentage = risk_probability * 100
                
                # Determine risk level
                if risk_percentage < 30:
                    risk_level = "Low Risk"
                    risk_class = "low-risk"
                    explanation = "Your profile suggests a lower likelihood of having diabetes based on the factors you provided."
                elif risk_percentage < 60:
                    risk_level = "Moderate Risk"
                    risk_class = "moderate-risk" 
                    explanation = "You have some risk factors that may increase your chance of having diabetes."
                else:
                    risk_level = "High Risk"
                    risk_class = "high-risk"
                    explanation = "Your profile indicates several significant risk factors associated with diabetes."
                
                # Get risk factors and recommendations
                risk_factors = get_risk_factors(user_data)
                common_recs, specific_recs, targeted_recs = get_recommendations(risk_level, risk_factors)
                
                # Display results
                st.markdown("---")
                st.subheader("Your Results")
                
                # Show risk meter (progress bar)
                st.progress(risk_probability)
                
                # Display risk category with custom styling
                st.markdown(f"""
                <div class="result-card {risk_class}">
                    <h3>{risk_level} ({risk_percentage:.1f}%)</h3>
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display identified risk factors
                if risk_factors:
                    st.markdown("### Risk Factors Identified")
                    for factor, description in risk_factors:
                        st.markdown(f"**{factor}**: {description}")
                else:
                    st.markdown("### No Significant Risk Factors Identified")
                    st.markdown("Based on your inputs, no major risk factors were identified.")
                
                # Display recommendations
                st.markdown("### Recommendations")
                
                # General recommendations
                st.markdown("#### General Health Recommendations")
                for rec in common_recs:
                    st.markdown(f"- {rec}")
                
                # Specific recommendations based on risk level
                st.markdown(f"#### Recommendations for {risk_level}")
                for rec in specific_recs:
                    st.markdown(f"- {rec}")
                
                # Targeted recommendations based on specific risk factors
                if targeted_recs:
                    st.markdown("#### Targeted Recommendations")
                    for rec in targeted_recs:
                        st.markdown(f"- {rec}")
                
                # Disclaimer
                st.markdown("---")
                st.caption("""
                **Disclaimer:** This assessment is for informational purposes only and does not constitute medical advice. 
                The prediction is based on statistical patterns observed in a specific population (Pima Indian heritage females) 
                and may not be applicable to all individuals. Always consult with a healthcare provider for proper evaluation 
                and guidance.
                """)
    
    with tab2:
        st.subheader("Understanding Diabetes Risk Factors")
        
        # Load and display feature importance
        try:
            feature_importance_img = display_image("model/feature_importance.png")
            if feature_importance_img:
                st.image(feature_importance_img, caption="Feature Importance in the Model", use_column_width=True)
                st.markdown("""
                <div class="explanation">
                    <h4>Understanding Feature Importance</h4>
                    <p>The chart above shows which factors have the strongest influence on diabetes prediction in our model. 
                    Longer bars indicate more important features for making predictions.</p>
                    <p>This can help you understand which health metrics have the most impact on diabetes risk.</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("Feature importance visualization will be shown here after model training is complete.")
        
        # Educational content about diabetes risk factors
        st.markdown("""
        ### Key Diabetes Risk Factors
        
        #### Glucose Levels
        High blood glucose is one of the strongest indicators of diabetes. Normal fasting glucose is below 100 mg/dL, 
        prediabetes is 100-125 mg/dL, and diabetes is 126 mg/dL or higher.
        
        #### Body Mass Index (BMI)
        BMI above 25 (overweight) increases diabetes risk, with risk rising significantly at BMI 30+ (obese). 
        Even modest weight loss of 5-10% can significantly reduce risk.
        
        #### Age
        Risk increases with age, particularly after 45 years. This is partly due to decreased physical activity, 
        loss of muscle mass, and increased insulin resistance as we age.
        
        #### Family History
        Having a parent or sibling with diabetes increases your risk. The Diabetes Pedigree Function in our model 
        captures the genetic influence of diabetes based on family history.
        
        #### Blood Pressure
        Hypertension (blood pressure ≥ 140/90 mmHg) often co-occurs with diabetes and can increase complications.
        
        #### Insulin Levels
        High insulin levels with normal or high glucose can indicate insulin resistance, a precursor to type 2 diabetes.
        
        #### Pregnancies
        For women, having had gestational diabetes during pregnancy increases future diabetes risk. Multiple pregnancies
        may also play a role in increasing risk.
        """)
        
        # Display additional visualizations if available
        try:
            training_loss_img = display_image("model/training_loss.png")
            if training_loss_img:
                st.image(training_loss_img, caption="Model Training Performance", use_column_width=True)
        except:
            pass

if __name__ == "__main__":
    main()
