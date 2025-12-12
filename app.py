import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration 
st.set_page_config(
    page_title="RetentionAI | Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# 2. Load Model (Cached for performance) 
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    try:
        model.load_model("churn_model.cbm")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 3. Main Title 
st.title("📊 RetentionAI: Intelligent Churn Prediction")
st.markdown("""
This application uses **Machine Learning (CatBoost)** and **Explainable AI (SHAP)** to predict if a customer is likely to churn and explains **why**.
""")
st.markdown("---")

# 4. Sidebar: User Inputs 
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Adjust the values below to simulate a customer:")

def user_input_features():
    # 1. Data received from the user (Sidebar)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)
    
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    
    # 2. DataFrame Oluşturma
    data = {
        'Contract': contract,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService': internet_service,
        'TechSupport': tech_support,
        'PaymentMethod': payment_method,
        'OnlineSecurity': online_security,
        # Other features we don't obtain from the user (Defaults)
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'Yes'
    }
    
    df = pd.DataFrame(data, index=[0])

    # 3. CRITICAL STEP: We arrange the columns in the order the model was trained! 
    # The model is waiting for this order (Telco Dataset Original Order)
    expected_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'MonthlyCharges', 'TotalCharges'
    ]
    
    # Rearrange the DataFrame in this order
    df = df[expected_order]
    
    return df

input_df = user_input_features()

# Display Selected Data (Optional)
with st.expander("Show Selected Customer Data", expanded=False):
    st.dataframe(input_df)

# Prediction & SHAP Logic 
if st.button("Analyze Risk", type="primary"):
    
    if model:
        # A. Calculate Probability
        # Returns [probability_stay, probability_churn] -> We take index 1
        prob_churn = model.predict_proba(input_df)[0][1]
        
        # B. Define Optimal Threshold (Update this with your notebook result!)
        THRESHOLD = 0.56 
        
        # C. Display Results
        col1, col2 = st.columns([1, 2])
        
        # Result Column
        with col1:
            st.subheader("Prediction Result")
            if prob_churn > THRESHOLD:
                st.error("⚠️ HIGH RISK (Churn)")
                st.metric(label="Churn Probability", value=f"{prob_churn:.2%}", delta="Risk Detected")
            else:
                st.success("✅ LOW RISK (Loyal)")
                st.metric(label="Churn Probability", value=f"{prob_churn:.2%}", delta="- Safe")
        
        # Explanation Column (SHAP)
        with col2:
            st.subheader("Why this result? (SHAP Explanation)")
            with st.spinner("Calculating feature importance..."):
                # Initialize Explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # Plot Waterfall Chart
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0], 
                        base_values=explainer.expected_value, 
                        data=input_df.iloc[0], 
                        feature_names=input_df.columns
                    ),
                    max_display=8, # Limit to top 8 factors
                    show=False
                )
                st.pyplot(fig)
                st.caption("Red bars increase churn risk, Blue bars decrease it.")

# Footer
st.markdown("---")
st.markdown("Developed for Data Science Portfolio")