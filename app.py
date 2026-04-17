import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
model = joblib.load(r'C:\Users\pc\Downloads\Olist project\best_model.pkl')
scaler = joblib.load(r'C:\Users\pc\Downloads\Olist project\scaler.pkl')
#sidebar
with st.sidebar:
    st.markdown("### About This App")
    st.markdown("---")
    st.markdown("This app uses a **Random Forest** machine learning model trained on the **Olist Brazilian E-commerce Dataset** to predict whether a customer will be satisfied with their order.")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Fill in the order details")
    st.markdown("2. Click **Predict Satisfaction**")
    st.markdown("3. View the customer insight result")
    st.markdown("---")
    st.markdown("**Model Performance:**")
    st.metric("Accuracy", "82.3%")
    st.metric("AUC Score", "93.9%")
#Heading 
st.markdown("""
    <div style="background-color:#001F5B; padding:20px 25px; border-radius:10px; margin-bottom:20px;">
        <h2 style="color:white; margin:0;">Olist Customer Satisfaction Predictor</h2>
        <p style="color:#B5D4F4; margin:5px 0 0 0; font-size:14px;">Brazilian e-commerce data</p>
    </div>
""", unsafe_allow_html=True)
#purchase details
st.markdown("**Purchase Details**")
col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Product Price (R$)", min_value=0.0, value=100.0)
    payment_value = st.number_input("Payment Value (R$)", min_value=0.0, value=120.0)
with col2:
    freight_value = st.number_input("Freight Value (R$)", min_value=0.0, value=20.0)
    payment_type = st.selectbox("Payment Type", ["Credit Card", "Boleto", "Voucher", "Debit Card"])

st.markdown("---")
# delivery detalils 
st.markdown("**Delivery Details**")
col3, col4 = st.columns(2)
with col3:
    delivery_time_days = st.number_input("Delivery Time (days)", min_value=0, value=10)
    shipping_delay_days = st.number_input("Shipping Delay (days)", min_value=0, value=0)
with col4:
    customer_state = st.selectbox("Customer State", [
        "SP - São Paulo",
        "MG - Minas Gerais",
        "PR - Paraná",
        "RS - Rio Grande do Sul",
        "Other"
    ])

st.markdown("---")

if st.button("Predict Satisfaction"):
    is_credit_card = 1 if payment_type == "Credit Card" else 0
    is_sp = 1 if customer_state == "SP - São Paulo" else 0
    is_mg = 1 if customer_state == "MG - Minas Gerais" else 0
    is_pr = 1 if customer_state == "PR - Paraná" else 0
    is_rs = 1 if customer_state == "RS - Rio Grande do Sul" else 0

    input_features = [[
        shipping_delay_days,
        payment_value,
        price,
        freight_value,
        delivery_time_days,
        is_credit_card,
        is_mg,
        is_sp,
        is_pr,
        is_rs
    ]]

    input_scaled = scaler.transform(input_features)
    probability = model.predict_proba(input_scaled)[0, 1]

    st.markdown("**Customer Insight Result**")
    if probability >= 0.7:
        st.markdown(f"""
            <div style="background-color:#EAF3DE; border-left:4px solid #639922; border-radius:8px; padding:16px 20px;">
                <p style="font-size:32px; font-weight:600; color:#27500A; margin:0;">{probability*100:.1f}%</p>
                <p style="color:#3B6D11; margin:4px 0 0 0; font-size:14px;">This customer is likely to be satisfied with their order</p>
            </div>
        """, unsafe_allow_html=True)
    elif probability >= 0.5:
        st.markdown(f"""
            <div style="background-color:#FAEEDA; border-left:4px solid #BA7517; border-radius:8px; padding:16px 20px;">
                <p style="font-size:32px; font-weight:600; color:#633806; margin:0;">{probability*100:.1f}%</p>
                <p style="color:#854F0B; margin:4px 0 0 0; font-size:14px;">This customer is moderately likely to be satisfied</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color:#FCEBEB; border-left:4px solid #A32D2D; border-radius:8px; padding:16px 20px;">
                <p style="font-size:32px; font-weight:600; color:#501313; margin:0;">{probability*100:.1f}%</p>
                <p style="color:#791F1F; margin:4px 0 0 0; font-size:14px;">This customer is unlikely to be satisfied</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>Built with Olist Brazilian E-commerce Dataset</p>", unsafe_allow_html=True)