import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#LOADING MODELS
lr_model = joblib.load('models/linear_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
features = joblib.load('models/features.pkl')
df = pd.read_csv('data/gold_final.csv', index_col=0, parse_dates=True)

#PAGE CONFIG
st.set_page_config(
    page_title='Gold Forecasting Dashboard',
    layout='wide'
)

#SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "Data Explorer",
        "Model Performance",
        "Feature Analysis",
        "Findings & Limitations",
        "Forecast"
    ]
)
#OVERVIEW PAGE
if page=="Overview":
    st.title('Gold Price Forecasting Dashboard')    
    st.markdown("""
    This project forecasts future 7-day gold returns using:
    - Market Indicators: 
                - Gold returns 
                - S&P 500 returns
                - USD index changes
    - Macroeconomic Variables: 
                - Inflation rates (CPI), 
                - Interest rates (FEDFUNDS)
    - Machine Learning Models that were trained using:
                - Lag Features
                - Rolling Means
                - Volatility Measures
                - Interaction Features
    """)
    st.subheader("Project Objective")
    st.write("""
    The goal of this project is to predict short-term gold returns
    using market and macroeconomic indicators while comparing
    linear and nonlinear machine learning models under proper
    time-series validation.
    """)
    col1,col2=st.columns(2)
    with col1:
        st.info("""
        ### Linear Regression
        
        Strengths:
        - Better directional robustness
        - More interpretable
        - More stable across temporal splits
        """)
    with col2:
        st.info("""
        ### XGBoost
        
        Strengths:
        - Lower Prediction Error (MSE)
        - Captures nonlinear relationships
        - More sensitive to feature importance
        - Better Magnitude Prediction
        """)
#DATA EXPLORER
elif page=="Data Explorer":
    st.title("Data Explorer")
    feature=st.selectbox(
        "Select Variable",
        [
            "Gold_ret",
            "SP500_ret",
            "USD_ret",
            "FEDFUNDS",
            "CPI_ret"
        ]
    )
    st.subheader(f'{feature} Over Time')
    st.line_chart(df[feature])
    st.subheader('Dataset Preview')
    st.dataframe(df.tail())
#MODEL PERFORMANCE 
elif page == "Model Performance":
    st.title("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Linear Direction Accuracy",
            "55.1%"
        )
    with col2:
        st.metric(
            "XGBoost MSE",
            "0.000751"
        )
    with col3:
        st.metric(
            "Validation Method",
            "TimeSeriesSplit"
        )
    st.subheader("Actual Gold Returns")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(
        df["Gold_ret"].tail(200),
        label="Actual Returns"
    )
    ax.legend()
    st.pyplot(fig)
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
         "Model": [
            "Linear Regression",
            "XGBoost"
        ],
        "Direction Accuracy": [
            "55.1%",
            "51.5%"
        ],
        "MSE": [
            "0.000868",
            "0.000751"
        ],
        "Strength": [
            "Directional robustness",
            "Magnitude prediction"
        ]
    })
    st.table(comparison)
#FEATURE ANALYSIS PAGE 
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    st.subheader("XGBoost Feature Importance")
    importance = pd.Series(
        xgb_model.feature_importances_,
        index=features
    )
    importance = importance.sort_values(
        ascending=False
    )
    st.bar_chart(importance)
    st.subheader("Linear Regression Coefficients")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr_model.coef_
    })
    coef_df = coef_df.sort_values(
        by="Coefficient",
        ascending=False
    )
    st.dataframe(coef_df)
#FINDINGS & LIMITATIONS
elif page == "Findings & Limitations":
    st.title("Findings & Limitations")
    st.subheader("Key Findings")
    st.markdown("""
    - Feature engineering contributed more to performance than increasing model complexity.
    - Linear Regression generalized more directionally.
    - XGBoost reduced prediction error but did not improve directional accuracy.
    - Market-derived features contributed more predictive information than macroeconomic variables.
    - Prediction quality varied significantly across splits, indicating sensitivity in financial forecasting.
    """)
    st.subheader("Project Limitations")
    st.markdown("""
    - Weak signal financial environments limit predictive performance.
    - No sentiment/news data was incorporated, which could enhance forecasts.
    - Models struggled with extreme market movements.
    - Performance instability across splits suggest sensitivity to training data composition.    
    """)
#FORECAST PAGE
elif page == "Forecast":
    st.title("Latest Forecast")
    latest_data = df[features].iloc[-1:]
    lr_pred = lr_model.predict(latest_data)[0]
    xgb_pred = xgb_model.predict(latest_data)[0]
    st.subheader("Linear Regression Prediction")
    if lr_pred > 0:
        st.success(
            f"Predicted 7-Day Gold Return: {lr_pred:.5f}"
        )
    else:
        st.warning(
            f"Predicted 7-Day Gold Return: {lr_pred:.5f}"
        )
    st.subheader("XGBoost Prediction")
    if xgb_pred > 0:
        st.success(
            f"Predicted 7-Day Gold Return: {xgb_pred:.5f}"
        )
    else:
        st.warning(
            f"Predicted 7-Day Gold Return: {xgb_pred:.5f}"
        )
    st.subheader("Latest Feature Values")
    st.dataframe(latest_data.T)
#FOOTER
st.markdown("---")
st.markdown("""
Built using:
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Yahoo Finance
- FRED API
""")