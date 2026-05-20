GOLD PRICE FORECASTING USING MACHINE LEARNING

PROJECT OBJECTIVE: Forecasting 7-day returns using market and macroeconomic features

DATA SOURCES: Yahoo Finance + FRED

FEATURE ENGINEERING: returns, lags, rolling means, volatility, interactions

MODELS: Linear Regression and XGBoost

VALIDATION: TimeSeriesSplit

FINDINGS:

1. Feature Engineering contributed more performance gains than increasing model complexity.
2. Linear Regression generalized more directionally.
3. XGBoost reduced prediction error but did not improve directional forecasting.
4. Market features contributed more predictive information than macroeconomic variables.
5. Prediction quality varied substantially across splits, suggesting sensitivity in financial forecasting.

LIMITATIONS:

1. Weak signal financial environment limit predictive performance.
2. No sentiment/news information **was incorporated, which could enhance forecasts.**
3. Models struggled with extreme market movements.
4. Performance instability across splits suggest sensitivity to training data composition.

DEPLOYMENT LINK:

