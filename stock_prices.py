import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

def predict_stock_prices(initial_data, forecast_days):
    forecast = arima_model.get_forecast(steps=forecast_days)
    forecasted_returns = forecast.predicted_mean
    
    predicted_prices = []
    last_price = initial_data[-1]  
    
    for return_value in forecasted_returns:
        next_price = last_price * (1 + return_value)
        predicted_prices.append(next_price)
        last_price = next_price  
    
    return predicted_prices

st.title("Stock Price Prediction with ARIMA")
st.sidebar.header("Input Parameters")

initial_price = st.sidebar.number_input("Enter last known stock price", value=100.0, min_value=1.0)

forecast_days = st.sidebar.slider("Select number of days to forecast", 1, 30, 7)

if st.sidebar.button("Predict Prices"):
    predicted_prices = predict_stock_prices([initial_price], forecast_days)
    prediction_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days + 1)]
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price': predicted_prices
    })
    st.write("Predicted Stock Prices for the next", forecast_days, "days")
    st.dataframe(prediction_df)
    
    plt.figure(figsize=(10, 5))
    plt.plot(prediction_df['Date'], prediction_df['Predicted Price'], marker='o')
    plt.title("Predicted Stock Prices (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.xticks(rotation=45)
    st.pyplot(plt)
