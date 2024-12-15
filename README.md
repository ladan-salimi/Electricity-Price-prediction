
# Day-Ahead Price Forecasting for NO1 Zone in Norway

This project implements a **Day-ahead price forecasting model** for the NO1 zone in Norway, focusing on predicting electricity prices using Long Short-Term Memory (LSTM) architecture.

## Project Overview

The primary objective of this project is to forecast electricity prices one day ahead for the NO1 zone in Norway. The prediction model uses historical data on electricity generation and load from various zones and regions.

### Input Features

The model utilizes the following features as input:
- **Actual Generation** for zones NO1, NO2, NO3, NO5, and SE3
  - Based on **Cross Border Physical Flow** data from ENTSO-E.
- **Actual Load** for NO1.

### Target Variable

- **Electricity Price** for the NO1 zone.

### Challenges

One of the significant challenges encountered during this project was data handling, especially due to the geopolitical situation, such as the **war between Ukraine and Russia**. The absence of data on gas and oil prices further complicated accurate forecasting.

## Model Description

- The forecasting model is built using **Recurrent Neural Networks (RNN)**, specifically utilizing the **LSTM (Long Short-Term Memory)** architecture.
- **LSTM** is chosen for its ability to capture long-term dependencies and trends in time-series data, making it suitable for electricity price forecasting.

## Data Source

- The data for this project is sourced from **ENTSO-E**, focusing on cross-border physical flow and actual load information.

## Usage

1. **Data Preprocessing**: The raw data is preprocessed to handle missing values and normalize the input features.
2. **Model Training**: The LSTM model is trained on the preprocessed data.
3. **Prediction**: The trained model is used to make day-ahead electricity price forecasts.

## Results

- The model outputs daily electricity price predictions for the NO1 zone.
- The results indicate the effectiveness of using historical generation and load data for price forecasting.

## Conclusion

This project demonstrates the feasibility of day-ahead electricity price forecasting using LSTM networks, despite the challenges posed by missing input variables like gas and oil prices due to geopolitical tensions.
