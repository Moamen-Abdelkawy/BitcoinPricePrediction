# BitcoinPricePredictionML

A comprehensive project focusing on Bitcoin price prediction using neural network models. This repository demonstrates a structured approach to time series analysis and forecasting, comparing the performance of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures on Bitcoin's closing price data.

## Project Overview

Bitcoin's price is notoriously volatile, making it a challenging yet valuable target for time series forecasting. This project involves:

1. **Data Collection and Preprocessing:**  
   - Data sourced from Yahoo Finance for the period 2018-01-01 to 2023-01-01.
   - Preprocessing techniques include log transformations, stationary differencing, and fractional differencing.

2. **Feature Engineering:**  
   - Lagged features are created to capture temporal dependencies in the dataset.

3. **Model Architectures:**  
   - **MLP:** A neural network with fully connected layers designed for regression tasks.
   - **CNN:** A convolutional architecture utilizing Gramian Angular Field (GAF) transformations to analyze time series as images.

4. **Performance Comparison:**  
   - Models are evaluated on datasets with varying levels of preprocessing.
   - Key metrics include Root Mean Square Error (RMSE).

## Key Results

- **MLP:** Best suited for datasets with global dependencies or long-term memory, such as fractionally differenced series.  
- **CNN:** Excels in capturing localized patterns in stationary data but struggles with raw, non-stationary inputs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
