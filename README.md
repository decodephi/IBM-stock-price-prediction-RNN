# IBM Stock Price Prediction — Time Series Forecasting

A time series forecasting project that predicts IBM's next-day closing stock price using ~20 years of historical daily data (2001–2022). The project benchmarks classical statistical and baseline methods against deep learning architectures to identify what actually adds predictive value on this dataset.

## Overview

This project goes beyond simply training a single model — it builds a full evaluation pipeline comparing six approaches (Naive Forecast, Linear Regression, ARIMA, SimpleRNN, LSTM, and GRU) on the same held-out test period, so model performance can be judged against honest baselines rather than in isolation.

## Features

- **Exploratory Data Analysis**: trend and volume visualization, daily returns distribution, moving averages, correlation heatmap
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test and first-order differencing
- **Seasonal Decomposition**: trend, seasonality, and residual analysis
- **Time-aware Preprocessing**: chronological train/test split, MinMax scaling, 60-day sliding window sequence generation (no data leakage)
- **Baseline Models**: Naive (persistence) forecast and Linear Regression
- **Classical Model**: ARIMA(5,1,0)
- **Deep Learning Models**: SimpleRNN, LSTM, and GRU (TensorFlow/Keras), with dropout regularization and early stopping
- **Evaluation**: RMSE, MAE, MAPE, and R² across all six models on a common held-out test set

## Dataset

- **Source**: `IBM.csv` — daily OHLCV (Open, High, Low, Close, Adj Close, Volume) data
- **Range**: July 23, 2001 – July 20, 2022
- **Target variable**: Close price (univariate time series)
- **Train/Test split**: chronological — train up to 2021-12-31, test on 2022 data

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- statsmodels (ADF test, seasonal decomposition, ARIMA)
- TensorFlow / Keras (SimpleRNN, LSTM, GRU)
- matplotlib, seaborn


## Results

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| Naive (persistence) | 2.13 | 1.55 | 1.16% | 0.845 |
| Linear Regression | 2.19 | 1.60 | 1.20% | 0.836 |
| GRU | 2.28 | 1.65 | 1.24% | 0.822 |
| SimpleRNN | 2.44 | 1.80 | 1.35% | 0.796 |
| LSTM | 2.63 | 1.97 | 1.47% | 0.763 |
| ARIMA(5,1,0) | 5.42 | 4.58 | 3.47% | -0.006 |

**Key finding**: The naive persistence baseline outperformed every other model, consistent with IBM's stock price behaving close to a random walk (confirmed via ACF/PACF analysis). Among the neural network architectures, GRU performed best, likely due to its smaller parameter count generalizing better on a moderately sized dataset.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn statsmodels tensorflow matplotlib seaborn
```

### Usage

1. Clone this repository
2. Place `IBM.csv` in the project directory
3. Open and run `IBM_stock_price_prediction_full.ipynb` in Jupyter Notebook or JupyterLab

## Future Improvements

- Predict returns instead of raw price levels to directly address non-stationarity
- Walk-forward validation for more realistic, production-like evaluation
- Add exogenous features (technical indicators, volume, macro data)
- Ensemble methods combining linear and nonlinear models
- Hyperparameter tuning with time-series cross-validation


