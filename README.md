# 📈 Gold Stock Price Prediction

This project uses deep learning and machine learning models to predict gold stock prices based on historical financial data. The notebook demonstrates a hybrid architecture combining LSTM (Long Short-Term Memory) networks with XGBoost to improve predictive accuracy and robustness.

## 🧠 Project Overview

Gold is a critical financial asset, often used as a hedge against inflation or economic uncertainty. Accurately forecasting its price is valuable for investors, traders, and financial analysts. This notebook presents a time series forecasting approach that:

Implements a PyTorch-based LSTM neural network.
Enhances predictions using a hybrid model by combining LSTM and XGBoost.
Performs data preprocessing, feature engineering, and evaluation with R² and loss metrics.

## 🔍 Features

Custom LSTM model built with PyTorch
Xavier weight initialization
Batch normalization and dropout for regularization
SmoothL1Loss as the loss function
Evaluation using R² Score and Loss
Hybrid LSTM-XGBoost model for improved predictions
Clean predictions export for further use

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- PyTorch
- XGBoost
- Pandas, NumPy, scikit-learn
- Matplotlib, Seaborn

## 📂 Project Structure

```
goldStockPricePrediction.ipynb   # Main notebook with all code and results
```

## 🧪 How It Works

- Data Preparation:
The dataset is preprocessed with scaling, shifting, and feature engineering.
- LSTM Modeling:
A custom LSTM architecture is trained on sequential data to model temporal dependencies in the gold price series.
- Hybrid Modeling:
Outputs from the LSTM are used as input features for an XGBoost model, leveraging the strength of both models.
- Evaluation:
The performance of each model is evaluated using R² score and loss curves.
- Prediction:
The final predictions are visualized and optionally exported for further analysis.

## 📊 Results

The hybrid LSTM-XGBoost model showed improved performance over standalone LSTM in terms of R² score.
Dropout and batch normalization significantly helped mitigate overfitting.

## 🚀 How to Run

```
Clone this repository or download the notebook.
Install dependencies (preferably in a virtual environment):
pip install torch xgboost scikit-learn pandas matplotlib seaborn
Open the notebook in Jupyter:
jupyter notebook goldStockPricePrediction.ipynb
Run all cells in sequence.
```

## 📌 Notes

All computations are done using PyTorch tensors—NumPy arrays are avoided to maintain consistency.
This is part of a larger effort to align the PyTorch model with a well-performing TensorFlow reference.

## 📬 Contact

For questions or feedback, feel free to reach out!
