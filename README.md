# Traffic Flow Prediction Using Linear Regression

## Project Overview
This project predicts traffic volume using Linear Regression and historical traffic and weather data. It uses the Metro Interstate Traffic Volume dataset.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Dataset
The dataset contains historical traffic volume, temperature, rainfall, snowfall, cloud coverage, weather conditions, and date-time information.

## Features
- Exploratory data analysis
- Traffic volume visualization
- Weather and time-based feature engineering
- Linear Regression model training
- Model evaluation using MAE, MSE, RMSE, and R²
- Traffic volume prediction for user-provided inputs

## Model Performance
- MAE: 847.05
- MSE: 1,208,736.12
- RMSE: 1,099.43
- R² Score: 0.6943

*These are the results from the current train-test split.*

## How to Run
1. Activate the virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Train the model using `python main.py`.
4. Run predictions using `python predict.py`.

## Project Structure
- `data/` — Dataset
- `graphs/` — Generated visualizations
- `models/` — Saved model and feature columns
- `main.py` — Model training and evaluation
- `predict.py` — Traffic prediction
- `requirements.txt` — Required Python libraries

## Disclaimer
Predictions are estimates based on historical data and are not real-time traffic measurements.