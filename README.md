# JSE-Portfolio-Optimization-
A python file that uses XGBoost model to make a maximum return portfolio on JSE stocks market. This problem is a being addressed after participating in a JSE University Challenge, one of the time consuming things during this challenge is how you need to deal with how much each stock should be weighted in a portfolio that you trying to build. 

These are the values that the model produced, even though there is some good ground work that can be done to make it better through using a LSTM Recurrent Neural Networks as these models perform better with time series forecasting. 

--- HRP Expected Portfolio Performance ---
Expected annual return: 14.4%
Annual volatility: 14.0%
Sharpe Ratio: 1.03

--- Discrete Allocation for R1,000,000.00 ---
Number of shares to BOUGHT for each stock:
{'BTI.JO': 3, 'CLS.JO': 2, 'BYI.JO': 5, 'WHL.JO': 10, 'SHP.JO': 1, 'GLN.JO': 5, 'MRP.JO': 2, 'MTN.JO': 3, 'PIK.JO': 11, 'FSR.JO': 4, 'SOL.JO': 3, 'AGL.JO': 1, 'NED.JO': 1, 'TKG.JO': 4, 'OMU.JO': 21, 'ABG.JO': 1, 'TGA.JO': 2, 'ANG.JO': 1, 'IMP.JO': 1, 'SSW.JO': 4, '4SI.JO': 241, 'HAR.JO': 1}
Funds remaining: R2884.07


<img width="1454" height="782" alt="Optimal_Portfolio" src="https://github.com/user-attachments/assets/ff9e199c-2d6a-46f4-8140-44f7ef775351" />
<img width="1470" height="800" alt="Normalized_Hist_Perf_Stocks" src="https://github.com/user-attachments/assets/4f1f1000-a474-48fd-8c98-3dfe8d948164" />
<img width="1470" height="800" alt="Forecast_Returns by Stock" src="https://github.com/user-attachments/assets/b44307f4-ef6c-4b72-a696-f2ccd7d69a52" />
<img width="1470" height="883" alt="Corr_StockPrice" src="https://github.com/user-attachments/assets/5c6e1580-9883-47c8-aef0-9840959b3609" />


The future Plans to improve the modelling in solving this problem is to have a look at these follwoing models:
1. LSTM Network for Regression
2. LSTM Network for Regression Using the Window Method + Time Steps
3. Multistep Time Series Forecasting with LSTMs. As much as the focus of the model is for portfoloi optimzation, the forecasting of the stocks is pretty important as well. As the portfoloi optimazation is generated, a forecasting model will be beneficial to get a good portfoloi optimised. 
