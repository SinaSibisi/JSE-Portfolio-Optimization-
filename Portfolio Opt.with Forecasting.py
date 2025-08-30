# Portfolio Optimization with Forecasting
# This script builds upon the initial data gathering and EDA.
# It introduces forecasting with XGBoost and optimization with PyPortfolioOpt.

# --- Step 1: Install Necessary Libraries ---
# Make sure you have these libraries installed. Open your terminal or command prompt and run:
# pip install pandas yfinance numpy scikit-learn xgboost PyPortfolioOpt matplotlib seaborn

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.hierarchical_portfolio import HRPOpt

# --- Step 2: Configuration ---
# List of tickers for stocks on the Johannesburg Stock Exchange (JSE)
TICKERS = [
    'MTN.JO', '4SI.JO', 'BYI.JO', 'NPN.JO', 'PPH.JO', 'SHP.JO', 'CPI.JO',
    'OMU.JO', 'SLM.JO', 'SBK.JO', 'ANG.JO', 'BTI.JO', 'CLS.JO',
    'HAR.JO', 'IMP.JO', 'MRP.JO', 'SOL.JO', 'SSW.JO', 'VOD.JO',
    'TGA.JO', 'GLN.JO', 'PIK.JO', 'WHL.JO', 'AGL.JO', 'TKG.JO',
    'NED.JO', 'ABG.JO', 'FSR.JO'
]

# Total amount to invest
INVESTMENT_AMOUNT = 1000000  # R1,000,000

# --- Step 3: Data Fetching and Feature Engineering ---
def get_stock_data(tickers):
    """
    Fetches historical stock data from Yahoo Finance and engineers features.
    """
    stocks_prices = pd.DataFrame()
    for ticker in tickers:
        print(f"Fetching data for: {ticker}")
        yticker = yf.Ticker(ticker)
        historyPrices = yticker.history(period='5y') # Fetching 5 years of data for relevance

        if historyPrices.empty:
            print(f"Could not fetch data for {ticker}. Skipping.")
            continue

        # Feature Engineering
        historyPrices['Ticker'] = ticker
        historyPrices['Year'] = historyPrices.index.year
        historyPrices['Month'] = historyPrices.index.month
        historyPrices['Weekday'] = historyPrices.index.weekday
        historyPrices['Date'] = historyPrices.index.date

        # Historical returns (log returns are often preferred for financial time series)
        historyPrices['log_return'] = np.log(historyPrices['Close'] / historyPrices['Close'].shift(1))

        # Lagged returns as features
        for i in [1, 3, 7, 14, 30]:
            historyPrices[f'return_{i}d'] = historyPrices['Close'].pct_change(i)

        # Future return to predict (our target variable)
        # We want to predict the percentage change 30 days into the future
        historyPrices['future_return_30d'] = historyPrices['Close'].pct_change(30).shift(-30)

        # 30-day rolling volatility
        historyPrices['volatility_30d'] = historyPrices['log_return'].rolling(window=30).std() * np.sqrt(252)

        # Moving Averages
        historyPrices['ma_10d'] = historyPrices['Close'].rolling(window=10).mean()
        historyPrices['ma_30d'] = historyPrices['Close'].rolling(window=30).mean()

        # Concatenate with the main dataframe
        stocks_prices = pd.concat([stocks_prices, historyPrices], ignore_index=False)

    # Clean up the data by dropping rows with NaN values created by shifts and rolling windows
    stocks_prices.dropna(inplace=True)
    return stocks_prices

# --- Step 4: Forecasting Future Returns ---
def get_forecasted_returns(stock_data):
    """
    Trains an XGBoost model for each stock to forecast future returns.
    """
    forecasted_returns = {}
    
    for ticker in stock_data['Ticker'].unique():
        print(f"Forecasting for: {ticker}")
        
        # Isolate data for the current stock
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        
        # Define features (X) and target (y)
        features = [
            'return_1d', 'return_3d', 'return_7d', 'return_14d', 'return_30d',
            'volatility_30d', 'ma_10d', 'ma_30d', 'Month', 'Weekday'
        ]
        target = 'future_return_30d'

        X = ticker_data[features]
        y = ticker_data[target]

        # We don't need a train-test split here because we are training on all available
        # data to make the best possible forecast for the *next* period.
        # A train-test split is for evaluating model performance, which is a separate task.

        # Initialize and train the XGBoost Regressor model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Get the most recent data point to make a prediction
        latest_data = X.iloc[[-1]]
        
        # Predict the future 30-day return and store it
        # The forecast is for a 30-day period, so we annualize it by multiplying by 12
        annualized_forecast = model.predict(latest_data)[0] * 12
        forecasted_returns[ticker] = annualized_forecast
        
    return pd.Series(forecasted_returns)


# --- Step 5: Portfolio Optimization ---
def optimize_portfolio(expected_rets, stock_data):
    """
    Uses PyPortfolioOpt to find the optimal portfolio weights that maximize the Sharpe Ratio.
    """
    # Pivot the data to get daily close prices for each ticker
    price_pivot = stock_data.pivot(columns='Ticker', values='Close')
    
    # Calculate the annualized sample covariance matrix of daily returns
    # This matrix represents the risk of the assets
    S = risk_models.sample_cov(price_pivot, frequency=252)
    
    # Initialize the Efficient Frontier object
    # This is the core of the optimization
    #ef = EfficientFrontier(expected_rets, S) #Gets Maximum Sharpe Ratio
    ef = EfficientFrontier(expected_rets, S, weight_bounds=(0, 0.15)) #Limits each stock to max 15% of portfolio
    
    # Find the portfolio that maximizes the Sharpe ratio

    # You can uncomment one of the methods below to choose your optimization strategy:
    # Method 1: Maximize the Sharpe ratio (best risk-adjusted return portfolio)
    #weights = ef.max_sharpe() 
    
    # Method 2: Minimize volatility (safest portfolio results in lowest returns)
    #weights = ef.min_volatility() 
    
    # Method 3: Get the best return for a target risk (min volatility 0.132)
    weights = ef.efficient_risk(target_volatility=0.20)

    # Clean the raw weights to remove tiny allocations
    cleaned_weights = ef.clean_weights()
    
    print("\n--- Optimal Portfolio Weights ---")
    print(cleaned_weights)
    
    print("\n--- Expected Portfolio Performance ---")
    ef.portfolio_performance(verbose=True)
    
    return cleaned_weights

# --- NEW FUNCTION: Alternative Optimization with HRP ---
def optimize_with_hrp(stock_data):
    """
    Optimizes the portfolio using Hierarchical Risk Parity (HRP).
    This method is often more robust and provides better diversification.
    """
    print("\n--- Optimizing with Hierarchical Risk Parity (HRP) ---")
    price_pivot = stock_data.pivot(columns='Ticker', values='Close')
    returns = price_pivot.pct_change().dropna()
    
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    
    print("\n--- HRP Optimal Portfolio Weights ---")
    print(hrp_weights)
    
    print("\n--- HRP Expected Portfolio Performance ---")
    hrp.portfolio_performance(verbose=True)
    
    return hrp_weights

# --- Step 6: Get Discrete Allocation ---
def get_discrete_allocation(weights, investment_amount,stock_data):
    """
    Calculates how many shares of each stock to buy based on the weights.
    """
    # Get the latest prices for all tickers
    latest_prices = get_latest_prices(stock_data.pivot(columns='Ticker', values='Close'))
    
    # Allocate the portfolio based on the total investment amount
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
    
    allocation, leftover = da.greedy_portfolio()
    
    print(f"\n--- Discrete Allocation for R{investment_amount:,.2f} ---")
    print("Number of shares to buy for each stock:")
    print(allocation)
    print(f"Funds remaining: R{leftover:.2f}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Fetch data and engineer features
    stock_data = get_stock_data(TICKERS)
    
    if not stock_data.empty:

        if not stock_data.empty:
        # --- VISUALIZATION: Plot Normalized Historical Prices ---
           print("\nPlotting normalized historical prices...")
           plt.figure(figsize=(15, 8))
           price_pivot = stock_data.pivot(columns='Ticker', values='Close')
           # Normalize prices to see performance on the same scale
           normalized_prices = price_pivot / price_pivot.iloc[0]
           normalized_prices.plot(ax=plt.gca(), legend=False)
           plt.title('Normalized Historical Performance of Stocks (5 Years)')
           plt.xlabel('Date')
           plt.ylabel('Normalized Price (Base 100)')
           plt.grid(True)
           plt.show()

        # 2. Get forecasted returns from the ML model
        mu = get_forecasted_returns(stock_data)

          # --- VISUALIZATION: Plot Forecasted Annual Returns ---
        print("\nPlotting forecasted annual returns...")
        plt.figure(figsize=(15, 8))
        mu.sort_values(ascending=False).plot(kind='bar')
        plt.title('Forecasted Annual Returns by Stock')
        plt.ylabel('Expected Annual Return (%)')
        plt.xlabel('Ticker')
        plt.grid(axis='y')
        plt.show()
        
        # 3. Run the portfolio optimization
        # CHOOSE WHICH OPTIMIZATION METHOD TO RUN:
        
        # Option A: Use the Efficient Frontier with your ML forecasts
        #print("\nRunning Mean-Variance Optimization with ML Forecasts...")
        #optimal_weights = optimize_portfolio(mu, stock_data)
        
        # Option B: Use the more robust Hierarchical Risk Parity
        print("\nRunning Hierarchical Risk Parity Optimization...")
        optimal_weights = optimize_with_hrp(stock_data) #Does not use ML forecasts history only price data
        
        # --- VISUALIZATION: Plot Optimal Portfolio Allocation ---
        print("\nPlotting optimal portfolio allocation...")
        plt.figure(figsize=(10, 10))
        # Filter out stocks with zero weight for a cleaner chart
        non_zero_weights = {k: v for k, v in optimal_weights.items() if v > 0}
        pd.Series(non_zero_weights).plot.pie(autopct='%1.1f%%', startangle=90)
        plt.title('Optimal Portfolio Allocation')
        plt.ylabel('') # Hide the y-label for pie charts
        plt.show()


        # 4. Calculate the discrete allocation of shares
        get_discrete_allocation(optimal_weights, INVESTMENT_AMOUNT,stock_data)
        
        # Optional: Plotting the correlation matrix from your original code
        corr_matrix = stock_data.pivot(columns='Ticker', values='Close').corr()
        plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title('Stock Price Correlation Matrix')
        plt.show()
    else:
        print("Could not fetch sufficient data to proceed. Please check tickers and internet connection.")
