#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:52:17 2023

@author: pierrejacquet
"""

#Before running the code make sure you have the following package: pip install arch
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from arch import arch_model
import random

# Define the list of stock symbols in the portfolio
symbols = ['AI.PA', 'MC.PA', 'TTE.PA','STM.PA','SU.PA','IW2.F','CAP.PA','OR.PA','HO.PA','RI.PA','SW.PA','CS.PA','DG.PA','SAF.PA','BN.PA']

# Retrieve share price data from Yahoo Finance
prices = pd.DataFrame()
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    data = ticker.history(start='2018-01-01')
    prices[symbol] = data['Close']

# Track the change in the closing price for each share
for symbol in symbols:
    prices[symbol].plot(label=symbol)

# Display graphic legend and title
plt.legend()
plt.title('Historical Stock Prices')
plt.show()

# Calculate log yields
returns = np.log(prices / prices.shift(1)).dropna()

# Optimization function
def portfolio_variance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_volatility

# Initialize weights randomly
np.random.seed(42)
weights = np.random.random(len(symbols))
weights /= np.sum(weights)

# Constraints for asset weights
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: x - 0.035},
               {'type': 'ineq', 'fun': lambda x: 0.12 - x})
 
# Maximize the Sharpe ratio
def portfolio_sharpe(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = (portfolio_return - 0.012) / portfolio_volatility
    return -sharpe

# Optimization
optimal_portfolio = minimize(portfolio_sharpe, weights, args=(returns,), method='SLSQP', constraints=constraints)

# Weight limits
bounds = [(0, 1) for i in range(len(symbols))]

# Recover optimal weights
optimal_weights = optimal_portfolio.x

# Display results
print('Optimal portfolio weight :\n', optimal_portfolio.x)
print('Expected return on the optimal portfolio :', np.sum(returns.mean() * optimal_portfolio.x) * 252)
print('Volatility of the optimal portfolio :', np.sqrt(np.dot(optimal_portfolio.x.T, np.dot(returns.cov() * 252, optimal_portfolio.x))))
print('Sharpe ratio of the optimal portfolio :', -(optimal_portfolio.fun))

# Calculate capital evolution
# Download historical CAC 40 data
cac40 = yf.download('^FCHI', start='2018-01-01')
capital = 10000
returns_portfolio = (returns * optimal_weights).sum(axis=1)
cumulative_returns = np.cumprod(1 + returns_portfolio) * capital

# Calculation of the CAC 40 returns
cac40_perf = np.log(cac40['Close'] / cac40['Close'].shift(1))

cac40_invest = np.cumprod(1 + cac40_perf) * capital


# Calculation of the volatility, the annual return and the Sharpe ratio of the CAC 40
cac40_volatility = np.sqrt(252) * cac40_perf.std()
cac40_annual_return = np.exp(np.sum(cac40_perf) * 252 / len(cac40_perf)) - 1
cac40_sharpe_ratio = (cac40_annual_return - 0.012) / cac40_volatility
print("CAC 40 :")
print("Annual volatility : ", round(cac40_volatility * 100, 2), "%")
print("Annual return : ", round(cac40_annual_return * 100, 2), "%")
print("Sharpe ratio : ", round(cac40_sharpe_ratio, 2))

# Draw the graph of the evolution of the capital
plt.plot(cumulative_returns, label='Optimal Portfolio')
plt.plot(cac40_invest, label='CAC40')
plt.xlabel('Date')
plt.ylabel('Capital (€)')
plt.title('Evolution of the capital for an optimal portfolio and the CAC40')
plt.legend()
plt.show()

# Linear regression
# Download your historical portfolio data
# Replace stock symbols with your portfolio symbols
symbols = ['AI.PA', 'MC.PA', 'TTE.PA','STM.PA','SU.PA','IW2.F','CAP.PA','OR.PA','HO.PA','RI.PA','SW.PA','CS.PA','DG.PA','SAF.PA','BN.PA']
portfolio = yf.download(symbols, start='2018-01-01')

# Compute the returns of the optimized portfolio
# Optimal weights are recovered for each asset
weights = optimal_portfolio.x

# We multiply the returns of each asset by its weight in the portfolio
weighted_returns = returns * weights
 
# Daily portfolio returns are calculated by summing the products for each day
portfolio_returns = weighted_returns.sum(axis=1)

# Daily portfolio returns are displayed
print(portfolio_returns)

# Merge the two sets of yields into a single DataFrame
returns = pd.concat([portfolio_returns, cac40_perf], axis=1)
returns.columns = ["Portfolio Returns", "CAC 40 Returns"]

# Deleting rows with missing values
returns.dropna(inplace=True)

# Create a linear regression object
reg = LinearRegression()

# Separate independent and dependent variables
X = returns["CAC 40 Returns"].values.reshape(-1, 1)
y = returns["Portfolio Returns"].values

# Train the linear regression model
reg.fit(X, y)

# Display the coefficients of the linear regression
print(f"Intercept: {reg.intercept_:.4f}")
print(f"Slope: {reg.coef_[0]:.4f}")

# Draw the scatter plot
plt.scatter(returns["CAC 40 Returns"], returns["Portfolio Returns"], alpha=0.5)

# Draw the linear regression line
x_range = np.linspace(returns["CAC 40 Returns"].min(), returns["CAC 40 Returns"].max(), 100)
y_range = reg.predict(x_range.reshape(-1,1))
plt.plot(x_range, y_range, c="red")

# Add axis labels
plt.xlabel("CAC 40 Returns")
plt.ylabel("Portfolio Returns")

# Add caption and titles
plt.legend()
plt.title('Linear regression between portfolio and CAC 40 returns')
plt.xlabel('CAC 40 returns')
plt.ylabel('Portfolio returns')

# Add the equation of the linear regression line
equation = f"Y = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} X"
plt.text(-0.12, 0.15, equation, fontsize=12)
 
# Add the graph
plt.show()

# Regression equation
print(f"Equation of the linear regression : y = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} * x")
 
# Remove rows with missing values
portfolio_returns.dropna(inplace=True)

#Creation of the EXMA & GARCH models to analyze adjusted volatility
# Concatenate portfolio and CAC 40 returns
data = pd.concat([portfolio_returns, cac40_perf], axis=1, join='inner')
data.columns = ['Portfolio', 'CAC40']

# Estimate the GARCH model
model = arch_model(data['Portfolio'], x=data['CAC40'], p=1, q=1, vol='GARCH')
result = model.fit()
print(result.summary())

# Display a volatility graph
fig = result.plot(annualize='D')

# Calculate the EWMA volatility for the portfolio
ewma_portfolio = pd.DataFrame()
ewma_portfolio['Returns'] = portfolio_returns
ewma_portfolio['Volatility'] = portfolio_returns.ewm(span=30).std()

# Calculate the EWMA volatility for the CAC 40 index
ewma_cac40 = pd.DataFrame()
ewma_cac40['Returns'] = cac40_perf
ewma_cac40['Volatility'] = cac40_perf.ewm(span=30).std()

# Plot the EWMA volatility curves
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ewma_portfolio['Volatility'], label='Portefeuille')
ax.plot(ewma_cac40['Volatility'], label='CAC 40')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.set_title('Volatility EWMA')
ax.legend()
plt.show()


# Average return and standard deviation of the optimized portfolio
mu = portfolio_returns.mean()
sigma = portfolio_returns.std()

# Initial capital and number of simulation periods
initial_capital = 10000
num_periods = 252

# Monte Carlo simulation
sim_returns = np.random.normal(mu, sigma, (num_periods, 1000))
sim_cumulative_returns = (sim_returns + 1).cumprod(axis=0) * initial_capital

# Draw the graph of the evolution of the simulated capital
plt.plot(sim_cumulative_returns)
plt.xlabel('Simulation period')
plt.ylabel('Capital (€)')
plt.title('Monte Carlo simulation for an optimal portfolio')
plt.show()

#Calculation of the CAPM of the portfolio
beta = reg.coef_[0]
print(f"Portfolio beta : {beta:.4f}")

# Download historical risk-free rate data (1.2%)
rf_data = yf.download('^FVX', start='2018-01-01')
rf_returns = rf_data['Adj Close'].pct_change()[1:]

# Average annual CAC 40 return
CAC40_return_annual = (1 + cac40_perf.mean()) ** 252 - 1

# Portfolio annual return
portfolio_return_annual = (1 + portfolio_returns.mean()) ** 252 - 1

# Portfolio beta
beta = reg.coef_[0]

# Expected portfolio return (CAPM)
risk_free_rate = 0.012
expected_return = risk_free_rate + beta * (CAC40_return_annual - risk_free_rate)
print(f"Expected return (CAPM): {expected_return:.4f}")

#Black-Litterman Model
# Define symbols for actions
symbols = ['AI.PA', 'MC.PA', 'TTE.PA','STM.PA','SU.PA','IW2.F','CAP.PA','OR.PA','HO.PA','RI.PA','SW.PA','CS.PA','DG.PA','SAF.PA','BN.PA']

# Download historical stock data
data = yf.download(symbols, start='2018-01-01')['Adj Close']

# Calculate logarithmic returns
returns = np.log(data/data.shift(1)).dropna()

# Compute the covariance matrix of returns
cov_matrix = returns.cov()

# Define initial portfolio weights
weights = np.array([0.05] * len(symbols))

# Define yield expectations
return_forecast = np.array([0.10, 0.05, 0.12, 0.08, 0.06, 0.03, 0.09, 0.11, 0.07, 0.13, 0.02, 0.04, 0.01, 0.08, 0.06])

# Define the uncertainty matrix of return expectations
confidences = np.array([0.05] * len(symbols))

# Define the risk-free interest rate
risk_free_rate = 0.012

# Calculate the balanced covariance matrix
pi = np.dot(np.linalg.inv(cov_matrix), np.ones(len(symbols)))
pi /= np.sum(pi)

# Compute the variance matrix of balanced returns
tau = 0.05
omega = np.diag(np.square(return_forecast) * confidences) + np.dot(np.dot(np.dot(tau, pi), cov_matrix), np.transpose(np.dot(tau, pi)))

# Calculate the expected return of the balanced portfolio
eq_return = np.dot(pi, return_forecast)

# Calculate optimal portfolio weights
lmbda = (eq_return - risk_free_rate) / np.dot(np.dot(weights, cov_matrix), weights)
opt_weights = np.dot(np.linalg.inv(lmbda * omega), np.dot(np.linalg.inv(cov_matrix), pi))
opt_weights /= np.sum(opt_weights)
for i, symbol in enumerate(symbols):
    print(f"{symbol}: {opt_weights[i]:.4f}")
   
# Calculate the portfolio retrun
portfolio_returnBL = np.dot(opt_weights, return_forecast)
 
# Calculate the portfolio volatility
portfolio_volatilityBL = np.sqrt(np.dot(np.dot(opt_weights, cov_matrix), opt_weights))

# Calculate the portfolio Sharpe ratio
sharpe_ratioBL = (portfolio_returnBL - risk_free_rate) / portfolio_volatilityBL
print("Performance of the Black-Litterman portfolio:", portfolio_returnBL)
print("volatility of the Black-Litterman portfolio:", portfolio_volatilityBL)
print("Sharpe ratio of the Black-Litterman portfolio:", sharpe_ratioBL)

#Investment of 10000 euros on the BL portfolio
#Initial investment of 10000 euros
initial_investment = 10000
returns_portfolioBL = (returns * opt_weights).sum(axis=1)
cumulative_returnsBL = np.cumprod(1 + returns_portfolioBL) * initial_investment
 
# Draw the graph of the evolution of the capital
plt.plot(cumulative_returnsBL, label='Black Litterman portfolio')
plt.xlabel('Date')
plt.ylabel('Capital (€)')
plt.title('Evolution of capital for a Black Litterman portfolio')
plt.legend()
plt.show()

#Implementation of the monte carlo model on the black litterman portfolio
#Average return and standard deviation of the optimized portfolio
mu = portfolio_returnBL.mean()
sigma = portfolio_returnBL.std()

# Initial capital and number of simulation periods
initial_capital = 10000
num_periods = 252

# Monte Carlo simulation
BLsim_returns = np.random.normal(mu, sigma, (num_periods, 1000))
BLsim_cumulative_returns = (BLsim_returns + 1).cumprod(axis=0) * initial_capital

# Draw the graph of the evolution of the simulated capital
plt.plot(BLsim_cumulative_returns)
plt.xlabel('Simulation period')
plt.ylabel('Capital (€)')
plt.title('Monte Carlo simulation for a Black Litterman portfolio')
plt.show()
 
# Comparison of BL and optimized portfolio and CAC 40
plt.plot(cumulative_returnsBL, label='Black Litterman portfolio')
plt.plot(cumulative_returns, label='Optimal portfolio')
plt.plot(cac40_invest, label='CAC40')
plt.title('Evolution of the capital for a Black Litterman portfolio of the optimized portfolio and the CAC 40')
plt.xlabel('Date')
plt.ylabel('Capital (€)')
plt.legend()
plt.show()

# Comparison of our two portfolios on a position of 10,000 euros with that of one of the largest managers.
# Recovering data from the Vanguard 500 ETF
Vanguard500 = yf.download('VOO', start='2018-01-01')

#Calculation of returns and investments.
Vang_perf = np.log(Vanguard500['Close'] / Vanguard500['Close'].shift(1))
Vang_invest = np.cumprod(1 + Vang_perf) * capital

# Calculation of the manager's volatility, annual return and Sharpe ratio
Vang_volatility = np.sqrt(252) * Vang_perf.std()
Vang_annual_return = np.exp(np.sum(Vang_perf) * 252 / len(Vang_perf)) - 1
Vang_sharpe_ratio = (Vang_annual_return - 0.012) / Vang_volatility
print("Vanguard500 :")
print("Annual volatility : ", round(Vang_volatility * 100, 2), "%")
print("Annual return : ", round(Vang_annual_return * 100, 2), "%")
print("Sharpe ratio : ", round(Vang_sharpe_ratio, 2))

# Draw the graph of the evolution of the capital
plt.plot(cumulative_returns, label='Optimal portfolio')
plt.plot(cac40_invest, label='CAC40')
plt.plot(Vang_invest, label='Vanguard500')
plt.plot(cumulative_returnsBL, label='Black Litterman portfolio')
plt.xlabel('Date')
plt.ylabel('Capital (€)')
plt.title('Capital evolution')
plt.legend()
plt.show()

# Creation of an investment strategy based mainly on
# the evolution of the CAC 40. If the CAC 40 grows by 4.5% then we invest
# massively on the optimal portfolio (85/15%) if it falls by 3% we invest
# mainly in Chinese bonds (32/65%) and if it is stable then we
# invested a little in the portfolio and Chinese bonds (65/35%).
def cac40_return():
    """
    This function generates a random return for the CAC 40, with a mean of 0.056 and a standard deviation of 0.21.
    """
    return random.gauss(0.056, 0.21)

def investment_strategy():
    """
    This function implements the investment strategy described above for the next 5 years.
    """
    portfolio_allocation = []
    capital_evolution = [10000]
    cac_returns = [cac40_return() for _ in range(6)]
   
    if cac_returns[0] >= 0.045:
        portfolio_allocation.append(0.85)
        portfolio_allocation.append(0.15)
    elif cac_returns[0] <= -0.03:
        portfolio_allocation.append(0.35)
        portfolio_allocation.append(0.65)
    else:
        portfolio_allocation.append(0.65)
        portfolio_allocation.append(0.35)
   
    for i in range(5):
        portfolio_return = random.gauss(0.1726, 0.2116)
        portfolio_value = capital_evolution[-1] * (1 + portfolio_return * portfolio_allocation[0])
        bond_value = capital_evolution[-1] * (1 + 0.03 * portfolio_allocation[1])
        total_value = portfolio_value + bond_value
        capital_evolution.append(total_value)
       
        if cac_returns[i+1] >= 0.045:
            portfolio_allocation[0] = 0.85
            portfolio_allocation[1] = 0.15
        elif cac_returns[i+1] <= -0.07:
            portfolio_allocation[0] = 0.35
            portfolio_allocation[1] = 0.65
        else:
            portfolio_allocation[0] = 0.65
            portfolio_allocation[1] = 0.35
       
    return capital_evolution

capital_evolution = investment_strategy()
print(capital_evolution)

capital_evolution = investment_strategy()

plt.plot(range(6), [10000] + capital_evolution[1:], label='Capital')
plt.xlabel('Years')
plt.ylabel('Capital (€)')
plt.title("Capital evolution over the next 5 years")
plt.legend()
plt.show()

# This strategy is based on random and hypothetical CAC40 stocks
# as well as on an evolution of the capital according to a Gaussian formula
# from where the impersonal growth. In reality, we have to base ourselves on
# values of the past year. Here, the objective was to present our investment
# investment strategy. The macroeconomic environment is also important for a
# good strategy. It remains a very basic strategy.