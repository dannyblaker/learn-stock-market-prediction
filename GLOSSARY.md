# Financial Terminology Glossary

A comprehensive guide to financial terms used throughout this repository, designed for developers without finance backgrounds.

## Basic Stock Market Concepts

### Stock (Equity)
A share representing ownership in a company. When you own a stock, you own a small piece of that company.

### Ticker Symbol
A unique series of letters representing a company's stock (e.g., AAPL for Apple, MSFT for Microsoft).

### Price
The current value of one share of stock, determined by supply and demand in the market.

### Volume
The total number of shares traded during a specific time period (e.g., daily volume). High volume can indicate strong interest or significant news.

### Market Capitalization (Market Cap)
Total value of all company shares: `Stock Price × Total Shares Outstanding`. Used to classify company size (small-cap, mid-cap, large-cap).

## Price Data Components

### OHLC (Open, High, Low, Close)
- **Open**: First price of the trading day
- **High**: Highest price reached during the day
- **Low**: Lowest price reached during the day
- **Close**: Final price when market closes

### Adjusted Close
Price adjusted for corporate actions like stock splits and dividends. Use this for analysis, not raw close price.

### Candlestick Chart
Visual representation showing OHLC data. Green/white candles indicate closing higher than opening (bullish), red/black indicate closing lower (bearish).

## Returns and Performance

### Returns
Percentage change in price over time:
```
Return = (Price_end - Price_start) / Price_start × 100%
```
Example: Stock goes from $100 to $110 = 10% return

### Log Returns
Natural logarithm of price ratios: `ln(Price_t / Price_t-1)`. Preferred in modeling because they're:
- Additive over time
- Approximately normally distributed
- Symmetric for gains and losses

### Cumulative Returns
Total return over multiple periods: `(1 + r1) × (1 + r2) × ... - 1`

### Annualized Returns
Return scaled to represent yearly performance, allowing fair comparison across different time periods.

## Risk Metrics

### Volatility
Standard deviation of returns, measuring price fluctuation. Higher volatility = more risk.
```
σ = sqrt(mean((returns - mean(returns))²))
```

### Beta
Measure of stock's volatility relative to the overall market. 
- Beta = 1: Moves with market
- Beta > 1: More volatile than market
- Beta < 1: Less volatile than market

### Sharpe Ratio
Risk-adjusted return metric: `(Return - Risk_free_rate) / Volatility`
Higher is better; compares return gained per unit of risk taken.

### Maximum Drawdown
Largest peak-to-trough decline in portfolio value. Shows worst-case historical loss.

### Value at Risk (VaR)
Statistical estimate of potential loss over a time period at a given confidence level.

## Market Conditions

### Bull Market
Extended period of rising prices, typically associated with economic growth and investor optimism.

### Bear Market
Extended period of falling prices (usually 20%+ decline), often during economic downturns.

### Market Regime
Distinct market behavior patterns (trending, mean-reverting, high/low volatility).

### Liquidity
How easily an asset can be bought/sold without affecting price. High-volume stocks are more liquid.

## Technical Indicators

### Moving Average (MA)
Average price over N periods, smoothing out short-term fluctuations.
- **SMA** (Simple): Unweighted average
- **EMA** (Exponential): Weighted toward recent prices

### Relative Strength Index (RSI)
Momentum indicator (0-100) showing overbought (>70) or oversold (<30) conditions.

### MACD (Moving Average Convergence Divergence)
Trend-following indicator showing relationship between two moving averages. Used to identify trend changes.

### Bollinger Bands
Volatility bands placed above/below moving average. Price touching bands can signal overbought/oversold conditions.

### Support and Resistance
- **Support**: Price level where buying interest prevents further decline
- **Resistance**: Price level where selling pressure prevents further rise

## Trading Concepts

### Long Position
Buying a stock expecting it to increase in value. You profit when price goes up.

### Short Position
Borrowing and selling a stock expecting it to decrease. You profit when price goes down (buy back cheaper).

### Bid and Ask
- **Bid**: Highest price buyers willing to pay
- **Ask**: Lowest price sellers willing to accept
- **Spread**: Difference between bid and ask

### Order Types
- **Market Order**: Buy/sell immediately at current price
- **Limit Order**: Buy/sell only at specified price or better
- **Stop Loss**: Automatically sell if price drops to certain level

### Portfolio
Collection of investments (stocks, bonds, etc.) held by an investor.

### Diversification
Spreading investments across different assets to reduce risk.

## Time Series Concepts

### Stationarity
When statistical properties (mean, variance) don't change over time. Required for many forecasting methods.

### Autocorrelation
Correlation of a time series with its own past values. Helps identify patterns and appropriate models.

### Seasonality
Regular patterns that repeat over fixed periods (daily, weekly, yearly).

### Trend
Long-term direction of price movement (upward, downward, or sideways).

### Noise
Random fluctuations that make prediction difficult. ML tries to separate signal from noise.

## Model-Specific Terms

### Features (Predictors)
Input variables used by ML models to make predictions (e.g., past prices, volume, indicators).

### Target (Label)
What we're trying to predict (e.g., next day's price, direction of movement).

### Training Set
Historical data used to teach the model patterns.

### Validation Set
Data used to tune model parameters and prevent overfitting.

### Test Set
Unseen data used to evaluate final model performance.

### Overfitting
When model learns training data too well, including noise, and performs poorly on new data.

### Backtesting
Testing a trading strategy on historical data to see how it would have performed.

### Walk-Forward Analysis
Realistic backtesting where model is retrained periodically as new data becomes available.

## Advanced Concepts

### Market Efficiency
Theory that stock prices reflect all available information. If true, prediction is very difficult.

### Arbitrage
Exploiting price differences between markets/assets for risk-free profit. Opportunities are rare and short-lived.

### Alpha
Return exceeding benchmark performance. "Generating alpha" means beating the market.

### Factor Models
Models explaining returns through risk factors (e.g., size, value, momentum).

### High-Frequency Trading (HFT)
Using algorithms to execute trades in microseconds, exploiting tiny price movements.

### Sentiment Analysis
Using news, social media, or other text data to gauge market sentiment (bullish/bearish).

## Statistical Terms

### Mean Reversion
Tendency for prices to return to average over time. Important for some trading strategies.

### Momentum
Tendency for rising (falling) prices to continue rising (falling) in near term.

### Heteroskedasticity
When volatility changes over time (non-constant variance). GARCH models handle this.

### Multicollinearity
When predictor variables are highly correlated, causing problems in some ML models.

### Autoregression
Predicting current value based on previous values of the same variable.

## Risk Management Terms

### Position Sizing
Determining how much capital to allocate to each trade based on risk.

### Stop Loss
Predetermined price at which to exit losing trade to limit losses.

### Risk-Reward Ratio
Ratio of potential profit to potential loss for a trade. 3:1 means risking $1 to potentially make $3.

### Margin
Borrowed money used to invest, amplifying both gains and losses (leveraged trading).

### Slippage
Difference between expected trade price and actual execution price, especially in fast markets.

## Data Quality Terms

### Survivorship Bias
Only analyzing stocks that still exist, ignoring delisted/bankrupt companies. Inflates historical performance.

### Look-Ahead Bias
Using information in model that wouldn't have been available at prediction time. Causes overly optimistic results.

### Data Snooping
Testing many strategies and only reporting best-performing ones, leading to overfitting.

---

## Quick Reference: Common Abbreviations

- **OHLC**: Open, High, Low, Close
- **ATR**: Average True Range (volatility measure)
- **ROI**: Return on Investment
- **P/E**: Price-to-Earnings ratio
- **EPS**: Earnings Per Share
- **IPO**: Initial Public Offering
- **ETF**: Exchange-Traded Fund
- **NYSE**: New York Stock Exchange
- **NASDAQ**: National Association of Securities Dealers Automated Quotations
- **S&P 500**: Standard & Poor's 500 stock index
- **YTD**: Year-to-Date
- **QoQ**: Quarter-over-Quarter
- **YoY**: Year-over-Year

---

*This glossary will be expanded as new concepts are introduced in the repository. Feel free to contribute additional terms!*
