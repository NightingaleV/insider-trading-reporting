import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional

from collections import defaultdict
from typing import TYPE_CHECKING

from agentic_stock_analysis.models.stock_info import StockInfo 

class FinanceToolset:
    """
    A toolset for financial analysis using yfinance library.
    This class provides methods to fetch stock financials, news, and technical indicators.
    """
    
    CACHE_EXPIRY = timedelta(days=15)  # Cache expiry time for API calls
    
    def __init__(self):
        """Initialize the FinanceToolset."""
        # Cache structure will have {'info_ticker': (info_dict, timestamp)
        self.cache = defaultdict(lambda: None)  # Cache to store API responses
        
    def get_stock_info(self, ticker: str) -> StockInfo:
        """
        Get general information about a stock.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        cache_key = f"info_{ticker}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            stock = yf.Ticker(ticker)
            from agentic_stock_analysis.models.stock_info import StockInfo
            stock_info = StockInfo(**stock.info)
            self.cache[cache_key] = stock_info
            return stock_info
        except Exception as e:
            return {"error": f"Failed to fetch stock info: {str(e)}"}
    
    def get_stock_news(self, ticker: str, days: int = 7) -> list[dict]:
        """
        Get recent news for a stock.
        
        Args:
            ticker: The stock ticker symbol
            days: Number of days to look back. Defaults to 7.
            
        Returns:
            List of news items
        """
        #TODO
        cache_key = f"news_{ticker}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            new = news[-1]
            new
            # Filter news by date if needed
            if days > 0:
                cutoff_date = datetime.now() - timedelta(days=days)
                filtered_news = [
                    item for item in news 
                    if datetime.fromtimestamp(item.get('providerPublishTime', 0)) > cutoff_date
                ]
                news = filtered_news
            
            self.cache[cache_key] = news
            return news
        except Exception as e:
            return [{"error": f"Failed to fetch news: {str(e)}"}]
    
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a stock.
        
        Args:
            ticker: The stock ticker symbol
            period: Time period to fetch. Defaults to "1y" (1 year).
            interval: Data interval. Defaults to "1d" (daily). 
            
        Valid periods: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
            
        Returns:
            DataFrame containing historical price data
        """
        cache_key = f"hist_{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.cache[cache_key] = hist
            return hist
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})
    
    def get_technical_indicators(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Calculate technical indicators for a stock.
        
        Args:
            ticker: The stock ticker symbol
            period: Time period to fetch. Defaults to "1y" (1 year). Valid periods: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
            interval: Data interval. Defaults to "1d" (daily).
            
        Returns:
            Dictionary containing technical indicators
        """
        try:
            # Get historical data
            hist = self.get_historical_data(ticker, period, interval)
            if "error" in hist.columns:
                return {"error": hist["error"][0]}
            
            # Calculate technical indicators
            indicators = pd.DataFrame(index=hist.index)
            
            # Simple Moving Averages
            indicators["SMA_20"] = hist["Close"].rolling(window=20).mean()
            indicators["SMA_50"] = hist["Close"].rolling(window=50).mean()
            indicators["SMA_200"] = hist["Close"].rolling(window=200).mean()
            
            # Exponential Moving Averages
            indicators["EMA_12"] = hist["Close"].ewm(span=12, adjust=False).mean()
            indicators["EMA_26"] = hist["Close"].ewm(span=26, adjust=False).mean()
            
            # MACD (Moving Average Convergence Divergence)
            indicators["MACD"] = indicators["EMA_12"] - indicators["EMA_26"]
            indicators["MACD_Signal"] = indicators["MACD"].ewm(span=9, adjust=False).mean()
            indicators["MACD_Histogram"] = indicators["MACD"] - indicators["MACD_Signal"]
                        
            # RSI (Relative Strength Index)
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            indicators["RSI"] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            indicators["BB_Middle"] = indicators["SMA_20"]
            std_dev = hist["Close"].rolling(window=20).std()
            indicators["BB_Upper"] = indicators["BB_Middle"] + (std_dev * 2)
            indicators["BB_Lower"] = indicators["BB_Middle"] - (std_dev * 2)
            
            # Add price and volume data
            indicators["Close"] = hist["Close"]
            indicators["Volume"] = hist["Volume"]
            # Convert index to date            
            indicators.index = indicators.index.date
            return indicators
        except Exception as e:
            return {"error": f"Failed to calculate technical indicators: {str(e)}"}
    
    def compare_stocks(self, tickers: list[str], metric: str = "Close", period: str = "1y", resample: str = "M") -> pd.DataFrame:
        """
        Compare price performance of multiple stocks. The performance is normalized to show percentage change from the start.
        
        Args:
            tickers: List of stock ticker symbols
            metric: Metric to compare. Defaults to "Close".
            period: Time period to fetch. Defaults to "1y" (1 year). Valid periods: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
            resample: Frequency to resample data. Options: 'D' (daily), 'W' (weekly), 'M' (monthly). Defaults to 'D'.
            
        Returns:
            DataFrame containing normalized performance data
        """
        try:
            # Validate resample frequency
            valid_freqs = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
            if resample not in valid_freqs:
                return pd.DataFrame({"error": [f"Invalid resample frequency. Use one of {list(valid_freqs.keys())}"]})

            # Get data for all tickers
            comparison_data = {}
            
            for ticker in tickers:
                hist = self.get_historical_data(ticker, period=period)
                if "error" not in hist.columns and metric in hist.columns:
                    # Normalize to show percentage change from start
                    first_value = hist[metric].iloc[0]
                    normalized = hist[metric] / first_value * 100
                    
                    # Create a Series with datetime index
                    series = pd.Series(normalized, index=hist.index)
                    
                    # Resample and forward fill missing values
                    resampled = series.resample(resample).mean().ffill()
                    comparison_data[ticker] = resampled
            
            if not comparison_data:
                return pd.DataFrame({"error": ["No valid data available for comparison"]})
            
            return pd.DataFrame(comparison_data)
        except Exception as e:
            return pd.DataFrame({"error": [f"Failed to compare stocks: {str(e)}"]})
    
    def get_earnings(self, ticker: str) -> dict[pd.DataFrame]:
        """
        Get earnings data for a stock.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            EarningsData model containing earnings dates and financial metrics
        """
            
        class FundamentalData(BaseModel):
            earnings_dates: Optional[pd.DataFrame] = Field(None, description="Earnings dates DataFrame")
            annual_income: Optional[pd.DataFrame] = Field(None, description="Annual net income DataFrame")
            quarterly_income: Optional[pd.DataFrame] = Field(None, description="Quarterly net income DataFrame")
            quaterly_cash_flow: Optional[pd.DataFrame] = Field(None, description="Cash flow DataFrame")
            
            model_config = {
                "arbitrary_types_allowed": True
            }
        
        cache_key = f"fundamentals_{ticker}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            stock = yf.Ticker(ticker)
            
            if hasattr(stock, 'earnings_dates') and stock.earnings_dates is not None:
                earnings_dates = stock.earnings_dates  
                earnings_dates = earnings_dates.dropna(axis=0, how='all')
            else:
                earnings_dates = None
                
            important_metrics = [
                    "Total Revenue",
                    "Gross Profit",
                    "Cost Of Revenue",
                    "Operating Expense",
                    "Research And Development",
                    "Selling General And Administration",
                    "Operating Income",
                    "EBITDA",
                    "Normalized EBITDA",
                    "Pretax Income",
                    "Net Income",
                    "Diluted EPS"
                ]
            annual_income = None
            if hasattr(stock, 'income_stmt') and stock.income_stmt is not None:
                try:
                    annual_income = stock.income_stmt.loc[important_metrics]
                    annual_income = annual_income.dropna(axis=1, how='all')
                                        
                except Exception as e:
                    print(f"Error processing annual income statement: {e}")
                
            quarterly_income = None
            if hasattr(stock, 'quarterly_income_stmt') and stock.quarterly_income_stmt is not None:
                try:
                    quarterly_income = stock.quarterly_income_stmt.loc[
                        [m for m in important_metrics if m in stock.quarterly_income_stmt.index]
                    ]
                    quarterly_income = quarterly_income.dropna(axis=1, how='all')
                except Exception as e:
                    print(f"Error processing quarterly income statement: {e}")
        
            quaterly_cash_flow = None
            if hasattr(stock, 'quarterly_cashflow') and stock.quarterly_cashflow is not None:
                try:
                    important_metrics = [
                    "Free Cash Flow",
                    "Operating Cash Flow",
                    "Capital Expenditure",
                    "Repurchase Of Capital Stock",
                    "Cash Dividends Paid",
                    "Net Issuance Payments Of Debt",
                    "Net Investment Purchase And Sale", # Important given the activity level in your data
                    "Net Income From Continuing Operations" # Starting point for OCF
                ]
                    quaterly_cash_flow = stock.quarterly_cashflow
                    quaterly_cash_flow = quaterly_cash_flow.loc[important_metrics]
                    quaterly_cash_flow = quaterly_cash_flow.dropna(axis=1, how='all')
                except Exception as e:
                    print(f"Error processing quarterly income statement: {e}")
                    
            fundamentals = FundamentalData(
                earnings_dates=earnings_dates,
                annual_net_income=annual_income,
                quarterly_income=quarterly_income,
                quaterly_cash_flow= quaterly_cash_flow,
            )
            self.cache[cache_key] = fundamentals
            return fundamentals
        except Exception as e:
            raise ValueError(f"Failed to fetch earnings data: {str(e)}")
    
    def clear_cache(self):
        """Clear the internal cache."""
        self.cache = {}
        
        
if __name__ == "__main__":
    # Example usage
    finance_tool = FinanceToolset()
    ticker = "AAPL"
    print("\n\nStock Information:")
    print(finance_tool.get_stock_info(ticker))
    # print("\n\nFinancial Statements:")
    # print(finance_tool.get_stock_financials(ticker))
    # print("\n\nRecent News:")
    # print(finance_tool.get_stock_news(ticker))
    # print("\n\nHistorical Data:")
    # print(finance_tool._get_historical_data(ticker))
    # print("\n\nTechnical Indicators:")
    # print(finance_tool.get_technical_indicators(ticker, period="1y", interval="1w"))
    # print("\n\nComparing Stocks:")
    # print(finance_tool.compare_stocks(["AAPL", "MSFT"], period='6mo', resample='W'))
    # print("\n\nEarnings Data:")
    # print(finance_tool.get_earnings(ticker))