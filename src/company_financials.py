import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional


class FinanceToolset:
    """
    A toolset for financial analysis using yfinance library.
    This class provides methods to fetch stock financials, news, and technical indicators.
    """
    
    CACHE_EXPIRY = timedelta(days=15)  # Cache expiry time for API calls
    
    def __init__(self):
        """Initialize the FinanceToolset."""
        self.cache = {}  # Simple cache to avoid redundant API calls
        
    def get_stock_info(self, ticker: str) -> dict:
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
            info = stock.info
            self.cache[cache_key] = info
            return info
        except Exception as e:
            return {"error": f"Failed to fetch stock info: {str(e)}"}
    
    def get_stock_financials(self, ticker: str, quarterly: bool = True) -> dict:
        """
        Get financial statements for a stock.
        
        Args:
            ticker: The stock ticker symbol
            quarterly (bool, optional): Whether to return quarterly statements. Defaults to True.
            
        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        cache_key = f"financials_{ticker}_{quarterly}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get income statement
            if quarterly:
                income_stmt = stock.quarterly_income_stmt
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow
            else:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            
            financials = {
                "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
                "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
            }
            
            self.cache[cache_key] = financials
            return financials
        except Exception as e:
            return {"error": f"Failed to fetch financials: {str(e)}"}
    
    def get_stock_news(self, ticker: str, days: int = 7) -> list[dict]:
        """
        Get recent news for a stock.
        
        Args:
            ticker: The stock ticker symbol
            days: Number of days to look back. Defaults to 7.
            
        Returns:
            List of news items
        """
        cache_key = f"news_{ticker}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
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
            
        Returns:
            DataFrame containing historical price data
        """
        cache_key = f"hist_{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            self.cache[cache_key] = hist
            return hist
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})
    
    def get_technical_indicators(self, ticker: str, period: str = "1y", interval: str = "1d") -> dict[str, pd.Series]:
        """
        Calculate technical indicators for a stock.
        
        Args:
            ticker: The stock ticker symbol
            period: Time period to fetch. Defaults to "1y" (1 year).
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
            indicators = {}
            
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
            
            return indicators
        except Exception as e:
            return {"error": f"Failed to calculate technical indicators: {str(e)}"}
    
    def compare_stocks(self, tickers: list[str], metric: str = "Close", period: str = "1y", resample: str = "M") -> pd.DataFrame:
        """
        Compare performance of multiple stocks. The performance is normalized to show percentage change from the start.
        
        Args:
            tickers: List of stock ticker symbols
            metric: Metric to compare. Defaults to "Close".
            period: Time period to fetch. Defaults to "1y" (1 year).
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
            
        class EarningsData(BaseModel):
            earnings_dates: Optional[pd.DataFrame] = Field(None, description="Earnings dates DataFrame")
            annual_net_income: Optional[pd.DataFrame] = Field(None, description="Annual net income DataFrame")
            quarterly_net_income: Optional[pd.DataFrame] = Field(None, description="Quarterly net income DataFrame")
        
        
        cache_key = f"earnings_{ticker}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            stock = yf.Ticker(ticker)
            
            # Process earnings dates
            if hasattr(stock, 'earnings_dates') and stock.earnings_dates is not None:
                earnings_dates = stock.earnings_dates  
                nulls = earnings_dates.isna().max(axis=1)          
                earnings_dates = earnings_dates[~nulls]
            else:
                earnings_dates = None
                
            important_cols = [
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
            if hasattr(stock, 'income_stmt'):
                annual_income = stock.income_stmt
                annual_income = annual_income.loc[important_cols]
            else:
                annual_income = None
                
            if hasattr(stock, 'quarterly_income_stmt'):
                quarterly_income = stock.quarterly_income_stmt
            else:
                quarterly_income = None
            
            earnings_data = EarningsData(
                earnings_dates=earnings_dates,
                annual_net_income=annual_income,
                quarterly_net_income=quarterly_income
            )
            self.cache[cache_key] = earnings_data
            return earnings_data
        except Exception as e:
            raise ValueError(f"Failed to fetch earnings data: {str(e)}")
    
    def clear_cache(self):
        """Clear the internal cache."""
        self.cache = {}
        
        
if __name__ == "__main__":
    # Example usage
    finance_tool = FinanceToolset()
    ticker = "AAPL"
    # print("\n\nStock Information:")
    # print(finance_tool.get_stock_info(ticker))
    # print("\n\nFinancial Statements:")
    # print(finance_tool.get_stock_financials(ticker))
    # print("\n\nRecent News:")
    # print(finance_tool.get_stock_news(ticker))
    # print("\n\nHistorical Data:")
    # print(finance_tool.get_historical_data(ticker))
    # print("\n\nTechnical Indicators:")
    # print(finance_tool.get_technical_indicators(ticker))
    # print("\n\nComparing Stocks:")
    # print(finance_tool.compare_stocks(["AAPL", "MSFT"], period='6mo', resample='W'))
    print("\n\nEarnings Data:")
    print(finance_tool.get_earnings(ticker))