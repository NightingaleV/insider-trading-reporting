#%%
import requests
import pandas as pd
import os
from datetime import datetime
from config.base import TIINGO_API_KEY

class TiingoClient:
    REQUEST_HEADERS = {
        'Content-Type': 'application/json'
    }

    def __init__(self, api_key: str):
        self.api_key: str = api_key

    def get_fundamentals(self, ticker: str) -> dict:
        """Fetches fundamental data for a given stock ticker."""
        requestResponse = requests.get(
            f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements?token={self.api_key}",
            headers=self.REQUEST_HEADERS)
        return requestResponse.json()

    def get_price(self, ticker: str, lookback: int = 31, resample='monthly') -> dict:
        """Fetches price data for a given stock ticker."""
        headers = {
            'Content-Type': 'application/json'
        }
        from datetime import datetime
        from io import StringIO
        END_DATE = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')
        START_DATE = pd.to_datetime(datetime.now() - pd.DateOffset(days=lookback)).strftime('%Y-%m-%d')
        requestResponse = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={START_DATE}&resampleFreq={resample}&format=csv&endDate={END_DATE}&token={self.api_key}",
            headers=headers)
        csv_data = requestResponse.text
        return pd.read_csv(StringIO(csv_data))
    
    def get_news(self, ticker: str, lookback:int = 30, limit:int = 500) -> dict:
        """Fetches news data for a given stock ticker."""
        source = 'bloomberg.com'
        END_DATE = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')
        START_DATE = pd.to_datetime(datetime.now() - pd.DateOffset(days=lookback)).strftime('%Y-%m-%d')
        request_url = f"https://api.tiingo.com/tiingo/news?tickers={ticker}&startDate={START_DATE}&source={source}&limit={limit}&token={self.api_key}"
        print(request_url)
        requestResponse = requests.get(
            request_url,
            headers=self.REQUEST_HEADERS)
        
        return requestResponse
#%%
# if __name__ == "__main__":
client = TiingoClient(api_key=TIINGO_API_KEY)
fundamentals = client.get_fundamentals('AAPL')
price_data = client.get_price('AAPL', lookback=30)
# news_data = client.get_news('AAPL', lookback=30)

print(fundamentals)
print(price_data.head())
# print(news_data.text)
# %%
