#%%
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic import BaseModel, Field, field_validator
from io import StringIO
import httpx
import pandas as pd
from typing import Optional
import logging

#%%
from agentic_stock_analysis.agents.utils import print_response, inspect_agent
        
def clean_capitol_trades_markdown(content):
    """
    Cleans the markdown content by removing links and unnecessary formatting.
    """
    import re

    # Remove markdown links
    cleaned_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # Remove any other unwanted markdown formatting if necessary
    cleaned_content = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_content)  # Bold
    cleaned_content = re.sub(r'_(.*?)_', r'\1', cleaned_content)        # Italics
    # Remove ### headings
    cleaned_content = re.sub(r'#+\s*', '', cleaned_content)  # Remove headings
    # Remove \n
    cleaned_content = re.sub(r'\n+', ' ', cleaned_content)  # Remove extra newlines
    # Remove "|  days " pattern with different number of spaces
    cleaned_content = re.sub(r'\|\s*days\s*', '', cleaned_content)  # Remove "| days |" pattern
    # Replace "Goto trade detail page." With "Next Trade:"
    cleaned_content = re.sub(r'Goto trade detail page\.', 'Next Trade:', cleaned_content)
    # Remove ":US" from tickers
    cleaned_content = re.sub(r':US', '', cleaned_content)
    # Remove \\n and replace with |
    cleaned_content = re.sub(r'\\n', ';', cleaned_content)
    return cleaned_content
        
        
async def scrape_capitol_trades(page=1):
    url = f"https://www.capitoltrades.com/trades?assetType=stock&assetType=stock-options&pageSize=96&page={page}"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://192.168.68.99:11235/md",
            json={
                "url": url,
                "f": "fit",
                "q": "Data regarding Recent Trades by US Congress Politician",
                "c": "0"
            }
        )
        return response.json()
# %%
# Mocks
from agentic_stock_analysis.config import ROOT_DIR
with open(ROOT_DIR / 'mocks' / 'capitol_trades_scrape_sample.txt', 'r') as f:
    capitol_trades_raw_data = f.read()
# %%
# Models
from datetime import date
# Models for trades data

from enum import StrEnum

class TradeDirection(StrEnum):
    """Enum for trade signals based on insider trading patterns."""
    BUY = 'buy'
    SELL = 'sell'

    @classmethod
    def _missing_(cls, value: str): # type: ignore
        """Handle invalid enum values."""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid TradeDirection value: {value}. Must be one of {cls.__members__.keys()}")

class Trade(BaseModel):
    """Represents a single stock trade made by a politician or insider.
    
    Contains details about the trade, including who made it, what stock was traded,
    the amount, transaction type, and when it occurred.
    """
    politician_name: str = Field(..., description="Name of the politician or insider who made the trade")
    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL", "GOOGL", "TSLA"],)
    company_name: str = Field(..., description="Name of the company")
    transaction_type: TradeDirection = Field(..., description="Type of transaction: buy or sell")
    amount: int = Field(..., description="Dollar amount of the trade. In case of ranges (e.g. 1000-5000), it will be the upper bound")
    price: Optional[float] = Field(None, description="Price per share at which the trade was executed. Optional if not available.")
    transaction_date: date  = Field(..., description="Date when the trade occurred. In ISO format (YYYY-MM-DD)")
    disclosure_date: date = Field(..., description="Date when the trade was disclosed")
    
    @field_validator('ticker')
    def validate_stock_symbol(cls, v):
        if ':US' in v:
            v = v.replace(':US', '')
        return v.upper() if v else v
    
    @field_validator('transaction_type', mode='before')
    def validate_transaction_type(cls, v):
        return TradeDirection._missing_(v)
    
    @field_validator('amount')
    def validate_amount(cls, v):
        if isinstance(v, str):
            # Handle ranges like "1000-5000"
            parts = v.split('-')
            if len(parts) > 1:
                try:
                    return int(parts[-1])  # Use the upper bound
                except ValueError:
                    raise ValueError("Invalid amount format")
            else:
                try:
                    return int(v)
                except ValueError:
                    raise ValueError("Invalid amount format, should be an integer amount in dollars")
        elif isinstance(v, int):
            return v
        else:
            raise ValueError("Amount must be an integer")
        
    
class TradesData(BaseModel):
    """Represents a collection of trades made by politicians or insiders.
    
    Contains a list of Trade objects and metadata about the source of the data.
    """
    trades: list[Trade] = Field(..., description="list of trades made by politicians or insiders")    
    
    def to_df(self):
        """Convert the trades data to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([trade.model_dump() for trade in self.trades])
    
class Signal(StrEnum):
    """Enum for investment recommendation for trade activity related the the company based on insider trading patterns."""
    STRONG_BUY = 'strong buy'
    BUY = 'buy'
    HOLD = 'hold'
    SELL = 'sell'
    STRONG_SELL = 'strong sell'

    @classmethod
    def _missing_(cls, value: str): # type: ignore
        """Handle invalid enum values."""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid AnalystRecommendation value: {value}. Must be one of {cls.__members__.keys()}")


class MarketTrend(StrEnum):
    """Enum for current market sentiment regarding a stock based on the trades or news provided."""
    GREEDY = 'greedy'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    FEARFUL = 'fearful'

    @classmethod
    def _missing_(cls, value: str): # type: ignore
        """Handle invalid enum values."""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid TradeDirection value: {value}. Must be one of {cls.__members__.keys()}")

class StockTradesAnalysis(BaseModel):
    """
    Analysis results for a stock based on politician and insider trading data.
    
    Provides sentiment analysis and summary statistics about trading activity.
    """
    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL", "GOOGL", "TSLA"])
    company_name: str = Field(..., description="Name of the company", examples=["Apple Inc.", "Alphabet Inc.", "Tesla Inc."])
    buy_count: int = Field(0, description="Number of buy transactions")
    sell_count: int = Field(0, description="Number of sell transactions")
    total_volume: int = Field(0, gt=0, description="Total volume of a trades, in dollars")
    recommendation: str = Field(
        ...,
        description=(
            "Freeform markdown-formatted text providing a detailed justification for the investment signal. "
            "Should include reasoning, supporting arguments and conditions under which the recommendation applies. "
            "LLM must use markdown syntax (e.g., headings, bullet points, bold/italic, links if relevant)."
        ),
        examples=[
            "### Recommendation\n**Buy**: Recent insider activity shows a strong trend of purchases by multiple senators. "
            "This suggests positive sentiment. \n\n- 5 buys vs 1 sell\n- Total volume: $1,200,000\n\n_Consider monitoring for further activity._"
        ]
    )
    trend: MarketTrend = Field(..., description="Market trend based on the data provided", examples=["greedy", "bullish", "neutral", "bearish", "fearful"])
    signal: str = Field(..., description="Signal based on insider trading patterns", examples=["strong buy", "buy", "hold", "sell", "strong sell"])
    
    @field_validator('recommendation')
    def validate_recommendation(cls, v: str) -> str:
        """
        Validate that the recommendation is a non-empty markdown string.

        Args:
            v: The recommendation string.

        Returns:
            The validated recommendation string.

        Raises:
            ValueError: If the recommendation is empty or not a string.
        """
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Recommendation must be a non-empty markdown string.")
        # Optional: check for basic markdown structure (e.g., heading or bullet)
        if not any(token in v for token in ['#', '-', '*', '**', '_']):
            raise ValueError("Recommendation should use markdown syntax for structure and emphasis.")
        return v
    
    @field_validator('ticker')
    def validate_ticker(cls, v: str) -> str:
        """Ensure ticker is uppercase and valid."""
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string.")
        v = v.upper().strip()
        if not v.isalpha():
            raise ValueError("Ticker must contain only alphabetic characters.")
        return v
    
    @field_validator('signal', mode='before')
    def validate_signal(cls, v: str) -> Signal:
        return Signal._missing_(v)
    
    @field_validator('trend', mode='before')
    def validate_trend(cls, v: str) -> MarketTrend:
        return MarketTrend._missing_(v)
    
class InsiderTradingResults(BaseModel):
    """Results container for ticker-based analysis.
    
    Contains analysis for multiple tickers based on insider trading data.
    """
    tickers: dict[str, StockTradesAnalysis] = Field(
        ..., 
        description="Dictionary mapping ticker symbols to their trading analysis"
    )
    
    def get_analysis_for_ticker(self, ticker: str) -> StockTradesAnalysis | None:
        """Get analysis for a specific ticker."""
        return self.tickers.get(ticker.upper())
    
    def get_top_recommendations(self, n: int = 5) -> list[StockTradesAnalysis]:
        """Get top N recommendations sorted by total volume."""
        return sorted(
            self.tickers.values(), 
            key=lambda x: x.total_volume, 
            reverse=True
        )[:n]

#%%
class InsiderAnalystAgent:
    """Factory class for creating a configured Agent for insider trading analysis.

    This class encapsulates all configuration and prompt logic for the Insider Analyst agent.
    Instantiating this class returns a configured Agent instance.

    Example:
        agent = InsiderAnalystAgent(name="Insider Analyst")
    
    !!! note
        This pattern avoids subclassing final classes and keeps configuration modular.
    """

    SYSTEM_PROMPT = """You are an expert financial analyst specializing in insider trading data from US politicians. 
    Your task is to analyze trades made by politicians and insiders, extract structured data, and provide investment recommendations based on trading patterns.
    You will scrape data from Capitol Trades, process it into structured trades, and analyze the trading patterns to generate insights.
    You will be provided with raw markdown content containing trade information. 

    # Goal: 
    Your goal is to extract the trades into a structured format and analyze them to provide insights on potential investment opportunities. 
    """

    DEFAULT_MODEL = "openai:gpt-4o-mini"
    DEFAULT_NAME = "Insider Analyst"

    def __init__(self, model: str = DEFAULT_MODEL, name: str = DEFAULT_NAME, **kwargs):
        """Initialize the Insider Analyst Agent with extraction and analysis capabilities.

        Args:
            model: The AI model to use for both extraction and analysis
            name: Name identifier for the agent
            **kwargs: Additional arguments passed to Agent initialization

        !!! warning
            This agent requires access to Capitol Trades mock data file.
            Ensure the mock data is available in the expected location.
        """
        self.model = model or self.DEFAULT_MODEL
        self.name = name or self.DEFAULT_NAME
        self.agent = Agent(system_prompt=self.SYSTEM_PROMPT, model=model, name=name, **kwargs)
        self._setup_tools()
        
    def _setup_tools(self):
        """Setup tools for the agent, including scraping trades."""
        self.agent.tool_plain(self.scrape_trades)
        
    def get_analysis_prompt(self, ticker, trades: list[Trade]) -> str:
        """Get the system prompt for the agent."""
        return f"""
        Analyze the following trades for the ticker {ticker} specifically:
        
        ## Target {ticker} Trades
        {trades}
        
        ## Task: Provide a comprehensive analysis including:
        - Buy/sell ratio and volume analysis
        - Market sentiment based on trades
        
        ## Guidance
        - Use the provided trades to generate insights
        - Dont mention information that is not present in the trades (e.g. no news, no earnings, no financials)
        
        ## Strategy
        - Multiple different insiders trading the same stock is a stronger signal
        - Latest trades are more relevant than older ones
        - Consider the total volume of trades, not just counts        
        """
    
    @staticmethod
    def scrape_trades() -> str:
        """
        Scrapes trades data from Capitol Trades for a given page number.
        
        Args:
            page (int): Page number to scrape trades from. Default is 1.
            
        Returns:
            TradesData: Structured data containing trades made by politicians or insiders.
        """
        from agentic_stock_analysis.config import ROOT_DIR
        with open(ROOT_DIR / 'mocks' / 'capitol_trades_scrape_sample.txt', 'r') as f:
            capitol_trades_raw_data = f.read()
        return clean_capitol_trades_markdown(capitol_trades_raw_data)
    
    async def extract_trades(self, query: Optional[str] = None) -> Optional[TradesData]:
        """
        Extract trades data from Capitol Trades for a specific page.

        Args:
            page (int): Page number to scrape trades from. Default is 1.

        Returns:
            TradesData: Structured data containing trades made by politicians or insiders.
        """
        query = query or "Collect me trades from the politician. All of them."
        result = await self.agent.run(query, output_type=TradesData)
        try:
            if result.output is not None:
                df_trades = result.output.to_df()
                if df_trades.empty:
                    logging.warning("No trades found in the extracted data.")
                    return None
                logging.info(f"Extracted {len(df_trades)} trades from Capitol Trades.")
                return result.output
            else:            
                logging.error("Failed to extract trades data.")
                return None
        except Exception as e:
            logging.error(f"Error during trade extraction: {e}")
            
    async def analyze_all_tickers(self, trades: TradesData) -> InsiderTradingResults:
        """Analyze all tickers present in the trades data.
        """
        df_trades = trades.to_df()
        if df_trades.empty:
            logging.warning("No trades data available for analysis.")
            return InsiderTradingResults(tickers={})
        tickers = list(df_trades['ticker'].unique())   
        logging.info(f"Starting analysis for {len(tickers)} unique tickers")
        
        df_trades_agg = (
            df_trades.groupby('ticker', as_index=False)
            .agg(
                total_amount=('amount', 'sum'),
                buy_count=('transaction_type', lambda x: (x == 'buy').sum()),
                sell_count=('transaction_type', lambda x: (x == 'sell').sum()),
                last_trade_date=('transaction_date', 'max'),
            )
            .sort_values(['buy_count', 'total_amount'], ascending=False)
            .head(10)
        )
        
        trades_summary_str = str(df_trades_agg.to_dict('records'))
        print(f"Trades summary: {trades_summary_str}")
        @self.agent.system_prompt
        def add_trades_summary() -> str:
            """Add summary statistics to the trades DataFrame."""
            print('trade summary:', trades_summary_str)
            return """
            ## Overall Trading Activity
            Here's overall trading activity on market for all stocks in recent time to put things into context when analysing specific stock.
            {trades_summary_str}
            """
        analyses: dict[str, StockTradesAnalysis] = {}
        for ticker in tickers:
            try:
                ticker_trades = [trade for trade in trades.trades if trade.ticker == ticker]
                analysis = await self.analyze_ticker(ticker, ticker_trades)
                if analysis is not None:
                    analyses[ticker] = analysis
                logging.info(f"✅ Completed analysis for {ticker}")
            except Exception as e:
                logging.error(f"❌ Failed to analyze {ticker}: {e}")
                continue
        logging.info(f"Analysis complete. Successfully analyzed {len(analyses)} tickers")
        return InsiderTradingResults(tickers=analyses)
            
    async def analyze_ticker(self, ticker: str, trades: list[Trade]) -> Optional[StockTradesAnalysis]:
        """Analyze trading patterns for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol to analyze
            trades: List of all trades to filter for this ticker
            
        Returns:
            StockTradesAnalysis object with insights and recommendations
            
        Raises:
            ValueError: If no trades found for the specified ticker
            Exception: If analysis fails
        """
        # If worthy of analysis, continue
        query = self.get_analysis_prompt(ticker, trades)
        result = await self.agent.run(query, output_type=StockTradesAnalysis)
        logging.info(f"Completed analysis for ticker {ticker}")
        return result.output


#%%
agent = InsiderAnalystAgent(
    model='openai:gpt-4o-mini',
    name='Insider Analyst',
)
#%%
trades = await agent.extract_trades(query="Collect me trades from the politician. All of them.")
#%%
trades_test = trades.copy()
trades_test.trades = [trade for trade in trades_test.trades if trade.ticker in ['AMD', 'MSFT']]
trades_test.trades
#%%
analyses: InsiderTradingResults = await agent.analyze_all_tickers(trades_test)

#%%
analyses.tickers['AMD'].recommendation

#%%
analyses
#%%

# %%
# Test the agent
async def main():
    # Initialize the agent
    insider_agent = InsiderAnalystAgent()

    # Extract trades
    print("Extracting trades...")
    extracted_trades = await insider_agent.extract_trades()
    if extracted_trades:
        print(f"Successfully extracted {len(extracted_trades.trades)} trades.")
        # Analyze all tickers
        print("Analyzing all tickers...")
        all_analyses = await insider_agent.analyze_all_tickers(extracted_trades)
        
        if all_analyses.tickers:
            print("\n--- Analysis Results ---")
            for ticker, analysis in all_analyses.tickers.items():
                print(f"\nTicker: {analysis.ticker} ({analysis.company_name})")
                print(f"  Signal: {analysis.signal.upper()}")
                print(f"  Trend: {analysis.trend.capitalize()}")
                print(f"  Buy Count: {analysis.buy_count}, Sell Count: {analysis.sell_count}")
                print(f"  Total Volume: ${analysis.total_volume:,}")
                print(f"  Recommendation:\n{analysis.recommendation}")
            
            print("\n--- Top Recommendations (by volume) ---")
            top_recs = all_analyses.get_top_recommendations(3)
            for rec in top_recs:
                print(f"- {rec.ticker}: {rec.signal.upper()} (Volume: ${rec.total_volume:,})")
        else:
            print("No analyses generated.")
    else:
        print("Failed to extract any trades.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    