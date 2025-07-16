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
    
class TickerAnalysisResults(BaseModel):
    """Results container for ticker-based analysis.
    
    Contains analysis for multiple tickers based on insider trading data.
    """
    analyses: dict[str, StockTradesAnalysis] = Field(
        ..., 
        description="Dictionary mapping ticker symbols to their trading analysis"
    )
    
    def get_analysis_for_ticker(self, ticker: str) -> StockTradesAnalysis | None:
        """Get analysis for a specific ticker."""
        return self.analyses.get(ticker.upper())
    
    def get_top_recommendations(self, n: int = 5) -> list[StockTradesAnalysis]:
        """Get top N recommendations sorted by total volume."""
        return sorted(
            self.analyses.values(), 
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

    def __new__(cls, *args, **kwargs):
        """
        Return a configured Agent instance for insider trading analysis.

        Args:
            *args: Positional arguments for Agent (unused).
            **kwargs: Keyword arguments for Agent.

        Returns:
            Agent: Configured pydantic_ai Agent instance.
        """
        kwargs.setdefault("system_prompt", cls.SYSTEM_PROMPT)
        kwargs.setdefault("model", cls.DEFAULT_MODEL)
        kwargs.setdefault("name", cls.DEFAULT_NAME)
        agent = Agent(**kwargs)
        agent.tool_plain(cls.scrape_trades)
        return agent
    
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
            
    async def analyze_all_tickers(self, trades: TradesData) -> TickerAnalysisResults:
        """Analyze all tickers present in the trades data.
        """
        df_trades = trades.to_df()
        if df_trades.empty:
            logging.warning("No trades data available for analysis.")
            return TickerAnalysisResults(analyses={})
        tickers = list(df_trades['ticker'].unique())   
        df_trades_agg = (
            df_trades.groupby('ticker')
            .agg(
                total_amount=('amount', 'sum'),
                buy_count=('transaction_type', lambda x: (x == 'buy').sum()),
                sell_count=('transaction_type', lambda x: (x == 'sell').sum()),
                last_trade_date=('transaction_date', 'max'),
            )
            .sort_values(['buy_count', 'total_amount'], ascending=False)
            .head(10)
        )     
        if df_trades_agg.empty:
            logging.warning(f"Same number of buy/sell trades found for ticker {ticker}. Skip analysis")
            return None
        logging.info(f"Starting analysis for {len(tickers)} unique tickers")
        analyses: dict[str, StockTradesAnalysis] = {}
        for ticker in tickers:
            try:
                ticker_trades = df_trades[df_trades['ticker'] == ticker].sort_values('transaction_date', ascending=False).reset_index(drop=True)
                all_ticker_summary = df_trades_agg
                analysis = await self.analyze_ticker(ticker, ticker_trades, df_tickers=all_ticker_summary)
                if analysis is not None:
                    analyses[ticker] = analysis
                logging.info(f"✅ Completed analysis for {ticker}")
            except Exception as e:
                logging.error(f"❌ Failed to analyze {ticker}: {e}")
                continue
        logging.info(f"Analysis complete. Successfully analyzed {len(analyses)} tickers")
        return TickerAnalysisResults(analyses=analyses)
            
    async def analyze_ticker(self, ticker: str, df_trades: pd.DataFrame, df_tickers) -> Optional[StockTradesAnalysis]:
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
        df_trades_agg = (
            df_trades.groupby('ticker')
            .agg(
                total_amount=('amount', 'sum'),
                buy_count=('transaction_type', lambda x: (x == 'buy').sum()),
                sell_count=('transaction_type', lambda x: (x == 'sell').sum()),
                last_trade_date=('transaction_date', 'max'),
            )
            .sort_values(['buy_count', 'total_amount'], ascending=False)
            .head(10)
        )
        
        # If worthy of analysis, continue
        trades_str = str(df_trades.to_dict('records'))
        trades_summary_str = str(df_trades_agg.to_dict('records'))
        # from agentic_stock_analysis.agents.utils import get_model_descriptions
        query = f"""
        Analyze the following trades for ticker {ticker}:
        
        ## Target Ticker Trades
        {trades_str}
        
        ## Field Description for Trades
        {get_model_descriptions(Trade)}
        
        Provide a comprehensive analysis including:
        - Buy/sell ratio and volume analysis
        - Market sentiment based on trading patterns
        - Investment recommendation with detailed reasoning
        - Risk factors and considerations
        
        ## Overall Trading Activity
        Here's overall trading activity in recent time to put things into context regarding trends on other stocks.
        {trades_summary_str}
        """
        
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
trades_test=trades.trades

#%%
analyses = await agent.analyze_all_tickers(trades_test)

#%%
analyses
#%%
@agent.tool_plain
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
#%%
clean_capitol_trades_markdown(scrape_trades())

#%%
# Example usage
inspect_agent(agent)
#%%
async def main():
        # Run the analysis for the first page of trades
        query = "Collect me trades from the politiciation. All of them."
        result = await agent.run(query, output_type=TradesData)
        print_response(result)
        return result

# Run the async main function
result = await main()
#%%
print_response(result)

type(result)
#%%
trades = result.output
#%%
trades.to_df().head(10)  # Display the first 10 trades

#%%
result = await agent.run("whats the biggest trade", output_type=str)
        print_response(result)
        return result

#%%
if __name__ == "__main__":
    import asyncio

    # Initialize the agent
    agent = InsiderAnalystAgent(
        model='openai:gpt-4o-mini',
        name='Insider Analyst',
        system_prompt=SYSTEM_PROMPT,
    )

    async def main():
        # Run the analysis for the first page of trades
        query = "Give me a mock trade?"
        result = await agent.run(query, output_type=Trade)
        print_response(result)
        return result

    # Run the async main function
    result = await main()
# %%
#%%
df_trades = trades.to_df()
#%%
df_trades.head(10)  # Display the first 10 trades
#%%
df_trades.value_counts('ticker')
# %%
summary = (
    df_trades.groupby('ticker')
    .agg(
        total_amount=('amount', 'sum'),
        buy_count=('transaction_type', lambda x: (x == 'buy').sum()),
        sell_count=('transaction_type', lambda x: (x == 'sell').sum()),
        buy_volume=('amount', lambda x: x[df_trades.loc[x.index, 'transaction_type'] == 'buy'].sum()),
        sell_volume=('amount', lambda x: x[df_trades.loc[x.index, 'transaction_type'] == 'sell'].sum()),
        last_trade_date=('transaction_date', 'max'),
    )
    .sort_values(['total_amount', 'buy_count'], ascending=False)
    .head(10)
)
summary
#%%
io = summary[summary.index == 'AMD']
#%%
io = io.to_dict('records')[0]
#%%
str(io)
#%%
io2 = df_trades[df_trades['ticker'] == 'AMD'].sort_values('transaction_date', ascending=False).reset_index(drop=True)
#%%
io2.to_dict('records')
#%%
for ticker, row in summary.iterrows():
    print(f"Ticker: {ticker}")
    print(f"Total Amount: {row['total_amount']}")
    print(f"Buy Count: {row['buy_count']}")
    print(f"Sell Count: {row['sell_count']}")
    print(f"Last Trade Date: {row['last_trade_date']}")
    print("-" * 40)

# %%
summary['buy_count'].sum(), summary['sell_count'].sum(), summary['total_amount'].sum()
#%%
io = Trade.model_json_schema()
#%%
from agentic_stock_analysis.agents.utils import get_model_descriptions
# %%
def get_model_descriptions(pydantic_model: BaseModel) -> dict:
    """
    Extracts field descriptions and examples from a Pydantic model JSON schema.

    Args:
        schema: The JSON schema dictionary from model.model_json_schema().

    Returns:
        Dictionary mapping field names to their description and examples (if present).
    """
    schema = pydantic_model.model_json_schema()
    result = {}
    properties = schema.get("properties", {})
    for field, meta in properties.items():
        desc = meta.get("description", 'Not available')
        examples = meta.get("examples", '')
        result[field] = f'{meta.get("description")}'
        if examples:
            result[field] += f'. Examples: {", ".join(examples)}'
    result_as_str = '\n'.join(f"{k}: {v}" for k, v in output.items())
    return result_as_str

# %%
output = get_model_descriptions(Trade)
#%%
output
# %%
'\n'.join(f"{k}: {v}" for k, v in output.items())
# %%
