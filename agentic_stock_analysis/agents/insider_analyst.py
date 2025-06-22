#%%
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic import BaseModel, Field, field_validator
from io import StringIO
import httpx
import pandas as pd
from typing import Optional
import logging

#%%
# Utils
def inspect_agent(agent: Agent) -> None:
    """
    Inspect the given agent by listing its methods, attributes, and tools.

    Args:
        agent: The agent instance to inspect.
    """
    # List methods of the class, excluding deprecated attributes
    methods = [
        method for method in dir(agent)
        if callable(getattr(agent, method, None)) and not method.startswith("__")
    ]
    print("Available methods:\n", methods)

    # List available attributes
    attributes = [
        attr for attr in dir(agent)
        if not callable(getattr(agent, attr, None)) and not attr.startswith("__")
    ]
    print("Available attributes:\n", "\n- ".join(attributes))

    # Print function tools
    print("Function tools:\n", agent._function_tools)

def print_response(response):
    """Helper function to print the response in a readable format."""
    if isinstance(response, str):
        print(response)
    elif hasattr(response, 'output'):
        print('Agent: ', response.output)
    elif hasattr(response, 'all_messages'):
        print('Agent: ', "\n".join(response.all_messages()))
    elif isinstance(response, BaseModel):
        print(response.model_dump_json(indent=2, exclude_none=False))
    else:
        print(response)
        
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
    def _missing_(cls, value: str):
        """Handle invalid enum values."""
        valid_values = {item.value for item in cls}
        if value.lower() in valid_values:
            return cls(value.lower())
        raise ValueError(f"Invalid TradeDirection value: {value}. Must be one of {valid_values}.")

class Trade(BaseModel):
    """Represents a single stock trade made by a politician or insider.
    
    Contains details about the trade, including who made it, what stock was traded,
    the amount, transaction type, and when it occurred.
    """
    politician_name: str = Field(..., description="Name of the politician or insider who made the trade")
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Name of the company")
    transaction_type: TradeDirection = Field(..., description="Type of transaction: buy or sell")
    amount: int = Field(..., description="Dollar amount of the trade. In case of ranges (e.g. 1000-5000), it will be the upper bound")
    price: Optional[float] = Field(None, description="Price per share at which the trade was executed. Optional if not available.")
    transaction_date: date  = Field(..., description="Date when the trade occurred. In ISO format (YYYY-MM-DD)")
    disclosure_date: date = Field(..., description="Date when the trade was disclosed")
    
    @field_validator('ticker')
    def validate_stock_symbol(cls, v):
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

# class TradeAnalysis(BaseModel):
#     """
#     Analysis results for a stock based on politician and insider trading data.
    
#     Provides sentiment analysis and summary statistics about trading activity.
#     """
#     ticker: str = Field(..., description="Stock ticker symbol")
#     company_name: str = Field(..., description="Name of the company")
#     buy_count: int = Field(0, description="Number of buy transactions")
#     sell_count: int = Field(0, description="Number of sell transactions")
#     total_volume: str = Field("", description="Total volume of a trades, in dollars")
#     trend: str = Field(..., description="Greedy, Bullish, Neutral, Bearish or Fearful, trends based on trading patterns of multiple politicians", enumerate=['greedy','bullish', 'neutral', 'bearish', 'fearful'])
#     recommendation: str = Field(..., description="Investment recommendation based on insider trading patterns, explain your reasoning in 1 paragraph analysis.")
#     signal: str = Field(..., description="Signal based on insider trading patterns", enumerate=['buy', 'sell', 'hold'])


#%%


class InsiderAnalystAgent(Agent):
    """
    Agent that analyzes insider trading data from politicians and generates investment recommendations.
    
    This agent scrapes trading data from Capitol Trades, processes it into structured data,
    analyzes trading patterns, and provides investment signals based on politician trading activity.
    
    !!! note
        This agent focuses on identifying patterns in politician trading behavior that may
        indicate potential market movements or investment opportunities.
    """
    
    class Config:
        model = "gpt-4"
    
    SYSTEM_PROMPT = """You are an expert financial analyst specializing in insider trading data from US politicians. 
    Your task is to analyze trades made by politicians and insiders, extract structured data, and provide investment recommendations based on trading patterns.
    You will scrape data from Capitol Trades, process it into structured trades, and analyze the trading patterns to generate insights.
    You will be provided with raw markdown content containing trade information. 

    # Goal: 
    Your goal is to extract the trades into a structured format and analyze them to provide insights on potential investment opportunities. 
    """
    def __new__(cls, *args, **kwargs):
        # Initialize the agent with the system prompt
        if 'system_prompt' not in kwargs:
            kwargs['system_prompt'] = cls.SYSTEM_PROMPT
        if 'model' not in kwargs:
            kwargs['model'] = 'openai:gpt-4o-mini',
        instance = super().__new__(cls)
        # instance.system_prompt = cls.SYSTEM_PROMPT
        return instance
    # Example usage

#%%
agent = InsiderAnalystAgent(
    model='openai:gpt-4o-mini',
    name='Insider Analyst',
)

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
trades.to_df().head(10)  # Display the first 10 trades
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
