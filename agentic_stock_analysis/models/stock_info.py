from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, root_validator

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict

class StockInfo(BaseModel):
    # Configure Pydantic settings for V2
    model_config = ConfigDict(
        populate_by_name=True,  # Equivalent to allow_population_by_field_name
        extra="ignore"
    )

    # ── Identity ─────────────────────────────────
    symbol: Optional[str] = Field(default=None, alias="symbol")
    short_name: Optional[str] = Field(default=None, alias="shortName")
    sector: Optional[str] = Field(default=None, alias="sector")
    industry: Optional[str] = Field(default=None, alias="industry")

    # ── Price action & liquidity ─────────────────
    # current_price: Optional[float] = Field(default=None, alias="currentPrice")
    # previous_close: Optional[float] = Field(default=None, alias="previousClose")
    # day_low: Optional[float] = Field(default=None, alias="dayLow")
    # day_high: Optional[float] = Field(default=None, alias="dayHigh")
    # volume: Optional[int] = Field(default=None, alias="volume")
    # average_volume: Optional[int] = Field(default=None, alias="averageVolume")

    # ── Valuation ────────────────────────────────
    market_cap: Optional[int] = Field(
        default=None, alias="marketCap",
        description="Total market value of outstanding shares"
    )
    enterprise_value: Optional[int] = Field(
        default=None, alias="enterpriseValue",
        description="Market cap + debt – cash: takeover price proxy"
    )
    trailing_pe: Optional[float] = Field(
        default=None, alias="trailingPE",
        description="Price ÷ EPS over last 12 months"
    )
    forward_pe: Optional[float] = Field(
        default=None, alias="forwardPE",
        description="Price ÷ projected next-12-month EPS"
    )

    # ── Profitability & cash flow ────────────────
    eps_ttm: Optional[float] = Field(
        default=None, alias="epsTrailingTwelveMonths",
        description="Earnings per share in last 12 months"
    )
    eps_forward: Optional[float] = Field(
        default=None, alias="epsForward",
        description="Analyst-forecast EPS for next year"
    )
    net_income: Optional[int] = Field(
        default=None, alias="netIncomeToCommon",
        description="Profit after all expenses & taxes"
    )
    ebitda: Optional[int] = Field(
        default=None, alias="ebitda",
        description="Earnings before interest, taxes, depreciation & amortization"
    )
    free_cash_flow: Optional[int] = Field(
        default=None, alias="freeCashflow",
        description="Cash from operations – capex"
    )

    # ── Capital structure & risk ──────────────────
    total_debt: Optional[int] = Field(
        default=None, alias="totalDebt",
        description="Sum of short- and long-term debt"
    )
    total_cash: Optional[int] = Field(
        default=None, alias="totalCash",
        description="Cash and equivalents on the balance sheet"
    )
    debt_to_equity: Optional[float] = Field(
        default=None, alias="debtToEquity",
        description="Leverage ratio: totalDebt ÷ shareholders’ equity"
    )
    beta: Optional[float] = Field(
        default=None, alias="beta",
        description="Volatility vs. market benchmark"
    )

    # ── Short interest ────────────────────────────
    shares_short: Optional[int] = Field(
        default=None, alias="sharesShort",
        description="Shares sold short (bearish bets)"
    )
    short_ratio: Optional[float] = Field(
        default=None, alias="shortRatio",
        description="Days to cover: sharesShort ÷ avg daily volume"
    )
    shares_short_prior_month: Optional[int] = Field(
        default=None, alias="sharesShortPriorMonth",
        description="Short interest one month ago"
    )
    short_percent_of_float: Optional[float] = Field(
        default=None, alias="shortPercentOfFloat",
        description="% of freely traded shares currently shorted"
    )

    # ── Next earnings dates ───────────────────────
    earnings_timestamp_start: Optional[datetime] = Field(default=None, alias="earningsTimestampStart")
    earnings_call_timestamp_start: Optional[datetime] = Field(default=None, alias="earningsCallTimestampStart")

    # ── Analyst targets & sentiment ───────────────
    target_mean_price: Optional[float] = Field(
        default=None, alias="targetMeanPrice",
        description="Average analyst price target"
    )
    recommendation_key: Optional[str] = Field(
        default=None, alias="recommendationKey",
        description="Consensus buy/sell/hold rating"
    )
    number_of_analyst_opinions: Optional[int] = Field(
        default=None, alias="numberOfAnalystOpinions",
        description="Count of analysts contributing estimates"
    )
    average_analyst_rating: Optional[str] = Field(
        default=None, alias="averageAnalystRating",
        description="Numeric rating scale (e.g. '2.1 - Buy')"
    )

    @model_validator(mode='before')
    @classmethod
    def _convert_epochs(cls, data):
        if isinstance(data, dict): # Ensure data is a dict before trying to get values
            for key in (
                "exDividendDate",
                "earningsTimestampStart",
                "earningsTimestampEnd",
                "earningsCallTimestampStart",
                "earningsCallTimestampEnd",
            ):
                epoch = data.get(key)
                if isinstance(epoch, (int, float)):
                    data[key] = datetime.fromtimestamp(epoch)
        return data