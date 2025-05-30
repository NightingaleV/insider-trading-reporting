from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, root_validator

class StockInfo(BaseModel):
    # ── Identity ─────────────────────────────────
    symbol: Optional[str] = Field(None, alias="symbol")
    short_name: Optional[str] = Field(None, alias="shortName")
    sector: Optional[str] = Field(None, alias="sector")
    industry: Optional[str] = Field(None, alias="industry")

    # ── Price action & liquidity ─────────────────
    current_price: Optional[float] = Field(None, alias="currentPrice")
    previous_close: Optional[float] = Field(None, alias="previousClose")
    # day_low: Optional[float] = Field(None, alias="dayLow")
    # day_high: Optional[float] = Field(None, alias="dayHigh")
    # volume: Optional[int] = Field(None, alias="volume")
    average_volume: Optional[int] = Field(None, alias="averageVolume")

    # ── Valuation ────────────────────────────────
    market_cap: Optional[int] = Field(
        None, alias="marketCap",
        description="Total market value of outstanding shares"
    )
    enterprise_value: Optional[int] = Field(
        None, alias="enterpriseValue",
        description="Market cap + debt – cash: takeover price proxy"
    )
    trailing_pe: Optional[float] = Field(
        None, alias="trailingPE",
        description="Price ÷ EPS over last 12 months"
    )
    forward_pe: Optional[float] = Field(
        None, alias="forwardPE",
        description="Price ÷ projected next-12-month EPS"
    )


    # ── Profitability & cash flow ────────────────
    eps_ttm: Optional[float] = Field(
        None, alias="epsTrailingTwelveMonths",
        description="Earnings per share in last 12 months"
    )
    eps_forward: Optional[float] = Field(
        None, alias="epsForward",
        description="Analyst-forecast EPS for next year"
    )
    net_income: Optional[int] = Field(
        None, alias="netIncomeToCommon",
        description="Profit after all expenses & taxes"
    )
    ebitda: Optional[int] = Field(
        None, alias="ebitda",
        description="Earnings before interest, taxes, depreciation & amortization"
    )
    free_cash_flow: Optional[int] = Field(
        None, alias="freeCashflow",
        description="Cash from operations – capex"
    )

    # ── Capital structure & risk ──────────────────
    total_debt: Optional[int] = Field(
        None, alias="totalDebt",
        description="Sum of short- and long-term debt"
    )
    total_cash: Optional[int] = Field(
        None, alias="totalCash",
        description="Cash and equivalents on the balance sheet"
    )
    debt_to_equity: Optional[float] = Field(
        None, alias="debtToEquity",
        description="Leverage ratio: totalDebt ÷ shareholders’ equity"
    )
    beta: Optional[float] = Field(
        None, alias="beta",
        description="Volatility vs. market benchmark"
    )

    # ── Short interest ────────────────────────────
    shares_short: Optional[int] = Field(
        None, alias="sharesShort",
        description="Shares sold short (bearish bets)"
    )
    short_ratio: Optional[float] = Field(
        None, alias="shortRatio",
        description="Days to cover: sharesShort ÷ avg daily volume"
    )
    shares_short_prior_month: Optional[int] = Field(
        None, alias="sharesShortPriorMonth",
        description="Short interest one month ago"
    )
    short_percent_of_float: Optional[float] = Field(
        None, alias="shortPercentOfFloat",
        description="% of freely traded shares currently shorted"
    )

    # ── Next earnings dates ───────────────────────
    earnings_timestamp_start: Optional[datetime] = Field(None, alias="earningsTimestampStart")
    earnings_call_timestamp_start: Optional[datetime] = Field(None, alias="earningsCallTimestampStart")

    # ── Analyst targets & sentiment ───────────────
    target_mean_price: Optional[float] = Field(
        None, alias="targetMeanPrice",
        description="Average analyst price target"
    )
    recommendation_key: Optional[str] = Field(
        None, alias="recommendationKey",
        description="Consensus buy/sell/hold rating"
    )
    number_of_analyst_opinions: Optional[int] = Field(
        None, alias="numberOfAnalystOpinions",
        description="Count of analysts contributing estimates"
    )
    average_analyst_rating: Optional[str] = Field(
        None, alias="averageAnalystRating",
        description="Numeric rating scale (e.g. '2.1 - Buy')"
    )

    @root_validator(pre=True)
    def _convert_epochs(cls, values):
        for key in (
            "exDividendDate",
            "earningsTimestampStart",
            "earningsTimestampEnd",
            "earningsCallTimestampStart",
            "earningsCallTimestampEnd",
        ):
            epoch = values.get(key)
            if isinstance(epoch, (int, float)):
                values[key] = datetime.fromtimestamp(epoch)
        return values

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"