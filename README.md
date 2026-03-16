# Binance New Listings Short Strategy

Backtest of a short-only strategy on newly listed Binance USDT perpetual futures.

## Idea
The strategy shorts new perpetual listings after a short delay, but only if the first daily candle after listing is red.

## Main rules
- Universe: Binance USDⓈ-M USDT perpetual futures
- Listings source: `exchangeInfo.onboardDate`
- Entry: 1 day after listing
- Condition: first daily candle must be red
- Max holding period: 30 days
- Max concurrent positions: 20
- Leverage: 2x
- Rebalancing: triggered when exposure deviates by more than 25%
- Fees: 10 bps

## Outputs
The script saves:
- listings file
- price panels
- daily equity curve
- trades log
- events log
- performance summary
- equity and risk charts

## Tech
Python, pandas, NumPy, matplotlib, requests

## How to run
```bash
pip install -r requirements.txt
python src/backtest_binance_new_listings_short.py
![Equity Curve](output/equity_vs_btc.png)
