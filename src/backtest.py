from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(OUTDIR, "_cache")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LISTING_START = "2023-01-01"
LISTING_END = "2026-03-31"
BACKTEST_END = "2026-03-31"

START_CAPITAL = 300.0
MAX_CONCURRENT_POS = 20
LEVERAGE = 2.0
REBALANCE_RESERVE_RATIO = 0.35
MIN_TARGET_MARGIN = 1.0
MAX_TARGET_MARGIN = 1e9
ENTRY_DELAY_DAYS = 1
MAX_HOLD_DAYS = 30
REBALANCE_THRESHOLD = 0.25
DONT_ADD_IF_DOWN_FROM_INCEPTION = 0.70
FEE_BPS = 10.0

BINANCE_FAPI = "https://fapi.binance.com"
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_CALLS = 0.10
KLINE_INTERVAL = "1d"

QUOTE_ASSET = "USDT"
CONTRACT_TYPE = "PERPETUAL"
BTC_SYMBOL = "BTCUSDT"

EXCLUDE_SYMBOLS: set[str] = set()


def dt_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_get(url: str, params: dict | None = None, max_tries: int = 6) -> requests.Response:
    last_err = None
    for attempt in range(max_tries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Request failed after {max_tries} tries: {url} | {last_err}")


def kline_cache_name(symbol: str, start_ms: int, end_ms: int) -> str:
    return f"kline_binance_{symbol}_{start_ms}_{end_ms}.csv"


def get_exchange_info() -> dict:
    path = cache_path("binance_exchange_info.json")
    cached = load_json(path)
    if cached is not None:
        return cached

    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    response = safe_get(url)
    payload = response.json()

    if "symbols" not in payload:
        raise RuntimeError(f"Unexpected exchangeInfo response: {str(payload)[:500]}")

    save_json(path, payload)
    return payload


def build_listings(
    listing_start: str,
    listing_end: str,
    quote_asset: str = QUOTE_ASSET,
    contract_type: str = CONTRACT_TYPE,
) -> pd.DataFrame:
    payload = get_exchange_info()

    start = pd.Timestamp(listing_start, tz="UTC")
    end = pd.Timestamp(listing_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    rows = []

    for item in payload.get("symbols", []):
        symbol = str(item.get("symbol", "")).upper()
        if not symbol or symbol in EXCLUDE_SYMBOLS:
            continue

        if str(item.get("contractType", "")).upper() != contract_type.upper():
            continue
        if str(item.get("quoteAsset", "")).upper() != quote_asset.upper():
            continue

        onboard = item.get("onboardDate")
        if onboard is None:
            continue

        onboard_ts = pd.to_datetime(int(onboard), unit="ms", utc=True)
        if not (start <= onboard_ts <= end):
            continue

        rows.append(
            {
                "symbol": symbol,
                "post_date": onboard_ts,
                "title": f"{symbol} onboardDate from exchangeInfo",
                "status_now": item.get("status"),
                "baseAsset": item.get("baseAsset"),
                "quoteAsset": item.get("quoteAsset"),
                "contractType": item.get("contractType"),
            }
        )

    listings = pd.DataFrame(rows)
    if listings.empty:
        raise RuntimeError("No Binance USDT perpetual listings found in the requested window.")

    listings["post_date"] = pd.to_datetime(listings["post_date"], utc=True)
    listings = (
        listings.sort_values(["post_date", "symbol"])
        .drop_duplicates(subset=["symbol"], keep="first")
        .reset_index(drop=True)
    )
    return listings


def fetch_daily_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    path = cache_path(kline_cache_name(symbol, start_ms, end_ms))
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    current = start_ms
    rows = []
    limit = 1500

    while current <= end_ms:
        params = {
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "startTime": current,
            "endTime": end_ms,
            "limit": limit,
        }

        response = safe_get(url, params=params)
        data = response.json()

        if isinstance(data, dict) and data.get("code") not in (None, 0):
            raise RuntimeError(f"Binance API error for {symbol}: {data}")

        if not data:
            break

        for row in data:
            rows.append(
                [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[7]),
                ]
            )

        last_open_time = int(data[-1][0])
        next_current = last_open_time + 24 * 3600 * 1000
        if next_current <= current:
            break

        current = next_current
        time.sleep(SLEEP_BETWEEN_CALLS)

        if len(data) < limit:
            break

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


@dataclass
class ShortLot:
    units: float
    entry_px: float


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    inception_px: float
    first_candle_date: pd.Timestamp
    target_margin: float
    target_exposure: float
    lots: list[ShortLot] = field(default_factory=list)
    margin: float = 0.0

    def units_total(self) -> float:
        return float(sum(lot.units for lot in self.lots))

    def exposure(self, px: float) -> float:
        return abs(self.units_total()) * px

    def unrealized_pnl(self, px: float) -> float:
        return float(sum(lot.units * (lot.entry_px - px) for lot in self.lots))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 5:
        return {}

    returns = equity.pct_change().fillna(0.0)
    days = max(1, (equity.index[-1] - equity.index[0]).days)
    years = days / 365.0

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_vol = float(returns.std() * math.sqrt(365.0))
    sharpe = float((returns.mean() * 365.0) / (returns.std() * math.sqrt(365.0))) if returns.std() > 0 else np.nan
    mdd = max_drawdown(equity)

    n = len(equity) - 1
    irr_daily = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(1, n)) - 1.0
    irr_annual = (1.0 + irr_daily) ** 365.0 - 1.0

    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "days": int(days),
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "irr_annual": float(irr_annual) if np.isfinite(irr_annual) else np.nan,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    peak = equity.cummax()
    return equity / peak - 1.0


def plot_monthly_returns(equity: pd.Series, path: str) -> None:
    equity = equity.dropna()
    if len(equity) < 10:
        return

    monthly = equity.resample("M").last().pct_change().dropna()

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.bar(monthly.index, monthly.values)
    ax.set_title("Monthly Returns")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.25)

    for x, y in zip(monthly.index, monthly.values):
        ax.text(
            x,
            y,
            f"{y * 100:.1f}%",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    save_fig(path)


def plot_rolling_vol(equity: pd.Series, path: str, ann_factor: float = 365.0) -> None:
    equity = equity.dropna()
    if len(equity) < 95:
        return

    returns = equity.pct_change().fillna(0.0)
    vol = returns.rolling(90).std() * math.sqrt(ann_factor)

    plt.figure(figsize=(12, 5))
    plt.plot(vol.index, vol.values)
    plt.title("Rolling 90D Annualized Volatility")
    plt.ylabel("Annualized vol")
    plt.grid(True, alpha=0.25)
    save_fig(path)


def plot_rolling_sharpe(equity: pd.Series, path: str, ann_factor: float = 365.0) -> None:
    equity = equity.dropna()
    if len(equity) < 95:
        return

    returns = equity.pct_change().fillna(0.0)
    mu = returns.rolling(90).mean() * ann_factor
    sigma = returns.rolling(90).std() * math.sqrt(ann_factor)
    sharpe = mu / sigma.replace(0.0, np.nan)

    plt.figure(figsize=(12, 5))
    plt.plot(sharpe.index, sharpe.values)
    plt.title("Rolling 90D Annualized Sharpe")
    plt.ylabel("Sharpe")
    plt.grid(True, alpha=0.25)
    save_fig(path)


def plot_holdings_count(n_open: pd.Series, path: str) -> None:
    n_open = n_open.dropna()
    if len(n_open) < 5:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(n_open.index, n_open.values)
    plt.title("Number of Open Positions")
    plt.ylabel("Positions")
    plt.grid(True, alpha=0.25)
    save_fig(path)


def plot_pct_invested(daily_df: pd.DataFrame, path: str) -> None:
    if "gross_exposure" not in daily_df.columns or "equity" not in daily_df.columns:
        return

    equity = daily_df["equity"].replace(0.0, np.nan)
    pct = (daily_df["gross_exposure"] / equity).replace([np.inf, -np.inf], np.nan)

    plt.figure(figsize=(12, 4))
    plt.plot(pct.index, pct.values * 100.0)
    plt.title("Gross Exposure as % of Equity")
    plt.ylabel("%")
    plt.grid(True, alpha=0.25)
    save_fig(path)


def plot_drawdown(equity: pd.Series, path: str) -> None:
    dd = drawdown_series(equity)
    if len(dd) < 5:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(dd.index, dd.values * 100.0)
    plt.title(f"Drawdown (MaxDD = {dd.min() * 100:.2f}%)")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.25)
    save_fig(path)


def build_open_close_panels(series: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_frames = []
    close_frames = []

    for symbol, df in series.items():
        tmp = df[["ts", "open", "close"]].dropna().drop_duplicates(subset=["ts"]).sort_values("ts")
        tmp = tmp.set_index("ts")
        open_frames.append(tmp["open"].astype(float).rename(symbol))
        close_frames.append(tmp["close"].astype(float).rename(symbol))

    if not open_frames:
        return pd.DataFrame(), pd.DataFrame()

    open_px = pd.concat(open_frames, axis=1).sort_index()
    close_px = pd.concat(close_frames, axis=1).sort_index()

    all_days = pd.date_range(
        min(open_px.index.min().floor("D"), close_px.index.min().floor("D")),
        max(open_px.index.max().floor("D"), close_px.index.max().floor("D")),
        freq="D",
        tz="UTC",
    )

    return open_px.reindex(all_days), close_px.reindex(all_days)


def backtest_strategy(
    listings: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    start_capital: float,
    max_concurrent_pos: int,
    leverage: float,
    entry_delay_days: int,
    max_hold_days: int,
    rebalance_threshold: float,
    dont_add_if_down_from_inception: float,
    fee_bps: float,
    reserve_ratio: float,
    min_target_margin: float,
    max_target_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if leverage <= 0:
        raise ValueError("leverage must be > 0")

    fee_rate = fee_bps / 10000.0

    listings = listings.copy()
    listings["post_date"] = pd.to_datetime(listings["post_date"], utc=True)
    listings["listing_date"] = listings["post_date"].dt.floor("D")
    listings["entry_date"] = listings["listing_date"] + pd.Timedelta(days=entry_delay_days)
    listings = listings.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    if close_px.empty:
        raise RuntimeError("Close price panel is empty.")

    bt_start = max(close_px.index.min().floor("D"), listings["listing_date"].min())
    bt_end = close_px.index.max().floor("D")
    days = pd.date_range(bt_start, bt_end, freq="D", tz="UTC")

    entry_map: dict[pd.Timestamp, list[str]] = {}
    listing_date_map: dict[str, pd.Timestamp] = {}

    for _, row in listings.iterrows():
        symbol = str(row["symbol"])
        listing_date = pd.Timestamp(row["listing_date"]).tz_convert("UTC").floor("D")
        entry_date = pd.Timestamp(row["entry_date"]).tz_convert("UTC").floor("D")
        entry_map.setdefault(entry_date, []).append(symbol)
        listing_date_map[symbol] = listing_date

    cash_free = float(start_capital)
    open_positions: dict[str, Position] = {}
    pending_rebalances: dict[str, dict] = {}

    daily_rows = []
    events = []
    trades = []

    def trading_fee(exposure_traded: float) -> float:
        return float(abs(exposure_traded) * fee_rate)

    def close_fifo_lots(position: Position, units_to_close: float, px: float) -> float:
        realized = 0.0
        remaining_to_close = units_to_close
        new_lots: list[ShortLot] = []

        for lot in position.lots:
            if remaining_to_close <= 0:
                new_lots.append(lot)
                continue

            if lot.units <= remaining_to_close + 1e-12:
                realized += lot.units * (lot.entry_px - px)
                remaining_to_close -= lot.units
            else:
                realized += remaining_to_close * (lot.entry_px - px)
                new_lots.append(ShortLot(units=lot.units - remaining_to_close, entry_px=lot.entry_px))
                remaining_to_close = 0.0

        position.lots = new_lots
        return float(realized)

    def first_candle_on_or_after_listing(symbol: str, listing_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None, float | None]:
        if symbol not in open_px.columns or symbol not in close_px.columns:
            return None, None, None

        tmp = pd.DataFrame({"open": open_px[symbol], "close": close_px[symbol]})
        tmp = tmp.loc[tmp.index >= listing_date.floor("D")].dropna()
        if tmp.empty:
            return None, None, None

        d0 = tmp.index[0]
        return d0, float(tmp.loc[d0, "open"]), float(tmp.loc[d0, "close"])

    def mark_equity(dt: pd.Timestamp) -> tuple[float, float, float, float]:
        px_close_today = close_px.loc[dt] if dt in close_px.index else pd.Series(dtype=float)

        locked_margin = 0.0
        unrealized = 0.0
        gross_exposure = 0.0

        for symbol, position in open_positions.items():
            px = float(px_close_today.get(symbol, np.nan))
            if not (np.isfinite(px) and px > 0):
                continue

            locked_margin += position.margin
            unrealized += position.unrealized_pnl(px)
            gross_exposure += position.exposure(px)

        equity = cash_free + locked_margin + unrealized
        return float(equity), float(locked_margin), float(unrealized), float(gross_exposure)

    def compute_target_margin(equity: float) -> float:
        target = equity / (max_concurrent_pos * (1.0 + reserve_ratio))
        target = max(min_target_margin, min(max_target_margin, target))
        return float(target)

    for i, dt in enumerate(days):
        px_open_today = open_px.loc[dt] if dt in open_px.index else pd.Series(dtype=float)
        px_close_today = close_px.loc[dt] if dt in close_px.index else pd.Series(dtype=float)

        if pending_rebalances:
            for symbol in list(pending_rebalances.keys()):
                if symbol not in open_positions:
                    pending_rebalances.pop(symbol, None)
                    continue

                position = open_positions[symbol]
                px = float(px_open_today.get(symbol, np.nan))
                if not (np.isfinite(px) and px > 0):
                    continue

                desired_exposure = float(position.target_exposure)
                current_exposure = position.exposure(px)
                delta = desired_exposure - current_exposure

                inception_px = float(position.inception_px) if position.inception_px > 0 else px
                down_frac = max(0.0, 1.0 - px / inception_px)
                allow_increase = down_frac < dont_add_if_down_from_inception

                if delta > 0 and not allow_increase:
                    events.append(
                        {
                            "date": dt,
                            "symbol": symbol,
                            "action": "rebalance_exec_blocked_70pct_rule",
                            "px": px,
                            "exposure_traded": 0.0,
                            "fee": 0.0,
                            "realized_pnl": 0.0,
                            "cash_free_after": cash_free,
                        }
                    )
                    pending_rebalances.pop(symbol, None)
                    continue

                if abs(delta) / max(1e-12, desired_exposure) < 1e-12:
                    pending_rebalances.pop(symbol, None)
                    continue

                if delta < 0:
                    reduce_exposure = -delta
                    units_to_close = reduce_exposure / px
                    realized = close_fifo_lots(position, units_to_close=units_to_close, px=px)
                    fee_paid = trading_fee(reduce_exposure)

                    reduce_margin = reduce_exposure / leverage
                    position.margin = max(0.0, position.margin - reduce_margin)
                    cash_free += reduce_margin + realized - fee_paid

                    events.append(
                        {
                            "date": dt,
                            "symbol": symbol,
                            "action": "rebalance_exec_reduce",
                            "px": px,
                            "exposure_traded": reduce_exposure,
                            "fee": fee_paid,
                            "realized_pnl": realized,
                            "cash_free_after": cash_free,
                        }
                    )
                    pending_rebalances.pop(symbol, None)

                else:
                    add_exposure = delta
                    add_margin = add_exposure / leverage
                    fee_paid = trading_fee(add_exposure)
                    required_cash = add_margin + fee_paid

                    if cash_free < required_cash:
                        events.append(
                            {
                                "date": dt,
                                "symbol": symbol,
                                "action": "rebalance_exec_add_skipped_insufficient_cash",
                                "px": px,
                                "exposure_traded": 0.0,
                                "fee": 0.0,
                                "realized_pnl": 0.0,
                                "cash_free_after": cash_free,
                                "required_cash": required_cash,
                            }
                        )
                        pending_rebalances.pop(symbol, None)
                        continue

                    add_units = add_exposure / px
                    position.lots.append(ShortLot(units=add_units, entry_px=px))
                    position.margin += add_margin
                    cash_free -= required_cash

                    events.append(
                        {
                            "date": dt,
                            "symbol": symbol,
                            "action": "rebalance_exec_add",
                            "px": px,
                            "exposure_traded": add_exposure,
                            "fee": fee_paid,
                            "realized_pnl": 0.0,
                            "cash_free_after": cash_free,
                        }
                    )
                    pending_rebalances.pop(symbol, None)

        to_exit = []
        for symbol, position in open_positions.items():
            held_days = int((dt - position.entry_date).days)
            if held_days >= max_hold_days:
                px = float(px_close_today.get(symbol, np.nan))
                if np.isfinite(px) and px > 0:
                    to_exit.append(symbol)

        for symbol in to_exit:
            position = open_positions.pop(symbol)
            px = float(px_close_today.get(symbol))
            units = position.units_total()
            exposure_now = abs(units) * px

            realized = close_fifo_lots(position, units_to_close=abs(units), px=px)
            fee_paid = trading_fee(exposure_now)
            cash_free += position.margin + realized - fee_paid

            events.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "action": "exit",
                    "px": px,
                    "exposure_traded": exposure_now,
                    "fee": fee_paid,
                    "realized_pnl": realized,
                    "cash_free_after": cash_free,
                }
            )

            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position.entry_date,
                    "exit_date": dt,
                    "entry_target_margin": position.target_margin,
                    "entry_target_exposure": position.target_exposure,
                    "exit_px": px,
                    "hold_days": int((dt - position.entry_date).days),
                    "reason": "time_30d",
                }
            )

        equity_close, _, _, _ = mark_equity(dt)

        for symbol in entry_map.get(dt, []):
            if symbol in open_positions:
                continue
            if len(open_positions) >= max_concurrent_pos:
                continue

            listing_date = listing_date_map.get(symbol)
            if listing_date is None:
                continue

            d0, o0, c0 = first_candle_on_or_after_listing(symbol, listing_date)
            if d0 is None or o0 is None or c0 is None:
                continue

            if not (c0 < o0):
                continue

            px_entry = float(px_close_today.get(symbol, np.nan))
            if not (np.isfinite(px_entry) and px_entry > 0):
                continue

            target_margin = compute_target_margin(equity_close)
            target_exposure = target_margin * leverage

            entry_margin = target_exposure / leverage
            fee_paid = trading_fee(target_exposure)
            required_cash = entry_margin + fee_paid

            if cash_free < required_cash:
                continue

            units = target_exposure / px_entry

            position = Position(
                symbol=symbol,
                entry_date=dt,
                inception_px=float(o0),
                first_candle_date=pd.Timestamp(d0),
                target_margin=float(target_margin),
                target_exposure=float(target_exposure),
                lots=[ShortLot(units=units, entry_px=px_entry)],
                margin=float(entry_margin),
            )

            open_positions[symbol] = position
            cash_free -= required_cash

            events.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "action": "entry",
                    "px": px_entry,
                    "exposure_traded": target_exposure,
                    "fee": fee_paid,
                    "realized_pnl": 0.0,
                    "cash_free_after": cash_free,
                    "entry_target_margin": target_margin,
                    "entry_target_exposure": target_exposure,
                    "equity_close_used": equity_close,
                }
            )

        for symbol, position in open_positions.items():
            px = float(px_close_today.get(symbol, np.nan))
            if not (np.isfinite(px) and px > 0):
                continue

            current_exposure = position.exposure(px)
            desired_exposure = float(position.target_exposure)
            if desired_exposure <= 0:
                continue

            deviation = (current_exposure - desired_exposure) / desired_exposure
            if abs(deviation) < rebalance_threshold:
                continue

            if i + 1 >= len(days):
                continue

            pending_rebalances[symbol] = {"decided_on_close": str(dt.date())}

            events.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "action": "rebalance_scheduled",
                    "px": px,
                    "dev": float(deviation),
                    "cur_exposure_close": float(current_exposure),
                    "desired_exposure": float(desired_exposure),
                }
            )

        equity_close, locked_margin, unrealized, gross_exposure = mark_equity(dt)

        daily_rows.append(
            {
                "date": dt,
                "cash_free": cash_free,
                "locked_margin": locked_margin,
                "unreal_pnl": unrealized,
                "equity": equity_close,
                "n_open": len(open_positions),
                "gross_exposure": gross_exposure,
                "pending_rebalances": len(pending_rebalances),
            }
        )

        if equity_close <= 0 and len(open_positions) > 0:
            for symbol, position in list(open_positions.items()):
                px = float(px_close_today.get(symbol, np.nan))
                if not (np.isfinite(px) and px > 0):
                    continue

                units = position.units_total()
                exposure_now = abs(units) * px
                realized = close_fifo_lots(position, units_to_close=abs(units), px=px)
                fee_paid = trading_fee(exposure_now)
                cash_free += position.margin + realized - fee_paid

                events.append(
                    {
                        "date": dt,
                        "symbol": symbol,
                        "action": "liquidate_portfolio",
                        "px": px,
                        "exposure_traded": exposure_now,
                        "fee": fee_paid,
                        "realized_pnl": realized,
                        "cash_free_after": cash_free,
                    }
                )

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": position.entry_date,
                        "exit_date": dt,
                        "entry_target_margin": position.target_margin,
                        "entry_target_exposure": position.target_exposure,
                        "exit_px": px,
                        "hold_days": int((dt - position.entry_date).days),
                        "reason": "portfolio_liquidation",
                    }
                )

            open_positions.clear()
            pending_rebalances.clear()
            cash_free = 0.0
            daily_rows[-1].update(
                {
                    "cash_free": 0.0,
                    "locked_margin": 0.0,
                    "unreal_pnl": 0.0,
                    "equity": 0.0,
                    "n_open": 0,
                    "gross_exposure": 0.0,
                    "pending_rebalances": 0,
                }
            )
            break

    daily_df = pd.DataFrame(daily_rows).set_index("date")
    events_df = pd.DataFrame(events).sort_values(["date", "symbol"]).reset_index(drop=True)
    trades_df = pd.DataFrame(trades)

    return daily_df, trades_df, events_df


def plot_equity_vs_btc(strategy_equity: pd.Series, btc_close: pd.Series, path: str) -> None:
    strategy_equity = strategy_equity.dropna()
    btc_close = btc_close.dropna()

    idx = strategy_equity.index.intersection(btc_close.index)
    if len(idx) < 5:
        return

    eq = strategy_equity.loc[idx]
    btc = btc_close.loc[idx]
    btc_scaled = (btc / btc.iloc[0]) * float(eq.iloc[0])

    plt.figure(figsize=(12, 6))
    plt.plot(eq.index, eq.values, label="Strategy equity")
    plt.plot(btc_scaled.index, btc_scaled.values, label="BTC buy & hold (scaled)", linestyle="--")
    plt.title("Equity vs BTC")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_fig(path)


def main() -> None:
    print("[1/7] Building listings...")
    listings = build_listings(
        listing_start=LISTING_START,
        listing_end=LISTING_END,
        quote_asset=QUOTE_ASSET,
        contract_type=CONTRACT_TYPE,
    )
    listings.to_csv(os.path.join(OUTDIR, "binance_perp_listings.csv"), index=False)
    print(f"Found {len(listings)} listings.")

    if listings.empty:
        raise RuntimeError("No listings found.")

    end_ms = dt_to_ms(BACKTEST_END) + 24 * 3600 * 1000 - 1
    earliest_listing = pd.to_datetime(listings["post_date"], utc=True).min().floor("D")
    start_need = (earliest_listing - pd.Timedelta(days=3)).floor("D")
    start_ms = dt_to_ms(str(start_need.date()))

    print("[2/7] Downloading BTC benchmark...")
    btc_df = fetch_daily_klines(BTC_SYMBOL, start_ms, end_ms)
    if btc_df.empty:
        raise RuntimeError("Failed to fetch BTC futures data.")

    btc_close = (
        btc_df.drop_duplicates(subset=["ts"])
        .set_index("ts")["close"]
        .astype(float)
        .sort_index()
    )

    print("[3/7] Downloading listing data...")
    series: dict[str, pd.DataFrame] = {}
    kept_rows = []

    for _, row in listings.iterrows():
        symbol = row["symbol"]
        listing_ts = pd.Timestamp(row["post_date"]).tz_convert("UTC")
        symbol_start_ms = dt_to_ms(str((listing_ts.floor("D") - pd.Timedelta(days=3)).date()))

        try:
            df = fetch_daily_klines(symbol, symbol_start_ms, end_ms)
            if df.empty or len(df) < 5:
                print(f"[SKIP] {symbol}: insufficient data")
                continue

            tmp = df[["ts", "open", "close"]].copy().dropna().sort_values("ts").reset_index(drop=True)
            if tmp.empty:
                print(f"[SKIP] {symbol}: empty open/close data")
                continue

            row_copy = row.to_dict()
            row_copy["first_kline_ts"] = tmp["ts"].min()
            row_copy["last_kline_ts"] = tmp["ts"].max()
            kept_rows.append(row_copy)

            series[symbol] = tmp
            print(f"[OK] {symbol}: {len(tmp)} candles")

        except Exception as e:
            print(f"[ERR] {symbol}: {e}")

    if not series:
        raise RuntimeError("No symbols had sufficient price data.")

    listings_with_data = pd.DataFrame(kept_rows)
    listings_with_data["post_date"] = pd.to_datetime(listings_with_data["post_date"], utc=True)

    print("[4/7] Building price panels...")
    open_px, close_px = build_open_close_panels(series)

    available_symbols = set(close_px.columns)
    listings_with_data = listings_with_data[listings_with_data["symbol"].isin(available_symbols)].copy().reset_index(drop=True)

    listings_with_data.to_csv(os.path.join(OUTDIR, "binance_perp_listings_with_kline_span.csv"), index=False)
    open_px.to_csv(os.path.join(OUTDIR, "prices_panel_open.csv"))
    close_px.to_csv(os.path.join(OUTDIR, "prices_panel_close.csv"))

    print("[5/7] Running backtest...")
    daily_df, trades_df, events_df = backtest_strategy(
        listings=listings_with_data,
        open_px=open_px,
        close_px=close_px,
        start_capital=START_CAPITAL,
        max_concurrent_pos=MAX_CONCURRENT_POS,
        leverage=LEVERAGE,
        entry_delay_days=ENTRY_DELAY_DAYS,
        max_hold_days=MAX_HOLD_DAYS,
        rebalance_threshold=REBALANCE_THRESHOLD,
        dont_add_if_down_from_inception=DONT_ADD_IF_DOWN_FROM_INCEPTION,
        fee_bps=FEE_BPS,
        reserve_ratio=REBALANCE_RESERVE_RATIO,
        min_target_margin=MIN_TARGET_MARGIN,
        max_target_margin=MAX_TARGET_MARGIN,
    )

    daily_df.to_csv(os.path.join(OUTDIR, "daily_equity.csv"))
    trades_df.to_csv(os.path.join(OUTDIR, "trades.csv"), index=False)
    events_df.to_csv(os.path.join(OUTDIR, "events.csv"), index=False)

    equity = daily_df["equity"].copy()
    strategy_stats = perf_stats(equity)

    idx = equity.dropna().index.intersection(btc_close.dropna().index)
    if len(idx):
        equity_for_plot = equity.loc[idx]
        btc_for_plot = btc_close.loc[idx]
    else:
        equity_for_plot = equity
        btc_for_plot = btc_close

    print("[6/7] Saving charts...")
    plot_equity_vs_btc(equity_for_plot, btc_for_plot, os.path.join(OUTDIR, "equity_vs_btc.png"))
    plot_monthly_returns(equity, os.path.join(OUTDIR, "monthly_returns.png"))
    plot_rolling_vol(equity, os.path.join(OUTDIR, "rolling_90d_ann_vol.png"))
    plot_holdings_count(daily_df["n_open"], os.path.join(OUTDIR, "holdings_count.png"))
    plot_pct_invested(daily_df, os.path.join(OUTDIR, "pct_equity_invested.png"))
    plot_rolling_sharpe(equity, os.path.join(OUTDIR, "rolling_90d_ann_sharpe.png"))
    plot_drawdown(equity, os.path.join(OUTDIR, "drawdown.png"))

    summary = {
        "exchange": "Binance USDⓈ-M Futures",
        "listings_source": "exchangeInfo.onboardDate",
        "listing_start": LISTING_START,
        "listing_end": LISTING_END,
        "backtest_end": BACKTEST_END,
        "quote_asset": QUOTE_ASSET,
        "contract_type": CONTRACT_TYPE,
        "start_capital": START_CAPITAL,
        "max_concurrent_pos": MAX_CONCURRENT_POS,
        "leverage": LEVERAGE,
        "reserve_ratio": REBALANCE_RESERVE_RATIO,
        "min_target_margin": MIN_TARGET_MARGIN,
        "rebalance_threshold": REBALANCE_THRESHOLD,
        "max_hold_days": MAX_HOLD_DAYS,
        "dont_add_if_down_from_inception": DONT_ADD_IF_DOWN_FROM_INCEPTION,
        "fee_bps": FEE_BPS,
        "n_symbols_with_prices": int(len(listings_with_data)),
        "n_trades": int(len(trades_df)),
        "first_listing": str(listings_with_data["post_date"].min()) if len(listings_with_data) else None,
        "last_listing": str(listings_with_data["post_date"].max()) if len(listings_with_data) else None,
        "final_equity": float(equity.dropna().iloc[-1]) if len(equity.dropna()) else None,
        "perf_strategy": strategy_stats,
        "note": (
            "Listings come from the current Binance exchangeInfo onboardDate field. "
            "That avoids scraping announcements, but it will not recover symbols no longer present in exchangeInfo."
        ),
    }

    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[7/7] Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
