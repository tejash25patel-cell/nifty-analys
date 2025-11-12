#!/usr/bin/env python3
# nifty_banknifty_vega_dashboard_simple.py
"""
Streamlit app: NIFTY / BANKNIFTY Vega Dashboard (simple)
+ Smart S&R (existing) + Intraday S&R (ATM ± 5 strikes) tab
"""
import os
import time
import json
import gzip
import brotli
import math
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
from math import log, sqrt
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "vega_oi_history.csv")
INTRADAY_SR_CSV = os.path.join(DATA_DIR, "intraday_sr.csv")
DEFAULT_R = 0.0675

st.set_page_config(page_title="NIFTY/BANKNIFTY Vega Dashboard (simple)", layout="wide")
st.title("NIFTY / BANKNIFTY — Option Chain + Vega Summary (Day Open / Current / Difference)")

# Sidebar
symbol = st.sidebar.selectbox("Index", ["NIFTY", "BANKNIFTY"], index=0)
refresh_seconds = st.sidebar.number_input("Auto-refresh interval (seconds)", min_value=30, max_value=600, value=60)
track_range = st.sidebar.number_input("Track strikes ATM ± N", min_value=0, max_value=4, value=0)
save_csv_each_run = st.sidebar.checkbox("Auto-save Vega+OI CSV (append each refresh)", value=True)
risk_free_rate = float(st.sidebar.number_input("Risk-free rate (annual decimal)", value=DEFAULT_R, format="%.4f"))
history_limit = int(st.sidebar.number_input("Max Vega history rows", min_value=10, max_value=10000, value=1440))
show_sr_charts_default = st.sidebar.checkbox("Show S&R charts by default (Tab 2)", value=False)

# autorefresh trigger
_count = st_autorefresh(interval=refresh_seconds * 1000, limit=None, key="autorefresh_simple_dashboard")

# -------------------------
# NSE fetch helpers (gzip / brotli)
# -------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
}
session = requests.Session()

def fetch_option_chain(sym="NIFTY", retries=3, backoff=1.5):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}"
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=8)
    except Exception:
        pass
    last_err = None
    for attempt in range(1, retries+1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                raw = resp.content
                ce = resp.headers.get("Content-Encoding", "").lower()
                try:
                    if "br" in ce:
                        text = brotli.decompress(raw).decode("utf-8", errors="ignore")
                    elif raw[:2] == b"\x1f\x8b":
                        text = gzip.decompress(raw).decode("utf-8", errors="ignore")
                    else:
                        text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = resp.text
                if text and text.strip().startswith("{"):
                    return json.loads(text), None
                last_err = f"Unexpected content len={len(raw)}"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"Request error: {e}"
        time.sleep(backoff ** attempt)
        try:
            session.get("https://www.nseindia.com", headers=HEADERS, timeout=6)
        except Exception:
            pass
    return None, f"Failed after {retries} attempts. Last error: {last_err}"

# -------------------------
# Black-Scholes Greeks
# -------------------------
def bs_d1_d2(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes_vega(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        d1, _ = bs_d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0   # per 1% IV
        return float(vega)
    except Exception:
        return 0.0

def black_scholes_gamma(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        d1, _ = bs_d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return float(gamma)
    except Exception:
        return 0.0

def black_scholes_delta_call(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        d1, _ = bs_d1_d2(S, K, T, r, sigma)
        return float(norm.cdf(d1))
    except Exception:
        return 0.0

def black_scholes_theta_call(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        d1, d2 = bs_d1_d2(S, K, T, r, sigma)
        first = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        second = - r * K * math.exp(-r * T) * norm.cdf(d2)
        theta = (first + second) / 365.0
        return float(theta)
    except Exception:
        return 0.0

def black_scholes_delta_put(S, K, T, r, sigma):
    return black_scholes_delta_call(S, K, T, r, sigma) - 1.0

def black_scholes_theta_put(S, K, T, r, sigma):
    try:
        if sigma <= 0 or T <= 0:
            return 0.0
        d1, d2 = bs_d1_d2(S, K, T, r, sigma)
        first = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        second = r * K * math.exp(-r * T) * norm.cdf(-d2)
        theta = (first + second) / 365.0
        return float(theta)
    except Exception:
        return 0.0

# -------------------------
# weekly expiry picker
# -------------------------
from datetime import datetime as _dt
def pick_weekly_expiry(expiry_list):
    dates = []
    for e in expiry_list:
        try:
            d = _dt.strptime(e, "%d-%b-%Y")
            dates.append(d)
        except Exception:
            pass
    if not dates:
        return None
    dates.sort()
    def is_monthly(d):
        same_month = [x for x in dates if x.month==d.month and x.year==d.year]
        thursdays = [x for x in same_month if x.weekday()==3]
        if not thursdays:
            return False
        last_thu = max(thursdays)
        return d == last_thu
    weekly = [d for d in dates if not is_monthly(d)]
    today = _dt.now()
    for d in weekly:
        if d.date() >= today.date():
            return d.strftime("%d-%b-%Y")
    return (weekly[0].strftime("%d-%b-%Y")) if weekly else dates[0].strftime("%d-%b-%Y")

# -------------------------
# Session state
# -------------------------
if "vega_history" not in st.session_state:
    st.session_state.vega_history = []
if "last_oi" not in st.session_state:
    st.session_state.last_oi = {}
if "day_open" not in st.session_state:
    st.session_state.day_open = None
# storage for option OI per strike used for delta calculation across refreshes
if "option_last_oi" not in st.session_state:
    st.session_state.option_last_oi = {}  # { strike: { 'CE': oi, 'PE': oi } }
# intraday sr log
if "intraday_sr_history" not in st.session_state:
    st.session_state.intraday_sr_history = []

def append_history(row):
    st.session_state.vega_history.insert(0, row)
    if len(st.session_state.vega_history) > history_limit:
        st.session_state.vega_history = st.session_state.vega_history[:history_limit]

# -------------------------
# Fetch live option chain
# -------------------------
data, err = fetch_option_chain(symbol)
if data is None:
    st.error(f"Live fetch failed: {err}")
    st.stop()

records = data.get("records", {})
expiry_list = records.get("expiryDates", [])
if not expiry_list:
    st.error("No expiry dates found.")
    st.stop()

weekly_expiry = pick_weekly_expiry(expiry_list) or expiry_list[0]
spot = records.get("underlyingValue", None)

df_all = pd.json_normalize(records.get("data", []))
if df_all.empty:
    st.error("No option chain rows returned.")
    st.stop()

# Filter rows for weekly expiry
mask = pd.Series(False, index=df_all.index)
for col in ["expiryDate", "CE.expiryDate", "PE.expiryDate"]:
    if col in df_all.columns:
        mask = mask | (df_all[col] == weekly_expiry)
df = df_all[mask].copy().reset_index(drop=True)

# Build CE & PE frames
def safe_col(src, name):
    return src[name] if name in src.columns else pd.Series([None]*len(src), index=src.index)

ce_df = pd.DataFrame({
    "Strike": safe_col(df, "CE.strikePrice"),
    "LTP": safe_col(df, "CE.lastPrice"),
    "OI": safe_col(df, "CE.openInterest"),
    "ChgOI": safe_col(df, "CE.changeinOpenInterest"),
    "IV": safe_col(df, "CE.impliedVolatility"),
})
pe_df = pd.DataFrame({
    "Strike": safe_col(df, "PE.strikePrice"),
    "LTP": safe_col(df, "PE.lastPrice"),
    "OI": safe_col(df, "PE.openInterest"),
    "ChgOI": safe_col(df, "PE.changeinOpenInterest"),
    "IV": safe_col(df, "PE.impliedVolatility"),
})

# numeric conversions
for d in (ce_df, pe_df):
    d["Strike"] = pd.to_numeric(d["Strike"], errors="coerce")
    d["LTP"] = pd.to_numeric(d["LTP"], errors="coerce")
    d["OI"] = pd.to_numeric(d["OI"], errors="coerce").fillna(0).astype(int)
    d["ChgOI"] = pd.to_numeric(d["ChgOI"], errors="coerce").fillna(0).astype(int)
    d["IV"] = pd.to_numeric(d["IV"], errors="coerce")

ce_df = ce_df.dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
pe_df = pe_df.dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)

# Support/resistance using max OI nearby (basic fallback)
try:
    support_strike = int(pe_df.loc[pe_df["OI"].idxmax(), "Strike"])
except Exception:
    support_strike = None
try:
    resistance_strike = int(ce_df.loc[ce_df["OI"].idxmax(), "Strike"])
except Exception:
    resistance_strike = None

# ATM & track strikes
step = 50 if symbol == "NIFTY" else 100
atm_strike = int(round((spot or 0) / step) * step) if spot else None
strikes_to_track = [atm_strike + i*step for i in range(-track_range, track_range+1)] if atm_strike else []

# T in years
T_years = 1/365.0
try:
    exp_dt = _dt.strptime(weekly_expiry, "%d-%b-%Y")
    T_days = max((exp_dt - _dt.now()).days, 0)
    T_years = max(T_days / 365.0, 1/365.0)
except Exception:
    pass

# compute aggregated Greeks across tracked strikes (sums) and ATM OI details
r = float(risk_free_rate)
agg = {"call_vega": 0.0, "put_vega": 0.0, "total_vega": 0.0,
       "call_theta": 0.0, "put_theta": 0.0,
       "call_gamma": 0.0, "put_gamma": 0.0,
       "call_delta": 0.0, "put_delta": 0.0,
       "iv_avg": 0.0, "coi": 0, "count": 0}

for strike in strikes_to_track:
    ce_row = ce_df[ce_df["Strike"] == strike]
    pe_row = pe_df[pe_df["Strike"] == strike]

    ce_iv = float(ce_row["IV"].iloc[0]) / 100.0 if (not ce_row.empty and pd.notna(ce_row["IV"].iloc[0])) else 0.20
    pe_iv = float(pe_row["IV"].iloc[0]) / 100.0 if (not pe_row.empty and pd.notna(pe_row["IV"].iloc[0])) else 0.20

    call_vega = black_scholes_vega(spot or 0.0, strike, T_years, r, ce_iv)
    put_vega = black_scholes_vega(spot or 0.0, strike, T_years, r, pe_iv)

    call_theta = black_scholes_theta_call(spot or 0.0, strike, T_years, r, ce_iv)
    put_theta  = black_scholes_theta_put(spot or 0.0, strike, T_years, r, pe_iv)

    call_gamma = black_scholes_gamma(spot or 0.0, strike, T_years, r, ce_iv)
    put_gamma  = black_scholes_gamma(spot or 0.0, strike, T_years, r, pe_iv)

    call_delta = black_scholes_delta_call(spot or 0.0, strike, T_years, r, ce_iv)
    put_delta  = black_scholes_delta_put(spot or 0.0, strike, T_years, r, pe_iv)

    ce_oi_val = int(ce_row["OI"].iloc[0]) if (not ce_row.empty) else 0
    pe_oi_val = int(pe_row["OI"].iloc[0]) if (not pe_row.empty) else 0

    agg["call_vega"] += call_vega
    agg["put_vega"]  += put_vega
    agg["total_vega"] += call_vega + put_vega

    agg["call_theta"] += call_theta
    agg["put_theta"] += put_theta
    agg["call_gamma"] += call_gamma
    agg["put_gamma"] += put_gamma
    agg["call_delta"] += call_delta
    agg["put_delta"] += put_delta

    agg["iv_avg"] += (ce_iv + pe_iv) / 2.0
    agg["coi"] += (ce_oi_val + pe_oi_val)
    agg["count"] += 1

# finalize iv_avg
if agg["count"] > 0:
    agg["iv_avg"] = agg["iv_avg"] / agg["count"] * 100.0  # show IV in percent
else:
    agg["iv_avg"] = 0.0

# ATM OI and changes
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
atm_call_oi = int(ce_df.loc[ce_df["Strike"] == atm_strike, "OI"].iloc[0]) if (atm_strike in ce_df["Strike"].values) else 0
atm_put_oi  = int(pe_df.loc[pe_df["Strike"] == atm_strike, "OI"].iloc[0]) if (atm_strike in pe_df["Strike"].values) else 0

prev_call_oi, prev_put_oi = st.session_state.last_oi.get(atm_strike, (None, None))
change_call_oi = atm_call_oi - prev_call_oi if prev_call_oi is not None else 0
change_put_oi  = atm_put_oi - prev_put_oi  if prev_put_oi is not None else 0
oi_diff = atm_put_oi - atm_call_oi
prev_diff = (prev_put_oi - prev_call_oi) if (prev_put_oi is not None and prev_call_oi is not None) else None
change_diff = oi_diff - prev_diff if prev_diff is not None else 0
st.session_state.last_oi[atm_strike] = (atm_call_oi, atm_put_oi)

# log row: include aggregated Greeks and OI
log_row = {
    "timestamp": timestamp,
    "symbol": symbol,
    "expiry": weekly_expiry,
    "strike": int(atm_strike) if atm_strike else None,
    "call_vega": round(agg["call_vega"], 6),
    "put_vega": round(agg["put_vega"], 6),
    "total_vega": round(agg["total_vega"], 6),
    "call_theta": round(agg["call_theta"], 6),
    "put_theta": round(agg["put_theta"], 6),
    "call_gamma": round(agg["call_gamma"], 6),
    "put_gamma": round(agg["put_gamma"], 6),
    "call_delta": round(agg["call_delta"], 6),
    "put_delta": round(agg["put_delta"], 6),
    "iv": round(agg["iv_avg"], 4),
    "coi": int(agg["coi"]),
    "count": int(agg["count"]),
    "call_oi": atm_call_oi,
    "put_oi": atm_put_oi,
    "change_call_oi": change_call_oi,
    "change_put_oi": change_put_oi,
    "oi_diff": oi_diff,
    "change_oi_diff": change_diff
}

# compute deltas relative to previous history entry (for Vega Δ etc.)
prev = st.session_state.vega_history[0] if st.session_state.vega_history and st.session_state.vega_history[0].get("strike") == log_row["strike"] else None
if prev:
    log_row["call_vega_delta"] = round(log_row["call_vega"] - prev.get("call_vega", 0.0), 6)
    log_row["put_vega_delta"]  = round(log_row["put_vega"]  - prev.get("put_vega", 0.0), 6)
    log_row["total_vega_delta"] = round(log_row["total_vega"] - prev.get("total_vega", 0.0), 6)
else:
    log_row["call_vega_delta"] = 0.0
    log_row["put_vega_delta"] = 0.0
    log_row["total_vega_delta"] = 0.0

# append history
append_history(log_row)

# set day_open if not set in session: first successful fetch becomes "Day Open"
if st.session_state.day_open is None:
    st.session_state.day_open = {
        "call_vega": log_row["call_vega"],
        "put_vega": log_row["put_vega"],
        "total_vega": log_row["total_vega"],
        "call_theta": log_row["call_theta"],
        "put_theta": log_row["put_theta"],
        "call_gamma": log_row["call_gamma"],
        "put_gamma": log_row["put_gamma"],
        "call_delta": log_row["call_delta"],
        "put_delta": log_row["put_delta"],
        "iv": log_row["iv"],
        "coi": log_row["coi"],
        "count": log_row["count"],
    }

# persist CSV append
if save_csv_each_run:
    try:
        row_df = pd.DataFrame([log_row])
        if os.path.exists(CSV_PATH):
            row_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            row_df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        st.sidebar.error(f"CSV save failed: {e}")

# -------------------------
# Intraday S&R calculation (ATM ± 5 strikes)
# -------------------------
def compute_intraday_sr(ce_df, pe_df, atm, step, top_n=3, weight=1.5):
    """
    Return support & resistance tables for ATM ± 5 strikes.
    Support uses PUT OI (pe_df) build-ups; Resistance uses CALL OI (ce_df) build-ups.
    Score = OI + weight * max(delta_OI, 0)
    """
    strikes = [atm + i*step for i in range(-5, 6)]  # ATM ± 5
    rows = []
    for s in strikes:
        ce_row = ce_df[ce_df["Strike"] == s]
        pe_row = pe_df[pe_df["Strike"] == s]
        ce_oi = int(ce_row["OI"].iloc[0]) if (not ce_row.empty) else 0
        pe_oi = int(pe_row["OI"].iloc[0]) if (not pe_row.empty) else 0

        # previous recorded OI for delta computation
        prev = st.session_state.option_last_oi.get(s, {"CE": None, "PE": None})
        prev_ce = prev.get("CE")
        prev_pe = prev.get("PE")

        ce_delta = ce_oi - prev_ce if prev_ce is not None else 0
        pe_delta = pe_oi - prev_pe if prev_pe is not None else 0

        ce_score = ce_oi + weight * max(ce_delta, 0)
        pe_score = pe_oi + weight * max(pe_delta, 0)

        rows.append({
            "Strike": int(s),
            "CE_OI": ce_oi,
            "CE_ΔOI": int(ce_delta),
            "CE_Score": int(ce_score),
            "PE_OI": pe_oi,
            "PE_ΔOI": int(pe_delta),
            "PE_Score": int(pe_score)
        })

        # update last oi for next run
        st.session_state.option_last_oi[s] = {"CE": ce_oi, "PE": pe_oi}

    df_sr = pd.DataFrame(rows)

    # build support and resistance top lists
    support_df = df_sr.sort_values("PE_Score", ascending=False).head(top_n)[["Strike","PE_OI","PE_ΔOI","PE_Score"]].rename(
        columns={"PE_OI":"OI","PE_ΔOI":"ΔOI","PE_Score":"Score"}
    )
    support_df["Type"] = "Support"

    resistance_df = df_sr.sort_values("CE_Score", ascending=False).head(top_n)[["Strike","CE_OI","CE_ΔOI","CE_Score"]].rename(
        columns={"CE_OI":"OI","CE_ΔOI":"ΔOI","CE_Score":"Score"}
    )
    resistance_df["Type"] = "Resistance"

    return support_df.reset_index(drop=True), resistance_df.reset_index(drop=True), df_sr

# append intraday SR to session history and CSV
def log_intraday_sr(support_df, resistance_df, atm, expiry):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for _, r in support_df.iterrows():
        rec = {
            "timestamp": now,
            "symbol": symbol,
            "expiry": expiry,
            "atm": atm,
            "type": "Support",
            "strike": int(r["Strike"]),
            "oi": int(r["OI"]),
            "delta_oi": int(r["ΔOI"]),
            "score": int(r["Score"])
        }
        records.append(rec)
    for _, r in resistance_df.iterrows():
        rec = {
            "timestamp": now,
            "symbol": symbol,
            "expiry": expiry,
            "atm": atm,
            "type": "Resistance",
            "strike": int(r["Strike"]),
            "oi": int(r["OI"]),
            "delta_oi": int(r["ΔOI"]),
            "score": int(r["Score"])
        }
        records.append(rec)
    # save to session history
    st.session_state.intraday_sr_history = records + st.session_state.intraday_sr_history
    # trim
    if len(st.session_state.intraday_sr_history) > 1000:
        st.session_state.intraday_sr_history = st.session_state.intraday_sr_history[:1000]
    # persist csv append
    try:
        df_save = pd.DataFrame(records)
        if not df_save.empty:
            if os.path.exists(INTRADAY_SR_CSV):
                df_save.to_csv(INTRADAY_SR_CSV, mode='a', header=False, index=False)
            else:
                df_save.to_csv(INTRADAY_SR_CSV, index=False)
    except Exception:
        pass

# Compute intraday S&R for ATM ±5
support_df, resistance_df, full_band_df = compute_intraday_sr(ce_df, pe_df, atm_strike, step, top_n=3, weight=1.5)
log_intraday_sr(support_df, resistance_df, atm_strike, weekly_expiry)

# -------------------------
# Tabs UI
# -------------------------
tab1, tab2, tab3 = st.tabs(["Vega Dashboard", "Smart S&R", "Intraday S&R (ATM ±5)"])

with tab1:
    st.markdown("### ATM Vega Summary (latest)")
    latest = st.session_state.vega_history[0] if st.session_state.vega_history else None
    if latest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time", latest["timestamp"])
        c2.metric("Call Vega", f"{latest['call_vega']:.6f}")
        c3.metric("Put Vega", f"{latest['put_vega']:.6f}")
        c4.metric("Total Vega", f"{latest['total_vega']:.6f}")
    else:
        st.info("Vega history will populate after the first successful fetch.")

    st.markdown("### Vega Log (latest first)")
    hist_df = pd.DataFrame(st.session_state.vega_history)
    if not hist_df.empty:
        st.dataframe(hist_df[[
            "timestamp","symbol","expiry","strike",
            "call_vega","put_vega","total_vega",
            "call_vega_delta","put_vega_delta","total_vega_delta",
            "call_oi","put_oi","change_call_oi","change_put_oi","oi_diff"
        ]], width="stretch", height=300)
        st.download_button("Download Vega+OI history CSV", hist_df.to_csv(index=False), file_name=f"vega_oi_history_{symbol}.csv", key="download_hist")
    else:
        st.write("No Vega history recorded yet.")

    st.markdown("---")
    col_left, col_mid, col_right = st.columns([4,1,4])
    with col_left:
        st.subheader("Calls (CE) — OI | ChgOI | LTP | IV")
        display_ce = ce_df[ce_df["Strike"].between(atm_strike - 10*step, atm_strike + 10*step)].copy() if atm_strike else ce_df.copy()
        display_ce = display_ce[["Strike","OI","ChgOI","LTP","IV"]].reset_index(drop=True)
        st.dataframe(display_ce.style.format({"LTP":"{:.2f}","IV":"{:.2f}"}), width="stretch", height=600)
    with col_mid:
        st.subheader("Strike")
        strikes_display = display_ce["Strike"].tolist() if (not display_ce.empty) else ce_df["Strike"].tolist()
        md = "### Strikes\n\n" + "\n".join([f"**{int(s)}**" for s in strikes_display])
        st.markdown(md)
    with col_right:
        st.subheader("Puts (PE) — OI | ChgOI | LTP | IV")
        display_pe = pe_df[pe_df["Strike"].between(atm_strike - 10*step, atm_strike + 10*step)].copy() if atm_strike else pe_df.copy()
        display_pe = display_pe[["Strike","OI","ChgOI","LTP","IV"]].reset_index(drop=True)
        st.dataframe(display_pe.style.format({"LTP":"{:.2f}","IV":"{:.2f}"}), width="stretch", height=600)

    st.markdown("---")
    st.subheader("ATM Vega Summary (latest) — aggregated")
    day_open = st.session_state.day_open
    current = {
        "call_vega": log_row["call_vega"],
        "put_vega": log_row["put_vega"],
        "total_vega": log_row["total_vega"],
        "call_theta": log_row["call_theta"],
        "put_theta": log_row["put_theta"],
        "call_gamma": log_row["call_gamma"],
        "put_gamma": log_row["put_gamma"],
        "call_delta": log_row["call_delta"],
        "put_delta": log_row["put_delta"],
        "iv": log_row["iv"],
        "coi": log_row["coi"],
        "count": log_row["count"]
    }

    diff = {}
    for k in ["call_vega","put_vega","total_vega","call_theta","put_theta",
              "call_gamma","put_gamma","call_delta","put_delta","iv","coi","count"]:
        diff[k] = current.get(k, 0.0) - (day_open.get(k, 0.0) if day_open else 0.0)

    summary_df = pd.DataFrame([
        {
            "Label": "Day Open",
            "Vega": day_open["call_vega"] + day_open["put_vega"] if day_open else 0.0,
            "Theta": (day_open["call_theta"] + day_open["put_theta"]) if day_open else 0.0,
            "Gamma": (day_open["call_gamma"] + day_open["put_gamma"]) if day_open else 0.0,
            "Delta": (day_open["call_delta"] + day_open["put_delta"]) if day_open else 0.0,
            "IV": day_open["iv"] if day_open else 0.0,
            "Coi": day_open["coi"] if day_open else 0,
            "Count": day_open["count"] if day_open else 0
        },
        {
            "Label": "Current",
            "Vega": current["call_vega"] + current["put_vega"],
            "Theta": current["call_theta"] + current["put_theta"],
            "Gamma": current["call_gamma"] + current["put_gamma"],
            "Delta": current["call_delta"] + current["put_delta"],
            "IV": current["iv"],
            "Coi": current["coi"],
            "Count": current["count"]
        },
        {
            "Label": "Difference",
            "Vega": diff["call_vega"] + diff["put_vega"],
            "Theta": diff["call_theta"] + diff["put_theta"],
            "Gamma": diff["call_gamma"] + diff["put_gamma"],
            "Delta": diff["call_delta"] + diff["put_delta"],
            "IV": diff["iv"],
            "Coi": diff["coi"],
            "Count": diff["count"]
        }
    ])
    display_cols = ["Label","Vega","Theta","Gamma","Delta","IV","Coi","Count"]
    st.table(summary_df[display_cols].round({"Vega":6,"Theta":6,"Gamma":6,"Delta":6,"IV":4}))

with tab2:
    st.markdown("### Smart Support & Resistance (based on cumulative OI & ΔOI)")
    # (reuse existing smart SR logic—uses cumulative OI)
    try:
        pe_df_sorted = pe_df.sort_values("Strike").reset_index(drop=True)
        pe_df_sorted["CumulativeOI"] = pe_df_sorted["OI"].cumsum()
    except Exception:
        pe_df_sorted = pe_df.copy(); pe_df_sorted["CumulativeOI"] = pe_df_sorted["OI"].cumsum()

    try:
        ce_df_sorted = ce_df.sort_values("Strike").reset_index(drop=True)
        ce_df_sorted["CumulativeOI"] = ce_df_sorted["OI"].iloc[::-1].cumsum().iloc[::-1]
    except Exception:
        ce_df_sorted = ce_df.copy(); ce_df_sorted["CumulativeOI"] = ce_df_sorted["OI"].iloc[::-1].cumsum().iloc[::-1]

    if "last_ce_oi" not in st.session_state:
        st.session_state.last_ce_oi = ce_df_sorted.set_index("Strike")["OI"].to_dict()
    if "last_pe_oi" not in st.session_state:
        st.session_state.last_pe_oi = pe_df_sorted.set_index("Strike")["OI"].to_dict()

    ce_df_sorted["OI_Delta"] = ce_df_sorted.apply(lambda x: int(x["OI"]) - int(st.session_state.last_ce_oi.get(x["Strike"], 0)), axis=1)
    pe_df_sorted["OI_Delta"] = pe_df_sorted.apply(lambda x: int(x["OI"]) - int(st.session_state.last_pe_oi.get(x["Strike"], 0)), axis=1)

    # update stored last
    st.session_state.last_ce_oi = ce_df_sorted.set_index("Strike")["OI"].to_dict()
    st.session_state.last_pe_oi = pe_df_sorted.set_index("Strike")["OI"].to_dict()

    band = 10
    band_pe = pe_df_sorted[(pe_df_sorted["Strike"] >= (atm_strike - band*step)) & (pe_df_sorted["Strike"] <= (atm_strike + band*step))].copy()
    band_ce = ce_df_sorted[(ce_df_sorted["Strike"] >= (atm_strike - band*step)) & (ce_df_sorted["Strike"] <= (atm_strike + band*step))].copy()
    band_pe["SupportScore"] = band_pe["CumulativeOI"] + band_pe["OI_Delta"].clip(lower=0) * 2.0
    band_ce["ResistanceScore"] = band_ce["CumulativeOI"] + band_ce["OI_Delta"].clip(lower=0) * 2.0

    try:
        smart_support = int(band_pe.loc[band_pe["SupportScore"].idxmax(), "Strike"])
    except Exception:
        smart_support = support_strike
    try:
        smart_resistance = int(band_ce.loc[band_ce["ResistanceScore"].idxmax(), "Strike"])
    except Exception:
        smart_resistance = resistance_strike

    st.metric("Smart Support", smart_support if smart_support else "-", help="PUT cumulative OI weighted")
    st.metric("Smart Resistance", smart_resistance if smart_resistance else "-", help="CALL cumulative OI weighted")
    show_charts = st.checkbox("Show OI & ΔOI charts (S&R tab)", value=show_sr_charts_default, key="sr_show_charts")
    if show_charts:
        try:
            ce_plot_df = band_ce.sort_values("Strike")[["Strike","OI","OI_Delta","ResistanceScore"]].copy()
            pe_plot_df = band_pe.sort_values("Strike")[["Strike","OI","OI_Delta","SupportScore"]].copy()
            fig_pe = px.bar(pe_plot_df, x="Strike", y=["OI","OI_Delta"], barmode="group", title="PUT OI and ΔOI (band)")
            fig_ce = px.bar(ce_plot_df, x="Strike", y=["OI","OI_Delta"], barmode="group", title="CALL OI and ΔOI (band)")
            st.plotly_chart(fig_pe, width="stretch", use_container_width=True, key="pe_sr_chart_tab2")
            st.plotly_chart(fig_ce, width="stretch", use_container_width=True, key="ce_sr_chart_tab2")
        except Exception:
            st.warning("Failed to render S&R charts.")

with tab3:
    st.markdown("### Intraday S&R Zones — ATM ± 5 strikes (short-term)")
    st.write(f"ATM strike: {atm_strike}  |  Spot: {spot}  |  Expiry: {weekly_expiry}  |  Updated: {timestamp}")

    # Support table (PUTs)
    st.subheader("Top Support Zones (PUT OI build-ups)")
    if not support_df.empty:
        # strength label
        def strength_label(score, max_score):
            if max_score == 0: return "Weak"
            p = score / max_score * 100
            if p > 80: return "Strong"
            if p > 50: return "Moderate"
            return "Weak"
        max_supp = support_df["Score"].max() if not support_df.empty else 1
        support_df["Strength"] = support_df["Score"].apply(lambda x: strength_label(x, max_supp))
        # show simple table
        st.dataframe(support_df[["Strike","OI","ΔOI","Score","Strength"]], width="stretch", height=200)
    else:
        st.write("No support data in ATM ±5 band.")

    # Resistance table (CALLs)
    st.subheader("Top Resistance Zones (CALL OI build-ups)")
    if not resistance_df.empty:
        max_res = resistance_df["Score"].max() if not resistance_df.empty else 1
        resistance_df["Strength"] = resistance_df["Score"].apply(lambda x: strength_label(x, max_res))
        st.dataframe(resistance_df[["Strike","OI","ΔOI","Score","Strength"]], width="stretch", height=200)
    else:
        st.write("No resistance data in ATM ±5 band.")

    st.markdown("---")
    st.write("Full ATM ±5 band (CE and PE scores):")
    st.dataframe(full_band_df.sort_values("Strike").reset_index(drop=True), width="stretch", height=300)

    # show/download intraday SR history for session
    st.markdown("### Intraday S&R Log (session recent entries)")
    intr_hist = pd.DataFrame(st.session_state.intraday_sr_history)
    if not intr_hist.empty:
        st.dataframe(intr_hist.head(200), width="stretch", height=300)
        st.download_button("Download Intraday S&R CSV (session)", intr_hist.to_csv(index=False), file_name=f"intraday_sr_{symbol}.csv", key="download_intraday")
    else:
        st.write("No intraday S&R records yet.")

st.caption("Notes: Intraday S&R uses short-term OI build (ΔOI) around ATM ±5 strikes. Score = OI + 1.5 * max(ΔOI,0). Logged to data/intraday_sr.csv.")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Refresh interval: {refresh_seconds} sec")
# End of file