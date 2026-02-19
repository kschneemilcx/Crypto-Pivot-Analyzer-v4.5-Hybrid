"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CRYPTO PIVOT ANALYZER v4.5 HYBRID — Multi-Asset Advanced System       ║
║  TIME + DISTANCE + P1/P2 + Flip Risk + Educational + Projections        ║
║  BTC/ETH · Weekly/Daily · ICT/SMC Statistical Validation                ║
╚══════════════════════════════════════════════════════════════════════════╝

SISTEMA HÍBRIDO:
  ✓ TIME validation (ICT): ¿High/low a hora H típicamente holds o taken?
  ✓ DISTANCE validation (ICT): ¿Displacement X% suficiente o extends?
  ✓ P1/P2 detection (SMC): Primer/segundo extremo semanal con Flip Risk
  ✓ Pivots tradicionales: PP/R1/S1/R2/S2
  ✓ Geometric bias: price vs WO vs PP
  ✓ Price projections: Escenarios bull/bear con probabilidades
  ✓ Decision synthesis: Semáforo multi-señal + score final
  ✓ Educational: Explicaciones inline para cada métrica

INSTALACIÓN:
    pip install pandas numpy requests ccxt colorama tqdm

EJECUCIÓN:
    python crypto_pivot_v4_5_hybrid.py
"""
import os, sys, json, warnings, webbrowser
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
from colorama import init, Fore, Style

warnings.filterwarnings("ignore")
init(autoreset=True)

CONFIG = {
    "OUTPUT_DIR": "pivot_v45_output",
    "CACHE_DIR":  "pivot_v45_cache",
    "ASSETS":     ["BTC", "ETH"],
}

DAY_NAMES = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
DAY_SHORT = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]

# ══════════════════════════════════════════════════════════════════════════
#  CONSOLA
# ══════════════════════════════════════════════════════════════════════════
def banner():
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════════════════════╗
║  🔷 CRYPTO PIVOT ANALYZER v4.5 HYBRID                                   ║
║  TIME · DISTANCE · P1/P2 · Projections · Educational                    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def section(t): print(Fore.YELLOW + Style.BRIGHT + "\n" + "="*72 + "\n  " + t + "\n" + "="*72)
def info(m):    print(Fore.CYAN   + "  ℹ  " + m)
def success(m): print(Fore.GREEN  + "  ✓  " + m)
def warn(m):    print(Fore.YELLOW + "  ⚠  " + m)
def err(m):     print(Fore.RED    + "  ✗  " + m)


# ══════════════════════════════════════════════════════════════════════════
#  FETCH DATA — Multi-asset CCXT con paginación
# ══════════════════════════════════════════════════════════════════════════
def fetch_ohlcv(asset: str, timeframe: str, months: int) -> pd.DataFrame:
    """Descarga OHLCV desde Binance via CCXT con paginación."""
    os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
    cache_file = os.path.join(CONFIG["CACHE_DIR"], f"{asset}_{timeframe}_{months}m.json")
    
    if os.path.exists(cache_file):
        try:
            df = pd.DataFrame(json.load(open(cache_file)))
            df["date"] = pd.to_datetime(df["date"])
            age_h = (datetime.now() - df["date"].max()).total_seconds() / 3600
            max_age = 1 if timeframe in ["1h","4h"] else 24
            if age_h < max_age:
                info(f"Cache: {asset} {timeframe} ({len(df)} velas)")
                return df
        except Exception:
            pass
    
    info(f"Descargando {asset} {timeframe} {months}m...")
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        symbol = f"{asset}/USDT"
        since = int((datetime.now(timezone.utc) - timedelta(days=30*months)).timestamp() * 1000)
        
        all_ohlcv = []
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not batch: break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 3600*1000:
                break
            since = last_ts + 1
            if len(all_ohlcv) > 5000: break
        
        df = pd.DataFrame(all_ohlcv, columns=["ts","open","high","low","close","volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        df = df[["date","open","high","low","close","volume"]].dropna()
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        
        tmp = df.copy(); tmp["date"] = tmp["date"].astype(str)
        json.dump(tmp.to_dict("records"), open(cache_file, "w"))
        success(f"{asset} {timeframe}: {len(df)} velas")
        return df
    except Exception as e:
        err(f"Error {asset} {timeframe}: {str(e)}")
        return pd.DataFrame()


def fetch_realtime_metrics(asset: str) -> Dict:
    """Métricas en tiempo real: precio, funding, volumen, spread."""
    m = {"asset": asset, "price": None, "change_24h": None, "volume_24h": None,
         "bid": None, "ask": None, "spread_pct": None, "funding_rate": None, "source": "unavailable"}
    
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        tk = ex.fetch_ticker(f"{asset}/USDT")
        
        m["price"]      = round(tk["last"], 2)
        m["change_24h"] = round(tk.get("percentage", 0), 2)
        m["volume_24h"] = round(tk.get("quoteVolume", 0) / 1e6, 1)
        m["bid"]        = round(tk.get("bid", 0), 2)
        m["ask"]        = round(tk.get("ask", 0), 2)
        if m["bid"] and m["ask"] and m["bid"] > 0:
            m["spread_pct"] = round((m["ask"] - m["bid"]) / m["bid"] * 100, 4)
        
        try:
            ef = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
            fr = ef.fetch_funding_rate(f"{asset}/USDT")
            m["funding_rate"] = round(float(fr.get("fundingRate", 0)) * 100, 4)
        except Exception:
            pass
        
        m["source"] = "ccxt_binance"
        return m
    except Exception as e:
        warn(f"Métricas {asset}: {str(e)}")
        return m


# ══════════════════════════════════════════════════════════════════════════
#  GEOMETRIC LEVELS — PP/R1/S1/R2/S2 + Opens
# ══════════════════════════════════════════════════════════════════════════
def calculate_pivots(yesterday_ohlc: Dict) -> Dict:
    """Pivots tradicionales: PP, R1, S1, R2, S2."""
    h, l, c = yesterday_ohlc["high"], yesterday_ohlc["low"], yesterday_ohlc["close"]
    pp = (h + l + c) / 3
    r1 = 2*pp - l
    s1 = 2*pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    return {"PP": round(pp,2), "R1": round(r1,2), "S1": round(s1,2),
            "R2": round(r2,2), "S2": round(s2,2)}


def get_session_opens(df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> Dict:
    """Calcula weekly open (domingo 00:00 UTC) y monthly open (1er día mes)."""
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    days_since_sun = (today.weekday() + 1) % 7
    wk_start = (today - timedelta(days=days_since_sun)).replace(hour=0,minute=0,second=0,microsecond=0)
    
    weekly_open = None
    if not df_weekly.empty:
        wk_candle = df_weekly[df_weekly["date"] <= wk_start].iloc[-1] if len(df_weekly[df_weekly["date"] <= wk_start]) > 0 else df_weekly.iloc[-1]
        weekly_open = round(float(wk_candle["open"]), 2)
    
    monthly_open = None
    month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not df_daily.empty:
        mon_candle = df_daily[df_daily["date"] >= month_start].iloc[0] if len(df_daily[df_daily["date"] >= month_start]) > 0 else df_daily.iloc[-1]
        monthly_open = round(float(mon_candle["open"]), 2)
    
    return {"weekly_open": weekly_open, "monthly_open": monthly_open, "week_start": wk_start.date()}


def geometric_bias(price: float, weekly_open: float, pp: float) -> str:
    """Bias geométrico: ALCISTA si price > WO y price > PP."""
    if price > weekly_open and price > pp:
        return "ALCISTA"
    elif price < weekly_open and price < pp:
        return "BAJISTA"
    else:
        return "NEUTRAL"


# ══════════════════════════════════════════════════════════════════════════
#  TIME VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def time_validation(current_session: pd.DataFrame, historical_sessions: List[pd.DataFrame]) -> Dict:
    """
    Analiza si high/low formado a hora H típicamente HOLDS o TAKEN.
    Returns: {'high': {...}, 'low': {...}}
    """
    if current_session.empty:
        return {"high": None, "low": None}
    
    cur = current_session.sort_values("date").reset_index(drop=True)
    hi_idx = cur["high"].idxmax(); lo_idx = cur["low"].idxmin()
    hi_ts = cur.loc[hi_idx, "date"]; lo_ts = cur.loc[lo_idx, "date"]
    
    session_start = cur["date"].iloc[0]
    session_duration_h = (cur["date"].iloc[-1] - session_start).total_seconds() / 3600
    hi_hours_in = (hi_ts - session_start).total_seconds() / 3600
    lo_hours_in = (lo_ts - session_start).total_seconds() / 3600
    early_threshold = session_duration_h * 0.2
    
    hi_result = _analyze_extreme_time(historical_sessions, "high", hi_ts.hour, hi_hours_in < early_threshold)
    lo_result = _analyze_extreme_time(historical_sessions, "low",  lo_ts.hour, lo_hours_in < early_threshold)
    
    return {"high": hi_result, "low": lo_result}


def _analyze_extreme_time(historical: List[pd.DataFrame], extreme_type: str, target_hour: int, is_early: bool) -> Dict:
    """Helper: analiza sesiones históricas donde extremo formado a hora similar."""
    held_count = 0; taken_count = 0; hours_to_takeout = []
    
    for sess in historical:
        if sess.empty: continue
        sess = sess.sort_values("date").reset_index(drop=True)
        
        if extreme_type == "high":
            ext_idx = sess["high"].idxmax()
            ext_val = sess.loc[ext_idx, "high"]
            ext_ts  = sess.loc[ext_idx, "date"]
        else:
            ext_idx = sess["low"].idxmin()
            ext_val = sess.loc[ext_idx, "low"]
            ext_ts  = sess.loc[ext_idx, "date"]
        
        if abs(ext_ts.hour - target_hour) > 2: continue
        
        after = sess[sess["date"] > ext_ts]
        if after.empty:
            held_count += 1
            continue
        
        breached = (after["high"] > ext_val).any() if extreme_type == "high" else (after["low"] < ext_val).any()
        
        if breached:
            taken_count += 1
            first_breach_idx = (after[after["high"] > ext_val].index[0] if extreme_type == "high"
                                else after[after["low"] < ext_val].index[0])
            breach_ts = sess.loc[first_breach_idx, "date"]
            hours_to_takeout.append((breach_ts - ext_ts).total_seconds() / 3600)
        else:
            held_count += 1
    
    total = held_count + taken_count
    return {
        "hour": target_hour,
        "held_pct": round(held_count / total * 100, 1) if total > 0 else 0,
        "taken_pct": round(taken_count / total * 100, 1) if total > 0 else 0,
        "n_sessions": total,
        "avg_hours_to_takeout": round(np.mean(hours_to_takeout), 1) if hours_to_takeout else None,
        "early_risk": is_early
    }


# ══════════════════════════════════════════════════════════════════════════
#  DISTANCE VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def distance_validation(current_session: pd.DataFrame, historical_sessions: List[pd.DataFrame], session_open: float) -> Dict:
    """Analiza si displacement actual suficiente o likely to extend."""
    if current_session.empty or session_open is None:
        return {"disp_pct": 0, "reversed_pct": 0, "continued_pct": 0, "n_sessions": 0,
                "percentile": 0, "small_wick": False, "p25": 0, "p50": 0, "p75": 0}
    
    cur = current_session.sort_values("date").reset_index(drop=True)
    current_price = cur["close"].iloc[-1]
    current_disp = abs(current_price - session_open) / session_open * 100
    
    all_disps = []
    reversed_count = 0; continued_count = 0
    
    for sess in historical_sessions:
        if sess.empty: continue
        sess = sess.sort_values("date").reset_index(drop=True)
        sess_open = sess["open"].iloc[0]
        max_disp = max(abs(sess["high"].max() - sess_open) / sess_open * 100,
                      abs(sess["low"].min() - sess_open) / sess_open * 100)
        all_disps.append(max_disp)
        
        if abs(max_disp - current_disp) / current_disp > 0.10: continue
        
        mid_idx = len(sess) // 2
        if mid_idx >= len(sess): continue
        mid_price = sess.loc[mid_idx, "close"]
        mid_disp = abs(mid_price - sess_open) / sess_open * 100
        
        if max_disp <= mid_disp * 1.1:
            reversed_count += 1
        else:
            continued_count += 1
    
    total = reversed_count + continued_count
    
    if all_disps:
        p25 = round(np.percentile(all_disps, 25), 2)
        p50 = round(np.percentile(all_disps, 50), 2)
        p75 = round(np.percentile(all_disps, 75), 2)
        pct_rank = int(sum(d <= current_disp for d in all_disps) / len(all_disps) * 100)
    else:
        p25 = p50 = p75 = 0; pct_rank = 0
    
    return {
        "disp_pct": round(current_disp, 2),
        "reversed_pct": round(reversed_count / total * 100, 1) if total > 0 else 0,
        "continued_pct": round(continued_count / total * 100, 1) if total > 0 else 0,
        "n_sessions": total,
        "percentile": pct_rank,
        "small_wick": current_disp < p25 if p25 > 0 else False,
        "p25": p25, "p50": p50, "p75": p75
    }


# ══════════════════════════════════════════════════════════════════════════
#  P1/P2 DETECTION + FLIP RISK (from v3.0)
# ══════════════════════════════════════════════════════════════════════════
def analyze_p1p2(df4h: pd.DataFrame) -> Dict:
    """
    Detecta P1 (primer extremo) y P2 (segundo extremo) semanal.
    Calcula Flip Risk, timing P1→P2, distancias, matriz.
    """
    if df4h.empty:
        return {"available": False}
    
    df = df4h.copy().sort_values("date").reset_index(drop=True)
    df["wk"] = df["date"].dt.to_period("W-SAT")
    
    rows = []
    for wk, grp in df.groupby("wk", sort=True):
        grp = grp.sort_values("date")
        if len(grp) < 4: continue
        
        wo = grp["open"].iloc[0]; wc = grp["close"].iloc[-1]
        wh = grp["high"].max(); wl = grp["low"].min()
        
        hi_idx = grp["high"].idxmax(); lo_idx = grp["low"].idxmin()
        hi_ts = grp.loc[hi_idx, "date"]; lo_ts = grp.loc[lo_idx, "date"]
        hi_day = hi_ts.weekday(); lo_day = lo_ts.weekday()
        
        if hi_ts <= lo_ts:
            p1_type, p1_day, p1_ts, p1_val = "high", hi_day, hi_ts, wh
            p2_day = lo_day
        else:
            p1_type, p1_day, p1_ts, p1_val = "low", lo_day, lo_ts, wl
            p2_day = hi_day
        
        delta_h = (grp.loc[lo_idx if p1_type == "high" else hi_idx, "date"] - p1_ts).total_seconds() / 3600
        dist_pct = (wl - wh) / wh * 100 if p1_type == "high" else (wh - wl) / wl * 100
        bullish = wc > wo
        p1_flip = (p1_type == "high" and bullish) or (p1_type == "low" and not bullish)
        
        rows.append({
            "bullish": bullish, "p1_type": p1_type, "p1_day": int(p1_day),
            "p2_day": int(p2_day), "hours_p1p2": round(delta_h, 1),
            "dist_pct": round(dist_pct, 3), "p1_flipped": p1_flip
        })
    
    if not rows:
        return {"available": False}
    
    w4 = pd.DataFrame(rows)
    n = len(w4)
    
    flip_total = w4["p1_flipped"].mean() * 100
    flip_hi = w4[w4["p1_type"]=="high"]["p1_flipped"].mean() * 100
    flip_lo = w4[w4["p1_type"]=="low"]["p1_flipped"].mean() * 100
    
    h_mean = w4["hours_p1p2"].mean(); h_med = w4["hours_p1p2"].median()
    
    ph = w4[w4["p1_type"]=="high"]; pl = w4[w4["p1_type"]=="low"]
    prob_bear_p1h = (ph["bullish"]==False).mean()*100 if len(ph)>0 else 50.0
    prob_bull_p1l = (pl["bullish"]==True).mean()*100 if len(pl)>0 else 50.0
    
    def dstats(s):
        if len(s) == 0: return {"mean":0,"med":0,"p25":0,"p75":0}
        return {"mean": round(s.mean(),2), "med": round(s.median(),2),
                "p25": round(s.quantile(.25),2), "p75": round(s.quantile(.75),2)}
    
    mat = [[int(len(w4[(w4["p1_day"]==p1d)&(w4["p2_day"]==p2d)])) for p2d in range(7)] for p1d in range(7)]
    
    return {
        "available": True,
        "n_weeks": n,
        "flip_total": round(flip_total,1),
        "flip_hi": round(flip_hi,1),
        "flip_lo": round(flip_lo,1),
        "h_mean": round(h_mean,1),
        "h_med": round(h_med,1),
        "prob_bear_p1h": round(prob_bear_p1h,1),
        "prob_bull_p1l": round(prob_bull_p1l,1),
        "dist_bull": dstats(w4[w4["bullish"]]["dist_pct"]),
        "dist_bear": dstats(w4[~w4["bullish"]]["dist_pct"]),
        "p1p2_matrix": mat,
        "p1_by_day": {d: round((w4["p1_day"]==d).sum()/n*100,1) for d in range(7)}
    }


def current_week_p1(df4h: pd.DataFrame, p1p2_stats: Dict) -> Dict:
    """Detecta P1 en semana actual con validación de estructura."""
    if df4h.empty or not p1p2_stats.get("available"):
        return {"active": False}
    
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    days_since_sun = (today.weekday() + 1) % 7
    wk_start = (today - timedelta(days=days_since_sun)).replace(hour=0,minute=0,second=0,microsecond=0)
    
    cur = df4h[df4h["date"] >= wk_start].copy()
    if len(cur) == 0:
        return {"active": False}
    
    cur = cur.sort_values("date").reset_index(drop=True)
    wo = cur["open"].iloc[0]; wc = cur["close"].iloc[-1]
    wh = cur["high"].max(); wl = cur["low"].min()
    
    hi_idx = cur["high"].idxmax(); lo_idx = cur["low"].idxmin()
    hi_ts = cur.loc[hi_idx, "date"]; lo_ts = cur.loc[lo_idx, "date"]
    
    if hi_ts <= lo_ts:
        p1_type, p1_day, p1_ts, p1_val = "high", hi_ts.weekday(), hi_ts, wh
    else:
        p1_type, p1_day, p1_ts, p1_val = "low", lo_ts.weekday(), lo_ts, wl
    
    hrs_since = (today - p1_ts).total_seconds() / 3600
    
    # Validación estructura
    p1_candle_idx = cur[cur["date"] == p1_ts].index
    structure_ok = False; accept_count = 0
    if len(p1_candle_idx) > 0:
        p1_i = p1_candle_idx[0]
        post = cur[cur.index > p1_i].head(4)
        if p1_type == "high":
            accept_count = int((post["close"] < p1_val).sum())
        else:
            accept_count = int((post["close"] > p1_val).sum())
        structure_ok = accept_count >= 2
    
    # Sesgo + proyección
    if p1_type == "high":
        prob = p1p2_stats["prob_bear_p1h"]; flip = p1p2_stats["flip_hi"]
        dist = p1p2_stats["dist_bear"]; bias = "BAJISTA"
    else:
        prob = p1p2_stats["prob_bull_p1l"]; flip = p1p2_stats["flip_lo"]
        dist = p1p2_stats["dist_bull"]; bias = "ALCISTA"
    
    flip_adj = flip
    if structure_ok:
        flip_adj = max(0, flip - 8)
    elif accept_count == 0 and hrs_since > 8:
        flip_adj = min(100, flip + 10)
    
    proj_p2 = p1_val * (1 + dist["med"] / 100)
    proj_cons = p1_val * (1 + dist["p25"] / 100)
    proj_agr = p1_val * (1 + dist["p75"] / 100)
    h_remain = max(0, p1p2_stats["h_mean"] - hrs_since)
    
    return {
        "active": True,
        "p1_type": p1_type,
        "p1_day": p1_day,
        "p1_hour": p1_ts.hour,
        "p1_val": round(p1_val, 2),
        "structure_ok": structure_ok,
        "accept_count": accept_count,
        "bias": bias,
        "prob": round(prob, 1),
        "flip": round(flip, 1),
        "flip_adj": round(flip_adj, 1),
        "proj_p2": round(proj_p2, 2),
        "proj_cons": round(proj_cons, 2),
        "proj_agr": round(proj_agr, 2),
        "hrs_since": round(hrs_since, 1),
        "h_remain": round(h_remain, 1),
        "dist_p25": dist["p25"],
        "dist_med": dist["med"],
        "dist_p75": dist["p75"]
    }


# ══════════════════════════════════════════════════════════════════════════
#  PRICE PROJECTIONS (NEW)
# ══════════════════════════════════════════════════════════════════════════
def price_projections(price: float, time_val: Dict, dist_val: Dict, p1_current: Dict, targets: Dict) -> Dict:
    """
    Calcula escenarios ALCISTA/BAJISTA con targets + probabilidades.
    CLARIFICA dirección actual y qué significa reversión/continuación.
    """
    # Escenario ALCISTA
    bull_targets = []
    for label in ["+2%", "+5%", "+10%"]:
        target_price = price * (1 + float(label.replace("%","").replace("+",""))/100)
        confidence = targets.get(label, 0)
        bull_targets.append({"label": label, "price": round(target_price, 2), "confidence": confidence})
    
    # Escenario BAJISTA
    bear_targets = []
    for label in ["-2%", "-5%", "-10%"]:
        target_price = price * (1 - float(label.replace("%","").replace("-",""))/100)
        confidence = targets.get(label, 0)
        bear_targets.append({"label": label, "price": round(target_price, 2), "confidence": confidence})
    
    # Detectar dirección ACTUAL basado en confidence de targets
    avg_bull_conf = sum(t["confidence"] for t in bull_targets) / len(bull_targets)
    avg_bear_conf = sum(t["confidence"] for t in bear_targets) / len(bear_targets)
    
    if avg_bull_conf > avg_bear_conf + 20:
        current_direction = "ALCISTA"
        direction_emoji = "📈"
    elif avg_bear_conf > avg_bull_conf + 20:
        current_direction = "BAJISTA"
        direction_emoji = "📉"
    else:
        current_direction = "NEUTRAL"
        direction_emoji = "➡"
    
    # Probabilidad de reversión vs continuación
    prob_reversal = dist_val.get("reversed_pct", 50)
    prob_continuation = dist_val.get("continued_pct", 50)
    
    # Ajustar por percentil
    pct = dist_val.get("percentile", 50)
    if pct > 70:
        prob_reversal = min(100, prob_reversal + 15)
        prob_continuation = max(0, prob_continuation - 15)
    elif pct < 30:
        prob_continuation = min(100, prob_continuation + 15)
        prob_reversal = max(0, prob_reversal - 15)
    
    # Generar lectura clara
    if prob_continuation > 60:
        if current_direction == "ALCISTA":
            reading = f"El precio está subiendo y tiene {prob_continuation}% de probabilidad de SEGUIR SUBIENDO. Los targets alcistas (+2%, +5%, +10%) son los relevantes."
        elif current_direction == "BAJISTA":
            reading = f"El precio está bajando y tiene {prob_continuation}% de probabilidad de SEGUIR BAJANDO. Los targets bajistas (-2%, -5%, -10%) son los relevantes."
        else:
            reading = f"El precio está en rango neutral con {prob_continuation}% de probabilidad de continuar sin dirección clara."
    else:
        if current_direction == "ALCISTA":
            reading = f"El precio está subiendo PERO tiene {prob_reversal}% de probabilidad de REVERSIÓN (giro bajista). Considerar tomar profit en targets alcistas."
        elif current_direction == "BAJISTA":
            reading = f"El precio está bajando PERO tiene {prob_reversal}% de probabilidad de REVERSIÓN (giro alcista). El rebote es probable."
        else:
            reading = f"El precio está neutral con {prob_reversal}% de probabilidad de reversión desde cualquier dirección."
    
    return {
        "bull_targets": bull_targets,
        "bear_targets": bear_targets,
        "current_direction": current_direction,
        "direction_emoji": direction_emoji,
        "prob_reversal": round(prob_reversal, 1),
        "prob_continuation": round(prob_continuation, 1),
        "reading": reading
    }


# ══════════════════════════════════════════════════════════════════════════
#  DECISION SYNTHESIS (NEW)
# ══════════════════════════════════════════════════════════════════════════
def decision_synthesis(time_val: Dict, dist_val: Dict, p1_current: Dict, geo_bias: str) -> Dict:
    """
    Semáforo multi-señal + score final + lectura narrativa.
    ✅ (green) = señal válida
    ⚠ (yellow) = señal mixta
    ❌ (red) = señal inválida
    """
    # TIME score
    tvh = time_val.get("high")
    tvl = time_val.get("low")
    time_held = max(tvh["held_pct"] if tvh else 0, tvl["held_pct"] if tvl else 0)
    if time_held >= 60:
        time_status = "✅"; time_color = "green"
    elif time_held >= 40:
        time_status = "⚠"; time_color = "yellow"
    else:
        time_status = "❌"; time_color = "red"
    
    # DISTANCE score
    pct = dist_val.get("percentile", 50)
    if 30 <= pct <= 70:
        dist_status = "✅"; dist_color = "green"
    elif pct < 30 or pct > 70:
        dist_status = "⚠"; dist_color = "yellow"
    else:
        dist_status = "✅"; dist_color = "green"
    
    # P1 STRUCTURE score
    if not p1_current.get("active"):
        p1_status = "⚠"; p1_color = "yellow"
    elif p1_current.get("structure_ok"):
        p1_status = "✅"; p1_color = "green"
    elif p1_current.get("accept_count") >= 1:
        p1_status = "⚠"; p1_color = "yellow"
    else:
        p1_status = "❌"; p1_color = "red"
    
    # GEOMETRIC BIAS score
    if geo_bias == "ALCISTA" or geo_bias == "BAJISTA":
        bias_status = "✅"; bias_color = "green"
    else:
        bias_status = "⚠"; bias_color = "yellow"
    
    # Combined score (0-4 stars)
    score_map = {"✅": 1, "⚠": 0.5, "❌": 0}
    total_score = (score_map[time_status] + score_map[dist_status] + 
                   score_map[p1_status] + score_map[bias_status])
    stars = int(total_score)
    
    # Lectura narrativa
    if stars >= 3:
        reading = f"Señal de alta confianza ({stars}/4 estrellas). TIME valida, DISTANCE adecuada, estructura P1 confirmada. Sesgo {geo_bias}. Escenario: operación con alta probabilidad de éxito."
    elif stars >= 2:
        reading = f"Señal moderada ({stars}/4 estrellas). Al menos 2 indicadores validan. Operar con cautela, esperar confirmación adicional en precio."
    else:
        reading = f"Señal débil ({stars}/4 estrellas). Múltiples indicadores en conflicto. No operar hasta tener mayor claridad."
    
    return {
        "time_status": time_status, "time_color": time_color,
        "dist_status": dist_status, "dist_color": dist_color,
        "p1_status": p1_status, "p1_color": p1_color,
        "bias_status": bias_status, "bias_color": bias_color,
        "score": stars,
        "reading": reading
    }


# ══════════════════════════════════════════════════════════════════════════
#  EDUCATIONAL CONTENT (NEW)
# ══════════════════════════════════════════════════════════════════════════
def generate_context_alert(proj: Dict, p1_current: Dict, geo_bias: str, dv: Dict) -> Dict:
    """
    Genera alerta de contexto automática detectando conflictos entre timeframes.
    
    Detecta:
    - Rebote dentro de estructura bajista
    - Pullback dentro de estructura alcista
    - Trampa alcista/bajista
    - Alineación completa
    """
    if not p1_current.get("active"):
        return {
            "has_alert": False,
            "alert_type": "none",
            "alert_title": "",
            "alert_message": "",
            "recommendation": ""
        }
    
    # Extraer datos
    mov_actual = proj.get("current_direction", "NEUTRAL")
    p1_bias = p1_current.get("bias", "NEUTRAL")
    p1_type = p1_current.get("p1_type", "")
    struct_ok = p1_current.get("structure_ok", False)
    flip_risk = p1_current.get("flip_adj", 50)
    prob_p1 = p1_current.get("prob", 50)
    hrs_since = p1_current.get("hrs_since", 0)
    h_remain = p1_current.get("h_remain", 0)
    small_wick = dv.get("small_wick", False)
    percentile = dv.get("percentile", 50)
    
    # CASO 1: Rebote dentro de estructura bajista
    if p1_bias == "BAJISTA" and mov_actual == "ALCISTA" and struct_ok and flip_risk < 30:
        return {
            "has_alert": True,
            "alert_type": "rebote_en_bajista",
            "alert_title": "⚠ REBOTE TÉCNICO DENTRO DE ESTRUCTURA BAJISTA",
            "alert_message": (
                f"<strong>ESTRUCTURA MACRO (semanal):</strong> BAJISTA<br>"
                f"• P1={p1_type.upper()} confirmado con estructura válida (Flip Risk: {flip_risk}%)<br>"
                f"• Probabilidad de cerrar bajista: {prob_p1}%<br>"
                f"• Horas desde P1: {hrs_since:.0f}h | Restantes estimadas: {h_remain:.0f}h<br>"
                f"• Proyección P2 (mínimo): ${p1_current['proj_p2']:,.0f}<br><br>"
                f"<strong>MOVIMIENTO ACTUAL (últimas horas):</strong> ALCISTA<br>"
                f"• Rebote técnico en progreso desde el LOW<br>"
                f"• Continuación del rebote corto plazo: {proj['prob_continuation']}%<br><br>"
                f"<strong>⚠ INTERPRETACIÓN:</strong><br>"
                f"Estás viendo un <strong>REBOTE DENTRO DE ESTRUCTURA BAJISTA</strong>. "
                f"El rebote puede continuar unas horas más, pero la probabilidad de que "
                f"la semana cierre bajista sigue siendo <strong>{prob_p1}%</strong>. "
                f"{'El movimiento bajista es small wick (P<30) — puede extenderse más abajo.' if small_wick else ''}"
            ),
            "recommendation": (
                f"<strong>OPCIONES:</strong><br>"
                f"1. <strong>Trading del rebote</strong> (corto plazo, riesgoso): "
                f"Comprar con targets en zona ${int(p1_current['p1_val']*0.97):,.0f}-${int(p1_current['p1_val']*0.96):,.0f}. "
                f"Stop bajo ${p1_current['proj_p2']:,.0f}. Salir rápido.<br>"
                f"2. <strong>Esperar fin del rebote y SHORT</strong> (recomendado): "
                f"Esperar rechazo en resistencia y entrar bajista. Target: P2 ${p1_current['proj_p2']:,.0f} o inferior.<br>"
                f"3. <strong>NO OPERAR</strong>: Si hay dudas, esperar claridad — el rebote puede ser trampa."
            )
        }
    
    # CASO 2: Pullback dentro de estructura alcista
    elif p1_bias == "ALCISTA" and mov_actual == "BAJISTA" and struct_ok and flip_risk < 30:
        return {
            "has_alert": True,
            "alert_type": "pullback_en_alcista",
            "alert_title": "⚠ PULLBACK TÉCNICO DENTRO DE ESTRUCTURA ALCISTA",
            "alert_message": (
                f"<strong>ESTRUCTURA MACRO (semanal):</strong> ALCISTA<br>"
                f"• P1={p1_type.upper()} confirmado con estructura válida (Flip Risk: {flip_risk}%)<br>"
                f"• Probabilidad de cerrar alcista: {prob_p1}%<br>"
                f"• Horas desde P1: {hrs_since:.0f}h | Restantes estimadas: {h_remain:.0f}h<br>"
                f"• Proyección P2 (máximo): ${p1_current['proj_p2']:,.0f}<br><br>"
                f"<strong>MOVIMIENTO ACTUAL (últimas horas):</strong> BAJISTA<br>"
                f"• Pullback (retroceso) técnico en progreso<br>"
                f"• Continuación del pullback corto plazo: {proj['prob_continuation']}%<br><br>"
                f"<strong>⚠ INTERPRETACIÓN:</strong><br>"
                f"Estás viendo un <strong>PULLBACK DENTRO DE ESTRUCTURA ALCISTA</strong>. "
                f"El retroceso puede continuar unas horas más (saludable en tendencias alcistas), "
                f"pero la probabilidad de que la semana cierre alcista sigue siendo <strong>{prob_p1}%</strong>. "
                f"{'El movimiento alcista es small wick (P<30) — puede extenderse más arriba después del pullback.' if small_wick else ''}"
            ),
            "recommendation": (
                f"<strong>OPCIONES:</strong><br>"
                f"1. <strong>Esperar fin del pullback y LONG</strong> (recomendado): "
                f"Esperar soporte en zona ${int(p1_current['p1_val']*1.03):,.0f}-${int(p1_current['p1_val']*1.05):,.0f} "
                f"y rebote para entrar alcista. Target: P2 ${p1_current['proj_p2']:,.0f}.<br>"
                f"2. <strong>Trading del pullback</strong> (corto plazo, riesgoso): "
                f"Vender con targets cortos en soportes clave. Stop ajustado.<br>"
                f"3. <strong>NO OPERAR</strong>: Si el pullback es muy fuerte, puede invalidar la estructura alcista."
            )
        }
    
    # CASO 3: Trampa alcista (P1=HIGH temprano, precio aún alto)
    elif p1_type == "high" and mov_actual == "ALCISTA" and hrs_since < 48 and not struct_ok:
        return {
            "has_alert": True,
            "alert_type": "trampa_alcista",
            "alert_title": "🚨 POSIBLE TRAMPA ALCISTA — P1=HIGH SIN CONFIRMAR",
            "alert_message": (
                f"<strong>SITUACIÓN:</strong><br>"
                f"• P1=HIGH detectado hace solo {hrs_since:.0f}h<br>"
                f"• Estructura NO confirmada aún ({p1_current['accept_count']} velas)<br>"
                f"• Precio sigue subiendo (movimiento alcista actual)<br>"
                f"• Flip Risk: {flip_risk}% — riesgo de señal falsa<br><br>"
                f"<strong>⚠ INTERPRETACIÓN:</strong><br>"
                f"El precio formó un máximo temprano pero <strong>NO tiene aceptación</strong>. "
                f"Esto puede ser una <strong>TRAMPA ALCISTA</strong> donde el precio sube primero "
                f"para barrer stops de shorts, y luego gira bajista. "
                f"Estadísticamente, P1=HIGH lleva a cierre bajista el {prob_p1}% de las veces."
            ),
            "recommendation": (
                f"<strong>RECOMENDACIÓN:</strong><br>"
                f"<strong>NO ENTRAR LARGO TODAVÍA.</strong> Esperar:<br>"
                f"1. Confirmación de estructura (2+ velas 4H cierran bajo ${p1_current['p1_val']:,.0f})<br>"
                f"2. Si el precio sigue subiendo y rompe el P1, la estructura cambió<br>"
                f"3. Si el precio rechaza y baja con volumen, considerar SHORT"
            )
        }
    
    # CASO 4: Trampa bajista (P1=LOW temprano, precio aún bajo)
    elif p1_type == "low" and mov_actual == "BAJISTA" and hrs_since < 48 and not struct_ok:
        return {
            "has_alert": True,
            "alert_type": "trampa_bajista",
            "alert_title": "🚨 POSIBLE TRAMPA BAJISTA — P1=LOW SIN CONFIRMAR",
            "alert_message": (
                f"<strong>SITUACIÓN:</strong><br>"
                f"• P1=LOW detectado hace solo {hrs_since:.0f}h<br>"
                f"• Estructura NO confirmada aún ({p1_current['accept_count']} velas)<br>"
                f"• Precio sigue bajando (movimiento bajista actual)<br>"
                f"• Flip Risk: {flip_risk}% — riesgo de señal falsa<br><br>"
                f"<strong>⚠ INTERPRETACIÓN:</strong><br>"
                f"El precio formó un mínimo temprano pero <strong>NO tiene aceptación</strong>. "
                f"Esto puede ser una <strong>TRAMPA BAJISTA (sweep de liquidez)</strong> donde el precio "
                f"baja primero para barrer stops de longs, y luego rebota alcista. "
                f"Estadísticamente, P1=LOW lleva a cierre alcista el {prob_p1}% de las veces."
            ),
            "recommendation": (
                f"<strong>RECOMENDACIÓN:</strong><br>"
                f"<strong>NO ENTRAR SHORT TODAVÍA.</strong> Esperar:<br>"
                f"1. Confirmación de rebote (2+ velas 4H cierran sobre ${p1_current['p1_val']:,.0f})<br>"
                f"2. Si el precio sigue bajando y rompe el P1, la estructura cambió<br>"
                f"3. Si el precio rechaza abajo y sube con volumen, considerar LONG"
            )
        }
    
    # CASO 5: Alineación completa (todo en sincronía)
    elif p1_bias == mov_actual and struct_ok and flip_risk < 25:
        return {
            "has_alert": True,
            "alert_type": "alineacion",
            "alert_title": "✅ ALINEACIÓN COMPLETA — SEÑAL CLARA",
            "alert_message": (
                f"<strong>ESTRUCTURA MACRO:</strong> {p1_bias}<br>"
                f"<strong>MOVIMIENTO ACTUAL:</strong> {mov_actual}<br>"
                f"<strong>GEOMETRIC BIAS:</strong> {geo_bias}<br><br>"
                f"• P1={p1_type.upper()} confirmado con estructura sólida<br>"
                f"• Flip Risk bajo: {flip_risk}% (fiabilidad {100-flip_risk}%)<br>"
                f"• Probabilidad direccional: {prob_p1}%<br>"
                f"• Todas las señales apuntan en la MISMA dirección<br><br>"
                f"<strong>✅ INTERPRETACIÓN:</strong><br>"
                f"<strong>SEÑAL DE ALTA CLARIDAD.</strong> Todos los timeframes e indicadores "
                f"están alineados en dirección {p1_bias}. Esta es una configuración operacional óptima."
            ),
            "recommendation": (
                f"<strong>RECOMENDACIÓN:</strong><br>"
                f"<strong>OPERAR CON CONFIANZA</strong> en dirección {p1_bias}:<br>"
                f"• Entry: Precio actual o en pullback a niveles clave<br>"
                f"• Target conservador: ${p1_current['proj_cons']:,.0f}<br>"
                f"• Target normal: ${p1_current['proj_p2']:,.0f}<br>"
                f"• Target agresivo: ${p1_current['proj_agr']:,.0f}<br>"
                f"• Stop: {'Sobre' if p1_bias=='BAJISTA' else 'Bajo'} P1 en ${p1_current['p1_val']:,.0f}"
            )
        }
    
    # CASO 6: Sin alerta (situación normal/neutral)
    else:
        return {
            "has_alert": False,
            "alert_type": "none",
            "alert_title": "",
            "alert_message": "",
            "recommendation": ""
        }


# ══════════════════════════════════════════════════════════════════════════
#  EDUCATIONAL CONTENT
# ══════════════════════════════════════════════════════════════════════════
def educational_content() -> Dict:
    """Explicaciones para usuario nuevo en lenguaje simple."""
    return {
        "pivots": """
<strong>¿Qué son los pivots tradicionales?</strong><br>
Los pivots (PP, R1, S1, R2, S2) son niveles de precio calculados matemáticamente desde el OHLC de ayer. 
Traders los usan como zonas de soporte/resistencia donde el precio tiende a reaccionar.<br><br>
<strong>Cómo usarlos:</strong><br>
• Si precio está sobre PP → sesgo alcista<br>
• Si precio está bajo PP → sesgo bajista<br>
• R1, R2 = resistencias (precio puede frenar ahí subiendo)<br>
• S1, S2 = soportes (precio puede rebotar ahí bajando)
""",
        "projections": """
<strong>¿Cómo leer las proyecciones de precio?</strong><br><br>

<strong>1. MOVIMIENTO ACTUAL:</strong> Lo primero que verás es si el precio está subiendo (📈 ALCISTA), bajando (📉 BAJISTA) o sin dirección clara (➡ NEUTRAL).<br><br>

<strong>2. ESCENARIOS:</strong><br>
• <strong>ESCENARIO ALCISTA</strong> = Qué pasaría si el precio sube (+2%, +5%, +10%)<br>
• <strong>ESCENARIO BAJISTA</strong> = Qué pasaría si el precio baja (-2%, -5%, -10%)<br><br>

<strong>3. Confidence %:</strong> Es la probabilidad histórica de alcanzar ese nivel.<br>
Ejemplo: "+5% con 52% confidence" = En el 52% de sesiones similares el precio subió al menos 5%.<br><br>

<strong>4. PROBABILIDAD REVERSIÓN vs CONTINUACIÓN (lo más importante):</strong><br>
• <strong style='color:#00ff9f'>CONTINUACIÓN alta (>60%)</strong> = El precio probablemente <strong>SIGUE en la MISMA dirección actual</strong>.<br>
&nbsp;&nbsp;→ Si está subiendo ahora, seguirá subiendo. Los targets ALCISTAS son relevantes.<br>
&nbsp;&nbsp;→ Si está bajando ahora, seguirá bajando. Los targets BAJISTAS son relevantes.<br><br>

• <strong style='color:#ff006e'>REVERSIÓN alta (>60%)</strong> = El precio probablemente <strong>GIRA en dirección OPUESTA</strong>.<br>
&nbsp;&nbsp;→ Si está subiendo ahora, probablemente baja. Tomar profit.<br>
&nbsp;&nbsp;→ Si está bajando ahora, probablemente rebota. Esperar compra.<br><br>

<strong>REGLA SIMPLE:</strong><br>
1. Mira "MOVIMIENTO ACTUAL" primero<br>
2. Si CONTINUACIÓN es alta → el precio sigue en esa dirección<br>
3. Si REVERSIÓN es alta → el precio gira en dirección opuesta<br>
4. Los targets con mayor confidence % son los más probables de alcanzarse
""",
        "time_validation": """
<strong>¿Por qué importa la HORA en que se forma un high/low?</strong><br>
No todos los extremos son iguales. Si un máximo (high) se forma muy temprano en la sesión (ejemplo: primera hora), 
históricamente tiene <strong>más probabilidad de ser liquidado</strong> (taken out) antes de que termine la sesión.<br><br>
<strong>Cómo leerlo:</strong><br>
• <span style='color:#00ff9f'>HELD >60%</span> = El extremo típicamente sostiene, es una zona válida de soporte/resistencia.<br>
• <span style='color:#ff006e'>TAKEN <50%</span> = El extremo típicamente es breached, probablemente es una trampa.<br>
• <span style='color:#ffd60a'>⚠ EARLY FORMATION</span> = Formado en primer 20% de la sesión, mayor riesgo de takeout.
""",
        "distance_validation": """
<strong>¿Qué significa "small wick" y por qué importa?</strong><br>
DISTANCE validation mide cuánto se movió el precio desde el open de la sesión (displacement). 
Lo comparamos con el historial para saber si es un movimiento "pequeño", "normal" o "grande".<br><br>
<strong>Percentiles:</strong><br>
• P25 = movimiento pequeño (solo 25% de sesiones movieron menos)<br>
• P50 = movimiento mediano (mitad de sesiones)<br>
• P75 = movimiento grande (solo 25% de sesiones movieron más)<br><br>
<strong>⚠ SMALL WICK (P<30):</strong> El precio se movió poco vs histórico. Estadísticamente, es probable que <strong>continúe moviéndose más</strong>.<br>
<strong>Exhaustion (P>70):</strong> El precio se movió mucho. Estadísticamente, es probable que <strong>revierta o frene</strong>.
""",
        "p1p2": """
<strong>¿Qué es P1/P2 y en qué se diferencia de TIME validation?</strong><br>
<strong>P1 (Pivot 1):</strong> El <strong>primer extremo</strong> que se forma en la semana (high o low). 
Si el high llega primero, P1=HIGH. Si el low llega primero, P1=LOW.<br>
<strong>P2 (Pivot 2):</strong> El <strong>segundo extremo</strong> (el opuesto a P1).<br><br>
<strong>Diferencia con TIME validation:</strong><br>
• TIME validation = "¿El high formado a las 14:00 UTC típicamente holds?"<br>
• P1/P2 = "¿Cuál extremo llegó PRIMERO (P1) y cuál llegó SEGUNDO (P2)?"<br><br>
<strong>P1 Flip Risk:</strong> Frecuencia con que el P1 resulta señal <strong>falsa</strong>. 
Ejemplo: P1=HIGH indica sesión bajista, pero si Flip Risk es alto (>30%), muchas veces la semana cierra alcista (flip).<br><br>
<strong>Cómo usarlo:</strong> Si P1=LOW con estructura confirmada y Flip Risk <20%, es señal alcista de alta fiabilidad.
""",
        "synthesis": """
<strong>¿Cómo integrar todas las señales?</strong><br>
Usamos un sistema de semáforo multi-señal:<br><br>
<strong>4 indicadores clave:</strong><br>
1. <strong>TIME:</strong> ¿El extremo formado a esta hora típicamente holds?<br>
2. <strong>DISTANCE:</strong> ¿El movimiento actual es suficiente o likely to extend?<br>
3. <strong>P1 STRUCTURE:</strong> ¿El P1 tiene aceptación (2+ velas confirman)?<br>
4. <strong>GEOMETRIC BIAS:</strong> ¿Precio está sobre/bajo WO y PP?<br><br>
<strong>Score final:</strong><br>
• <span style='color:#00ff9f'>4/4 estrellas</span> = Todos los indicadores alineados, señal de alta confianza.<br>
• <span style='color:#ffd60a'>2-3/4 estrellas</span> = Señal moderada, operar con cautela.<br>
• <span style='color:#ff006e'>0-1/4 estrellas</span> = Señal débil o conflictiva, no operar.<br><br>
<strong>Regla de oro:</strong> No operar si el score es <2 estrellas.
"""
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════
def analyze_asset_full(asset: str) -> Dict:
    """Análisis completo de un asset: fetch data + todos los módulos."""
    info(f"Analizando {asset}...")
    
    # Fetch data
    df_weekly = fetch_ohlcv(asset, "1w", 24)
    df_daily  = fetch_ohlcv(asset, "1d", 12)
    df_4h     = fetch_ohlcv(asset, "4h", 12)
    df_1h     = fetch_ohlcv(asset, "1h", 3)
    metrics   = fetch_realtime_metrics(asset)
    
    # Current sessions
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    days_since_sun = (today.weekday() + 1) % 7
    wk_start = (today - timedelta(days=days_since_sun)).replace(hour=0,minute=0,second=0,microsecond=0)
    day_start = today.replace(hour=0,minute=0,second=0,microsecond=0)
    
    df_cur_week = df_4h[df_4h["date"] >= wk_start].copy() if not df_4h.empty else pd.DataFrame()
    df_cur_day  = df_1h[df_1h["date"] >= day_start].copy() if not df_1h.empty else pd.DataFrame()
    
    # Pivots & Opens
    if not df_daily.empty and len(df_daily) > 1:
        yesterday = df_daily.iloc[-2]
        pivots = calculate_pivots({"high": yesterday["high"], "low": yesterday["low"], "close": yesterday["close"]})
    else:
        pivots = {"PP": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
    
    opens = get_session_opens(df_daily, df_weekly)
    
    # Current price & bias
    current_price = metrics["price"] if metrics["price"] else (float(df_cur_week["close"].iloc[-1]) if not df_cur_week.empty else 0)
    wo = opens["weekly_open"] if opens["weekly_open"] else current_price
    pp = pivots["PP"] if pivots["PP"] else current_price
    geo_bias = geometric_bias(current_price, wo, pp)
    
    # Split historical into sessions
    df_hist_week = df_4h.copy()
    df_hist_week["week"] = df_hist_week["date"].dt.to_period("W-SAT")
    sessions_week = [grp.sort_values("date") for _, grp in df_hist_week.groupby("week")]
    
    df_hist_day = df_1h.copy()
    df_hist_day["day"] = df_hist_day["date"].dt.date
    sessions_day = [grp.sort_values("date") for _, grp in df_hist_day.groupby("day")]
    
    # TIME validation (weekly)
    time_val_week = time_validation(df_cur_week, sessions_week)
    
    # DISTANCE validation (weekly)
    session_open_week = float(df_cur_week["open"].iloc[0]) if not df_cur_week.empty else wo
    dist_val_week = distance_validation(df_cur_week, sessions_week, session_open_week)
    
    # Confidence targets (weekly)
    targets_week = {}
    for label in ["+2%", "+5%", "+10%", "-2%", "-5%", "-10%"]:
        target_price = current_price * (1 + float(label.replace("%","").replace("+","").replace("-",""))/100 * (1 if "+" in label else -1))
        count_reached = 0; total = 0
        for sess in sessions_week:
            if sess.empty: continue
            sess = sess.sort_values("date").reset_index(drop=True)
            sess_open = sess["open"].iloc[0]
            current_ratio = current_price / session_open_week if session_open_week else 1
            sess_ratio = sess["close"].iloc[-1] / sess_open
            if abs(sess_ratio - current_ratio) / current_ratio > 0.05: continue
            total += 1
            if "+" in label:
                if sess["high"].max() >= target_price: count_reached += 1
            else:
                if sess["low"].min() <= target_price: count_reached += 1
        targets_week[label] = round(count_reached / total * 100, 1) if total > 0 else 0
    
    # P1/P2 analysis
    p1p2_stats = analyze_p1p2(df_4h)
    p1_current = current_week_p1(df_4h, p1p2_stats)
    
    # Price projections
    projections = price_projections(current_price, time_val_week, dist_val_week, p1_current, targets_week)
    
    # Decision synthesis
    synthesis = decision_synthesis(time_val_week, dist_val_week, p1_current, geo_bias)
    
    # Context alert (NEW)
    context_alert = generate_context_alert(projections, p1_current, geo_bias, dist_val_week)
    
    return {
        "asset": asset,
        "price": current_price,
        "ohlc_week": {
            "open": float(df_cur_week["open"].iloc[0]) if not df_cur_week.empty else 0,
            "high": float(df_cur_week["high"].max()) if not df_cur_week.empty else 0,
            "low": float(df_cur_week["low"].min()) if not df_cur_week.empty else 0,
            "close": float(df_cur_week["close"].iloc[-1]) if not df_cur_week.empty else 0
        },
        "pivots": pivots,
        "weekly_open": wo,
        "monthly_open": opens["monthly_open"],
        "geometric_bias": geo_bias,
        "time_validation": time_val_week,
        "distance_validation": dist_val_week,
        "targets": targets_week,
        "p1p2_stats": p1p2_stats,
        "p1_current": p1_current,
        "projections": projections,
        "synthesis": synthesis,
        "context_alert": context_alert,
        "metrics": metrics
    }


# ══════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML v4.5 HYBRID
# ══════════════════════════════════════════════════════════════════════════
def build_dashboard_hybrid(btc_data: Dict, eth_data: Dict, edu_content: Dict, out_dir: str) -> str:
    """Dashboard híbrido completo: TIME/DISTANCE + P1/P2 + Educational + Projections + Synthesis."""
    
    info("Construyendo dashboard v4.5 híbrido...")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    def asset_section_hybrid(data: Dict, asset_name: str, edu: Dict) -> str:
        if not data:
            return f"<div class='card'><p>{asset_name} no disponible</p></div>"
        
        price = data["price"]
        ohlc = data["ohlc_week"]
        pivots = data["pivots"]
        wo = data["weekly_open"]
        mo = data["monthly_open"]
        geo_bias = data["geometric_bias"]
        tv = data["time_validation"]
        dv = data["distance_validation"]
        targets = data["targets"]
        p1p2 = data["p1p2_stats"]
        p1_cur = data["p1_current"]
        proj = data["projections"]
        synth = data["synthesis"]
        alert = data["context_alert"]
        
        # Colors
        bias_col = "#00ff9f" if geo_bias == "ALCISTA" else "#ff006e" if geo_bias == "BAJISTA" else "#ffd60a"
        
        # Section 1: LIVE PRICE & LEVELS
        sec1 = (
            "<div class='section'>"
            "<div class='sec-header'><h2>📊 PRECIO EN VIVO & NIVELES</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-pivots\")'>📖 Explicar</button></div>"
            f"<div id='edu-pivots' class='edu-box' style='display:none'>{edu['pivots']}</div>"
            "<div class='price-card'>"
            f"<div class='price-main'>${price:,.2f}</div>"
            "<div class='ohlc-mini'>"
            f"<span>O: ${ohlc['open']:,.0f}</span>"
            f"<span style='color:#00ff9f'>H: ${ohlc['high']:,.0f}</span>"
            f"<span style='color:#ff006e'>L: ${ohlc['low']:,.0f}</span>"
            f"<span>C: ${ohlc['close']:,.0f}</span>"
            "</div></div>"
            "<div class='levels-grid'>"
            f"<div class='level r2'><span>R2</span><span>${pivots['R2']:,.0f}</span></div>"
            f"<div class='level r1'><span>R1</span><span>${pivots['R1']:,.0f}</span></div>"
            f"<div class='level pp'><span>PP</span><span>${pivots['PP']:,.0f}</span></div>"
            f"<div class='level s1'><span>S1</span><span>${pivots['S1']:,.0f}</span></div>"
            f"<div class='level s2'><span>S2</span><span>${pivots['S2']:,.0f}</span></div>"
            "</div>"
            "<div class='opens-row'>"
            f"<div class='open-item'><span>WEEKLY OPEN</span><span style='color:#00d4ff'>${wo:,.0f}</span></div>"
            f"<div class='open-item'><span>MONTHLY OPEN</span><span style='color:#b388ff'>${mo:,.0f}</span></div>"
            "</div>"
            f"<div class='bias-badge' style='border-left:4px solid {bias_col}'>"
            f"<span class='bias-label'>GEOMETRIC BIAS</span>"
            f"<span class='bias-value' style='color:{bias_col}'>{geo_bias}</span>"
            "</div></div>"
        )
        
        # Section 2: PRICE PROJECTIONS
        bull_rows = "".join([
            f"<div class='proj-row'><span>{t['label']}</span>"
            f"<div class='proj-bar'><div class='proj-fill bull' style='width:{t['confidence']}%'></div></div>"
            f"<span class='proj-target'>${t['price']:,.0f}</span>"
            f"<span class='proj-conf'>{t['confidence']}%</span></div>"
            for t in proj["bull_targets"]
        ])
        bear_rows = "".join([
            f"<div class='proj-row'><span>{t['label']}</span>"
            f"<div class='proj-bar'><div class='proj-fill bear' style='width:{t['confidence']}%'></div></div>"
            f"<span class='proj-target'>${t['price']:,.0f}</span>"
            f"<span class='proj-conf'>{t['confidence']}%</span></div>"
            for t in proj["bear_targets"]
        ])
        
        # Direction badge color
        dir_col = "#00ff9f" if proj["current_direction"] == "ALCISTA" else "#ff006e" if proj["current_direction"] == "BAJISTA" else "#ffd60a"
        
        sec2 = (
            "<div class='section'>"
            "<div class='sec-header'><h2>🎯 PROYECCIONES DE PRECIO</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-proj\")'>📖 Explicar</button></div>"
            f"<div id='edu-proj' class='edu-box' style='display:none'>{edu['projections']}</div>"
            
            # DIRECCIÓN ACTUAL (nuevo)
            f"<div class='direction-badge' style='background:linear-gradient(135deg,{dir_col}22,{dir_col}05);border-left:4px solid {dir_col}'>"
            f"<div class='dir-label'>MOVIMIENTO ACTUAL</div>"
            f"<div class='dir-value' style='color:{dir_col}'>{proj['direction_emoji']} {proj['current_direction']}</div>"
            f"<div class='dir-explain'>{proj['reading']}</div>"
            "</div>"
            
            "<div class='proj-grid'>"
            "<div class='proj-scenario bull'>"
            "<h3>📈 ESCENARIO ALCISTA</h3>"
            + bull_rows +
            "</div>"
            "<div class='proj-scenario bear'>"
            "<h3>📉 ESCENARIO BAJISTA</h3>"
            + bear_rows +
            "</div></div>"
            
            "<div class='prob-row'>"
            f"<div class='prob-card reversal'>"
            "<span class='prob-label'>🔄 PROBABILIDAD REVERSIÓN</span>"
            f"<span class='prob-value' style='color:#ff006e'>{proj['prob_reversal']}%</span>"
            "<span class='prob-hint'>Precio gira en dirección opuesta</span>"
            "</div>"
            f"<div class='prob-card continuation'>"
            "<span class='prob-label'>➡ PROBABILIDAD CONTINUACIÓN</span>"
            f"<span class='prob-value' style='color:#00ff9f'>{proj['prob_continuation']}%</span>"
            "<span class='prob-hint'>Precio sigue en misma dirección</span>"
            "</div>"
            "</div></div>"
        )
        
        # CONTEXT ALERT (NEW) — inserted after projections
        alert_section = ""
        if alert.get("has_alert"):
            alert_type = alert["alert_type"]
            # Color coding by alert type
            if alert_type in ["rebote_en_bajista", "pullback_en_alcista"]:
                alert_color = "#ffd60a"  # yellow
                alert_icon = "⚠"
            elif alert_type in ["trampa_alcista", "trampa_bajista"]:
                alert_color = "#ff006e"  # red
                alert_icon = "🚨"
            elif alert_type == "alineacion":
                alert_color = "#00ff9f"  # green
                alert_icon = "✅"
            else:
                alert_color = "#00d4ff"  # cyan
                alert_icon = "ℹ"
            
            alert_section = (
                f"<div class='section context-alert' style='border-left:4px solid {alert_color};background:rgba(15,22,41,1);border:2px solid {alert_color}'>"
                f"<div class='alert-header' style='color:{alert_color};font-size:1.1rem;font-weight:700;margin-bottom:14px'>"
                f"{alert_icon} {alert['alert_title']}"
                "</div>"
                f"<div class='alert-body' style='background:rgba(0,0,0,.3);padding:14px;border-radius:8px;margin-bottom:12px;line-height:1.7'>{alert['alert_message']}</div>"
                f"<div class='alert-recommendation' style='background:rgba(0,212,255,.05);border-left:3px solid #00d4ff;padding:14px;border-radius:0 8px 8px 0;line-height:1.7'>{alert['recommendation']}</div>"
                "</div>"
            )
        
        sec2 += alert_section
        
        # Section 3: TIME VALIDATION
        tvh = tv.get("high"); tvl = tv.get("low")
        time_blocks = ""
        if tvh:
            held_col = "#00ff9f" if tvh["held_pct"] > 60 else "#ffd60a" if tvh["held_pct"] > 40 else "#ff006e"
            time_blocks += (
                f"<div class='val-block' style='border-left:4px solid {held_col}'>"
                f"<div class='val-head'>HIGH formado a las {tvh['hour']:02d}:00 UTC</div>"
                "<div class='val-stats'>"
                f"<div class='stat'><span>✓ HELD</span><span style='color:#00ff9f'>{tvh['held_pct']}%</span></div>"
                f"<div class='stat'><span>✗ TAKEN</span><span style='color:#ff006e'>{tvh['taken_pct']}%</span></div>"
                f"<div class='stat'><span>N</span><span>{tvh['n_sessions']}</span></div>"
                "</div>"
            )
            if tvh["early_risk"]:
                time_blocks += "<div class='warning'>⚠ EARLY FORMATION — Mayor riesgo de takeout</div>"
            time_blocks += "</div>"
        
        if tvl:
            held_col = "#00ff9f" if tvl["held_pct"] > 60 else "#ffd60a" if tvl["held_pct"] > 40 else "#ff006e"
            time_blocks += (
                f"<div class='val-block' style='border-left:4px solid {held_col};margin-top:12px'>"
                f"<div class='val-head'>LOW formado a las {tvl['hour']:02d}:00 UTC</div>"
                "<div class='val-stats'>"
                f"<div class='stat'><span>✓ HELD</span><span style='color:#00ff9f'>{tvl['held_pct']}%</span></div>"
                f"<div class='stat'><span>✗ TAKEN</span><span style='color:#ff006e'>{tvl['taken_pct']}%</span></div>"
                f"<div class='stat'><span>N</span><span>{tvl['n_sessions']}</span></div>"
                "</div>"
            )
            if tvl["early_risk"]:
                time_blocks += "<div class='warning'>⚠ EARLY FORMATION — Mayor riesgo de takeout</div>"
            time_blocks += "</div>"
        
        sec3 = (
            "<div class='section'>"
            "<div class='sec-header'><h2>⏱ TIME VALIDATION</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-time\")'>📖 Explicar</button></div>"
            f"<div id='edu-time' class='edu-box' style='display:none'>{edu['time_validation']}</div>"
            + time_blocks +
            "</div>"
        )
        
        # Section 4: DISTANCE VALIDATION
        dist_col = "#ff006e" if dv.get("small_wick") else "#00ff9f" if dv.get("percentile", 0) > 60 else "#ffd60a"
        sec4 = (
            "<div class='section'>"
            "<div class='sec-header'><h2>📏 DISTANCE VALIDATION</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-dist\")'>📖 Explicar</button></div>"
            f"<div id='edu-dist' class='edu-box' style='display:none'>{edu['distance_validation']}</div>"
            f"<div class='val-block' style='border-left:4px solid {dist_col}'>"
            f"<div class='val-head'>Displacement desde open: {dv.get('disp_pct', 0)}%</div>"
            "<div class='val-stats'>"
            f"<div class='stat'><span>↩ REVERSED</span><span>{dv.get('reversed_pct', 0)}%</span></div>"
            f"<div class='stat'><span>↗ CONTINUED</span><span>{dv.get('continued_pct', 0)}%</span></div>"
            f"<div class='stat'><span>PERCENTILE</span><span>P{dv.get('percentile', 0)}</span></div>"
            "</div>"
            f"<div class='pct-bar'><div class='pct-fill' style='width:{dv.get('percentile', 0)}%;background:{dist_col}'></div></div>"
            f"<div class='pct-labels'><span>P25:{dv.get('p25', 0)}%</span><span>P50:{dv.get('p50', 0)}%</span><span>P75:{dv.get('p75', 0)}%</span></div>"
        )
        if dv.get("small_wick"):
            sec4 += "<div class='warning' style='background:rgba(255,0,110,.1);border-color:#ff006e'>⚠ SMALL WICK — Likely to extend</div>"
        sec4 += "</div></div>"
        
        # Section 5: P1/P2 WEEKLY
        p1_section = ""
        if p1_cur.get("active"):
            p1_col = "#ff006e" if p1_cur["p1_type"] == "high" else "#00ff9f"
            struct_col = "#00ff9f" if p1_cur["structure_ok"] else "#ffd60a" if p1_cur["accept_count"] >= 1 else "#ff006e"
            struct_lbl = ("✅ ACEPTACIÓN CONFIRMADA" if p1_cur["structure_ok"] else
                         "⚠ ACEPTACIÓN PARCIAL" if p1_cur["accept_count"] >= 1 else
                         "❌ SIN CONFIRMACIÓN")
            
            p1_section = (
                "<div class='p1-grid'>"
                f"<div class='p1-card' style='border-left:4px solid {p1_col}'>"
                f"<h4>P1 DETECTADO: {p1_cur['p1_type'].upper()}</h4>"
                f"<p>{DAY_NAMES[p1_cur['p1_day']]} {p1_cur['p1_hour']:02d}:00 UTC — ${p1_cur['p1_val']:,.0f}</p>"
                f"<p>Horas desde P1: <strong>{p1_cur['hrs_since']:.0f}h</strong> / Restantes: <strong>{p1_cur['h_remain']:.0f}h</strong></p>"
                f"<div class='struct-badge' style='color:{struct_col}'>{struct_lbl} ({p1_cur['accept_count']} velas)</div>"
                "</div>"
                f"<div class='p1-card' style='border-left:4px solid {bias_col}'>"
                f"<h4>SESGO: {p1_cur['bias']}</h4>"
                f"<p>Prob histórica: <strong>{p1_cur['prob']:.0f}%</strong></p>"
                f"<p>Flip Risk: <strong style='color:#ffd60a'>{p1_cur['flip_adj']:.0f}%</strong> | Fiabilidad: <strong style='color:#00ff9f'>{100-p1_cur['flip_adj']:.0f}%</strong></p>"
                "</div>"
                "<div class='p1-card' style='border-left:4px solid #60a5fa'>"
                "<h4>PROYECCIÓN P2</h4>"
                f"<p>Normal: <strong>${p1_cur['proj_p2']:,.0f}</strong></p>"
                f"<p>Conservador: ${p1_cur['proj_cons']:,.0f} | Agresivo: ${p1_cur['proj_agr']:,.0f}</p>"
                f"<p>Distancias: P25={p1_cur['dist_p25']}% | Med={p1_cur['dist_med']}% | P75={p1_cur['dist_p75']}%</p>"
                "</div></div>"
            )
        
        if p1p2.get("available"):
            nw = p1p2["n_weeks"]
            p1_section += (
                f"<div class='p1-stats'>"
                f"<p><strong>Estadísticas P1/P2 ({nw} semanas):</strong></p>"
                f"<p>• P1=HIGH → Bajista: {p1p2['prob_bear_p1h']}% (Flip Risk: {p1p2['flip_hi']}%)</p>"
                f"<p>• P1=LOW → Alcista: {p1p2['prob_bull_p1l']}% (Flip Risk: {p1p2['flip_lo']}%)</p>"
                f"<p>• Timing P1→P2: promedio {p1p2['h_mean']}h, mediana {p1p2['h_med']}h</p>"
                "</div>"
            )
        
        sec5 = (
            "<div class='section'>"
            "<div class='sec-header'><h2>📍 P1/P2 WEEKLY ANALYSIS</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-p1\")'>📖 Explicar</button></div>"
            f"<div id='edu-p1' class='edu-box' style='display:none'>{edu['p1p2']}</div>"
            + p1_section +
            "</div>"
        )
        
        # Section 6: DECISION SYNTHESIS
        stars = "⭐" * synth["score"]
        sec6 = (
            "<div class='section synthesis'>"
            "<div class='sec-header'><h2>🎯 SÍNTESIS DE DECISIÓN</h2>"
            "<button class='btn-edu' onclick='toggleEdu(\"edu-synth\")'>📖 Explicar</button></div>"
            f"<div id='edu-synth' class='edu-box' style='display:none'>{edu['synthesis']}</div>"
            "<div class='traffic-grid'>"
            f"<div class='traffic-light'><span>TIME</span><span class='{synth['time_color']}'>{synth['time_status']}</span></div>"
            f"<div class='traffic-light'><span>DISTANCE</span><span class='{synth['dist_color']}'>{synth['dist_status']}</span></div>"
            f"<div class='traffic-light'><span>P1 STRUCTURE</span><span class='{synth['p1_color']}'>{synth['p1_status']}</span></div>"
            f"<div class='traffic-light'><span>BIAS</span><span class='{synth['bias_color']}'>{synth['bias_status']}</span></div>"
            "</div>"
            f"<div class='score-card'>"
            f"<div class='score-stars'>{stars}</div>"
            f"<div class='score-text'>{synth['score']}/4 ESTRELLAS</div>"
            "</div>"
            f"<div class='reading-card'><strong>LECTURA FINAL:</strong><br>{synth['reading']}</div>"
            "</div>"
        )
        
        return sec1 + sec2 + sec3 + sec4 + sec5 + sec6
    
    # CSS
    css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
:root{--bg:#0a0e1a;--surface:#0f1629;--border:#1e2d4a;--text:#e2e8f0;--muted:#64748b;
      --lime:#00ff9f;--magenta:#ff006e;--yellow:#ffd60a;--cyan:#00d4ff;--purple:#b388ff;}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;}
.container{max-width:1600px;margin:0 auto;padding:20px;}
.header{padding:24px 0;border-bottom:1px solid var(--border);margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;}
.header h1{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700;color:var(--lime);letter-spacing:-.5px;}
.header .meta{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--muted);}
.asset-tabs{display:flex;gap:12px;margin-bottom:24px;}
.asset-tab{font-family:'IBM Plex Mono',monospace;font-size:.875rem;font-weight:600;padding:10px 24px;
           background:var(--surface);border:1px solid var(--border);border-radius:6px;color:var(--muted);cursor:pointer;transition:.2s;}
.asset-tab:hover{border-color:var(--lime);color:var(--lime);}
.asset-tab.active{background:linear-gradient(135deg,rgba(0,255,159,.1),transparent);border-color:var(--lime);color:var(--lime);}
.asset-content{display:none;}
.asset-content.active{display:block;}
.section{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:20px;}
.sec-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;}
.sec-header h2{font-size:1rem;font-weight:600;color:var(--lime);}
.btn-edu{font-size:.75rem;padding:6px 12px;background:rgba(0,212,255,.15);border:1px solid var(--cyan);
         border-radius:6px;color:var(--cyan);cursor:pointer;transition:.2s;}
.btn-edu:hover{background:rgba(0,212,255,.25);}
.edu-box{background:rgba(0,212,255,.05);border-left:3px solid var(--cyan);border-radius:0 6px 6px 0;
         padding:12px;margin-bottom:16px;font-size:.8125rem;color:var(--muted);line-height:1.5;}
.edu-box strong{color:var(--cyan);}
.price-card{display:flex;justify-content:space-between;align-items:center;padding:16px;
           background:rgba(0,0,0,.3);border-radius:8px;margin-bottom:16px;}
.price-main{font-family:'IBM Plex Mono',monospace;font-size:2.5rem;font-weight:700;color:var(--lime);}
.ohlc-mini{display:flex;gap:16px;font-family:'IBM Plex Mono',monospace;font-size:.875rem;}
.levels-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px;}
.level{background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:6px;padding:10px;
       display:flex;flex-direction:column;align-items:center;gap:4px;}
.level span:first-child{font-size:.7rem;color:var(--muted);font-weight:600;}
.level span:last-child{font-family:'IBM Plex Mono',monospace;font-size:.875rem;font-weight:600;}
.level.r2,.level.r1{border-left:3px solid var(--lime);}
.level.s2,.level.s1{border-left:3px solid var(--magenta);}
.level.pp{border-left:3px solid var(--yellow);}
.opens-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;}
.open-item{background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:6px;padding:10px;
           display:flex;justify-content:space-between;align-items:center;}
.open-item span:first-child{font-size:.7rem;color:var(--muted);font-weight:600;}
.open-item span:last-child{font-family:'IBM Plex Mono',monospace;font-size:.875rem;font-weight:600;}
.bias-badge{padding:14px;border-radius:6px;display:flex;justify-content:space-between;align-items:center;
           background:rgba(0,0,0,.2);}
.bias-label{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);}
.bias-value{font-family:'IBM Plex Mono',monospace;font-size:1.125rem;font-weight:700;}
.direction-badge{padding:16px;border-radius:8px;margin-bottom:16px;background:rgba(0,0,0,.2);}
.dir-label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:6px;}
.dir-value{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700;margin-bottom:8px;}
.dir-explain{font-size:.8125rem;color:var(--text);line-height:1.6;padding:10px;background:rgba(0,0,0,.3);border-radius:6px;}
.proj-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
.proj-scenario{background:rgba(0,0,0,.2);border-radius:8px;padding:16px;}
.proj-scenario h3{font-size:.875rem;margin-bottom:12px;}
.proj-row{display:grid;grid-template-columns:60px 1fr 80px 60px;gap:12px;align-items:center;margin-bottom:8px;}
.proj-bar{background:rgba(255,255,255,.05);border-radius:4px;height:8px;overflow:hidden;}
.proj-fill{height:100%;transition:width .6s ease;}
.proj-fill.bull{background:var(--lime);}
.proj-fill.bear{background:var(--magenta);}
.proj-target{font-family:'IBM Plex Mono',monospace;font-size:.8125rem;font-weight:600;}
.proj-conf{font-family:'IBM Plex Mono',monospace;font-size:.8125rem;font-weight:600;text-align:right;}
.prob-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.prob-card{background:rgba(0,0,0,.3);border-radius:8px;padding:16px;text-align:center;}
.prob-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:8px;}
.prob-value{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700;display:block;margin-bottom:6px;}
.prob-hint{font-size:.75rem;color:var(--muted);font-style:italic;display:block;}
.val-block{background:rgba(0,0,0,.2);border-radius:8px;padding:16px;}
.val-head{font-family:'IBM Plex Mono',monospace;font-size:.875rem;font-weight:600;margin-bottom:12px;}
.val-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;}
.stat{display:flex;flex-direction:column;gap:4px;}
.stat span:first-child{font-size:.7rem;color:var(--muted);font-weight:600;}
.stat span:last-child{font-family:'IBM Plex Mono',monospace;font-size:1.25rem;font-weight:700;}
.warning{background:rgba(255,214,10,.15);border:1px solid var(--yellow);border-radius:6px;
         padding:8px 12px;font-size:.75rem;font-weight:600;color:var(--yellow);margin-top:12px;}
.pct-bar{background:rgba(255,255,255,.05);border-radius:4px;height:8px;margin:12px 0;overflow:hidden;}
.pct-fill{height:100%;transition:width .6s ease;}
.pct-labels{display:flex;justify-content:space-between;font-size:.7rem;font-family:'IBM Plex Mono',monospace;color:var(--muted);}
.p1-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;}
.p1-card{background:rgba(0,0,0,.2);border-radius:8px;padding:14px;}
.p1-card h4{font-size:.875rem;margin-bottom:8px;}
.p1-card p{font-size:.8125rem;color:var(--muted);margin-bottom:6px;}
.struct-badge{font-size:.75rem;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:8px;}
.p1-stats{background:rgba(0,0,0,.3);border-radius:8px;padding:14px;font-size:.8125rem;color:var(--muted);}
.p1-stats p{margin:4px 0;}
.synthesis{border:2px solid var(--lime);}
.traffic-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;}
.traffic-light{background:rgba(0,0,0,.3);border-radius:8px;padding:14px;text-align:center;}
.traffic-light span:first-child{font-size:.7rem;color:var(--muted);display:block;margin-bottom:8px;}
.traffic-light span:last-child{font-size:1.5rem;}
.traffic-light .green{color:var(--lime);}
.traffic-light .yellow{color:var(--yellow);}
.traffic-light .red{color:var(--magenta);}
.score-card{background:rgba(0,0,0,.3);border-radius:8px;padding:20px;text-align:center;margin-bottom:16px;}
.score-stars{font-size:2rem;margin-bottom:8px;}
.score-text{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:var(--lime);}
.reading-card{background:rgba(0,212,255,.05);border-left:3px solid var(--cyan);border-radius:0 6px 6px 0;
             padding:14px;font-size:.8125rem;line-height:1.6;}
.context-alert{margin:20px 0;animation:alertPulse 2s ease-in-out infinite;}
@keyframes alertPulse{0%,100%{box-shadow:0 0 0 0 rgba(255,214,10,.4);}50%{box-shadow:0 0 20px 5px rgba(255,214,10,.1);}}
.alert-header{font-size:1.1rem;font-weight:700;margin-bottom:14px;font-family:'IBM Plex Mono',monospace;}
.alert-body{background:rgba(0,0,0,.3);padding:14px;border-radius:8px;margin-bottom:12px;line-height:1.7;font-size:.8125rem;}
.alert-recommendation{background:rgba(0,212,255,.05);border-left:3px solid var(--cyan);padding:14px;border-radius:0 8px 8px 0;line-height:1.7;font-size:.8125rem;}
"""
    
    btc_sec = asset_section_hybrid(btc_data, "BTC", edu_content)
    eth_sec = asset_section_hybrid(eth_data, "ETH", edu_content)
    
    html = f"""
<!DOCTYPE html>
<html lang='es'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1.0'>
<title>Crypto Pivot v4.5 Hybrid</title>
<style>{css}</style>
</head>
<body>
<div class='container'>
<div class='header'>
<div>
<h1>🔷 CRYPTO PIVOT ANALYZER v4.5 HYBRID</h1>
<p style='font-size:.75rem;color:var(--muted);margin-top:4px'>TIME + DISTANCE + P1/P2 + Projections + Educational</p>
</div>
<div class='meta'>{now_str}</div>
</div>

<div class='asset-tabs'>
<div class='asset-tab active' onclick='showAsset("btc")'>BTC/USDT</div>
<div class='asset-tab' onclick='showAsset("eth")'>ETH/USDT</div>
</div>

<div id='btc-content' class='asset-content active'>
{btc_sec}
</div>

<div id='eth-content' class='asset-content'>
{eth_sec}
</div>
</div>

<script>
function showAsset(asset) {{
document.querySelectorAll('.asset-tab').forEach(t => t.classList.remove('active'));
document.querySelectorAll('.asset-content').forEach(c => c.classList.remove('active'));
if(asset === 'btc') {{
document.querySelector('.asset-tab:first-child').classList.add('active');
document.getElementById('btc-content').classList.add('active');
}} else {{
document.querySelector('.asset-tab:last-child').classList.add('active');
document.getElementById('eth-content').classList.add('active');
}}
}}

function toggleEdu(id) {{
const el = document.getElementById(id);
el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

setTimeout(() => location.reload(), 3600000);
</script>
</body>
</html>
"""
    
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    success(f"Dashboard: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    banner()
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    section("📊 ANÁLISIS BTC")
    btc_data = analyze_asset_full("BTC")
    
    section("📊 ANÁLISIS ETH")
    eth_data = analyze_asset_full("ETH")
    
    section("📚 CONTENIDO EDUCATIVO")
    edu = educational_content()
    
    section("🎨 GENERANDO DASHBOARD")
    dash_path = build_dashboard_hybrid(btc_data, eth_data, edu, CONFIG["OUTPUT_DIR"])
    
    # Console output
    print(f"\n  {'─'*70}")
    print(f"  BTC — Bias: {btc_data['geometric_bias']} | Score: {btc_data['synthesis']['score']}/4 ⭐")
    print(f"  Precio: ${btc_data['price']:,.2f}")
    print(f"  Reversión: {btc_data['projections']['prob_reversal']}% | Continuación: {btc_data['projections']['prob_continuation']}%")
    if btc_data['context_alert'].get('has_alert'):
        print(f"  ⚠ ALERTA: {btc_data['context_alert']['alert_type']}")
    print(f"  {'─'*70}\n")
    
    section("✅ COMPLETADO")
    print(f"\n  Dashboard: {os.path.abspath(dash_path)}\n")
    
    # Preguntar si quiere servidor web
    print(Fore.CYAN + "\n  ¿Quieres iniciar un servidor web local para ver el dashboard? (s/n): " + Fore.RESET, end="")
    try:
        respuesta = input().strip().lower()
        if respuesta in ['s', 'si', 'y', 'yes']:
            start_web_server(CONFIG["OUTPUT_DIR"])
        else:
            webbrowser.open(f"file:///{os.path.abspath(dash_path)}")
    except:
        webbrowser.open(f"file:///{os.path.abspath(dash_path)}")


def start_web_server(output_dir: str, port: int = 8080):
    """
    Inicia servidor HTTP simple para servir el dashboard.
    Accesible desde cualquier dispositivo en la red local.
    """
    import http.server
    import socketserver
    import threading
    
    os.chdir(output_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Intentar varios puertos si el primero está ocupado
    for p in range(port, port + 10):
        try:
            with socketserver.TCPServer(("", p), Handler) as httpd:
                port = p
                break
        except OSError:
            continue
    
    def serve():
        with socketserver.TCPServer(("", port), Handler) as httpd:
            success(f"Servidor web iniciado en http://localhost:{port}")
            print(Fore.YELLOW + f"\n  {'─'*70}")
            print(Fore.YELLOW + f"  📱 ACCESO DESDE OTROS DISPOSITIVOS:")
            print(Fore.YELLOW + f"  {'─'*70}")
            
            # Obtener IP local
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(Fore.CYAN + f"  📱 Desde tu celular/tablet en la misma red WiFi:")
                print(Fore.CYAN + f"     http://{local_ip}:{port}/dashboard.html")
            except:
                pass
            
            print(Fore.CYAN + f"\n  💻 Desde esta computadora:")
            print(Fore.CYAN + f"     http://localhost:{port}/dashboard.html")
            print(Fore.YELLOW + f"\n  {'─'*70}")
            print(Fore.RED + f"\n  ⚠  Presiona Ctrl+C para detener el servidor")
            print(Fore.YELLOW + f"  {'─'*70}\n" + Fore.RESET)
            
            # Abrir en navegador
            webbrowser.open(f"http://localhost:{port}/dashboard.html")
            
            # Servir indefinidamente
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n  Servidor detenido.\n")
                httpd.shutdown()
    
    # Iniciar servidor en thread separado
    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    
    # Mantener programa vivo
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n  Cerrando...\n")


if __name__ == "__main__":
    main()
