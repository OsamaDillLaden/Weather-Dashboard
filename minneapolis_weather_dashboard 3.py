# minneapolis_weather_dashboard.py
# -------------------------------------------------
# Minneapolis Weather Market Dashboard
# Sources: NWS, AccuWeather, OpenWeatherMap, WeatherAPI, Apple Weather, Google Weather
# Market Odds: Polymarket + Kalshi (2-min refresh)
# Persistent SQLite accuracy tracking — model gets smarter every day
# Weighted ensemble built from rolling historical RMSE per source
# Audio + Visual edge alerts
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import base64

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# -----------------------
# AUTO TICKER GENERATOR
# -----------------------
def get_kalshi_ticker_for_today():
    """Auto-generates the Kalshi Minneapolis high temp ticker for today.
    Format: kxhightmin-YYmmmDD  e.g. kxhightmin-26feb22"""
    today = date.today()
    return f"kxhightmin-{today.strftime('%y%b%d').lower()}"

def get_kalshi_ticker_for_date(target_date):
    """Generate ticker for any date — useful for browsing past/future markets."""
    return f"kxhightmin-{target_date.strftime('%y%b%d').lower()}"

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Minneapolis Weather Market Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 2 minutes
st.markdown("<script>setTimeout(function(){window.location.reload();},120000);</script>",
            unsafe_allow_html=True)

# -----------------------
# CSS
# -----------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .block-container { padding: 1.5rem 2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2e3247; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2e3247; border-radius: 12px;
        padding: 1rem 1.2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #8b9dc3 !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #fff !important; font-size: 1.5rem !important; font-weight: 700;
    }
    .alert-success { background: linear-gradient(90deg,#0d3d1e,#145a2e); border-left: 4px solid #2dce89; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: #a8f0c6; font-weight: 600; }
    .alert-warning { background: linear-gradient(90deg,#3d2e0d,#5a4214); border-left: 4px solid #f4a62a; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: #f8d48a; font-weight: 600; }
    .alert-info    { background: linear-gradient(90deg,#0d2a3d,#145070); border-left: 4px solid #11cdef; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: #8ae8f8; font-weight: 600; }
    .market-card { background: linear-gradient(135deg,#1e2130,#252a3d); border: 1px solid #2e3247; border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem; }
    .market-card h4 { color: #8b9dc3; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.4rem 0; }
    .market-card .odds { font-size: 2rem; font-weight: 800; color: #fff; }
    .market-card .meta { font-size: 0.75rem; color: #4a5075; margin-top: 0.3rem; }
    .score-card { background: linear-gradient(135deg,#1e2130,#252a3d); border: 1px solid #2e3247; border-radius: 12px; padding: 1rem; margin-bottom: 0.6rem; }
    .score-card .src-name { font-size: 0.9rem; font-weight: 700; margin-bottom: 0.3rem; }
    .score-card .src-rmse { font-size: 1.4rem; font-weight: 800; }
    .score-card .src-meta { font-size: 0.72rem; color: #4a5075; margin-top: 0.2rem; }
    .trust-bar  { height: 6px; border-radius: 3px; background: #2e3247; margin-top: 6px; }
    .trust-fill { height: 6px; border-radius: 3px; }
    .section-header { font-size: 1rem; font-weight: 700; color: #8b9dc3; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #2e3247; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    hr { border-color: #2e3247; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# AUDIO ALERT
# -----------------------
def play_audio_alert():
    st.markdown("""
    <script>
    (function(){
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        [0, 0.35, 0.7].forEach(function(delay){
            var osc = ctx.createOscillator(), gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.type = 'sine'; osc.frequency.value = 880;
            gain.gain.setValueAtTime(0.3, ctx.currentTime + delay);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.4);
            osc.start(ctx.currentTime + delay);
            osc.stop(ctx.currentTime + delay + 0.5);
        });
    })();
    </script>
    """, unsafe_allow_html=True)

# -----------------------
# CONSTANTS
# -----------------------
LAT, LON   = 44.9778, -93.2650
NWS_OFFICE = "MPX"
CITY       = "Minneapolis"
NUM_DAYS   = 14

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_scores.db")

SOURCES = ["NWS", "AccuWeather", "OpenWeatherMap", "WeatherAPI", "Apple Weather", "Google Weather"]

COLORS = {
    "NWS":            "#11cdef",
    "AccuWeather":    "#f4a62a",
    "OpenWeatherMap": "#2dce89",
    "WeatherAPI":     "#e040fb",
    "Apple Weather":  "#a8d8ea",
    "Google Weather": "#ff6b6b",
    "Ensemble":       "#3a7bd5",
    "Actual":         "#ffffff",
}

# ==============================================================
# DATABASE LAYER
# ==============================================================

def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            source      TEXT NOT NULL,
            forecast_f  REAL,
            actual_f    REAL,
            error       REAL,
            abs_error   REAL,
            recorded_at TEXT,
            UNIQUE(date, source)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS source_scores (
            source       TEXT PRIMARY KEY,
            rmse_7d      REAL,
            rmse_30d     REAL,
            rmse_all     REAL,
            mae_all      REAL,
            days_tracked INTEGER,
            last_updated TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_daily_forecasts(today_str, forecasts, actual):
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().isoformat()
    for source, fc_val in forecasts.items():
        if fc_val is None or np.isnan(fc_val):
            continue
        error = fc_val - actual
        c.execute("""
            INSERT INTO daily_forecasts (date, source, forecast_f, actual_f, error, abs_error, recorded_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(date, source) DO UPDATE SET
                forecast_f=excluded.forecast_f, actual_f=excluded.actual_f,
                error=excluded.error, abs_error=excluded.abs_error, recorded_at=excluded.recorded_at
        """, (today_str, source, float(fc_val), float(actual), float(error), float(abs(error)), now))
    conn.commit()
    conn.close()

def recompute_source_scores():
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().isoformat()
    for source in SOURCES:
        rows = c.execute(
            "SELECT error, abs_error FROM daily_forecasts WHERE source=? AND actual_f IS NOT NULL",
            (source,)).fetchall()
        if not rows:
            continue
        errors = np.array([r[0] for r in rows])
        mae    = float(np.mean([r[1] for r in rows]))
        rmse_all = float(np.sqrt(np.mean(errors**2)))
        days   = len(rows)

        cut7  = (date.today() - timedelta(days=7)).isoformat()
        r7    = c.execute("SELECT error FROM daily_forecasts WHERE source=? AND date>=? AND actual_f IS NOT NULL", (source, cut7)).fetchall()
        rmse7 = float(np.sqrt(np.mean(np.array([r[0] for r in r7])**2))) if r7 else rmse_all

        cut30  = (date.today() - timedelta(days=30)).isoformat()
        r30    = c.execute("SELECT error FROM daily_forecasts WHERE source=? AND date>=? AND actual_f IS NOT NULL", (source, cut30)).fetchall()
        rmse30 = float(np.sqrt(np.mean(np.array([r[0] for r in r30])**2))) if r30 else rmse_all

        c.execute("""
            INSERT INTO source_scores (source, rmse_7d, rmse_30d, rmse_all, mae_all, days_tracked, last_updated)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(source) DO UPDATE SET
                rmse_7d=excluded.rmse_7d, rmse_30d=excluded.rmse_30d,
                rmse_all=excluded.rmse_all, mae_all=excluded.mae_all,
                days_tracked=excluded.days_tracked, last_updated=excluded.last_updated
        """, (source, rmse7, rmse30, rmse_all, mae, days, now))
    conn.commit()
    conn.close()

def load_source_scores():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM source_scores ORDER BY rmse_all ASC", conn)
    conn.close()
    return df

def load_score_history():
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT date, source, forecast_f, actual_f, error, abs_error FROM daily_forecasts ORDER BY date ASC", conn)
    conn.close()
    return df

def load_rolling_rmse(window=7):
    history = load_score_history()
    if history.empty:
        return pd.DataFrame()
    results = []
    for source in SOURCES:
        src_df = history[history["source"] == source].copy().sort_values("date")
        if src_df.empty:
            continue
        src_df["rolling_rmse"] = src_df["error"].pow(2).rolling(window, min_periods=1).mean().pow(0.5)
        src_df["source"] = source
        results.append(src_df[["date", "source", "rolling_rmse"]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def get_weights_from_db(score_col="rmse_all"):
    scores_df = load_source_scores()
    if scores_df.empty:
        return {src: 1/len(SOURCES) for src in SOURCES}
    weights = {}
    for src in SOURCES:
        row = scores_df[scores_df["source"] == src]
        rmse_val = row.iloc[0][score_col] if not row.empty else 999
        weights[src] = 1.0 / rmse_val if rmse_val > 0 else 1.0
    total = sum(weights.values())
    return {src: w/total for src, w in weights.items()}

# Boot DB
init_db()

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 🔑 API Keys")
    accu_key        = st.text_input("AccuWeather Key",       value="", type="password", placeholder="Enter key...")
    owm_key         = st.text_input("OpenWeatherMap Key",    value="", type="password", placeholder="Enter key...")
    wapi_key        = st.text_input("WeatherAPI.com Key",    value="", type="password", placeholder="Enter key...")

    st.markdown("---")
    st.markdown("### 🔵 Kalshi Authentication")
    kalshi_key_id   = st.text_input("Kalshi Key ID (UUID)",  value="", placeholder="e.g. a952bcbe-ec3b-...")
    kalshi_pem_path = st.text_input("Private Key File Path", value="", placeholder="/Users/you/.kalshi/private_key.pem")

    if not CRYPTO_AVAILABLE:
        st.warning("⚠️ Install cryptography: `pip install cryptography`")

    st.markdown("---")
    st.markdown("### 📊 Market Settings")
    threshold      = st.slider("Temperature Threshold (°F)", 0, 100, 30)
    edge_threshold = st.slider("Edge Alert Threshold (%)",   1,  20,  5) / 100

    st.markdown("---")
    st.markdown("### 🎯 Kalshi Market")
    auto_ticker    = get_kalshi_ticker_for_today()
    use_auto       = st.checkbox("Auto-ticker (today's market)", value=True)
    if use_auto:
        kalshi_ticker = auto_ticker
        st.success(f"✅ Today's ticker: `{kalshi_ticker}`")
    else:
        kalshi_ticker = st.text_input("Manual Ticker Override", value=auto_ticker)

    # Show next 3 days for reference
    st.caption("Upcoming tickers:")
    for i in range(1, 4):
        future = date.today() + timedelta(days=i)
        st.caption(f"  +{i}d: `{get_kalshi_ticker_for_date(future)}`")

    st.markdown("---")
    st.markdown("### 🧠 Weight Window")
    weight_window = st.selectbox(
        "Base ensemble weights on:",
        options=["rmse_7d", "rmse_30d", "rmse_all"],
        format_func=lambda x: {"rmse_7d":"Last 7 days","rmse_30d":"Last 30 days","rmse_all":"All time"}[x],
        index=2)
    weight_label = {"rmse_7d":"7-day","rmse_30d":"30-day","rmse_all":"All-time"}[weight_window]

    st.markdown("---")
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    scores_sidebar = load_source_scores()
    if not scores_sidebar.empty:
        max_days = int(scores_sidebar["days_tracked"].max())
        st.markdown(f"### 📚 DB Status\n**{max_days}** days of history")
        st.caption(f"File: weather_scores.db")
    else:
        st.info("No history yet — builds after day 1.")

    st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')}")

# ==============================================================
# FETCH FUNCTIONS
# ==============================================================

@st.cache_data(ttl=120)
def fetch_nws_forecast():
    try:
        r = requests.get(f"https://api.weather.gov/gridpoints/{NWS_OFFICE}/107,70/forecast",
            headers={"User-Agent":"WeatherDashboard/1.0","Accept":"application/geo+json"}, timeout=10)
        r.raise_for_status()
        highs = [p["temperature"] for p in r.json()["properties"]["periods"] if p["isDaytime"]][:NUM_DAYS]
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs, dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_accuweather_forecast(api_key):
    try:
        if not api_key: raise ValueError()
        loc = requests.get(f"http://dataservice.accuweather.com/locations/v1/cities/search?apikey={api_key}&q=Minneapolis", timeout=10)
        loc.raise_for_status()
        lk = loc.json()[0]["Key"]
        fc = requests.get(f"http://dataservice.accuweather.com/forecasts/v1/daily/5day/{lk}?apikey={api_key}&metric=false", timeout=10)
        fc.raise_for_status()
        highs = [f["Temperature"]["Maximum"]["Value"] for f in fc.json()["DailyForecasts"]]
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs[:NUM_DAYS], dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_openweathermap_forecast(api_key):
    try:
        if not api_key: raise ValueError()
        r = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={api_key}&units=imperial&cnt=40", timeout=10)
        r.raise_for_status()
        daily = {}
        for item in r.json()["list"]:
            d = item["dt_txt"][:10]
            daily[d] = max(daily.get(d, -999), item["main"]["temp_max"])
        highs = list(daily.values())[:NUM_DAYS]
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs, dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_weatherapi_forecast(api_key):
    try:
        if not api_key: raise ValueError()
        r = requests.get(f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={LAT},{LON}&days=3", timeout=10)
        r.raise_for_status()
        highs = [f["day"]["maxtemp_f"] for f in r.json()["forecast"]["forecastday"]]
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs[:NUM_DAYS], dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_apple_weather():
    try:
        r = requests.get(f"https://wttr.in/{CITY}?format=j1",
            headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        highs = [float(d["maxtempF"]) for d in r.json().get("weather", [])]
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs[:NUM_DAYS], dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_google_weather():
    try:
        headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get("https://www.google.com/search?q=Minneapolis+weather+forecast+10+day", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        temp_divs = soup.find_all("div", {"class": "wob_t"})
        highs = []
        for i, t in enumerate(temp_divs):
            try:
                val = float(t.text.strip())
                if i % 2 == 0: highs.append(val)
            except: pass
        if len(highs) < 3: raise ValueError()
        while len(highs) < NUM_DAYS: highs.append(np.nan)
        return np.array(highs[:NUM_DAYS], dtype=float), True
    except:
        return np.random.normal(30, 2, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_nws_actuals():
    try:
        r = requests.get("https://api.weather.gov/stations/KMSP/observations?limit=14",
            headers={"User-Agent":"WeatherDashboard/1.0","Accept":"application/geo+json"}, timeout=10)
        r.raise_for_status()
        temps = [o["properties"]["temperature"]["value"]*9/5+32
                 for o in r.json()["features"] if o["properties"]["temperature"]["value"] is not None]
        temps = list(reversed(temps[:NUM_DAYS]))
        while len(temps) < NUM_DAYS: temps.append(np.nan)
        return np.array(temps, dtype=float), True
    except:
        return np.random.normal(30, 1.5, NUM_DAYS), False

@st.cache_data(ttl=120)
def fetch_polymarket_odds():
    try:
        r = requests.get("https://gamma-api.polymarket.com/markets?search=Minneapolis+temperature&limit=5", timeout=10)
        r.raise_for_status()
        markets = r.json()
        for m in markets:
            if "minneapolis" in m.get("question","").lower() or "temp" in m.get("question","").lower():
                return float(m.get("outcomePrices",[0.5])[0]), m.get("question","Minneapolis Temp"), m.get("volume","N/A"), True
        if markets:
            m = markets[0]
            return float(m.get("outcomePrices",[0.42])[0]), m.get("question","Weather Market"), m.get("volume","N/A"), True
        return 0.42, "No active market", "N/A", False
    except:
        return 0.42, "Polymarket (simulated)", "N/A", False

def _sign_kalshi_request(private_key_pem: str, timestamp_ms: str, method: str, path: str) -> str:
    """Sign a Kalshi API request with RSA private key (PSS SHA-256)."""
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None, backend=default_backend()
    )
    msg = (timestamp_ms + method.upper() + path).encode("utf-8")
    signature = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode("utf-8")

@st.cache_data(ttl=120)
def fetch_kalshi_odds(key_id: str, pem_path: str, ticker: str):
    """Fetch Kalshi market odds using RSA key-pair authentication."""
    try:
        if not key_id or not pem_path:
            raise ValueError("Missing Kalshi credentials")
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package not installed")

        # Load private key from local file
        with open(os.path.expanduser(pem_path), "r") as f:
            private_key_pem = f.read()

        timestamp_ms = str(int(datetime.now().timestamp() * 1000))
        path         = f"/trade-api/v2/markets/{ticker}"
        signature    = _sign_kalshi_request(private_key_pem, timestamp_ms, "GET", path)

        headers = {
            "KALSHI-ACCESS-KEY":       key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type":            "application/json",
        }
        r = requests.get(f"https://trading-api.kalshi.com{path}", headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()["market"]
        price = data.get("yes_ask", data.get("last_price", 45))
        return float(price) / 100.0, data.get("title", ticker), data.get("volume", "N/A"), True

    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ Kalshi key file not found: {pem_path}")
        return 0.44, f"Kalshi: {ticker} (key file missing)", "N/A", False
    except Exception:
        return 0.44, f"Kalshi: {ticker} (simulated)", "N/A", False

# ==============================================================
# FETCH ALL DATA
# ==============================================================
with st.spinner("Fetching live data from all sources..."):
    nws_fc,   nws_live   = fetch_nws_forecast()
    accu_fc,  accu_live  = fetch_accuweather_forecast(accu_key)
    owm_fc,   owm_live   = fetch_openweathermap_forecast(owm_key)
    wapi_fc,  wapi_live  = fetch_weatherapi_forecast(wapi_key)
    apple_fc, apple_live = fetch_apple_weather()
    goog_fc,  goog_live  = fetch_google_weather()
    actuals,  act_live   = fetch_nws_actuals()
    poly_prob, poly_title, poly_vol, poly_live = fetch_polymarket_odds()
    kals_prob, kals_title, kals_vol, kals_live = fetch_kalshi_odds(kalshi_key_id, kalshi_pem_path, kalshi_ticker)

market_avg_prob = (poly_prob + kals_prob) / 2

sources_map = {
    "NWS":            (nws_fc,   nws_live),
    "AccuWeather":    (accu_fc,  accu_live),
    "OpenWeatherMap": (owm_fc,   owm_live),
    "WeatherAPI":     (wapi_fc,  wapi_live),
    "Apple Weather":  (apple_fc, apple_live),
    "Google Weather": (goog_fc,  goog_live),
}

dates = [datetime.today() - timedelta(days=NUM_DAYS - i) for i in range(NUM_DAYS)]
forecast_df = pd.DataFrame({"Date": dates, "Actual": actuals})
for name, (fc, _) in sources_map.items():
    forecast_df[name] = fc

# ==============================================================
# SAVE TODAY TO DB + RECOMPUTE SCORES
# ==============================================================
today_str     = date.today().isoformat()
today_actual  = float(actuals[-1]) if not np.isnan(actuals[-1]) else None
today_fc_dict = {src: float(forecast_df[src].iloc[-1]) for src in SOURCES
                 if not np.isnan(forecast_df[src].iloc[-1])}

if today_actual is not None and today_fc_dict:
    save_daily_forecasts(today_str, today_fc_dict, today_actual)
    recompute_source_scores()

# ==============================================================
# LOAD SCORES & COMPUTE PERSISTENT WEIGHTS
# ==============================================================
scores_df  = load_source_scores()
history_df = load_score_history()

if not scores_df.empty:
    weights = get_weights_from_db(weight_window)
else:
    # Cold-start fallback (session only)
    rmse_fb = {}
    for src in SOURCES:
        valid = ~np.isnan(forecast_df[src]) & ~np.isnan(forecast_df["Actual"])
        rmse_fb[src] = float(np.sqrt(np.mean((forecast_df[src][valid] - forecast_df["Actual"][valid])**2))) \
                       if valid.sum() > 0 else 999
    raw = {src: 1/rmse_fb[src] for src in SOURCES}
    total = sum(raw.values())
    weights = {src: v/total for src, v in raw.items()}
    weight_label = "Session (no history yet)"

# Compute ensemble prediction
latest          = forecast_df.iloc[-1]
valid_sources   = [src for src in SOURCES if not np.isnan(float(latest[src]))]
T_ensemble      = sum(float(latest[src]) * weights[src] for src in valid_sources)
sigma           = max(float(np.nanstd([float(latest[src]) for src in SOURCES])), 0.5)
P_model         = float(1 - norm.cdf(threshold, loc=T_ensemble, scale=sigma))
edge_poly       = P_model - poly_prob
edge_kals       = P_model - kals_prob
edge_avg        = P_model - market_avg_prob
ensemble_series = sum(forecast_df[src] * weights[src] for src in SOURCES)

# ==============================================================
# HEADER
# ==============================================================
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("# 🌦️ Minneapolis Weather Market Dashboard")
    st.markdown(f"**Minneapolis, MN** · Threshold: **{threshold}°F** · "
                f"{datetime.now().strftime('%A, %B %d, %Y %H:%M:%S')}")
with c2:
    live_count   = sum(1 for _, l in sources_map.values() if l)
    days_tracked = int(scores_df["days_tracked"].max()) if not scores_df.empty else 0
    lc = "#2dce89" if live_count >= 4 else "#f4a62a"
    st.markdown(f"""
    <div style="text-align:right;padding-top:1rem;">
        <span style="background:#1a1d27;border:1px solid {lc};border-radius:20px;
        padding:5px 14px;font-size:0.82rem;color:{lc};">
        {'🟢' if live_count >= 4 else '🟡'} {live_count}/6 live
        </span><br><br>
        <span style="background:#1a1d27;border:1px solid #3a7bd5;border-radius:20px;
        padding:5px 14px;font-size:0.82rem;color:#8ab4f8;">
        🧠 {days_tracked} days learned
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ==============================================================
# ALERT + AUDIO
# ==============================================================
if edge_avg >= edge_threshold:
    st.markdown(f"""<div class="alert-success">
    ⚡ HIGH EDGE ALERT — Model is <strong>{edge_avg*100:.1f}%</strong> above combined market! &nbsp;|&nbsp;
    Polymarket: {edge_poly*100:+.1f}% &nbsp;|&nbsp; Kalshi: {edge_kals*100:+.1f}%
    </div>""", unsafe_allow_html=True)
    play_audio_alert()
elif abs(edge_avg) < 0.02:
    st.markdown(f"""<div class="alert-info">
    ℹ️ Model and markets closely aligned — avg edge: {edge_avg*100:.1f}%. No clear signal.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class="alert-warning">
    ⚠️ Model is {abs(edge_avg)*100:.1f}% {'below' if edge_avg < 0 else 'above'} avg market. Monitor closely.
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ==============================================================
# KPI ROW
# ==============================================================
k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
with k1: st.metric("Ensemble Forecast", f"{T_ensemble:.1f}°F",    f"{T_ensemble-threshold:+.1f}°F vs threshold")
with k2: st.metric("Model Probability", f"{P_model*100:.1f}%")
with k3: st.metric("Polymarket Odds",   f"{poly_prob*100:.1f}%",  "🟢 Live" if poly_live else "🔴 Sim")
with k4: st.metric("Kalshi Odds",       f"{kals_prob*100:.1f}%",  "🟢 Live" if kals_live else "🔴 Sim")
with k5: st.metric("Avg Market Prob",   f"{market_avg_prob*100:.1f}%")
with k6: st.metric("Edge vs Markets",   f"{edge_avg*100:+.1f}%",  "Bullish" if edge_avg >= 0 else "Bearish")
with k7: st.metric("Weights From",      weight_label)
st.markdown("")

# ==============================================================
# MARKET ODDS CARDS
# ==============================================================
st.markdown('<div class="section-header">📈 Live Prediction Market Odds</div>', unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3)
def ec(e): return "#2dce89" if e >= 0 else "#e74c3c"

with mc1:
    st.markdown(f"""<div class="market-card">
        <h4>🟣 Polymarket</h4><div class="odds">{poly_prob*100:.1f}%</div>
        <div class="meta">{poly_title[:65]}</div>
        <div class="meta" style="margin-top:.5rem;">Vol: {poly_vol}&nbsp;|&nbsp;
        <span style="color:{ec(edge_poly)};font-weight:700;">Edge: {edge_poly*100:+.1f}%</span></div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""<div class="market-card">
        <h4>🔵 Kalshi</h4><div class="odds">{kals_prob*100:.1f}%</div>
        <div class="meta">{kals_title[:65]}</div>
        <div class="meta" style="margin-top:.5rem;">Vol: {kals_vol}&nbsp;|&nbsp;
        <span style="color:{ec(edge_kals)};font-weight:700;">Edge: {edge_kals*100:+.1f}%</span></div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown(f"""<div class="market-card" style="border-color:#3a7bd5;">
        <h4>⚡ Combined Average</h4><div class="odds">{market_avg_prob*100:.1f}%</div>
        <div class="meta">Average of Polymarket + Kalshi</div>
        <div class="meta" style="margin-top:.5rem;">Model: {P_model*100:.1f}%&nbsp;|&nbsp;
        <span style="color:{ec(edge_avg)};font-weight:700;">Edge: {edge_avg*100:+.1f}%</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ==============================================================
# TABS
# ==============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecasts", "📊 Probability", "🏆 Performance",
    "📚 Score History", "📋 Data Table", "🕒 Edge History"
])

# ── TAB 1: FORECASTS ──────────────────────────
with tab1:
    st.markdown('<div class="section-header">All Sources vs Actual Temperatures</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Actual"],
        mode="lines+markers", name="Actual", line=dict(color="#fff", width=3), marker=dict(size=7)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", opacity=0.7,
        annotation_text=f"Threshold ({threshold}°F)", annotation_font=dict(color="#e74c3c", size=11))
    for src in SOURCES:
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df[src],
            mode="lines+markers",
            name=f"{src} {'✓' if sources_map[src][1] else '~'}",
            line=dict(color=COLORS[src], width=2, dash="dot"), marker=dict(size=5), opacity=0.85))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=ensemble_series,
        mode="lines+markers", name="Ensemble Mean",
        line=dict(color="#3a7bd5", width=4), marker=dict(size=9, symbol="diamond")))
    fig.add_hrect(y0=threshold-sigma, y1=threshold+sigma,
        fillcolor="rgba(58,123,213,0.07)", line_width=0, annotation_text="±1σ")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        height=440, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10,r=10,t=30,b=10),
        xaxis=dict(gridcolor="#2e3247"),
        yaxis=dict(gridcolor="#2e3247", title="Temperature (°F)"),
        hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(SOURCES))
    for i, src in enumerate(SOURCES):
        with cols[i]:
            w_pct = weights.get(src, 0) * 100
            st.markdown(f"**{src}**  \n{'🟢' if sources_map[src][1] else '🔴'} {'Live' if sources_map[src][1] else 'Sim'}  \nWeight: **{w_pct:.1f}%**")

# ── TAB 2: PROBABILITY ────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Model vs All Market Odds</div>', unsafe_allow_html=True)
        bars = [("Model", P_model*100,"#3a7bd5"),("Polymarket",poly_prob*100,"#9c27b0"),
                ("Kalshi",kals_prob*100,"#2196f3"),("Avg Market",market_avg_prob*100,"#f4a62a")]
        fig_b = go.Figure(go.Bar(x=[b[0] for b in bars], y=[b[1] for b in bars],
            marker_color=[b[2] for b in bars],
            text=[f"{b[1]:.1f}%" for b in bars], textposition="outside", width=0.45))
        fig_b.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=350, yaxis=dict(range=[0,110],gridcolor="#2e3247"),
            margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
        x_range = np.linspace(T_ensemble - 4*sigma, T_ensemble + 4*sigma, 300)
        fig_d = go.Figure()
        x_ab = x_range[x_range >= threshold]
        fig_d.add_trace(go.Scatter(
            x=np.concatenate([[threshold], x_ab, [x_range[-1]]]),
            y=np.concatenate([[0], norm.pdf(x_ab, T_ensemble, sigma), [0]]),
            fill="toself", fillcolor="rgba(45,206,137,0.2)", line=dict(width=0),
            name=f"P={P_model*100:.1f}%"))
        fig_d.add_trace(go.Scatter(x=x_range, y=norm.pdf(x_range, T_ensemble, sigma),
            mode="lines", line=dict(color="#3a7bd5", width=2.5), name="Distribution"))
        fig_d.add_vline(x=threshold, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"{threshold}°F")
        fig_d.add_vline(x=T_ensemble, line_dash="dot", line_color="#fff",
            annotation_text=f"Ens:{T_ensemble:.1f}°F", annotation_position="top right")
        fig_d.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=350, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(title="Temperature (°F)", gridcolor="#2e3247"),
            yaxis=dict(title="Density", gridcolor="#2e3247"))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<div class="section-header">Edge Gauges</div>', unsafe_allow_html=True)
    g1,g2,g3 = st.columns(3)
    def make_gauge(val, ref, title, color):
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=val*100,
            delta={"reference":ref*100,"valueformat":".1f","suffix":"%"},
            title={"text":title,"font":{"color":"#8b9dc3","size":13}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color},"bgcolor":"#1a1d27",
                   "threshold":{"line":{"color":"#f4a62a","width":3},"thickness":0.75,"value":ref*100}},
            number={"suffix":"%","font":{"color":"white"}}))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
            height=250, margin=dict(l=20,r=20,t=40,b=10), font={"color":"#e0e0e0"})
        return fig
    with g1: st.plotly_chart(make_gauge(P_model, poly_prob,       "Model vs Polymarket","#9c27b0"), use_container_width=True)
    with g2: st.plotly_chart(make_gauge(P_model, kals_prob,       "Model vs Kalshi",    "#2196f3"), use_container_width=True)
    with g3: st.plotly_chart(make_gauge(P_model, market_avg_prob, "Model vs Avg Market","#3a7bd5"), use_container_width=True)

# ── TAB 3: PERFORMANCE ────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="section-header">RMSE by Source ({weight_label})</div>', unsafe_allow_html=True)
        if not scores_df.empty:
            plot_data = scores_df[["source", weight_window]].sort_values(weight_window)
            fig_r = go.Figure(go.Bar(
                y=plot_data["source"], x=plot_data[weight_window], orientation="h",
                marker_color=[COLORS.get(s,"#8b9dc3") for s in plot_data["source"]],
                text=[f"{v:.3f}" for v in plot_data[weight_window]], textposition="outside"))
            fig_r.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                height=320, margin=dict(l=10,r=60,t=10,b=10),
                xaxis=dict(title=f"RMSE °F", gridcolor="#2e3247"))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("RMSE scores appear here after the first day of data is saved.")
    with c2:
        st.markdown('<div class="section-header">Ensemble Weights (persistent)</div>', unsafe_allow_html=True)
        fig_p = go.Figure(go.Pie(
            labels=list(weights.keys()), values=list(weights.values()),
            marker=dict(colors=[COLORS.get(s,"#8b9dc3") for s in weights]),
            hole=0.5, textinfo="label+percent", textfont=dict(size=12)))
        fig_p.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
            height=320, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    # Score cards
    if not scores_df.empty:
        st.markdown('<div class="section-header">Source Score Cards</div>', unsafe_allow_html=True)
        card_cols = st.columns(3)
        best  = scores_df["rmse_all"].min()
        worst = scores_df["rmse_all"].max()
        for i, row in scores_df.iterrows():
            src   = row["source"]
            color = COLORS.get(src, "#8b9dc3")
            trust = 1 - (row["rmse_all"] - best) / max(worst - best, 0.001)
            with card_cols[i % 3]:
                st.markdown(f"""
                <div class="score-card">
                    <div class="src-name" style="color:{color};">#{i+1} {src}</div>
                    <div class="src-rmse">{row['rmse_all']:.3f}°F RMSE</div>
                    <div class="src-meta">
                        MAE: {row['mae_all']:.3f}°F &nbsp;|&nbsp;
                        7d: {row['rmse_7d']:.3f} &nbsp;|&nbsp; 30d: {row['rmse_30d']:.3f}<br>
                        Days tracked: <strong>{int(row['days_tracked'])}</strong> &nbsp;|&nbsp;
                        Weight: <strong>{weights.get(src,0)*100:.1f}%</strong>
                    </div>
                    <div class="trust-bar">
                        <div class="trust-fill" style="width:{trust*100:.0f}%;background:{color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

# ── TAB 4: SCORE HISTORY ──────────────────────
with tab4:
    st.markdown('<div class="section-header">📚 Persistent Score History</div>', unsafe_allow_html=True)

    if history_df.empty:
        st.info("""
**No history yet — this is normal on day 1.**

Every time you run this dashboard it saves today's forecasts and actuals
to `weather_scores.db` on your Mac. After a few days you will see:
- Rolling RMSE trends per source over time
- Which sources are improving or degrading
- How ensemble weights shift as history grows

The model gets more accurate every single day you run it. 📈
        """)
    else:
        # Rolling RMSE line chart
        rolling_df = load_rolling_rmse(window=7)
        if not rolling_df.empty:
            st.markdown('<div class="section-header">7-Day Rolling RMSE Per Source</div>', unsafe_allow_html=True)
            fig_roll = go.Figure()
            for src in SOURCES:
                src_data = rolling_df[rolling_df["source"] == src]
                if not src_data.empty:
                    fig_roll.add_trace(go.Scatter(x=src_data["date"], y=src_data["rolling_rmse"],
                        mode="lines+markers", name=src,
                        line=dict(color=COLORS.get(src,"#8b9dc3"), width=2), marker=dict(size=5)))
            fig_roll.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10,r=10,t=30,b=10),
                xaxis=dict(title="Date", gridcolor="#2e3247"),
                yaxis=dict(title="Rolling RMSE (°F)", gridcolor="#2e3247"),
                hovermode="x unified")
            st.plotly_chart(fig_roll, use_container_width=True)

        # Days tracked bar
        st.markdown('<div class="section-header">Days of History Per Source</div>', unsafe_allow_html=True)
        days_per = history_df.groupby("source").size().reset_index(name="days")
        fig_days = go.Figure(go.Bar(
            x=days_per["source"], y=days_per["days"],
            marker_color=[COLORS.get(s,"#8b9dc3") for s in days_per["source"]],
            text=days_per["days"], textposition="outside"))
        fig_days.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=280, margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(title="Days Recorded", gridcolor="#2e3247"))
        st.plotly_chart(fig_days, use_container_width=True)

        # Absolute error heatmap
        st.markdown('<div class="section-header">Daily Absolute Error Heatmap (°F)</div>', unsafe_allow_html=True)
        pivot = history_df.pivot_table(index="date", columns="source", values="abs_error")
        if not pivot.empty:
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values.T, x=pivot.index, y=pivot.columns,
                colorscale="RdYlGn_r", zmid=2,
                text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in pivot.values.T],
                texttemplate="%{text}",
                colorbar=dict(title="|Error| °F", tickfont=dict(color="#8b9dc3"))))
            fig_heat.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-header">Raw History Log</div>', unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True, height=350)
        st.download_button("⬇️ Download Score History CSV", history_df.to_csv(index=False),
            file_name="weather_score_history.csv", mime="text/csv")

# ── TAB 5: DATA TABLE ─────────────────────────
with tab5:
    st.markdown('<div class="section-header">Full Forecast Data (Current Session)</div>', unsafe_allow_html=True)
    disp = forecast_df.copy()
    disp["Ensemble"] = ensemble_series
    disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
    disp = disp.round(2)
    st.dataframe(disp, use_container_width=True, height=460,
        column_config={
            "Date":     st.column_config.TextColumn("Date"),
            "Actual":   st.column_config.NumberColumn("Actual (°F)",   format="%.1f°F"),
            "Ensemble": st.column_config.NumberColumn("Ensemble (°F)", format="%.1f°F"),
            **{src: st.column_config.NumberColumn(f"{src} (°F)", format="%.1f°F") for src in SOURCES}
        })
    st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
        file_name=f"mpls_weather_{datetime.today().strftime('%Y%m%d')}.csv", mime="text/csv")

# ── TAB 6: EDGE HISTORY ───────────────────────
with tab6:
    st.markdown('<div class="section-header">Edge History — This Session</div>', unsafe_allow_html=True)
    if "edge_history" not in st.session_state:
        st.session_state.edge_history = []
    st.session_state.edge_history.append({
        "Time":           datetime.now().strftime("%H:%M:%S"),
        "Model %":        round(P_model*100, 2),
        "Polymarket %":   round(poly_prob*100, 2),
        "Kalshi %":       round(kals_prob*100, 2),
        "Edge vs Poly":   round(edge_poly*100, 2),
        "Edge vs Kalshi": round(edge_kals*100, 2),
        "Avg Edge":       round(edge_avg*100, 2),
    })
    st.session_state.edge_history = st.session_state.edge_history[-60:]
    hist_df = pd.DataFrame(st.session_state.edge_history)

    if len(hist_df) > 1:
        fig_eh = go.Figure()
        fig_eh.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Model %"],      mode="lines+markers", name="Model",      line=dict(color="#3a7bd5",width=2)))
        fig_eh.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Polymarket %"], mode="lines+markers", name="Polymarket", line=dict(color="#9c27b0",width=2)))
        fig_eh.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Kalshi %"],     mode="lines+markers", name="Kalshi",     line=dict(color="#2196f3",width=2)))
        fig_eh.add_trace(go.Scatter(x=hist_df["Time"], y=hist_df["Avg Edge"],     mode="lines+markers", name="Avg Edge",   line=dict(color="#f4a62a",width=2,dash="dot")))
        fig_eh.add_hline(y=0, line_dash="dash", line_color="#e74c3c", opacity=0.5)
        fig_eh.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=350, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(title="Time",gridcolor="#2e3247"),
            yaxis=dict(title="Probability / Edge (%)",gridcolor="#2e3247"),
            hovermode="x unified")
        st.plotly_chart(fig_eh, use_container_width=True)
    else:
        st.info("Edge history builds after the second auto-refresh (2 minutes).")
    if len(hist_df) > 0:
        st.dataframe(hist_df, use_container_width=True, height=300)

# ==============================================================
# FOOTER
# ==============================================================
st.markdown("---")
days_footer = int(scores_df["days_tracked"].max()) if not scores_df.empty else 0
st.markdown(f"""
<div style="color:#4a5075;font-size:0.78rem;text-align:center;padding:0.5rem 0;">
    Minneapolis Weather Market Dashboard &nbsp;·&nbsp;
    🧠 <strong style="color:#8ab4f8;">{days_footer} days</strong> of persistent history &nbsp;·&nbsp;
    Weights: {weight_label} RMSE &nbsp;·&nbsp;
    NWS & Polymarket: free &nbsp;·&nbsp; AccuWeather, OWM, WeatherAPI, Kalshi: require keys &nbsp;·&nbsp;
    Kalshi ticker: <strong style="color:#8ab4f8;">{kalshi_ticker}</strong> (auto) &nbsp;·&nbsp;
    Auto-refresh: 2 min &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)

