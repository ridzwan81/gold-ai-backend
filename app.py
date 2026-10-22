"""
FastAPI Live LightGBM Cloud Prediction Microservice for XAUUSD
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime

app = FastAPI(title="XAUUSD LightGBM Live Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features
model = lgb.Booster(model_file="xauusd_lgbm_model.txt")
feature_cols = joblib.load("model_features.pkl")

def fetch_and_prepare_features():
    gold = yf.download(tickers="GC=F", interval="1h", period="5d", auto_adjust=True, progress=False)
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = [col[0].lower() for col in gold.columns]
    else:
        gold.columns = [c.lower() for c in gold.columns]

    gold = gold[['open', 'high', 'low', 'close', 'volume']].dropna()
    data = gold.copy()

    data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['trend_spread_fast'] = (data['ema_9'] - data['ema_21']) / data['close']
    data['trend_spread_slow'] = (data['ema_21'] - data['ema_50']) / data['close']

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi_14'] = 100 - (100 / (1 + rs))

    data['return_1h'] = data['close'].pct_change(1)
    data['return_3h'] = data['close'].pct_change(3)
    data['return_6h'] = data['close'].pct_change(6)

    hl = data['high'] - data['low']
    hc = (data['high'] - data['close'].shift()).abs()
    lc = (data['low'] - data['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(window=14).mean()
    data['atr_rel'] = data['atr_14'] / data['close']

    rolling_mean = data['close'].rolling(20).mean()
    rolling_std = data['close'].rolling(20).std()
    bb_upper = rolling_mean + (2 * rolling_std)
    bb_lower = rolling_mean - (2 * rolling_std)
    data['bb_pos'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    data.dropna(inplace=True)
    latest_row = data.iloc[[-1]]
    latest_features = latest_row[feature_cols]
    latest_close = float(latest_row['close'].iloc[0])
    latest_atr = float(latest_row['atr_14'].iloc[0])

    return latest_features, latest_close, latest_atr

@app.get("/")
def home():
    return {"status": "online", "model": "XAUUSD LightGBM H1", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/signal")
def get_live_signal():
    try:
        features, spot_price, atr = fetch_and_prepare_features()
        probabilities = model.predict(features)[0]
        best_class = int(np.argmax(probabilities))
        confidence = float(probabilities[best_class]) * 100

        tp_distance = round(max(atr * 1.5, 8.0), 2)
        sl_distance = round(tp_distance / 2.0, 2)
        tp_pips = int(round(tp_distance * 10))
        sl_pips = int(round(sl_distance * 10))

        if best_class == 2:
            action = "BUY"
            entry = round(spot_price, 2)
            tp = round(entry + tp_distance, 2)
            sl = round(entry - sl_distance, 2)
        elif best_class == 0:
            action = "SELL"
            entry = round(spot_price, 2)
            tp = round(entry - tp_distance, 2)
            sl = round(entry + sl_distance, 2)
        else:
            action = "WAIT"
            entry = round(spot_price, 2)
            tp = round(entry + tp_distance, 2)
            sl = round(entry - sl_distance, 2)

        return {
            "pair": "XAUUSD",
            "timeframe": "H1",
            "action": action,
            "confidence_pct": round(confidence, 1),
            "spot_price": round(spot_price, 2),
            "entry": entry,
            "take_profit": tp,
            "stop_loss": sl,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "risk_reward": "1:2.0",
            "atr": round(atr, 2),
            "probabilities": {
                "sell": round(float(probabilities[0]) * 100, 1),
                "neutral": round(float(probabilities[1]) * 100, 1),
                "buy": round(float(probabilities[2]) * 100, 1)
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
