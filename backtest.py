import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURASI BACKTEST V15.0 (GLOBAL QUANT ENGINE)
# ==========================================
IHSG_ALPHA_BASKET = [
    # --- Saham Liquid Utama (LQ45 & High Volume Basket) ---
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "ICBP.JK", 
    "INDF.JK", "AMRT.JK", "ADRO.JK", "PTBA.JK", "ITMG.JK", "PGAS.JK", "BRIS.JK", "KLBF.JK", 
    "MDKA.JK", "ANTM.JK", "INCO.JK", "BREN.JK", "PTRO.JK", "TPIA.JK", "ARTO.JK", "BRPT.JK", 
    "CUAN.JK", "TOWR.JK", "UNTR.JK", "AALI.JK", "AUTO.JK", "LSIP.JK", "BSDE.JK", "INKP.JK", 
    "TKIM.JK", "DSSA.JK", "ADMR.JK", "AMMN.JK", "CPIN.JK", "JPFA.JK", "SIDO.JK", "HMSP.JK", 
    "GGRM.JK", "MYOR.JK", "ISAT.JK", "EXCL.JK", "MTEL.JK", "BDMN.JK", "BNGA.JK", "BBTN.JK", 
    "NISP.JK", "SMGR.JK", "INTP.JK", "CTRA.JK", "PWON.JK", "SMRA.JK", "JSMR.JK", "MEDC.JK", 
    "AKRA.JK", "HRUM.JK", "BYAN.JK", "MIKA.JK", "HEAL.JK", "ACES.JK", "MAPI.JK", "MAPA.JK", 
    "CMRY.JK", "ULTJ.JK", "TBIG.JK", "EMTK.JK", "SCMA.JK", "SRTG.JK", "MBMA.JK", "NCKL.JK", 
    "PGEO.JK", "PTPP.JK", "WIKA.JK", "ADHI.JK", "ENRG.JK", "PNBN.JK", "WIIM.JK", "HILL.JK", 
    "LPPF.JK", "BRMS.JK", "DOID.JK", "ELSA.JK", "SMDR.JK", "TOTL.JK", "AGRO.JK", "WIRG.JK", 
    "DEWA.JK", "CLEO.JK", "WOOD.JK", "SAME.JK", "CARE.JK", "RALS.JK", "BJBR.JK", "BJTM.JK", 
    "TINS.JK", "SSMS.JK", "TAPG.JK", "DSNG.JK", "MARK.JK", "RAJA.JK", "ESSA.JK", "AVIA.JK", 
    "TPMA.JK", "ASSA.JK", "MPMX.JK", "GJTL.JK", "CSRA.JK", "TBLA.JK", "MAIN.JK", "JRPT.JK", 
    "PNIN.JK", "MTDL.JK", "STAA.JK", "PSSI.JK", "TUGU.JK", "ARNA.JK", "RMKE.JK", "POWR.JK", 
    "SGER.JK", "TOBA.JK", "BFIN.JK"
]

# FILTER LIKUIDITAS INSTITUSIONAL
MIN_TURNOVER = 5_000_000_000  # Minimal Turnover Rp 5 Miliar / hari
MIN_PRICE = 200               # Minimal harga Rp 200 (Bebas Saham Gorengan)
TRANSACTION_FEE = 0.003        # Fee & Slippage 0.3%
MAX_HOLD_DAYS = 7              # Maksimal hold 7 hari bursa
MIN_AI_PROBABILITY = 0.58      # Threshold Keyakinan AI Minimal 58%

print("==================================================")
print("🚀 RUNNING QUANT BACKTEST V15.0 (GLOBAL QUANT ENGINE)")
print(f"📦 Total Target Basket: {len(IHSG_ALPHA_BASKET)} Tickers")
print("==================================================\n")

# 1. DOWNLOAD DATA IHSG & REGIME MARKER
ihsg_data = yf.download("^JKSE", period="2y", interval="1d", progress=False)
if isinstance(ihsg_data.columns, pd.MultiIndex):
    ihsg_data.columns = ihsg_data.columns.get_level_values(0)

ihsg_data.index = ihsg_data.index.tz_localize(None)
ihsg_sma50 = ihsg_data.ta.sma(length=50)
ihsg_ret = ihsg_data["Close"].pct_change()
ihsg_status = pd.DataFrame(
    {
        "IHSG_Bullish": (ihsg_data["Close"] > ihsg_sma50).astype(int),
        "IHSG_Return": ihsg_ret
    },
    index=ihsg_data.index,
)

# POTONG TANGGAL UTAMA (70% TRAIN / 30% TEST GLOBAL TIME-SPLIT)
unique_dates = ihsg_data.index.sort_values()
split_date_idx = int(len(unique_dates) * 0.70)
CUTOFF_DATE = unique_dates[split_date_idx]

print(f"📅 Tanggal Pemisah Out-of-Sample: {CUTOFF_DATE.strftime('%Y-%m-%d')}")
print("⏳ Mengumpulkan Feature & Membangun Dataset Global...")

all_ticker_dfs = {}
train_frames = []

fitur_cols = [
    "Return_Interval", "Rel_Strength", "Dist_SMA20", "Dist_SMA50", 
    "Norm_MACDh", "Norm_ATR", "Vol_Ratio", "OBV_EMA_Dist",
    "BBP_20", "RSI_14_Norm", "Rolling_Volatility_14", "IHSG_Bullish"
]

# 2. FEATURE ENGINEERING DAN PELEBURAN DATA (DATA POOLING)
for symbol in IHSG_ALPHA_BASKET:
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty or len(df) < 100:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = df.index.tz_localize(None)
        df = df.join(ihsg_status, how="left")
        df["IHSG_Bullish"] = df["IHSG_Bullish"].ffill().fillna(0)
        df["IHSG_Return"] = df["IHSG_Return"].ffill().fillna(0)

        df["SMA_20"] = df.ta.sma(length=20)
        df["SMA_50"] = df.ta.sma(length=50)
        df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()
        df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-8)
        
        df["ATR_Raw"] = df.ta.atr(length=14)
        df["Norm_ATR"] = df["ATR_Raw"] / df["Close"]

        # Target AI Kriteria Risk-Adjusted (+2.0 ATR Win vs -1.0 ATR Risk dalam 5 hari)
        future_max_high = df["High"].shift(-5).rolling(5).max()
        future_min_low = df["Low"].shift(-5).rolling(5).min()

        df["Target"] = (
            ((future_max_high - df["Close"]) >= (2.0 * df["ATR_Raw"])) &
            ((df["Close"] - future_min_low) < (1.0 * df["ATR_Raw"]))
        ).astype(int)

        macd_df = df.ta.macd(fast=12, slow=26, signal=9)
        if macd_df is not None:
            col_macdh = [c for c in macd_df.columns if "MACDh_" in c][0]
            df["Norm_MACDh"] = macd_df[col_macdh] / df["Close"]
        else:
            df["Norm_MACDh"] = 0

        bb_df = df.ta.bbands(length=20, std=2)
        if bb_df is not None:
            col_bbp = [c for c in bb_df.columns if "BBP_" in c][0]
            df["BBP_20"] = bb_df[col_bbp]
        else:
            df["BBP_20"] = 0.5

        res_obv = df.ta.obv()
        df["OBV"] = (res_obv if isinstance(res_obv, pd.Series) else res_obv.iloc[:, 0])

        df["Return_Interval"] = df["Close"].pct_change()
        df["Rel_Strength"] = df["Return_Interval"] - df["IHSG_Return"]
        df["Dist_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
        df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
        df["OBV_EMA_Dist"] = (df["OBV"] - df["OBV"].ewm(span=10).mean()) / (df["Volume"] + 1e-8)
        
        res_rsi = df.ta.rsi(length=14)
        df["RSI_14_Norm"] = (res_rsi if isinstance(res_rsi, pd.Series) else res_rsi.iloc[:, 0]) / 100.0
        df["Rolling_Volatility_14"] = df["Return_Interval"].rolling(14).std()
        df["Turnover_5D"] = (df["Close"] * df["Volume"]).rolling(5).mean()

        df = df.dropna()
        all_ticker_dfs[symbol] = df

        train_part = df[df.index < CUTOFF_DATE]
        if not train_part.empty and train_part["Target"].nunique() > 1:
            train_frames.append(train_part)
    except Exception:
        pass

# 3. TRAINING KONSISTEN SATU MODEL GLOBAL
print("🧠 Memulai Pelatihan Single Global AI Model...")
full_train_df = pd.concat(train_frames, ignore_index=True)
X_global_train = full_train_df[fitur_cols]
y_global_train = full_train_df["Target"]

global_model = HistGradientBoostingClassifier(
    max_iter=150,
    max_depth=5,
    learning_rate=0.03,
    l2_regularization=5.0,
    class_weight="balanced",
    random_state=42,
)
global_model.fit(X_global_train, y_global_train)
print(f"✅ Model Global Berhasil Dilatih dengan {len(X_global_train):,} Baris Data Sample!\n")

# 4. SIMULASI TRADING ON OUT-OF-SAMPLE TEST DATA
all_trades = []

for symbol, df in all_ticker_dfs.items():
    test_data = df[df.index >= CUTOFF_DATE].copy()
    if len(test_data) < 10:
        continue

    X_test = test_data[fitur_cols]
    test_data["Prob_Naik"] = global_model.predict_proba(X_test)[:, 1]

    in_trade = False
    entry_price = 0
    tp_price = 0
    sl_price = 0
    max_price_seen = 0
    entry_date = None
    days_in_trade = 0
    atr_at_entry = 0

    for i in range(len(test_data) - 1):
        row = test_data.iloc[i]

        if in_trade:
            days_in_trade += 1
            curr_high = test_data.iloc[i + 1]["High"]
            curr_low = test_data.iloc[i + 1]["Low"]
            curr_close = test_data.iloc[i + 1]["Close"]

            max_price_seen = max(max_price_seen, curr_high)
            
            # Dynamic Trailing Stop aktif setelah floating profit >= 1.0x ATR
            if max_price_seen >= entry_price + (1.0 * atr_at_entry):
                trailing_sl = max_price_seen - (1.2 * atr_at_entry)
                effective_sl = max(sl_price, trailing_sl)
            else:
                effective_sl = sl_price

            exit_price = None
            result_type = None

            if curr_high >= tp_price:
                exit_price = tp_price
                result_type = "WIN (TP)"
            elif curr_low <= effective_sl:
                exit_price = effective_sl
                result_type = "WIN (TS)" if exit_price > entry_price else "LOSS (SL)"
            elif days_in_trade >= MAX_HOLD_DAYS:
                exit_price = curr_close
                result_type = "TIME EXIT"

            if exit_price:
                pnl_pct = ((exit_price - entry_price) / entry_price) - TRANSACTION_FEE
                all_trades.append({
                    "Ticker": symbol,
                    "Entry_Date": entry_date,
                    "Exit_Date": test_data.index[i + 1],
                    "Entry_Price": entry_price,
                    "Exit_Price": exit_price,
                    "PnL_Pct": pnl_pct,
                    "Result": result_type,
                })
                in_trade = False
                days_in_trade = 0

        else:
            is_market_bullish = row["IHSG_Bullish"] == 1
            is_trend_ok = row["Close"] > row["SMA_20"]
            is_volume_ok = row["Volume"] > row["Vol_SMA20"]
            is_ai_ok = row["Prob_Naik"] >= MIN_AI_PROBABILITY

            if (
                is_market_bullish
                and is_trend_ok
                and is_volume_ok
                and is_ai_ok
                and row["Turnover_5D"] >= MIN_TURNOVER
                and row["Close"] >= MIN_PRICE
            ):
                in_trade = True
                entry_price = test_data.iloc[i + 1]["Open"]
                atr_at_entry = row["ATR_Raw"]
                max_price_seen = entry_price

                # RASIO RISK-REWARD 2:1 (TP 2.0x ATR vs SL 1.0x ATR)
                tp_price = entry_price + (2.0 * atr_at_entry)
                sl_price = entry_price - (1.0 * atr_at_entry)
                entry_date = test_data.index[i + 1]
                days_in_trade = 0

print("📊 ================= EVALUASI KINERJA BACKTEST V15.0 =================")
trades_df = pd.DataFrame(all_trades)

if trades_df.empty:
    print("🚨 Tidak ada sinyal yang lolos kriteria ketat Model Global V15.0.")
else:
    total_trades = len(trades_df)
    wins = trades_df[trades_df["PnL_Pct"] > 0]
    losses = trades_df[trades_df["PnL_Pct"] <= 0]

    win_rate = (len(wins) / total_trades) * 100
    total_return = trades_df["PnL_Pct"].sum() * 100
    avg_trade = trades_df["PnL_Pct"].mean() * 100

    gross_profit = wins["PnL_Pct"].sum()
    gross_loss = abs(losses["PnL_Pct"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.nan

    trades_df["Cumulative"] = (1 + trades_df["PnL_Pct"]).cumprod()
    peak = trades_df["Cumulative"].cummax()
    drawdown = (trades_df["Cumulative"] - peak) / peak
    max_drawdown = drawdown.min() * 100

    print(f"Total Eksekusi Signal : {total_trades} Transaksi")
    print(f"Win Rate               : {win_rate:.2f}% ({len(wins)} Win / {len(losses)} Loss)")
    print(f"Profit Factor          : {profit_factor:.2f}")
    print(f"Total Return Kumulatif : {total_return:.2f}%")
    print(f"Rata-rata Return/Trade : {avg_trade:.2f}%")
    print(f"Maximum Drawdown (MaxDD): {max_drawdown:.2f}%")
    print("=================================================================\n")

    if not wins.empty:
        print("🏆 TOP 5 TRANSAKSI PALING CUAN:")
        print(
            trades_df.sort_values(by="PnL_Pct", ascending=False)[
                ["Ticker", "Entry_Date", "PnL_Pct", "Result"]
            ]
            .head(5)
            .to_string(index=False)
        )