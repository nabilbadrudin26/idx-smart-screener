import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURASI BACKTEST V13.1 (OPTIMIZED BASKET)
# ==========================================
IHSG_ALPHA_BASKET = [
    # --- 100 Saham Utama (Blue-chip, Mid-cap, & Likuid) ---
    "BBCA.JK",
    "BBRI.JK",
    "BMRI.JK",
    "BBNI.JK",
    "TLKM.JK",
    "ASII.JK",
    "UNVR.JK",
    "ICBP.JK",
    "INDF.JK",
    "AMRT.JK",
    "ADRO.JK",
    "PTBA.JK",
    "ITMG.JK",
    "PGAS.JK",
    "GOTO.JK",
    "BRIS.JK",
    "KLBF.JK",
    "MDKA.JK",
    "ANTM.JK",
    "INCO.JK",
    "BREN.JK",
    "PTRO.JK",
    "TPIA.JK",
    "BNBR.JK",
    "ARTO.JK",
    "CDIA.JK",
    "BUMI.JK",
    "BRPT.JK",
    "CUAN.JK",
    "TOWR.JK",
    "UNTR.JK",
    "AALI.JK",
    "AUTO.JK",
    "LSIP.JK",
    "BSDE.JK",
    "INKP.JK",
    "TKIM.JK",
    "DSSA.JK",
    "ADMR.JK",
    "LPKR.JK",
    "SILO.JK",
    "MLPL.JK",
    "AMMN.JK",
    "CPIN.JK",
    "JPFA.JK",
    "SIDO.JK",
    "HMSP.JK",
    "GGRM.JK",
    "MYOR.JK",
    "ISAT.JK",
    "EXCL.JK",
    "MTEL.JK",
    "BDMN.JK",
    "BNGA.JK",
    "BBTN.JK",
    "NISP.JK",
    "SMGR.JK",
    "INTP.JK",
    "CTRA.JK",
    "PWON.JK",
    "SMRA.JK",
    "JSMR.JK",
    "MEDC.JK",
    "AKRA.JK",
    "HRUM.JK",
    "BYAN.JK",
    "MIKA.JK",
    "HEAL.JK",
    "ACES.JK",
    "MAPI.JK",
    "MAPA.JK",
    "ERAA.JK",
    "CMRY.JK",
    "ULTJ.JK",
    "ROTI.JK",
    "TBIG.JK",
    "BUKA.JK",
    "EMTK.JK",
    "SCMA.JK",
    "MNCN.JK",
    "SRTG.JK",
    "MBMA.JK",
    "NCKL.JK",
    "PGEO.JK",
    "KEEN.JK",
    "PTPP.JK",
    "WIKA.JK",
    "ADHI.JK",
    "WSKT.JK",
    "ENRG.JK",
    "PNBN.JK",
    "BBYB.JK",
    "WIIM.JK",
    "KAEF.JK",
    "WEGE.JK",
    "HILL.JK",
    "SMMA.JK",
    "ASRI.JK",
    "LPPF.JK",
    # --- 50 Saham Tambahan Tahap 1 ---
    "BRMS.JK",
    "DOID.JK",
    "ELSA.JK",
    "SMDR.JK",
    "TMAS.JK",
    "WTON.JK",
    "PPRE.JK",
    "APLN.JK",
    "DILD.JK",
    "KIJA.JK",
    "BEST.JK",
    "BKSL.JK",
    "MDLN.JK",
    "TOTL.JK",
    "AGRO.JK",
    "BABP.JK",
    "BKSW.JK",
    "PNBS.JK",
    "NOBU.JK",
    "AMAR.JK",
    "BBKP.JK",
    "BMTR.JK",
    "BHIT.JK",
    "KPIG.JK",
    "MARI.JK",
    "VIVA.JK",
    "WIRG.JK",
    "BIPI.JK",
    "DEWA.JK",
    "IPCM.JK",
    "SOCI.JK",
    "LEAD.JK",
    "HOKI.JK",
    "GOOD.JK",
    "CLEO.JK",
    "CAMP.JK",
    "WOOD.JK",
    "SIMP.JK",
    "NSSS.JK",
    "GIAA.JK",
    "SAME.JK",
    "BMHS.JK",
    "OMED.JK",
    "INAF.JK",
    "PEHA.JK",
    "CARE.JK",
    "RALS.JK",
    "SDRA.JK",
    "META.JK",
    # --- 50 Saham Tambahan Tahap 2 ---
    "CPRO.JK",
    "BCAP.JK",
    "IATA.JK",
    "ACST.JK",
    "NRCA.JK",
    "SSIA.JK",
    "GWSA.JK",
    "GPRA.JK",
    "DART.JK",
    "MTLA.JK",
    "BVIC.JK",
    "INPC.JK",
    "BGTG.JK",
    "MCOR.JK",
    "CFIN.JK",
    "BIMA.JK",
    "VRNA.JK",
    "BWPT.JK",
    "GZCO.JK",
    "PALM.JK",
    "JAWA.JK",
    "WMPP.JK",
    "WMUU.JK",
    "DSFI.JK",
    "TRUK.JK",
    "WEHA.JK",
    "CMPP.JK",
    "TAXI.JK",
    "ABBA.JK",
    "MSKY.JK",
    "KBLI.JK",
    "KBLM.JK",
    "VOKS.JK",
    "BAJA.JK",
    "GDST.JK",
    "ISSP.JK",
    "IGAR.JK",
    "KDSI.JK",
    "SPMA.JK",
    "TRST.JK",
    "ALDO.JK",
    "POLA.JK",
    "BSBK.JK",
    "ZATA.JK",
    "KRYA.JK",
    "BPFI.JK",
    "TRAM.JK",
    "RIMO.JK",
    "COWL.JK",
    # --- 50 Saham Tambahan Tahap 3 ---
    "BJBR.JK",
    "BJTM.JK",
    "TINS.JK",
    "SSMS.JK",
    "TAPG.JK",
    "DSNG.JK",
    "MARK.JK",
    "RAJA.JK",
    "ESSA.JK",
    "AVIA.JK",
    "TPMA.JK",
    "ASSA.JK",
    "MPMX.JK",
    "GJTL.JK",
    "IRRA.JK",
    "BSIM.JK",
    "BNLI.JK",
    "ANJT.JK",
    "CSRA.JK",
    "TBLA.JK",
    "MAIN.JK",
    "LPCK.JK",
    "JRPT.JK",
    "PNIN.JK",
    "MRAT.JK",
    "CSAP.JK",
    "SMMT.JK",
    "BALI.JK",
    "MTDL.JK",
    "STAA.JK",
    "PSSI.JK",
    "LION.JK",
    "TFCO.JK",
    "TIFA.JK",
    "TUGU.JK",
    "SPTO.JK",
    "BTON.JK",
    "CASA.JK",
    "ARNA.JK",
    "RMKE.JK",
    "POWR.JK",
    "ASGR.JK",
    "SGER.JK",
    "TEBE.JK",
    "RAAM.JK",
    "PYFA.JK",
    "INRU.JK",
    "SUNI.JK",
    "TOBA.JK",
    "BFIN.JK",
]

# Parameter Dikalibrasi (Sweet Spot Execution)
MIN_TURNOVER = 1_000_000_000  # Dipangkas ke Rp 1 Miliar agar saham mid-small cap terserap
MIN_PRICE = 100  # Dipangkas ke Rp 100 menyesuaikan saham basket < Rp 500
TRANSACTION_FEE = 0.003  # Fee & Slippage 0.3%
MAX_HOLD_DAYS = 10  # Ditingkatkan ke 10 hari bursa (memberi ruang trend berkembang)

print("==================================================")
print("🚀 RUNNING QUANT BACKTEST V13.1 (CALIBRATED TREND & BREAKOUT)")
print(f"📦 Total Target Basket: {len(IHSG_ALPHA_BASKET)} Tickers")
print("==================================================\n")

# 1. DOWNLOAD & CLEAN DATA IHSG INDEX
print("📥 Fetching & Processing Data IHSG Index (^JKSE)...")
ihsg_data = yf.download("^JKSE", period="2y", interval="1d", progress=False)
if isinstance(ihsg_data.columns, pd.MultiIndex):
  ihsg_data.columns = ihsg_data.columns.get_level_values(0)

ihsg_data.index = ihsg_data.index.tz_localize(None)
ihsg_sma50 = ihsg_data.ta.sma(length=50)
ihsg_status = pd.DataFrame(
    {"IHSG_Bullish": (ihsg_data["Close"] > ihsg_sma50).astype(int)},
    index=ihsg_data.index,
)

all_trades = []


def run_backtest_on_ticker(ticker):
  try:
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 100:  # Membutuhkan minimal 100 bar
      return

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    df.index = df.index.tz_localize(None)
    df = df.join(ihsg_status, how="left")
    df["IHSG_Bullish"] = df["IHSG_Bullish"].ffill().fillna(0)

    # Indikator Tren & Volume
    df["SMA_20"] = df.ta.sma(length=20)
    df["SMA_50"] = df.ta.sma(length=50)
    df["Vol_SMA20"] = df.ta.sma(df["Volume"], length=20)

    df["Norm_ATR"] = df.ta.atr(length=14) / df["Close"]
    df["ATR_Raw"] = df.ta.atr(length=14)

    # Level Breakout 5 hari terakhir
    df["High_5D_Max"] = df["High"].shift(1).rolling(5).max()

    # Target Training AI: Kenaikan +2.0x ATR dalam 5 hari
    future_max_high = df["High"].shift(-5).rolling(5).max()
    df["Target"] = (
        (future_max_high - df["Close"]) >= (2.0 * df["ATR_Raw"])
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
    df["OBV"] = (
        res_obv if isinstance(res_obv, pd.Series) else res_obv.iloc[:, 0]
    )

    df["Return_Interval"] = df["Close"].pct_change()
    df["Dist_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
    df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
    df["OBV_EMA_Dist"] = (df["OBV"] - df["OBV"].ewm(span=10).mean()) / (
        df["Volume"] + 1e-8
    )
    df["Return_Lag1"] = df["Return_Interval"].shift(1)
    df["Return_Lag2"] = df["Return_Interval"].shift(2)

    res_rsi = df.ta.rsi(length=14)
    df["RSI_14"] = (
        res_rsi if isinstance(res_rsi, pd.Series) else res_rsi.iloc[:, 0]
    )
    df["RSI_14_Norm"] = df["RSI_14"] / 100.0
    df["Rolling_Volatility_14"] = df["Return_Interval"].rolling(14).std()
    df["Turnover_5D"] = (df["Close"] * df["Volume"]).rolling(5).mean()

    df = df.dropna()
    if len(df) < 50:
      return

    fitur = [
        "Return_Interval",
        "Dist_SMA20",
        "Dist_SMA50",
        "Norm_MACDh",
        "Norm_ATR",
        "Return_Lag1",
        "Return_Lag2",
        "OBV_EMA_Dist",
        "BBP_20",
        "RSI_14_Norm",
        "Rolling_Volatility_14",
    ]

    # Split Data (70% Train / 30% Test Out-of-Sample)
    split_idx = int(len(df) * 0.7)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:].copy()

    if (
        len(train_data) < 30
        or len(test_data) < 10
        or train_data["Target"].nunique() < 2
    ):
      return

    X_train, y_train = train_data[fitur], train_data["Target"]
    X_test = test_data[fitur]

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.03,
        l2_regularization=3.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_data["Prob_Naik"] = model.predict_proba(X_test)[:, 1]

    # Filter AI: Ambil Top 25% Sinyal Terkuat (Quantile 0.75)
    prob_cutoff = test_data["Prob_Naik"].quantile(0.75)

    # Eksekusi Trading
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
        # Trailing SL lebih longgar (1.5x ATR) agar tidak gampang terkena noise
        trailing_sl = max_price_seen - (1.5 * atr_at_entry)
        effective_sl = max(sl_price, trailing_sl)

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
          pnl_pct = (
              (exit_price - entry_price) / entry_price
          ) - TRANSACTION_FEE
          all_trades.append({
              "Ticker": ticker,
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
        # ATURAN ENTRY V13.1:
        # 1. Macro Filter: IHSG > SMA50
        # 2. Medium-Term Trend: Close > SMA50 DAN SMA20 > SMA50
        # 3. Price Breakout: Close >= Level Tertinggi 5 Hari Lalu
        # 4. Volume Surge: Volume >= 1.15x Volume SMA20
        # 5. Top AI Sinyal: Prob_Naik >= Quantile 0.75
        is_ihsg_bull = row["IHSG_Bullish"] == 1
        is_trend_ok = (row["Close"] > row["SMA_50"]) and (
            row["SMA_20"] > row["SMA_50"]
        )
        is_breakout = row["Close"] >= row["High_5D_Max"]
        is_volume_ok = row["Volume"] >= (1.15 * row["Vol_SMA20"])

        if (
            is_ihsg_bull
            and is_trend_ok
            and is_breakout
            and is_volume_ok
            and row["Prob_Naik"] >= prob_cutoff
            and row["Turnover_5D"] >= MIN_TURNOVER
            and row["Close"] >= MIN_PRICE
        ):
          in_trade = True
          entry_price = test_data.iloc[i + 1]["Open"]
          atr_at_entry = row["ATR_Raw"]
          max_price_seen = entry_price

          # RASIO RISK-REWARD 2.3 : 1 (TP 2.8x ATR vs SL 1.2x ATR)
          tp_price = entry_price + (2.8 * atr_at_entry)
          sl_price = entry_price - (1.2 * atr_at_entry)
          entry_date = test_data.index[i + 1]
          days_in_trade = 0

  except Exception as e:
    pass


# Jalankan Engine Backtest
for idx, symbol in enumerate(IHSG_ALPHA_BASKET):
  print(
      f"⏳ Testing [{idx+1}/{len(IHSG_ALPHA_BASKET)}]: {symbol}...", end="\r"
  )
  run_backtest_on_ticker(symbol)

print(
    "\n\n📊 ================= EVALUASI KINERJA BACKTEST V13.1 ================="
)
trades_df = pd.DataFrame(all_trades)

if trades_df.empty:
  print(
      "🚨 Tidak ada sinyal yang memenuhi kriteria filter. Coba kurangi"
      " ketatnya filter."
  )
else:
  total_trades = len(trades_df)
  wins = trades_df[trades_df["PnL_Pct"] > 0]
  losses = trades_df[trades_df["PnL_Pct"] <= 0]

  win_rate = (len(wins) / total_trades) * 100
  total_return = trades_df["PnL_Pct"].sum() * 100
  avg_trade = trades_df["PnL_Pct"].mean() * 100

  gross_profit = wins["PnL_Pct"].sum()
  gross_loss = abs(losses["PnL_Pct"].sum())
  profit_factor = (
      (gross_profit / gross_loss) if gross_loss != 0 else np.nan
  )

  trades_df["Cumulative"] = (1 + trades_df["PnL_Pct"]).cumprod()
  peak = trades_df["Cumulative"].cummax()
  drawdown = (trades_df["Cumulative"] - peak) / peak
  max_drawdown = drawdown.min() * 100

  print(f"Total Eksekusi Signal : {total_trades} Transaksi")
  print(
      f"Win Rate               : {win_rate:.2f}% ({len(wins)} Win /"
      f" {len(losses)} Loss)"
  )
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