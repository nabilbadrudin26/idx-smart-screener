import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman Wide-Mode (Tetap Dipertahankan)
st.set_page_config(page_title="AI Institutional Market Scanner", layout="wide")

# Custom CSS Executive Dark & Slate Blue Box (Tampilan yang Lu Suka)
st.markdown("""
    <style>
    .reportview-container { background: #f8fafc; }
    .main-header {
        background-color: #012981;
        padding: 22px;
        border-radius: 6px;
        color: white;
        margin-bottom: 25px;
        text-align: center;
    }
    .kpi-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 6px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .recom-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border-top: 4px solid #10b981;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# BANNER UTAMA
st.markdown("""
    <div class="main-header">
        <h2 style='margin:0; font-weight:800; letter-spacing: 0.5px;'>INDONESIA STOCK EXCHANGE (IDX) AI QUANT SCANNER</h2>
        <p style='margin:5px 0 0 0; opacity:0.8; font-size:14px;'>LIQUIDITY-PROTECTED BOOSTING ENGINE | MULTI-INTERVAL RADAR | ANTI-NOISE ALGORITHM</p>
    </div>
""", unsafe_allow_html=True)

# SIDEBAR PANEL KONTROL QUANT
st.sidebar.markdown("### 🛰️ CONFIG RADAR AI")

IHSG_ALPHA_BASKET = [
    # --- 100 Saham Sebelumnya (Campuran Blue-chip, Mid-cap, & Likuid) ---
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK",
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "PGAS.JK", "GOTO.JK",
    "BRIS.JK", "KLBF.JK", "MDKA.JK", "ANTM.JK", "INCO.JK",
    "BREN.JK", "PTRO.JK", "TPIA.JK", "BNBR.JK", "ARTO.JK",
    "CDIA.JK", "BUMI.JK", "BRPT.JK", "CUAN.JK", "TOWR.JK",
    "BLIB.JK", "UNTR.JK", "AALI.JK", "AUTO.JK", "LSIP.JK",
    "BSDE.JK", "INKP.JK", "TKIM.JK", "DSSA.JK", "ADMR.JK",
    "LPKR.JK", "SILO.JK", "MLPL.JK", "AMMN.JK", "CPIN.JK",
    "JPFA.JK", "SIDO.JK", "HMSP.JK", "GGRM.JK", "MYOR.JK",
    "ISAT.JK", "EXCL.JK", "MTEL.JK", "BDMN.JK", "BNGA.JK",
    "BBTN.JK", "NISP.JK", "SMGR.JK", "INTP.JK", "CTRA.JK",
    "PWON.JK", "SMRA.JK", "JSMR.JK", "MEDC.JK", "AKRA.JK",
    "HRUM.JK", "BYAN.JK", "MIKA.JK", "HEAL.JK", "ACES.JK",
    "MAPI.JK", "MAPA.JK", "ERAA.JK", "CMRY.JK", "ULTJ.JK",
    "ROTI.JK", "TBIG.JK", "BUKA.JK", "EMTK.JK", "SCMA.JK",
    "MNCN.JK", "SRTG.JK", "MBMA.JK", "NCKL.JK", "PGEO.JK",
    "KEEN.JK", "PTPP.JK", "WIKA.JK", "ADHI.JK", "WSKT.JK",
    "ENRG.JK", "PNBN.JK", "BBYB.JK", "WIIM.JK", "KAEF.JK",
    "WEGE.JK", "HILL.JK", "SMMA.JK", "ASRI.JK", "LPPF.JK",

    # --- 50 Saham Tambahan Tahap 1 (Fokus harga di bawah Rp 500) ---
    "BRMS.JK", "DOID.JK", "ELSA.JK", "SMDR.JK", "TMAS.JK",
    "WTON.JK", "PPRE.JK", "APLN.JK", "DILD.JK", "KIJA.JK",
    "BEST.JK", "BKSL.JK", "MDLN.JK", "TOTL.JK", "AGRO.JK",
    "BABP.JK", "BKSW.JK", "PNBS.JK", "NOBU.JK", "AMAR.JK",
    "BBKP.JK", "BMTR.JK", "BHIT.JK", "KPIG.JK", "MARI.JK",
    "VIVA.JK", "WIRG.JK", "BIPI.JK", "DEWA.JK", "IPCM.JK",
    "SOCI.JK", "LEAD.JK", "HOKI.JK", "GOOD.JK", "CLEO.JK",
    "CAMP.JK", "WOOD.JK", "SIMP.JK", "NSSS.JK", "GIAA.JK",
    "SAME.JK", "BMHS.JK", "OMED.JK", "INAF.JK", "PEHA.JK",
    "CARE.JK", "RALS.JK", "SDRA.JK", "PNLG.JK", "META.JK",

    # --- 50 Saham Tambahan Tahap 2 (Fokus harga di bawah Rp 500) ---
    "FREN.JK", "CPRO.JK", "BCAP.JK", "IATA.JK", "ACST.JK",
    "NRCA.JK", "SSIA.JK", "GWSA.JK", "GPRA.JK", "DART.JK",
    "MTLA.JK", "BVIC.JK", "INPC.JK", "BGTG.JK", "MCOR.JK",
    "CFIN.JK", "BIMA.JK", "VRNA.JK", "BWPT.JK", "GZCO.JK",
    "PALM.JK", "JAWA.JK", "WMPP.JK", "WMUU.JK", "DSFI.JK",
    "TRUK.JK", "WEHA.JK", "CMPP.JK", "TAXI.JK", "ABBA.JK",
    "MSKY.JK", "KBLI.JK", "KBLM.JK", "VOKS.JK", "BAJA.JK",
    "GDST.JK", "ISSP.JK", "IGAR.JK", "KDSI.JK", "SPMA.JK",
    "TRST.JK", "ALDO.JK", "POLA.JK", "BSBK.JK", "ZATA.JK",
    "KRYA.JK", "BPFI.JK", "TRAM.JK", "RIMO.JK", "COWL.JK",

    # --- 50 Saham Tambahan Tahap 3 (Fokus harga di atas Rp 500 dan di bawah Rp 1.000) ---
    "BJBR.JK", "BJTM.JK", "TINS.JK", "SSMS.JK", "TAPG.JK",
    "DSNG.JK", "MARK.JK", "RAJA.JK", "ESSA.JK", "AVIA.JK",
    "TPMA.JK", "ASSA.JK", "MPMX.JK", "GJTL.JK", "IRRA.JK",
    "BSIM.JK", "BNLI.JK", "ANJT.JK", "CSRA.JK", "TBLA.JK",
    "MAIN.JK", "LPCK.JK", "JRPT.JK", "PNIN.JK", "MRAT.JK",
    "CSAP.JK", "SMMT.JK", "BALI.JK", "MTDL.JK", "STAA.JK",
    "PSSI.JK", "LION.JK", "TFCO.JK", "TIFA.JK", "TUGU.JK",
    "SPTO.JK", "BTON.JK", "CASA.JK", "ARNA.JK", "RMKE.JK",
    "POWR.JK", "ASGR.JK", "SGER.JK", "TEBE.JK", "RAAM.JK",
    "PYFA.JK", "INRU.JK", "SUNI.JK", "TOBA.JK", "BFIN.JK"
]

opsi_interval = {
    "30 Menit (Hold: 2-4 Jam)": {"int": "30m", "per": "60d", "horizon": "2 s.d. 4 Jam (Intraday)"},
    "1 Jam (Hold: 1-2 Hari)": {"int": "1h", "per": "730d", "horizon": "1 s.d. 2 Hari Bursa"},
    "1 Hari (Hold: 3-10 Hari)": {"int": "1d", "per": "5y", "horizon": "3 s.d. 10 Hari (Swing)"},
    "1 Minggu (Hold: 1-3 Bulan)": {"int": "1wk", "per": "5y", "horizon": "1 s.d. 3 Bulan (Position)"}
}

pilihan_user = st.sidebar.selectbox("PILIH INTERVAL & ESTIMASI PROFIT:", list(opsi_interval.keys()), index=2)
interval_terpilih = opsi_interval[pilihan_user]["int"]
periode_terpilih = opsi_interval[pilihan_user]["per"]
horizon_waktu = opsi_interval[pilihan_user]["horizon"]

st.sidebar.markdown("---")
st.sidebar.markdown("### 💸 TRANSACTION FRICTION")
biaya_broker = st.sidebar.slider("FEES BROKER TOTAL (%)", 0.00, 0.50, 0.20, step=0.05) / 100
slippage_input = st.sidebar.slider("PENALTI SLIPPAGE RIIL (%)", 0.00, 0.50, 0.10, step=0.05) / 100

st.sidebar.markdown("---")
tombol_scan = st.sidebar.button("📡 JALANKAN RADAR EMITEN", use_container_width=True)

if tombol_scan:
    progress_bar = st.progress(0)
    status_text = st.empty()

    rekomendasi_list = []

    # SINKRONISASI DATA MAKRO (IHSG, Rupiah, Wall Street)
    status_text.markdown("⏳ **Sinkronisasi Intelijen Makro Eksogenus...**")
    try:
        dji = yf.download("^DJI", period=periode_terpilih, interval="1d", progress=False)
        ihsg = yf.download("^JKSE", period=periode_terpilih, interval="1d", progress=False)
        kurs = yf.download("IDR=X", period=periode_terpilih, interval="1d", progress=False)

        if isinstance(dji.columns, pd.MultiIndex): dji.columns = dji.columns.get_level_values(0)
        if isinstance(ihsg.columns, pd.MultiIndex): ihsg.columns = ihsg.columns.get_level_values(0)
        if isinstance(kurs.columns, pd.MultiIndex): kurs.columns = kurs.columns.get_level_values(0)

        dji['DJI_Lag_Return'] = dji['Close'].pct_change().shift(1)
        ihsg['IHSG_Lag_Return'] = ihsg['Close'].pct_change().shift(1)
        kurs['Kurs_Lag_Return'] = kurs['Close'].pct_change().shift(1)

        dji_c = dji[['DJI_Lag_Return']].dropna()
        ihsg_c = ihsg[['IHSG_Lag_Return']].dropna()
        kurs_c = kurs[['Kurs_Lag_Return']].dropna()

        dji_c.index = dji_c.index.date
        ihsg_c.index = ihsg_c.index.date
        kurs_c.index = kurs_c.index.date
    except Exception as macro_err:
        st.error(f"Gagal mengunduh parameter makro bursa: {macro_err}")
        dji_c, ihsg_c, kurs_c = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- SCANNING PIPELINE ANTI-NOISE ---
    for idx, ticker in enumerate(IHSG_ALPHA_BASKET):
        status_text.markdown(f"**Memindai & Menganalisis Likuiditas:** `{ticker}`... ({idx+1}/{len(IHSG_ALPHA_BASKET)})")
        progress_bar.progress(int((idx + 1) / len(IHSG_ALPHA_BASKET) * 100))

        try:
            # LAPIS 1: PROTEKSI KETAT LIKUIDITAS MINIMAL 750 JUTA RUPIAH (DIHITUNG BERDASARKAN RATA-RATA HARIAN)
            data_likuiditas = yf.download(ticker, period="5d", interval="1d", progress=False)
            if isinstance(data_likuiditas.columns, pd.MultiIndex): data_likuiditas.columns = data_likuiditas.columns.get_level_values(0)

            if data_likuiditas.empty or len(data_likuiditas) < 3:
                continue

            rata_rata_turnover_harian = (data_likuiditas['Close'] * data_likuiditas['Volume']).mean()

            # Download data utama sesuai interval pilihan user
            data = yf.download(ticker, period=periode_terpilih, interval=interval_terpilih, progress=False)
            if data.empty or len(data) < 100:
                continue

            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            # Integrasi Fitur Makro
            data['Tanggal'] = data.index.date
            if not dji_c.empty: data = data.join(dji_c, on='Tanggal')
            if not ihsg_c.empty: data = data.join(ihsg_c, on='Tanggal')
            if not kurs_c.empty: data = data.join(kurs_c, on='Tanggal')

            data['DJI_Lag_Return'] = data['DJI_Lag_Return'].ffill().fillna(0)
            data['IHSG_Lag_Return'] = data['IHSG_Lag_Return'].ffill().fillna(0)
            data['Kurs_Lag_Return'] = data['Kurs_Lag_Return'].ffill().fillna(0)

            # Target & Basis Teknikal
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data['SMA_10'] = data.ta.sma(length=10)
            data['SMA_50'] = data.ta.sma(length=50)

            macd_df = data.ta.macd(fast=12, slow=26, signal=9)
            col_macdh = [c for c in macd_df.columns if 'MACDh_' in c][0]
            data['Norm_MACDh'] = macd_df[col_macdh] / data['Close']

            data['Norm_ATR'] = data.ta.atr(length=14) / data['Close']
            data['ATR_Raw'] = data.ta.atr(length=14)

            bb_df = data.ta.bbands(length=20, std=2)
            col_bbp = [c for c in bb_df.columns if 'BBP_' in c][0]
            data['BBP_20'] = bb_df[col_bbp]

            res_obv = data.ta.obv()
            data['OBV'] = res_obv if isinstance(res_obv, pd.Series) else res_obv.iloc[:, 0]

            data['Return_Interval'] = data['Close'].pct_change()
            data['Dist_SMA10'] = (data['Close'] - data['SMA_10']) / data['SMA_10']
            data['Dist_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
            data['OBV_EMA_Dist'] = (data['OBV'] - data['OBV'].ewm(span=10).mean()) / (data['Volume'] + 1e-8)
            data['Return_Lag1'] = data['Return_Interval'].shift(1)
            data['Return_Lag2'] = data['Return_Interval'].shift(2)

            res_rsi = data.ta.rsi(length=14)
            data['RSI_14_Norm'] = (res_rsi if isinstance(res_rsi, pd.Series) else res_rsi.iloc[:, 0]) / 100.0
            data['Rolling_Volatility_14'] = data['Return_Interval'].rolling(14).std()

            data = data.dropna()

            fitur = [
                'Return_Interval', 'Dist_SMA10', 'Dist_SMA50', 'Norm_MACDh', 'Norm_ATR',
                'Return_Lag1', 'Return_Lag2', 'OBV_EMA_Dist', 'BBP_20',
                'DJI_Lag_Return', 'IHSG_Lag_Return', 'Kurs_Lag_Return',
                'RSI_14_Norm', 'Rolling_Volatility_14'
            ]

            X = data[fitur]
            y = data['Target']

            split_idx = int(len(data) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # LAPIS 2: MENAIKKAN L2 REGULARIZATION UNTUK MEREDAM NOISE HARGA SAHAM SEPI
            model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=5,
                learning_rate=0.04,
                l2_regularization=2.5,  # Diperketat dari 1.5 ke 2.5 agar anti-overfitting
                random_state=42
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc_lokal = accuracy_score(y_test, preds)

            data_terakhir = data.tail(1)

            # KEPUTUSAN KETAT RADAR JIKA SAHAM LOLOS ATAU GAGAL SKRINING LIKUIDITAS Rp 750 JUTA
            if rata_rata_turnover_harian < 750000000:
                prob_naik = 0.0  # Paksa probabilitas menjadi nol demi mengamankan modal
                aksi_final = "⚠️ LOW LIQUIDITY"
            else:
                prob_besok = model.predict_proba(data_terakhir[fitur])[0]
                prob_naik = prob_besok[1]
                aksi_final = "⚠️ HOLD / WAIT" if prob_naik < 0.56 else "🔥 STRONG BUY"

            harga_sekarang = data_terakhir['Close'].iloc[0]
            atr_sekarang = data_terakhir['ATR_Raw'].iloc[0]

            # Hitung Strategi Alokasi Uang Kelly Criterion
            b_ratio = 1.5
            formula_kelly = (prob_naik * b_ratio - (1.0 - prob_naik)) / b_ratio
            alokasi_kelly = max(0.0, min(formula_kelly * 0.5, 0.15)) * 100

            stop_loss = harga_sekarang - (2.0 * atr_sekarang)
            target_profit = harga_sekarang + (3.0 * atr_sekarang)

            rekomendasi_list.append({
                "KODE": ticker,
                "AI CONFIDENCE": prob_naik * 100,
                "AKURASI HISTORY": acc_lokal * 100,
                "AKSI": aksi_final,
                "HARGA MASUK": int(harga_sekarang),
                "TARGET PROFIT (TP)": int(target_profit),
                "STOP LOSS (SL)": int(stop_loss),
                "ALOKASI KELLY": f"{alokasi_kelly:.1f}%" if aksi_final == "🔥 STRONG BUY" else "0.0%",
                "DAILY TURNOVER": rata_rata_turnover_harian
            })
        except:
            continue

    status_text.empty()

    # --- RENDER DASHBOARD (KONSISTEN 100% DENGAN TAMPILAN ASLI) ---
    df_hasil = pd.DataFrame(rekomendasi_list)

    if not df_hasil.empty:
        df_buy = df_hasil[df_hasil['AKSI'] == "🔥 STRONG BUY"].sort_values(by="AI CONFIDENCE", ascending=False)
        # Gabungkan Wait dan Low Liquidity ke tabel bawah agar visual rapi
        df_hold = df_hasil[df_hasil['AKSI'] != "🔥 STRONG BUY"].sort_values(by="AI CONFIDENCE", ascending=False)

        st.markdown(f"### 🎯 HASIL RADAR UTAMA: SAHAM DIREKOMENDASIKAN (HORIZON: {horizon_waktu.upper()})")
        st.markdown(f"> **Anti-Noise Engine Status:** Skrining 44 emiten selesai. Saham dengan turnover harian di bawah Rp 750 Juta otomatis diblokir dari sinyal beli untuk mencegah manipulasi harga semu.")

        if df_buy.empty:
            st.warning("🚨 Tidak ada saham dari radar yang memenuhi kriteria beli hari ini. Saringan ketat likuiditas & volatilitas mendeteksi risiko tinggi. Amankan cash Anda.")
        else:
            st.markdown("#### 🏆 TOP 3 RADAR MATCHES (PROBABILITAS GRADIENT-BOOSTING TERTINGGI)")
            kol_kartu = st.columns(min(3, len(df_buy)))
            for i, kartu in enumerate(kol_kartu):
                with kartu:
                    row = df_buy.iloc[i]
                    st.markdown(f"""
                    <div class="recom-card">
                        <span style="background-color: #10b981; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">RANK #{i+1}</span>
                        <h3 style="margin: 10px 0 5px 0; color: #0f172a;">{row['KODE']}</h3>
                        <p style="margin: 0; font-size: 13px; color: #64748b;">Keyakinan Tren Naik: <b style="color:#10b981; font-size:16px;">{row['AI CONFIDENCE']:.1f}%</b></p>
                        <p style="margin: 0; font-size: 11px; color: #94a3b8;">Turnover Harian: Rp {row['DAILY TURNOVER']:,.0f}</p>
                        <hr style="margin: 12px 0; border: 0; border-top: 1px solid #e2e8f0;">
                        <table style="width:100%; font-size:12.5px; color:#334155;">
                            <tr><td><b>Harga Entry:</b></td><td style="text-align:right;">Rp {row['HARGA MASUK']}</td></tr>
                            <tr><td><b>Target Profit:</b></td><td style="text-align:right; color:#10b981;"><b>Rp {row['TARGET PROFIT (TP)']}</b></td></tr>
                            <tr><td><b>Stop Loss:</b></td><td style="text-align:right; color:#ef4444;"><b>Rp {row['STOP LOSS (SL)']}</b></td></tr>
                            <tr><td><b>Porsi Modal:</b></td><td style="text-align:right; color:#3b82f6;"><b>{row['ALOKASI KELLY']}</b></td></tr>
                        </table>
                        <p style="margin: 8px 0 0 0; font-size: 11px; color: #94a3b8; text-align: center;">Estimasi Rentang Waktu Profit: <br><b>{horizon_waktu}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 DATABASE SKRINING GABUNGAN RADAR BUY SAHAM")
            st.dataframe(df_buy[["KODE", "AI CONFIDENCE", "AKURASI HISTORY", "HARGA MASUK", "TARGET PROFIT (TP)", "STOP LOSS (SL)", "ALOKASI KELLY"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### ⏳ DAFTAR EMITEN STATUS WAIT / HOLD / LOW LIQUIDITY (BELUM MOMENTUM)")
        st.dataframe(df_hold[["KODE", "AI CONFIDENCE", "AKURASI HISTORY", "HARGA MASUK", "AKSI"]], use_container_width=True, hide_index=True)

    else:
        st.error("Gagal melakukan proses skrining pasar. Periksa koneksi data emiten.")