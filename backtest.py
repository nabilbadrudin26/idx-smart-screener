import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ENSEMBLE CLASSIFIER V25.0
# ==========================================
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        
        self.model_hgb = HistGradientBoostingClassifier(
            class_weight='balanced',
            l2_regularization=1.2,
            min_samples_leaf=25,
            random_state=self.random_state
        )
        self.model_rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_depth=7, 
            min_samples_leaf=15,
            class_weight='balanced', 
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_hgb.fit(X, y)
        self.model_rf.fit(X, y)
        return self

    def predict_proba(self, X):
        p1 = self.model_hgb.predict_proba(X)
        p2 = self.model_rf.predict_proba(X)
        return (p1 + p2) / 2.0

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.505).astype(int)

# ==========================================
# 2. FEATURE ENGINEERING V25.0
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    
    # ATR (14) & Relative Volatility
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-9)
    
    # Trend Dynamics
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / (df['SMA_50'] + 1e-9)
    df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / (df['SMA_200'] + 1e-9)
    
    # Breakout & Normalized RSI
    df['High_20'] = df['High'].rolling(20).max()
    df['High_20_Ratio'] = (df['Close'] - df['High_20']) / (df['High_20'] + 1e-9)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['RSI_Norm'] = (rsi - 50.0) / 50.0  # Scale between -1 and 1
    
    # Volume Z-Score
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Vol_STD20'] = df['Volume'].rolling(20).std()
    df['Vol_ZScore'] = (df['Volume'] - df['Vol_SMA20']) / (df['Vol_STD20'] + 1e-9)
    
    # Turnover Liquidity Filter
    df['Turnover_20MA'] = (df['Close'] * df['Volume']).rolling(20).mean()
    
    return df

def apply_triple_barrier(df, hold_days=10, tp_mult=2.4, sl_mult=1.3):
    labels = np.zeros(len(df))
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    
    for i in range(len(df) - hold_days):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        entry = close[i]
        tp = entry + (tp_mult * atr[i])
        sl = entry - (sl_mult * atr[i])
        
        label = 0
        for h in range(1, hold_days + 1):
            curr_high = high[i + h]
            curr_low = low[i + h]
            
            if curr_high >= tp:
                label = 1
                break
            if curr_low <= sl:
                label = 0
                break
                
        labels[i] = label
        
    df['Target'] = labels
    return df

# ==========================================
# 3. BACKTEST ENGINE V25.0 (WITH DYNAMIC TRAILING STOP)
# ==========================================
class QuantBacktesterV25:
    def __init__(self, tickers, start_date, end_date, initial_capital=100_000_000):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        
        # Hyperparameters V25.0
        self.prob_threshold = 0.505
        self.max_hold_days = 10
        self.sl_atr_mult = 1.3
        self.tp_atr_mult = 2.4
        self.max_positions = 8
        self.max_daily_entries = 3
        self.risk_per_trade = 0.015  # Terkalibrasi ke 1.5%
        
    def prepare_data(self):
        print("Downloading Data & Engineering Features (V25.0)...")
        self.data = {}
        feature_cols = ['Dist_SMA50', 'Dist_SMA200', 'Vol_ZScore', 'ATR_Pct', 'High_20_Ratio', 'RSI_Norm']
        
        for t in self.tickers:
            df = yf.download(t, start=self.start_date, end=self.end_date, progress=False)
            if df.empty or len(df) < 200:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = calculate_indicators(df)
            df = apply_triple_barrier(df, hold_days=self.max_hold_days, 
                                     tp_mult=self.tp_atr_mult, sl_mult=self.sl_atr_mult)
            
            # Liquidity & Price Filters
            df = df[(df['Turnover_20MA'] >= 300_000_000) & (df['Close'] >= 50)]
            df = df.dropna(subset=feature_cols + ['Target', 'ATR'])
            
            if len(df) > 50:
                self.data[t] = df
            
        self.feature_cols = feature_cols

    def run(self):
        self.prepare_data()
        
        train_dfs, test_dfs = [], {}
        for t, df in self.data.items():
            split_idx = int(len(df) * 0.7)
            train_dfs.append(df.iloc[:split_idx])
            test_dfs[t] = df.iloc[split_idx:]
            
        full_train = pd.concat(train_dfs)
        X_train = full_train[self.feature_cols]
        y_train = full_train['Target']
        
        print(f"Pelatihan Balanced Ensemble Model V25.0 pada {len(X_train)} baris sampel...")
        model = EnsembleClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        print("Menjalankan Simulasi Backtest V25.0...")
        
        all_dates = sorted(list(set.union(*[set(df.index) for df in test_dfs.values()])))
        
        portfolio = {'cash': self.capital, 'positions': {}}
        equity_curve = []
        trades_history = []
        
        for current_date in all_dates:
            # 1. Evaluasi Posisi Aktif & Trailing Stop Update
            active_tickers = list(portfolio['positions'].keys())
            for t in active_tickers:
                pos = portfolio['positions'][t]
                df_t = test_dfs[t]
                
                if current_date not in df_t.index:
                    continue
                    
                row = df_t.loc[current_date]
                curr_price = row['Close']
                curr_high = row['High']
                days_held = (current_date - pos['entry_date']).days
                
                # Dynamic Trailing Stop Activation (+1.0x ATR profit)
                pos['highest_price'] = max(pos['highest_price'], curr_high)
                if (pos['highest_price'] - pos['entry_price']) >= pos['atr']:
                    trailing_sl = pos['highest_price'] - (1.2 * pos['atr'])
                    pos['sl'] = max(pos['sl'], trailing_sl)
                
                hit_sl = row['Low'] <= pos['sl']
                hit_tp = curr_high >= pos['tp']
                expired = days_held >= self.max_hold_days
                
                if hit_sl or hit_tp or expired:
                    exit_price = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else curr_price)
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    portfolio['cash'] += pos['shares'] * exit_price
                    
                    trades_history.append({
                        'ticker': t, 
                        'pnl': pnl, 
                        'ret': (exit_price / pos['entry_price']) - 1,
                        'reason': 'SL' if hit_sl else ('TP' if hit_tp else 'TIME')
                    })
                    del portfolio['positions'][t]

            # 2. Generasi Sinyal & Eksekusi Entry
            daily_signals = []
            for t, df_t in test_dfs.items():
                if t in portfolio['positions'] or current_date not in df_t.index:
                    continue
                    
                row = df_t.loc[[current_date]]
                prob = model.predict_proba(row[self.feature_cols])[0][1]
                
                if prob >= self.prob_threshold:
                    daily_signals.append((t, prob, row['Close'].values[0], row['ATR'].values[0]))
            
            daily_signals.sort(key=lambda x: x[1], reverse=True)
            top_targets = daily_signals[:self.max_daily_entries]
            
            for t, prob, price, atr in top_targets:
                if len(portfolio['positions']) >= self.max_positions:
                    break
                    
                risk_amount = portfolio['cash'] * self.risk_per_trade
                sl_distance = self.sl_atr_mult * atr
                shares = int(risk_amount / sl_distance) if sl_distance > 0 else 0
                
                cost = shares * price
                if shares > 0 and cost <= portfolio['cash']:
                    portfolio['cash'] -= cost
                    portfolio['positions'][t] = {
                        'entry_price': price,
                        'shares': shares,
                        'entry_date': current_date,
                        'sl': price - sl_distance,
                        'tp': price + (self.tp_atr_mult * atr),
                        'highest_price': price,
                        'atr': atr
                    }

            total_equity = portfolio['cash']
            for t, pos in portfolio['positions'].items():
                if current_date in test_dfs[t].index:
                    total_equity += pos['shares'] * test_dfs[t].loc[current_date, 'Close']
                else:
                    total_equity += pos['shares'] * pos['entry_price']
            equity_curve.append(total_equity)

        self._print_metrics(trades_history, equity_curve, model, X_train, y_train)

    def _print_metrics(self, trades, equity, model, X, y):
        df_trades = pd.DataFrame(trades)
        equity = np.array(equity)
        
        wins = df_trades[df_trades['pnl'] > 0] if not df_trades.empty else []
        losses = df_trades[df_trades['pnl'] < 0] if not df_trades.empty else []
        
        win_rate = (len(wins) / len(df_trades)) * 100 if len(df_trades) > 0 else 0
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss
        
        total_return = ((equity[-1] - self.capital) / self.capital) * 100
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100
        
        print("\n==================================================")
        print("STATISTIK HASIL BACKTEST V25.0")
        print("==================================================")
        print(f"Total Eksekusi Signal           : {len(df_trades)} Transaksi")
        print(f"Win Rate                        : {win_rate:.2f}% ({len(wins)} Win / {len(losses)} Loss)")
        print(f"Profit Factor                   : {profit_factor:.2f}")
        print(f"Total Return Portofolio         : {total_return:.2f}%")
        print(f"Maximum Drawdown (Portofolio)   : {max_dd:.2f}%")
        print(f"Ekuitas Akhir                   : Rp {equity[-1]:,.0f}")
        print("==================================================")
        
        print("\n FEATURE IMPORTANCE (Permutation Importance):")
        perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        for i in perm.importances_mean.argsort()[::-1]:
            print(f"- {self.feature_cols[i]:<15}: {perm.importances_mean[i]:.4f}")

# ==========================================
# 4. EKSEKUSI UTAMA (CLEAN BASKET IDX)
# ==========================================
if __name__ == '__main__':
    # Diperbarui: Ticker delisted/bermasalah dihapus & diganti dengan saham aktif
    idx_basket = [
        # --- 100 Saham Utama ---
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
        "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK",
        "ADRO.JK", "PTBA.JK", "ITMG.JK", "PGAS.JK", "GOTO.JK",
        "BRIS.JK", "KLBF.JK", "MDKA.JK", "ANTM.JK", "INCO.JK",
        "BREN.JK", "PTRO.JK", "TPIA.JK", "BNBR.JK", "ARTO.JK",
        "CDIA.JK", "BUMI.JK", "BRPT.JK", "CUAN.JK", "TOWR.JK",
        "MIDI.JK", "UNTR.JK", "AALI.JK", "AUTO.JK", "LSIP.JK",
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
        "KEEN.JK", "PTPP.JK", "WIKA.JK", "ADHI.JK", "DRMA.JK",
        "ENRG.JK", "PNBN.JK", "BBYB.JK", "WIIM.JK", "KAEF.JK",
        "WEGE.JK", "HILL.JK", "SMMA.JK", "ASRI.JK", "LPPF.JK",

        # --- 50 Saham Tambahan Tahap 1 ---
        "BRMS.JK", "DOID.JK", "ELSA.JK", "SMDR.JK", "TMAS.JK",
        "WTON.JK", "PPRE.JK", "APLN.JK", "DILD.JK", "KIJA.JK",
        "BEST.JK", "BKSL.JK", "MDLN.JK", "TOTL.JK", "AGRO.JK",
        "BABP.JK", "BKSW.JK", "PNBS.JK", "NOBU.JK", "AMAR.JK",
        "BBKP.JK", "BMTR.JK", "BHIT.JK", "KPIG.JK", "MARI.JK",
        "VIVA.JK", "WIRG.JK", "BIPI.JK", "DEWA.JK", "IPCM.JK",
        "SOCI.JK", "LEAD.JK", "HOKI.JK", "GOOD.JK", "CLEO.JK",
        "CAMP.JK", "WOOD.JK", "SIMP.JK", "NSSS.JK", "GIAA.JK",
        "SAME.JK", "BMHS.JK", "OMED.JK", "INAF.JK", "PEHA.JK",
        "CARE.JK", "RALS.JK", "SDRA.JK", "SMSM.JK", "MLBI.JK",

        # --- 50 Saham Tambahan Tahap 2 ---
        "MBSS.JK", "CPRO.JK", "BCAP.JK", "IATA.JK", "ACST.JK",
        "NRCA.JK", "SSIA.JK", "GWSA.JK", "GPRA.JK", "DART.JK",
        "MTLA.JK", "BVIC.JK", "INPC.JK", "BGTG.JK", "MCOR.JK",
        "CFIN.JK", "BIMA.JK", "VRNA.JK", "BWPT.JK", "GZCO.JK",
        "PALM.JK", "JAWA.JK", "WMPP.JK", "WMUU.JK", "DSFI.JK",
        "TRUK.JK", "WEHA.JK", "CMPP.JK", "TAXI.JK", "ABBA.JK",
        "MSKY.JK", "KBLI.JK", "KBLM.JK", "VOKS.JK", "BAJA.JK",
        "GDST.JK", "ISSP.JK", "IGAR.JK", "KDSI.JK", "SPMA.JK",
        "TRST.JK", "ALDO.JK", "POLA.JK", "BSBK.JK", "ZATA.JK",
        "KRYA.JK", "BPFI.JK", "PORT.JK", "SHIP.JK", "MAIN.JK",

        # --- 50 Saham Tambahan Tahap 3 ---
        "BJBR.JK", "BJTM.JK", "TINS.JK", "SSMS.JK", "TAPG.JK",
        "DSNG.JK", "MARK.JK", "RAJA.JK", "ESSA.JK", "AVIA.JK",
        "TPMA.JK", "ASSA.JK", "MPMX.JK", "GJTL.JK", "IRRA.JK",
        "BSIM.JK", "BNLI.JK", "ANJT.JK", "CSRA.JK", "TBLA.JK",
        "PSSI.JK", "LPCK.JK", "JRPT.JK", "PNIN.JK", "MRAT.JK",
        "CSAP.JK", "SMMT.JK", "BALI.JK", "MTDL.JK", "STAA.JK",
        "SPTO.JK", "LION.JK", "TFCO.JK", "TIFA.JK", "TUGU.JK",
        "TEBE.JK", "BTON.JK", "CASA.JK", "ARNA.JK", "RMKE.JK",
        "POWR.JK", "ASGR.JK", "SGER.JK", "SUNI.JK", "RAAM.JK",
        "PYFA.JK", "INRU.JK", "TOBA.JK", "BFIN.JK", "CITA.JK"
    ]
    
    engine = QuantBacktesterV25(
        tickers=idx_basket,
        start_date="2022-01-01",
        end_date="2026-01-01",
        initial_capital=100_000_000
    )
    engine.run()