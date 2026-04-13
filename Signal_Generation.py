# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from scipy.signal import hilbert, savgol_filter
from collections import Counter

warnings.filterwarnings(action='ignore', message=".*SettingWithCopyWarning.*")

#-----------------------------config-------------------------------------------#

MAC_DATA_PATH = r"\\jy305_server\ResearchCenter\MacroShare\Macro_database\Macro_db\data_origin\mac_data"
DATA_PATH = r"\\jy305_server\LiveTrack\EQTools\zhengyang_tools\timing\data"
DATA_PATH_SERVER = r"\\eqserver\data\trade"
SIGNAL_PATH = r"\\jy305_server\LiveTrack\EQTools\zhengyang_tools\timing\signal"

INPUT_DATA_PATHS = {
                    'index_amt': r'\\eqserver\data\trade\index_amt.pkl',
                    'fturn': r'\\eqserver\data\trade\fturn.pkl',
                    'vol': r'\\eqserver\data\trade\vol.pkl',
                    'close': r'\\eqserver\data\trade\close.pkl',
                    'ret': r'\\eqserver\data\trade\ret.pkl',
                    'I000905': r'\\eqserver\data\univ\I000905.pkl',
                    'amt': r'\\eqserver\data\trade\amt.pkl',
                    'nm1_idy': r'\\eqserver\data\idy\nm1_idy.pkl',
                    }


DB_ENGINE_URL, INDEX_CODE = "mysql+pymysql://tbyang:tbyang0209@rm-uf61r18bm23g32vzgzo.mysql.rds.aliyuncs.com:3306", 'I000905'

#-----------------------------data---------------------------------------------#

class DataPreloader:
    def __init__(self):
        self.data_cache, self.db_connection = {}, None
        
#    def get_trading_days(self, start='2015-01-01'):
#        df = pd.read_csv(os.path.join(DATA_PATH, '000905_close.csv'))
#        dates = pd.to_datetime(df.iloc[:, 0])
#        return sorted(dates[dates >= start].dt.strftime('%Y-%m-%d').tolist())
    
    def get_trading_days(self, start='2015-01-01'):
        df = pd.read_pickle(os.path.join(DATA_PATH_SERVER, 'dt.pkl'))
        df=df[pd.to_datetime(df)>=pd.to_datetime(start)]
        return list(map(lambda x:str(x)[:10],df.tolist()))
    
    def calculate_required_history_days(self, target_date, lookback_days):
        all_trading_days_str = self.get_trading_days()
        all_trading_days_dt = pd.to_datetime(all_trading_days_str)
        target_dt = pd.to_datetime(target_date)
        if target_dt not in all_trading_days_dt: raise ValueError(f"目标日期 {target_date} 不是有效交易日")
        idx = all_trading_days_dt.get_loc(target_dt)
        start_idx = max(0, idx - lookback_days + 1)
        return all_trading_days_dt[start_idx:idx + 1].strftime('%Y-%m-%d').tolist()
    
    def preload_all_data(self):
        print("开始加载原始数据")
        trading_days_str = self.get_trading_days()
        trading_days_dt = pd.to_datetime(trading_days_str)
        trading_days_set = set(trading_days_dt)

        def align_to_trading_days(df):
            if df.empty: return pd.DataFrame({'Date': sorted(trading_days_dt)})
            date_col = next((col for col in df.columns if str(col).lower() == 'date'), df.columns[0])
            df = df.copy()
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['Date'])
            value_cols = [c for c in df.columns if c not in [date_col, 'Date']]
            if not value_cols: return pd.DataFrame({'Date': sorted(trading_days_dt)})
            df.set_index('Date', inplace=True)
            df = df[value_cols].apply(pd.to_numeric, errors='coerce')
            df_aligned = df.reindex(sorted(trading_days_dt), method='ffill')
            df_aligned.reset_index(inplace=True)
            df_aligned.rename(columns={'index': 'Date'}, inplace=True)
            return df_aligned

        # xlsx 数据
        mac_file = os.path.join(MAC_DATA_PATH, 'mac_data_all.xlsx')
        sheet_map = {
            'growth': 'growth', 'inflation': 'inflation_202112',
            'monetarycondition': 'monetarycondition', 'monetarypolicy': 'monetarypolicy'
        }
        for key, sheet in sheet_map.items():
            try:
                self.data_cache[key] = align_to_trading_days(pd.read_excel(mac_file, sheet_name=sheet))
            except Exception as e:
                print(f"⚠️ 宏观数据 {sheet} 加载失败: {e}")
                self.data_cache[key] = pd.DataFrame(columns=['Date'])

        # csv 数据
        csv_files = {
                    'pe': '000905_pe_ttm.csv',
                    'pb': '000905_pb_lf.csv',
                    'trend_d1': 'H00905_close.csv',
                    'trend_d2': 'H11025_close.csv',
                    'std': 'H00905_close.csv',
                    'basis_d1': 'SH_5100501MSF_close.csv',
                    'basis_d2': '510800_nav.csv',
                    'vol_d1': 'SH_510050IV_close.csv',
                    'vol_d2': '510800_nav.csv',
                    'skew_d1': 'SH_510050SKEW_close.csv',
                    'skew_d2': '510800_nav.csv',
                    'usdcnh': 'USDCNH_close.csv',
                    'cn_yield': 'cn_yields_10y.csv',
                    'us_yield': 'us_yields_10y.csv',
                    'ic_spot': '000905_close.csv',
                    'ic_fut': 'IC00_pct_chg.csv',
                    'pcr': 'option_callput_voloi.csv',
                    'margin': 'margin_short_balance.csv',
                    'cds': 'CDS_5Y.csv',
                    }
        for key, filename in csv_files.items():
            file_path = os.path.join(DATA_PATH, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ 警告：{key} 文件不存在: {file_path}")
                self.data_cache[key] = pd.DataFrame(columns=['Date'])
                continue
            try:
                self.data_cache[key] = align_to_trading_days(pd.read_csv(file_path))
            except Exception as e:
                print(f"⚠️ CSV {filename} 加载失败: {e}")
                self.data_cache[key] = pd.DataFrame(columns=['Date'])

        # pickle 数据
        for key, path in INPUT_DATA_PATHS.items():
            if not os.path.exists(path):
                print(f"⚠️ 警告：{key} Pickle 文件不存在: {path}")
                self.data_cache[key] = pd.DataFrame(columns=['Date'])
                continue
            try:
                df = pd.read_pickle(path).reset_index()
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                self.data_cache[key] = align_to_trading_days(df)
            except Exception as e:
                print(f"⚠️ Pickle {path} 加载失败: {e}")
                self.data_cache[key] = pd.DataFrame(columns=['Date'])

        # 数据库连接
        self.db_connection = create_engine(DB_ENGINE_URL)
        print("✅ 原始数据加载完成，并已与交易日对齐")

preloader = DataPreloader()
preloader.preload_all_data()

def get_data_for_date(data_key, target_date, lookback_days=240):
    required_dates = preloader.calculate_required_history_days(target_date, lookback_days)
    if data_key in preloader.data_cache:
        df = preloader.data_cache[data_key]
        return df[df['Date'].isin(pd.to_datetime(required_dates))].sort_values('Date')
    return None

#-----------------------------signal-------------------------------------------#

def gen_growth_signal(date):
    df = get_data_for_date('growth', date, 240)
    if df is None: return None
    target_row = df[df['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    target_value = target_row.iloc[0, 4]
    if pd.isna(target_value): return None
    return pd.DataFrame({'Date': [pd.to_datetime(date)],'growth_signal': {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0}.get(int(target_value), 0.5)})

def gen_inflation_signal(date):
    df = get_data_for_date('inflation', date, 240)
    if df is None: return None
    target_row = df[df['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    target_value = target_row.iloc[0, 4]
    if pd.isna(target_value): return None
    return pd.DataFrame({'Date': [pd.to_datetime(date)],'inflation_signal': {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}.get(int(target_value), 0.5)})

def gen_monetarycondition_signal(date):
    df = get_data_for_date('monetarycondition', date, 240)
    if df is None: return None
    target_row = df[df['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    target_value = target_row.iloc[0, 4]
    if pd.isna(target_value): return None
    return pd.DataFrame({'Date': [pd.to_datetime(date)], 'monetarycondition_signal': {0:1.0, 1:1.0, 2:0.0, 3:1.0}.get(int(target_value), 0.5)})

def gen_monetarypolicy_signal(date):
    df = get_data_for_date('monetarypolicy', date, 240)
    if df is None: return None
    target_row = df[df['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    target_value = target_row.iloc[0, 3]
    if pd.isna(target_value): return None
    return pd.DataFrame({'Date': [pd.to_datetime(date)], 'monetarypolicy_signal': {0:1.0, 1:0.0, 2:1.0, 3:1.0}.get(int(target_value), 0.5)})

def gen_pe_signal(date):
    df = get_data_for_date('pe', date, 240)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    v = df.iloc[:, 0]
    ma, q = v.rolling(5).mean(), v.rolling(60).quantile(0.5)
    try:
        target_ma, target_q = ma.loc[target_dt], q.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_ma) or pd.isna(target_q): return None
    pe_signal = 1.0 if target_ma < target_q else 0.0 if target_ma > target_q else 0.5
    return pd.DataFrame({'Date': [target_dt], 'pe_signal': [pe_signal]})

def gen_pb_signal(date):
    df = get_data_for_date('pb', date, 240)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    v = df.iloc[:, 0]
    ma, q = v.rolling(5).mean(), v.rolling(60).quantile(0.5)
    try:
        target_ma, target_q = ma.loc[target_dt], q.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_ma) or pd.isna(target_q): return None
    pb_signal = 1.0 if target_ma < target_q else 0.0 if target_ma > target_q else 0.5
    return pd.DataFrame({'Date': [target_dt], 'pb_signal': [pb_signal]})

def gen_AIAE_signal(date):
    if preloader.db_connection is None: return None
    all_days_str = preloader.get_trading_days()
    all_days = pd.to_datetime(all_days_str)
    target_dt = pd.to_datetime(date)
    all_days = all_days[all_days <= target_dt]
    if len(all_days) == 0: return None
    target_dt = pd.to_datetime(date)
    if target_dt not in all_days: return None
    dfs = {}
    for key in ['fturn', 'vol', 'close']:
        if key not in preloader.data_cache or preloader.data_cache[key].empty: return None
        df = preloader.data_cache[key].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        dfs[key] = df.set_index('Date').sort_index()
    idx = dfs['vol'].index.intersection(dfs['fturn'].index).intersection(dfs['close'].index)
    df_fshares = (dfs['vol'].loc[idx] / dfs['fturn'].loc[idx] * dfs['close'].loc[idx]).ffill()
    valid_dates = all_days[all_days.isin(df_fshares.index)]
    if valid_dates.empty: return None
    sdt = (all_days[0] - timedelta(days=int(1.25 * 365))).strftime("%Y-%m-%d")
    sql = f"SELECT stock_security_code ticker, report_period date, total_liabilities AS val FROM eqchina.bankfinancials WHERE report_period > '{sdt}' AND report_period <= '{target_dt.strftime('%Y-%m-%d')}' AND report_date <= '{target_dt.strftime('%Y-%m-%d')}' AND total_liabilities != 0"
    try:
        df_lia = pd.read_sql(sql, preloader.db_connection)
    except: return None
    if df_lia.empty: return None
    df_lia['date'] = pd.to_datetime(df_lia['date'], errors='coerce')
    df_lia = df_lia.dropna(subset=['date'])
    df_lia_pivot = df_lia.pivot_table(index='ticker', columns='date', values='val').sort_index(axis=1)
    def _data_align(ts):
        if len(ts) >= 2 and pd.isna(ts.iat[-1]) and pd.isna(ts.iat[-2]): return ts.shift(2)
        elif len(ts) >= 1 and pd.isna(ts.iat[-1]): return ts.shift(1)
        return ts
    df_lia_pivot = df_lia_pivot.apply(_data_align, axis=1)
    df_lia_pivot.columns = pd.to_datetime(df_lia_pivot.columns)
    common_tickers = df_fshares.columns.intersection(df_lia_pivot.index)
    if common_tickers.empty: return None
    df_f_sub, df_l_sub = df_fshares[common_tickers], df_lia_pivot.loc[common_tickers]
    aiae_list, date_list = [], []
    for d in valid_dates:
        cols = df_l_sub.columns[df_l_sub.columns <= d]
        if cols.empty: continue
        latest_lia = df_l_sub[cols[-1]]
        if d not in df_f_sub.index: continue
        fsh = df_f_sub.loc[d]
        valid = latest_lia.notna() & fsh.notna()
        if not valid.any(): continue
        total = fsh[valid].sum() * 100 + latest_lia[valid].sum() * 1_000_000
        if total > 0:
            aiae_list.append(fsh[valid].sum() * 100 / total)
            date_list.append(d)
    if not date_list: return None
    df_aiae = pd.merge(pd.DataFrame({'Date': all_days}), pd.DataFrame({'Date': date_list, 'AIAE': aiae_list}), on='Date', how='left')
    df_aiae['AIAE'] = df_aiae['AIAE'].ffill()
    v = df_aiae['AIAE']
    df_aiae['AIAE_signal'] = np.where(v.rolling(5, min_periods=1).mean() < v.rolling(60, min_periods=1).quantile(0.5), 1.0, 0.0)
    row = df_aiae[df_aiae['Date'] == target_dt]
    if row.empty or pd.isna(row['AIAE_signal'].iloc[0]): return None
    return pd.DataFrame({'Date': [target_dt], 'AIAE_signal': [float(row['AIAE_signal'].iloc[0])]})

def gen_margin_signal(date):
    df = get_data_for_date('margin', date, 240)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    v = df.iloc[:, 0] - df.iloc[:, 1]
    ma5, ma60 = v.rolling(5).mean(), v.rolling(60).mean()
    try:
        target_ma5, target_ma60 = ma5.loc[target_dt], ma60.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_ma5) or pd.isna(target_ma60): return None
    margin_signal = 1.0 if target_ma5 <= target_ma60 else 0.0
    return pd.DataFrame({'Date': [target_dt], 'margin_signal': [margin_signal]})

def gen_amt_signal(date):
    df = get_data_for_date('index_amt', date, 240)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    amt_series = df['I000905']
    if amt_series.isna().all(): return None
    v = np.log(amt_series)
    ma20, ma60 = v.rolling(20).mean(), v.rolling(60).mean()
    ratio = ma20 / ma60 - 1
    ma20_ratio, std20_ratio = ratio.rolling(20).mean(), ratio.rolling(20).std()
    up, lw = ma20_ratio + std20_ratio, ma20_ratio - std20_ratio
    smooth_ratio = ratio.rolling(5).mean()
    try:
        target_smooth, target_up, target_lw = smooth_ratio.loc[target_dt], up.loc[target_dt], lw.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_smooth) or pd.isna(target_up) or pd.isna(target_lw): return None
    if target_smooth > target_up: amt_signal = 1.0
    elif target_smooth < target_lw: amt_signal = 0.0
    else: amt_signal = 0.5
    return pd.DataFrame({'Date': [target_dt], 'amt_signal': [amt_signal]})

def gen_CDS_signal(date):
    df = get_data_for_date('cds', date, 240)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    v = df.iloc[:, 0]
    ma5, ma20 = v.rolling(5).mean(), v.rolling(20).mean()
    try:
        target_ma5, target_ma20 = ma5.loc[target_dt], ma20.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_ma5) or pd.isna(target_ma20): return None
    cds_signal = 1.0 if target_ma5 <= target_ma20 else 0.0
    return pd.DataFrame({'Date': [target_dt], 'CDS_signal': [cds_signal]})

def gen_sc_amt_pct_signal(date):
    df = get_data_for_date('amt', date, 240)
    if df is None or df.empty or 'Date' not in df.columns: return None
    df_temp = df.set_index('Date')
    cols = df_temp.columns
    target_cols = [col for col in cols if str(col).startswith(('A300', 'A688'))]
    if not target_cols: return None
    total_amt = df_temp.sum(axis=1)
    target_amt = df_temp[target_cols].sum(axis=1)
    cc_pct = (target_amt / total_amt).replace([np.inf, -np.inf], np.nan)
    df_pct = pd.DataFrame({'Date': df_temp.index, 'cc_pct': cc_pct.values}).reset_index(drop=True)
    v, ma20, std20, smooth = df_pct['cc_pct'], df_pct['cc_pct'].rolling(20).mean(), df_pct['cc_pct'].rolling(20).std(), df_pct['cc_pct'].rolling(5).mean()
    target_row = df_pct[df_pct['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    pos = target_row.index[0]
    try:
        target_smooth, target_ma20, target_std20 = smooth.iloc[pos], ma20.iloc[pos], std20.iloc[pos]
    except IndexError: return None
    if pd.isna(target_smooth) or pd.isna(target_ma20) or pd.isna(target_std20): return None
    threshold_up, threshold_down = target_ma20 + target_std20, target_ma20 - target_std20
    if target_smooth > threshold_up: sc_amt_pct_signal = 1.0
    elif target_smooth < threshold_down: sc_amt_pct_signal = 0.0
    else: sc_amt_pct_signal = 0.5
    return pd.DataFrame({'Date': [pd.to_datetime(date)], 'sc_amt_pct_signal': [sc_amt_pct_signal]})

def gen_idy_amt_corr_signal(date):
    df_idy = get_data_for_date('nm1_idy', date, 240)
    df_amt = get_data_for_date('amt', date, 240)
    if df_idy is None or df_amt is None or df_idy.empty or df_amt.empty: return None
    if 'Date' not in df_idy.columns or 'Date' not in df_amt.columns: return None
    common_cols = df_idy.columns.intersection(df_amt.columns).difference(['Date'])
    if len(common_cols) == 0: return None
    df_idy_indexed = df_idy.set_index('Date').sort_index()
    df_amt_indexed = df_amt.set_index('Date').sort_index()
    target_dt = pd.to_datetime(date)
    all_dates_in_window = df_idy_indexed.index
    if target_dt not in all_dates_in_window: return None
    try:
        target_pos = all_dates_in_window.get_loc(target_dt)
        if target_pos == 0: return None
        prev_dt = all_dates_in_window[target_pos - 1]
    except KeyError: return None
    try:
        idy_curr = df_idy_indexed.loc[target_dt, common_cols]
        amt_curr = df_amt_indexed.loc[target_dt, common_cols]
        idy_prev = df_idy_indexed.loc[prev_dt, common_cols]
        amt_prev = df_amt_indexed.loc[prev_dt, common_cols]
    except KeyError: return None
    curr_df = pd.DataFrame({'idy': idy_curr, 'amt': amt_curr})
    prev_df = pd.DataFrame({'idy': idy_prev, 'amt': amt_prev})
    curr_grp = curr_df.groupby('idy')['amt'].sum()
    prev_grp = prev_df.groupby('idy')['amt'].sum()
    all_idy = prev_grp.index.union(curr_grp.index)
    prev_rank = prev_grp.reindex(all_idy, fill_value=0).rank(method='min')
    curr_rank = curr_grp.reindex(all_idy, fill_value=0).rank(method='min')
    try:
        corr, _ = pearsonr(prev_rank, curr_rank)
        if not np.isfinite(corr): return None
    except Exception: return None
    full_corr_list, full_corr_dates = [], []
    for i in range(1, len(all_dates_in_window)):
        d_curr = all_dates_in_window[i]
        d_prev = all_dates_in_window[i - 1]
        try:
            i_c = df_idy_indexed.loc[d_curr, common_cols]
            a_c = df_amt_indexed.loc[d_curr, common_cols]
            i_p = df_idy_indexed.loc[d_prev, common_cols]
            a_p = df_amt_indexed.loc[d_prev, common_cols]
        except KeyError: continue
        c_df = pd.DataFrame({'idy': i_c, 'amt': a_c})
        p_df = pd.DataFrame({'idy': i_p, 'amt': a_p})
        c_grp = c_df.groupby('idy')['amt'].sum()
        p_grp = p_df.groupby('idy')['amt'].sum()
        all_i = p_grp.index.union(c_grp.index)
        if len(all_i) == 0: continue
        p_r = p_grp.reindex(all_i, fill_value=0).rank(method='min')
        c_r = c_grp.reindex(all_i, fill_value=0).rank(method='min')
        try:
            c_val, _ = pearsonr(p_r, c_r)
            if np.isfinite(c_val):
                full_corr_list.append(c_val)
                full_corr_dates.append(d_curr)
        except Exception: continue
    if not full_corr_dates: return None
    corr_series = pd.Series(full_corr_list, index=pd.to_datetime(full_corr_dates), name='corr').sort_index()
    target_dt = pd.to_datetime(date).tz_localize(None)
    corr_series.index = corr_series.index.tz_localize(None)
    if target_dt not in corr_series.index: return None
    ma20, ma60, std60 = corr_series.rolling(20, min_periods=20).mean(), corr_series.rolling(60, min_periods=60).mean(), corr_series.rolling(60, min_periods=60).std()
    try:
        t_ma20, t_ma60 = ma20.loc[target_dt], ma60.loc[target_dt]
    except KeyError: return None
    if pd.isna(t_ma20) or pd.isna(t_ma60): return None
    s1 = 1.0 if t_ma20 > t_ma60 else 0.0
    s2 = np.nan
    try:
        t_std60 = std60.loc[target_dt]
        if pd.notna(t_std60) and std60.dropna().shape[0] >= 20:
            std_ma5 = std60.rolling(5, min_periods=5).mean()
            std_ma20 = std60.rolling(20, min_periods=20).mean()
            t_std_ma5, t_std_ma20 = std_ma5.loc[target_dt], std_ma20.loc[target_dt]
            if pd.notna(t_std_ma5) and pd.notna(t_std_ma20):
                s2 = 1.0 if t_std_ma5 <= t_std_ma20 else 0.0
    except (KeyError, Exception): pass
    if pd.isna(s2): return None
    final_signal = (s1 + s2) / 2.0
    return pd.DataFrame({'Date': [target_dt], 'idy_amt_corr_signal': [final_signal]})

def gen_idy_turn_pct_signal(date):
    df_idy, df_fturn, df_amt = get_data_for_date('nm1_idy', date, 240), get_data_for_date('fturn', date, 240), get_data_for_date('amt', date, 240)
    if any(x is None or x.empty for x in [df_idy, df_fturn, df_amt]) or not all('Date' in df.columns for df in [df_idy, df_fturn, df_amt]): return None
    common_cols = (set(df_idy.columns) & set(df_fturn.columns) & set(df_amt.columns)) - {'Date'}
    if not common_cols: return None
    common_cols = sorted(common_cols)
    cols_to_merge = ['Date'] + list(common_cols)
    df_idy_sub, df_fturn_sub, df_amt_sub = df_idy[cols_to_merge], df_fturn[cols_to_merge], df_amt[cols_to_merge]
    df_mkt = df_amt_sub.copy()
    df_mkt[list(common_cols)] = (df_amt_sub[list(common_cols)] / df_fturn_sub[list(common_cols)] * 100).replace([np.inf, -np.inf], np.nan)
    date_list, rate_list = [], []
    for idx in range(len(df_idy_sub)):
        date_val = df_idy_sub.iloc[idx]['Date']
        idy_vals = df_idy_sub.iloc[idx][common_cols].values
        amt_vals = df_amt_sub.iloc[idx][common_cols].values
        mkt_vals = df_mkt.iloc[idx][common_cols].values
        temp = pd.DataFrame({'idy': idy_vals, 'amt': amt_vals, 'mkt': mkt_vals}).dropna()
        if temp.empty or (temp['mkt'] <= 0).any(): continue
        ind_amt, ind_mkt = temp.groupby('idy')['amt'].sum(), temp.groupby('idy')['mkt'].sum()
        ind_turn = ind_amt / ind_mkt
        market_amt, market_mkt = temp['amt'].sum(), temp['mkt'].sum()
        if market_mkt <= 0: continue
        market_turn = market_amt / market_mkt
        top3_mean = ind_turn.nlargest(3).mean()
        rate = top3_mean / market_turn
        if np.isfinite(rate):
            date_list.append(date_val)
            rate_list.append(rate)
    if not date_list: return None
    df_rate = pd.DataFrame({'Date': date_list, 'rate': rate_list}).sort_values('Date').reset_index(drop=True)
    v, ma60, std60, smooth = df_rate['rate'], df_rate['rate'].rolling(60).mean(), df_rate['rate'].rolling(60).std(), df_rate['rate'].rolling(5).mean()
    target_dt = pd.to_datetime(date)
    target_row = df_rate[df_rate['Date'] == target_dt]
    if target_row.empty: return None
    pos = target_row.index[0]
    try:
        t_smooth, t_ma60, t_std60 = smooth.iloc[pos], ma60.iloc[pos], std60.iloc[pos]
    except IndexError: return None
    if pd.isna(t_smooth) or pd.isna(t_ma60) or pd.isna(t_std60): return None
    threshold_up, threshold_down = t_ma60 + 0.25 * t_std60, t_ma60 - 0.25 * t_std60
    if t_smooth > threshold_up: signal = 0.0
    elif t_smooth < threshold_down: signal = 1.0
    else: signal = 0.5
    return pd.DataFrame({'Date': [target_dt], 'idy_turn_pct_signal': [signal]})

def gen_trend_signal(date):
    df1, df2 = get_data_for_date('trend_d1', date, 360), get_data_for_date('trend_d2', date, 360)
    if df1 is None or df2 is None: return None
    res = pd.DataFrame({
        'Date': pd.to_datetime(df1['Date']),
        'p1': df1.iloc[:, 1].values,
        'p2': df2.iloc[:, 1].values
    }).reset_index(drop=True)
    target_dt = pd.to_datetime(date)
    if res.empty or res['Date'].iloc[-1] != target_dt: return None
    ratio = res['p1'] / res['p2']
    ma120, ma240 = ratio.rolling(120, min_periods=120).mean(), ratio.rolling(240, min_periods=240).mean()
    dist = ma120 / ma240 - 1
    max20, max60, min20, min60 = dist.rolling(20).max(), dist.rolling(60).max(), dist.rolling(20).min(), dist.rolling(60).min()
    try:
        target_dist, target_max_20, target_max_60, target_min_20, target_min_60 = dist.iloc[-1], max20.iloc[-1], max60.iloc[-1], min20.iloc[-1], min60.iloc[-1]
    except IndexError: return None
    if any(pd.isna(x) for x in [target_dist, target_max_20, target_max_60, target_min_20, target_min_60]): return None
    if target_max_20 == target_max_60: s = 1
    elif target_min_20 == target_min_60: s = -1
    else: s = 0
    total = np.sign(target_dist) + s
    trend_signal = 0.0 if total == -2 else (0.5 if total == 0 else 1.0)
    return pd.DataFrame({'Date': [target_dt], 'trend_signal': [trend_signal]})

def gen_highlow_signal(date):
    if 'ret' not in preloader.data_cache or 'I000905' not in preloader.data_cache: return None
    df_ret_raw, df_univ_raw = preloader.data_cache['ret'], preloader.data_cache['I000905']
    df_ret = df_ret_raw.set_index('Date') if 'Date' in df_ret_raw.columns else df_ret_raw.copy()
    df_univ = df_univ_raw.set_index('Date') if 'Date' in df_univ_raw.columns else df_univ_raw.copy()
    df_ret, df_univ = df_ret.apply(pd.to_numeric, errors='coerce'), df_univ.apply(pd.to_numeric, errors='coerce')
    df_ret.index, df_univ.index = pd.to_datetime(df_ret.index), pd.to_datetime(df_univ.index)
    if any(x is None or x.empty for x in [df_ret, df_univ]): return None
    target_dt = pd.to_datetime(date)
    try:
        all_trading_days = pd.to_datetime(preloader.get_trading_days())
        target_idx = all_trading_days.get_loc(target_dt)
    except KeyError: return None
    if target_idx < 299: return None
    hist_dates = all_trading_days[target_idx - 299 : target_idx + 1]
    df_nav = (df_ret + 1).cumprod()
    high_vals, low_vals, valid_dates = [], [], []
    for d in hist_dates:
        if d not in df_ret.index or d not in df_univ.index: continue
        nav_hist = df_nav[df_nav.index <= d].iloc[-240:]
        if len(nav_hist) < 240: continue
        try: univ_cols = df_univ.loc[[d]].dropna(axis=1).columns.tolist()
        except KeyError: continue
        if not univ_cols: continue
        nav_hist = nav_hist[univ_cols]
        if nav_hist.empty or len(nav_hist) < 240: continue
        last_prices = nav_hist.iloc[-1]
        high_count, low_count = (last_prices == nav_hist.max()).sum(), (last_prices == nav_hist.min()).sum()
        high_vals.append(high_count)
        low_vals.append(low_count)
        valid_dates.append(d)
    if len(valid_dates) < 60: return None
    df_hl = pd.DataFrame({'Date': valid_dates,'high': high_vals,'low': low_vals}).sort_values('Date').reset_index(drop=True)
    df_hl['diff'] = df_hl['high'] - df_hl['low']
    ma20, ma60 = df_hl['diff'].rolling(20).mean(), df_hl['diff'].rolling(60).mean()
    target_row = df_hl[df_hl['Date'] == target_dt]
    if target_row.empty: return None
    pos = target_row.index[0]
    try: val20, val60 = ma20.iloc[pos], ma60.iloc[pos]
    except IndexError: return None
    if pd.isna(val20) or pd.isna(val60): return None
    highlow_signal = 1.0 if val20 > val60 else 0.0
    return pd.DataFrame({'Date': [target_dt],'highlow_signal': [highlow_signal]})

def gen_std_signal(date):
    df = get_data_for_date('std', date, 360)
    if df is None or df.empty: return None
    target_dt = pd.to_datetime(date)
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index()
    if target_dt not in df.index: return None
    price = df.iloc[:, 0]
    ret = price.pct_change()
    rolling_std_240 = ret.rolling(window=240, min_periods=240).std()
    rolling_mean_20 = rolling_std_240.rolling(window=20, min_periods=20).mean()
    if target_dt not in rolling_mean_20.index: return None
    target_std, target_mean = rolling_std_240.loc[target_dt], rolling_mean_20.loc[target_dt]
    if pd.isna(target_std) or pd.isna(target_mean): return None
    std_signal = 1.0 if target_std < target_mean else 0.0
    return pd.DataFrame({'Date': [target_dt], 'std_signal': [std_signal]})

def gen_basis_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'basis_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df1, df2 = get_data_for_date('basis_d1', date, 360), get_data_for_date('basis_d2', date, 360)
    if df1 is None or df2 is None: return None
    res = pd.merge(df1[['Date', df1.columns[-1]]], df2[['Date', df2.columns[-1]]], on='Date')
    res.columns = ['Date', 'f', 's']
    res = res.set_index('Date').sort_index()
    pct_change_f, pct_change_s = res['f'].pct_change(), res['s'].pct_change()
    b = pct_change_f - pct_change_s
    rolling_q30, rolling_q70 = b.rolling(240).quantile(0.3), b.rolling(240).quantile(0.7)
    s_pct_3 = res['s'].pct_change(3)
    target_dt = pd.to_datetime(date)
    if target_dt not in b.index: return None
    try:
        target_b, target_q30, target_q70, target_s3 = b.loc[target_dt], rolling_q30.loc[target_dt], rolling_q70.loc[target_dt], s_pct_3.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_b) or pd.isna(target_q30) or pd.isna(target_q70) or pd.isna(target_s3): return None
    cond1, cond2 = (target_b < target_q30) and (target_s3 < 0), (target_b > target_q70) and (target_s3 > 0)
    if cond1: basis_signal = 0.0
    elif cond2: basis_signal = 1.0
    else: basis_signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'basis_signal': [basis_signal]})

def gen_vol_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'vol_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df1, df2 = get_data_for_date('vol_d1', date, 240), get_data_for_date('vol_d2', date, 240)
    if df1 is None or df2 is None: return None
    res = pd.merge(df1[['Date', df1.columns[-1]]], df2[['Date', df2.columns[-1]]], on='Date')
    res.columns = ['Date', 'iv', 's']
    res = res.set_index('Date').sort_index()
    rolling_q05, rolling_q95 = res['iv'].rolling(240).quantile(0.05), res['iv'].rolling(240).quantile(0.95)
    s_pct_3 = res['s'].pct_change(3)
    target_dt = pd.to_datetime(date)
    if target_dt not in res.index: return None
    try:
        target_iv, target_q05, target_q95, target_s3 = res['iv'].loc[target_dt], rolling_q05.loc[target_dt], rolling_q95.loc[target_dt], s_pct_3.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_iv) or pd.isna(target_q05) or pd.isna(target_q95) or pd.isna(target_s3): return None
    cond1, cond2 = (target_iv < target_q05) and (target_s3 < 0), (target_iv > target_q95) and (target_s3 > 0)
    if cond1: vol_signal = 0.0
    elif cond2: vol_signal = 1.0
    else: vol_signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'vol_signal': [vol_signal]})

def gen_skew_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'skew_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df1, df2 = get_data_for_date('skew_d1', date, 240), get_data_for_date('skew_d2', date, 240)
    if df1 is None or df2 is None: return None
    res = pd.merge(df1[['Date', df1.columns[-1]]], df2[['Date', df2.columns[-1]]], on='Date')
    res.columns = ['Date', 'sk', 's']
    res = res.set_index('Date').sort_index()
    rolling_q05, rolling_q95 = res['sk'].rolling(240).quantile(0.05), res['sk'].rolling(240).quantile(0.95)
    s_pct_3 = res['s'].pct_change(3)
    target_dt = pd.to_datetime(date)
    if target_dt not in res.index: return None
    try:
        target_sk, target_q05, target_q95, target_s3 = res['sk'].loc[target_dt], rolling_q05.loc[target_dt], rolling_q95.loc[target_dt], s_pct_3.loc[target_dt]
    except KeyError: return None
    if pd.isna(target_sk) or pd.isna(target_q05) or pd.isna(target_q95) or pd.isna(target_s3): return None
    cond1, cond2 = (target_sk < target_q05) and (target_s3 < 0), (target_sk > target_q95) and (target_s3 > 0)
    if cond1: skew_signal = 0.0
    elif cond2: skew_signal = 1.0
    else: skew_signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'skew_signal': [skew_signal]})

def gen_usdcnh_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'usdcnh_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df = get_data_for_date('usdcnh', date, 240)
    if df is None or df.empty: return None
    df = df.set_index('Date').sort_index()
    v = df.iloc[:, -1]
    ma20, ma120 = v.rolling(20).mean(), v.rolling(120).mean()
    diff = ma20 / ma120 - 1
    q15, q85 = diff.rolling(120).quantile(0.15), diff.rolling(120).quantile(0.85)
    target_dt = pd.to_datetime(date)
    if target_dt not in diff.index: return None
    try:
        t_diff, t_q15, t_q85 = diff.loc[target_dt], q15.loc[target_dt], q85.loc[target_dt]
    except KeyError: return None
    if pd.isna(t_diff) or pd.isna(t_q15) or pd.isna(t_q85): return None
    if t_diff < t_q15: signal = 1.0
    elif t_diff > t_q85: signal = 0.0
    else: signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'usdcnh_signal': [signal]})

def gen_spread_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'spread_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors='coerce')
            hist_df = hist_df.dropna(subset=['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = float(last_valid.iloc[-1])
        except Exception as e: print(f"⚠️ 读取历史信号失败: {e}")
    df_cn, df_us = get_data_for_date('cn_yield', date, 240), get_data_for_date('us_yield', date, 240)
    if df_cn is None or df_us is None or df_cn.empty or df_us.empty: return None
    target_dt = pd.to_datetime(date)
    for df in [df_cn, df_us]:
        if 'Date' not in df.columns:
            date_col = next((col for col in df.columns if str(col).lower() == 'date'), None)
            if date_col: df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
    def extract_data_series(df, name):
        data_col = next((col for col in df.columns if str(col).lower() == 'data'), None)
        if data_col is None: raise ValueError(f"{name} 数据中未找到 'data' 列，可用列: {list(df.columns)}")
        series = df.set_index('Date')[data_col]
        series = pd.to_numeric(series, errors='coerce')
        return series.sort_index()
    try:
        cn_series, us_series = extract_data_series(df_cn, "CN"), extract_data_series(df_us, "US")
    except Exception as e:
        print(f"❌ 提取数值列失败: {e}")
        return None
    common_dates = cn_series.index.intersection(us_series.index)
    if common_dates.empty or target_dt not in common_dates: return None
    sp = cn_series.loc[common_dates] - us_series.loc[common_dates]
    rolling_mean_20, rolling_mean_40 = sp.rolling(window=20, min_periods=20).mean(), sp.rolling(window=40, min_periods=40).mean()
    md = rolling_mean_20 - rolling_mean_40
    d = md.diff()
    rolling_std_40 = d.rolling(window=40, min_periods=40).std()
    try:
        target_md, target_d, target_std_40 = md.loc[target_dt], d.loc[target_dt], rolling_std_40.loc[target_dt]
    except KeyError: return None
    if any(pd.isna(x) for x in [target_d, target_md, target_std_40]): return None
    condition1, condition2 = (target_d > 2 * target_std_40) and (target_md > 0), (target_d < -2 * target_std_40) and (target_md < 0)
    if condition1: spread_signal = 1.0
    elif condition2: spread_signal = 0.0
    else: spread_signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'spread_signal': [spread_signal]})

def gen_IC_basis_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'IC_basis_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df_spot, df_fut = get_data_for_date('ic_spot', date, 360), get_data_for_date('ic_fut', date, 360)
    if df_spot is None or df_fut is None: return None
    spot = df_spot[['Date', df_spot.columns[1]]].rename(columns={df_spot.columns[1]: 'spot'})
    fut = df_fut[['Date', df_fut.columns[1]]].rename(columns={df_fut.columns[1]: 'fut_ret'})
    res = pd.merge(spot, fut, on='Date').set_index('Date').sort_index()
    
    res['spot_ret'] = res['spot'].pct_change()
    res['daily_spread'] = res['fut_ret'] - res['spot_ret']
    res['basis_sentiment'] = res['daily_spread'].rolling(window=240).sum()
    mu = res['basis_sentiment'].rolling(window=120).mean()
    std = res['basis_sentiment'].rolling(window=40).std()
    res['z_score'] = (res['basis_sentiment'] - mu) / (std + 1e-9)
    
    target_dt = pd.to_datetime(date)
    if target_dt not in res.index: return None
    try:
        z_score = res.loc[target_dt,'z_score']
    except KeyError: return None
    if pd.isna(z_score): return None
    if z_score > 1.0: signal = 1.0
    elif z_score < -1.0: signal = 0.0
    else: signal = last_signal
    return pd.DataFrame({'Date': [target_dt], 'IC_basis_signal': [signal]})

def gen_pcr_signal(date):
    signal_file = os.path.join(SIGNAL_PATH, 'pcr_signal.xlsx')
    last_signal = 0.0
    if os.path.exists(signal_file):
        try:
            hist_df = pd.read_excel(signal_file, dtype={'Date': str})
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df[hist_df['Date'] < pd.to_datetime(date)]
            if not hist_df.empty:
                last_valid = hist_df['Signal_Value'].dropna()
                if not last_valid.empty: last_signal = last_valid.iloc[-1]
        except Exception: pass
    df = get_data_for_date('pcr', date, 240)
    if df is None: return None
    v = df.iloc[:,6]
    rolling_mean_20, rolling_mean_40 = v.rolling(20).mean(), v.rolling(40).mean()
    md = rolling_mean_20 - rolling_mean_40
    md_diff = md.diff()
    target_row = df[df['Date'] == pd.to_datetime(date)]
    if target_row.empty: return None
    target_md = md[df['Date'] == pd.to_datetime(date)].iloc[0] if len(md[df['Date'] == pd.to_datetime(date)]) > 0 else np.nan
    target_md_diff = md_diff[df['Date'] == pd.to_datetime(date)].iloc[0] if len(md_diff[df['Date'] == pd.to_datetime(date)]) > 0 else np.nan
    if pd.isna(target_md) or pd.isna(target_md_diff): return None
    condition1, condition2 = (target_md_diff < 0) & (target_md < 0), (target_md_diff > 0) & (target_md > 0)
    if condition1: pcr_signal = 1.0
    elif condition2: pcr_signal = 0.0
    else: pcr_signal = last_signal
    pcr_signal = pcr_signal if pd.notna(pcr_signal) else 0
    return pd.DataFrame({'Date': [pd.to_datetime(date)], 'pcr_signal': [pcr_signal]})

def gen_ht_signal(date):

    LOOKBACK_WINDOW = 40
    VOTE_WINDOW = 20
    SAVGOL_WINDOW = 11
    SAVGOL_POLYORDER = 2
    AVG_WINDOW = 5
    USE_SMALL_VOTE = True  # 与VOTE6一致：均值>=0.5 -> 1，否则0（输出仓位信号）

    # 为避免未来函数，只使用截止到当日（含当日）的历史数据
    hist_prices = get_data_for_date('ic_spot', date, 60)
    hist_prices = pd.DataFrame(hist_prices['000905'].tolist(),index = hist_prices['Date'],columns = ['close'])

    close_prices = hist_prices["close"].astype(float).values
    dates = hist_prices.index
    n = len(close_prices)

    # 1) Hilbert+象限投票：生成日频象限序列（1/2/3/4）
    rolling_votes = np.zeros(n, dtype=int)
    for t in range(LOOKBACK_WINDOW, n + 1):
        raw_window = close_prices[t - LOOKBACK_WINDOW : t].copy()

        if np.isnan(raw_window).any():
            continue

        try:
            if len(raw_window) >= SAVGOL_WINDOW:
                window_smoothed = savgol_filter(
                    raw_window,
                    SAVGOL_WINDOW,
                    SAVGOL_POLYORDER,
                    mode="mirror",
                )
            else:
                window_smoothed = raw_window.copy()

            processed_signal = np.diff(window_smoothed)
            processed_signal = processed_signal - np.mean(processed_signal)

            analytic_signal = hilbert(processed_signal)
            inst_phases = np.angle(analytic_signal)

            window_quadrants = []
            for p in inst_phases:
                if p >= 0:
                    q = 1 if p <= np.pi / 2 else 2
                else:
                    q = 4 if p >= -np.pi / 2 else 3
                window_quadrants.append(q)

            vote_quadrants = (
                window_quadrants[-VOTE_WINDOW:]
                if len(window_quadrants) > VOTE_WINDOW
                else window_quadrants
            )

            counts = Counter(vote_quadrants)
            max_votes = max(counts.values())
            winners = [q for q, c in counts.items() if c == max_votes]

            if len(winners) == 1:
                final_winner = winners[0]
            else:
                last_q = vote_quadrants[-1]
                final_winner = last_q if last_q in winners else winners[0]

            rolling_votes[t - 1] = final_winner
        except Exception:
            continue

    vote_series = pd.Series(rolling_votes, index=dates).replace(0, np.nan)

    # 2) 象限 → 日频 0/1 信号（Q2 或 Q4 → 1，其余 0），并前向填充
    signals = pd.Series(0, index=vote_series.index, dtype=int)
    signals[(vote_series == 2) | (vote_series == 4)] = 1
    signals = signals.reindex(hist_prices.index, method="ffill").fillna(0).astype(int)

    # 3) 方案：目标仓位 = 近N日信号均值 r 的 shift(1)，并可选小投票二值化
    r = signals.rolling(AVG_WINDOW, min_periods=1).mean()
    target_weight = r.shift(1).fillna(0.0)
    if USE_SMALL_VOTE:
        target_weight = (target_weight >= 0.5).astype(float)

    # 4) 对齐回测起始日：方案2的 backtest_dates = all_dates[all_dates > first_signal_date]
    if len(dates) < LOOKBACK_WINDOW:
        pcr_signal = 0.0
    else:
        first_signal_date = dates[LOOKBACK_WINDOW - 1]
        pcr_signal = 0.0 if pd.to_datetime(date) <= first_signal_date else float(target_weight.loc[date])

    # 5) 数值清理：与VOTE6落盘一致（clip + isclose吸附）
    pcr_signal = float(np.clip(pcr_signal, 0.0, 1.0))
    if np.isclose(pcr_signal, 0.0, atol=1e-10):
        pcr_signal = 0.0
    if np.isclose(pcr_signal, 1.0, atol=1e-10):
        pcr_signal = 1.0

    # 6) 返回单日结果，格式与要求一致
    return pd.DataFrame({"Date": [pd.to_datetime(date)], "pcr_signal": [pcr_signal]})

#-----------------------------cmb----------------------------------------------#

def combine_signal(dict_sig,ls_wgt):
    df_sig_all = pd.DataFrame()
    for cat in dict_sig.keys():
        df_sig_temp = pd.DataFrame()
        for sig in dict_sig[cat]:
            df_sig = pd.read_excel(os.path.join(SIGNAL_PATH,sig+'_signal.xlsx'),index_col = 'Date')
            df_sig.columns = [sig]
            df_sig_temp = df_sig_temp.join(df_sig,how = 'outer')
            df_cat = df_sig_temp.mean(axis = 1)
        df_cat = pd.DataFrame(df_cat,columns = [cat])
        df_cat[cat] = df_cat[cat].apply(lambda x:(x-0.5)/0.2*0.5+0.5).clip(0,1)
        df_cat.to_excel(os.path.join(SIGNAL_PATH,cat+'_cat_signal.xlsx'))
        print(f"大类信号{cat}合成完成")
        df_sig_all = df_sig_all.join(df_cat,how = 'outer')
    df_cmb = df_sig_all.dot(ls_wgt)/len(ls_wgt)
    df_cmb = pd.DataFrame(df_cmb,columns = ['cmb'])
    df_cmb['cmb'] = df_cmb['cmb'].apply(lambda x:(x-0.5)/0.2*0.5+0.5).clip(0,1)
    df_cmb.to_excel(os.path.join(SIGNAL_PATH,'cmb_signal.xlsx'))
    print(f"最终信cm1合成完成")

#-----------------------------main---------------------------------------------#

def main():
    dt = str(datetime.now().date())
    END_DATE = preloader.calculate_required_history_days(dt, 2)[0]
#    END_DATE = '2026-03-18'
    TARGET_START = preloader.calculate_required_history_days(dt, 6)[0]
#    TARGET_START = '2015-01-05'
    
    full_dates = preloader.get_trading_days()
    if len(full_dates) < 360: raise ValueError("历史交易日不足360天")
    target_start_dt, target_end_dt = pd.to_datetime(TARGET_START), pd.to_datetime(END_DATE)
    target_dates = [date for date in full_dates if target_start_dt <= pd.to_datetime(date) <= target_end_dt]
    tasks = [
            ("growth", gen_growth_signal), 
            ("inflation", gen_inflation_signal),
            ("monetarycondition", gen_monetarycondition_signal), 
            ("monetarypolicy", gen_monetarypolicy_signal),
            ("pe", gen_pe_signal), 
            ("pb", gen_pb_signal), 
            ("AIAE", gen_AIAE_signal),
            ("margin", gen_margin_signal), 
            ("amt", gen_amt_signal), 
            ("CDS", gen_CDS_signal),
            ("sc_amt_pct", gen_sc_amt_pct_signal), 
            ("idy_amt_corr", gen_idy_amt_corr_signal),
            ("idy_turn_pct", gen_idy_turn_pct_signal), 
            ("trend", gen_trend_signal),
            ("highlow", gen_highlow_signal), 
            ("std", gen_std_signal), 
            ("basis", gen_basis_signal),
            ("vol", gen_vol_signal), 
            ("skew", gen_skew_signal), 
            ("usdcnh", gen_usdcnh_signal),
            ("spread", gen_spread_signal), 
            ("IC_basis", gen_IC_basis_signal), 
            ("pcr", gen_pcr_signal),
            ("ht", gen_ht_signal),
            ]
    print(f"开始处理 {len(target_dates)} 天的信号，从 {TARGET_START} 到 {END_DATE}")
    for date in target_dates:
        print(f"\n处理日期: {date}")
        success = 0
        for name, func in tasks:
            try:
                df = func(date)
                if df is None or df.empty:
                    print(f"  ⚠️ 跳过: {name}")
                    continue
                df['Date'] = pd.to_datetime(df['Date'])
                col = next(c for c in df.columns if c != 'Date')
                out = df[['Date', col]].rename(columns={col: 'Signal_Value'})
                out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
                path = os.path.join(SIGNAL_PATH, f'{name}_signal.xlsx')
                old = pd.read_excel(path, dtype={'Date': str}) if os.path.exists(path) else pd.DataFrame(columns=['Date', 'Signal_Value'])
                if not {'Date', 'Signal_Value'}.issubset(old.columns): old = pd.DataFrame(columns=['Date', 'Signal_Value'])
                final = pd.concat([old, out], ignore_index=True).drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
                final.to_excel(path, index=False)
                success += 1
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        print(f"  ✅ 日期 {date} 成功更新 {success} 个信号")
    print("\n" + "="*50)
    print("所有日期处理完成！")

    # signal combine
    dict_sig = {
                'macro':['growth','inflation','monetarycondition','monetarypolicy','CDS','usdcnh','spread'],
                'value':['pe','pb','AIAE'],
                'option':['basis','vol','skew','pcr'],
                'moneyflow':['margin','amt','sc_amt_pct','idy_amt_corr','idy_turn_pct'],
                'price':['trend','highlow','std','IC_basis','ht'],
                }

    ls_wgt = [1,1/2,1/2,1,1]
    combine_signal(dict_sig,ls_wgt)

if __name__ == "__main__":
    main()
