# -*- coding: utf-8 -*-
import os
import pandas as pd

path = r'V:\EQTools\zhengyang_tools\timing\output\cmb'

ls_sig = [
#        "growth_signal",
#        "inflation_signal",
#        "monetarycondition_signal",
#        "monetarypolicy_signal",
#        "CDS_signal",
#        "usdcnh_signal",
#        "spread_signal",
#        
#        "pe_signal",
#        "pb_signal",
#        "AIAE_signal",
#        
#        "basis_signal",
#        "vol_signal",
#        "skew_signal",
#        "pcr_signal",
#        
#        "margin_signal",
#        "amt_signal",
#        "sc_amt_pct_signal",
#        "idy_amt_corr_signal",
#        "idy_turn_pct_signal",
#        "trend_signal",
#        "highlow_signal",
#        "std_signal",
#        "IC_basis_signal",
#        "ht_signal",
        
        "macro_signal",
        "value_signal",
        "option_signal",
        "tech_signal",
        "cmb1_signal",
        "macro_score_signal",
        "value_score_signal",
        "option_score_signal",
        "tech_score_signal",
        "cmb2_signal",
        "moneyflow_score_signal",
        "price_score_signal",
        "cmb3_signal",
        
        ]

col = [
       'year',
       'ret',
       'vol',
       'sharpe',
       'dd',
       'bmk_ret',
       'bmk_vol',
       'bmk_sharpe',
       'bmk_dd',
       'alpha_ret',
       'alpha_vol',
       'alpha_sharpe',
       'alpha_dd',
       'to',
       ]

backtest = 'monthly_mthend_open'

df_sig_all = pd.DataFrame()
for sig in ls_sig:
    df_sig = pd.read_excel(os.path.join(path,backtest,sig+'_report.xlsx'))
    df_sig.columns = col
    df_sig_all = df_sig_all.append(df_sig.iloc[-1,:])
    print(sig)
df_sig_all.index = ls_sig
df_sig_all = df_sig_all[col]