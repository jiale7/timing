# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import datetime
import json
import requests
from sqlalchemy import create_engine

from WindPy import w
import iFinDPy as ifind
w.start()

#-----------------------Config-------------------------------------------------#

data_path = r'V:\EQTools\zhengyang_tools\timing\data'
BBG_SERVER_URL = 'http://47.103.56.255:3008/'
ifind.THS_iFinDLogin('shjytz015','kk6Y54rW') # inv
engine = None

#-----------------------DB-----------------------------------------------------#

def get_engine():
    global engine
    if engine == None:
        try:
            engine = create_engine("mysql+pymysql://tbyang:tbyang0209@rm-uf61r18bm23g32vzgzo.mysql.rds.aliyuncs.com:3306")
        except:
            return None
    return engine

def get_sql_data(sql, index_col=None):
    if index_col is None:
        return pd.read_sql(sql, get_engine())
    else:
        return pd.read_sql(sql, get_engine(), index_col=index_col)

def get_tradedate_data(sdt,edt):
    sql = "select date_format(trading_day, '%%Y-%%m-%%d') dt from eqchina.tradingday order by trading_day asc"
    ts = get_sql_data(sql)
    ts = ts[(pd.to_datetime(ts['dt']) >= pd.to_datetime(sdt))&\
            (pd.to_datetime(ts['dt']) <= pd.to_datetime(edt))]
    return ts['dt'].tolist()

def get_next_ndt(dt, nday=1):
    sql = "select date_format(trading_day, '%%Y-%%m-%%d') dt from eqchina.tradingday order by trading_day asc"
    dts = pd.read_sql(sql, get_engine())['dt']
    return dts[np.where(dts >= dt)[0][0] + nday]

def get_data_macro(sec_id,sdt,edt):
    sql = "select security_id sec_id, date_format(date, '%%Y-%%m-%%d') date, \
           date_format(update_date, '%%Y-%%m-%%d') update_date, data \
           from macro.macroindicator where security_id = '"+sec_id+"' and \
           date >= '"+sdt+"'  and date <= '"+edt+"' order by date asc"
    ts = get_sql_data(sql)
    return ts

def update_macro_data(ticker,sdt,edt):
    dict_t = {
              'cn_yields_10y':'GROWTH_CN_00806',
              'us_yields_10y':'US_CN_00133',
              }
    df = pd.read_csv(os.path.join(data_path,ticker+'.csv'))
    df_new = get_data_macro(dict_t[ticker],sdt,edt)
    df = df.append(df_new)
    df = df[~df['date'].duplicated(keep='last')]
    df.to_csv(os.path.join(data_path,ticker+'.csv'),index = None)
    print(ticker+' from '+sdt+' to '+edt+' update!')

def update_IC00_ret(sdt,edt):
    df_return_ori = pd.read_csv(os.path.join(data_path,'IC00_pct_chg.csv'))
    ls_dt = get_tradedate_data(sdt,edt)
    df_return = pd.DataFrame()
    for date in ls_dt:
        sql = "select future_security_code ticker, date_format(last_trade_date, '%%Y-%%m-%%d') ls_date from eqchina.futureinfo \
               where benchmark = '000905.SH' and contract_issue_date <= '"+date+"' and last_trade_date >= '"+date+"'"
        df_info = get_sql_data(sql)
        IC00_ticker = df_info['ticker'].tolist()[0]
        sql = "select date_format(trade_date, '%%Y-%%m-%%d') date, close  from eqchina.futureprice \
               where future_security_code = '"+IC00_ticker+"' and trade_date in ('"+date+"', '"+get_next_ndt(date, -1)+"')"
        df_price_temp = get_sql_data(sql)
        df_return_temp = pd.DataFrame([date,df_price_temp['close'].tolist()[1]/df_price_temp['close'].tolist()[0]-1],index = ['date','IC00']).T
        df_return = df_return.append(df_return_temp)
    df_return_ori = df_return_ori.append(df_return)
    df_return_ori = df_return_ori[~df_return_ori['date'].duplicated(keep='last')]
    df_return_ori.to_csv(os.path.join(data_path,'IC00_pct_chg.csv'),index = None)
    print('IC00 from '+sdt+' to '+edt+' update!')

#-----------------------BBG----------------------------------------------------#

def us_macro(start_date, end_date, tickers, flds):
    """
    取美国宏观数据
    ticker 和 flds 都是 list, 因为可以一次取多个
    """
    tickers = ','.join(tickers)
    flds = ','.join(flds)

    bdh = http_post(
        url=BBG_SERVER_URL + "BDH",
        data={
            "code": tickers,
            "flds": flds,
            "startDate": start_date,
            "endDate": end_date
        }
    )
    try:
        data_dict = json.loads(bdh.text)
        flds_name = ['/'.join(i).rstrip() for i in data_dict['columns']]
        price = pd.DataFrame(data_dict['data'], columns=flds_name)
        price.set_index('Date/', inplace=True)
        price = price.T.stack().reset_index()
        price.columns = ['NAME', 'DATE', 'VALUE']
        price = price.assign(
            FIELD=price.NAME.apply(lambda x: x.split('/')[1]),
            TICKER=price.NAME.apply(lambda x: x.split('/')[0]),
        )
    except Exception as e:
        print(e)
        print(f'{tickers}:取数失败！')
        return pd.DataFrame()
    
    return price


def http_post(url, data):
    response = ''
    for it in range(0, 5):
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            response = requests.post(url=url, data=data, headers=headers, timeout=40)
            break
        except Exception as e:
            print(e)
            continue

    return response

def update_CDS(ticker,sdt,edt):
    # ticker: CHINAGOV CDS USD SR 5Y Corp
    tickers = [ticker]
    flds = ['PX_LAST']
    df = pd.read_csv(os.path.join(data_path,'CDS_5Y.csv'),index_col = 'Unnamed: 0')
    df.index = list(map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),df.index.tolist()))
    data_ori = us_macro(sdt,edt,tickers,flds)
    data = pd.DataFrame()
    data['Last Price'] = data_ori['VALUE']
    data.index = data_ori['DATE']
    df = df.append(data)
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(os.path.join(data_path,'CDS_5Y.csv'))
    print('CDS 5Y from '+sdt+' to '+edt+' update!')

#-----------------------IFIND--------------------------------------------------

def update_option_callput_voloi(ticker,sdt,edt):
    # ticker: 510050.SH
    ls_dt = get_tradedate_data(sdt,edt)
    df = pd.read_csv(os.path.join(data_path,'option_callput_voloi.csv'),index_col = 'Unnamed: 0')
    df.index = list(map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),df.index.tolist()))
    data = pd.DataFrame()
    for date in ls_dt:
        data_ori = ifind.THS_BD(ticker,\
                                'ths_option_call_volume_option;ths_option_put_volume_option;ths_option_call_oi_option;ths_option_put_oi_option;ths_option_total_oi_pcr_option',\
                                date+';'+date+';'+date+';'+date+';'+date).data
        data_ori.index = [date]
        data = data.append(data_ori)
    df = df.append(data)
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(os.path.join(data_path,'option_callput_voloi.csv'))
    print('option call&put vol&oi from '+sdt+' to '+edt+' update!')                  
      
#-----------------------WIND---------------------------------------------------#

def update_index_field(ticker,field,sdt,edt,days = 'td'):
    # ticker: H11025.CSI, H00905.CSI, 000905.SH. 
    #         510800.SH,
    #         SH_5100501MSF.WI, SH_510050IV.WI, SH_510050SKEW.WI
    #         IC00.CFE
    #         USDCNH.FX
    # field: close, pct_chg, pr_ttm, pb_lf, nav
    df = pd.read_csv(os.path.join(data_path,ticker+'_'+field+'.csv'),index_col = 'Unnamed: 0')
    df.index = list(map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),df.index.tolist()))
    if ticker in ['H11025','H00905']:
        ex = '.CSI'
    elif ticker in ['000905','510800']:
        ex = '.SH'
    elif ticker in ['SH_5100501MSF','SH_510050IV','SH_510050SKEW']:
        ex = '.WI'
    elif ticker in ['IC00']:
        ex = '.CFE'
    elif ticker in ['USDCNH']:
        ex = '.FX'
    if days == 'td':
        data_ori = w.wsd(ticker+ex,field,sdt,edt,'',usedf = True)[1]
    elif days == 'ad':
        data_ori = w.wsd(ticker+ex,field,sdt,edt,'Days=Alldays',usedf = True)[1]
    if len(data_ori) == 1:
        data_ori.index = [sdt]
    data_ori.columns = df.columns
    data_ori.index = list(map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),data_ori.index.tolist()))
    df = df.append(data_ori)
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(os.path.join(data_path,ticker+'_'+field+'.csv'))
    print(ticker+'_'+field+' from '+sdt+' to '+edt+' update!')
    
def update_margin_short(sdt,edt):
    df = pd.read_csv(os.path.join(data_path,'margin_short_balance.csv'),index_col = 'Unnamed: 0')
    df.index = list(map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),df.index.tolist()))
    data_ori = w.wset('markettradingstatistics(value)',\
                      'startdate='+sdt+';enddate='+edt+';frequency=day;sort=asc;field=end_date,margin_balance,short_balance',\
                      usedf = True)[1]
    data_ori.index = map(lambda x:pd.to_datetime(x).strftime('%Y-%m-%d'),data_ori['end_date'].tolist())
    data_ori = data_ori[df.columns]
    df = df.append(data_ori)
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(os.path.join(data_path,'margin_short_balance.csv'))
    print('margin&short balance from '+sdt+' to '+edt+' update!')


if __name__ == "__main__":
#    edt = '2026-03-09'
#    sdt = '2026-03-03'
    dt = str(datetime.datetime.now().date())
    edt = str(w.tdaysoffset(-1, dt, "").Data[0][0])[:10]
    sdt = str(w.tdaysoffset(-6, dt, "").Data[0][0])[:10]
    # sql
    update_macro_data('cn_yields_10y',sdt,edt)
    update_macro_data('us_yields_10y',sdt,edt)
    update_IC00_ret(sdt,edt)
    # bbg
    update_CDS('CHINAGOV CDS USD SR 5Y Corp',sdt,edt)
    # ifind
    update_option_callput_voloi('510050.SH',sdt,edt)
    # wind
    update_index_field('H11025','close',sdt,edt)
    update_index_field('H00905','close',sdt,edt)
    update_index_field('000905','close',sdt,edt)
    update_index_field('000905','pe_ttm',sdt,edt)
    update_index_field('000905','pb_lf',sdt,edt)
    update_index_field('510800','nav',sdt,edt)
    update_index_field('SH_5100501MSF','close',sdt,edt)
    update_index_field('SH_510050IV','close',sdt,edt)
    update_index_field('SH_510050SKEW','close',sdt,edt)
    update_index_field('USDCNH','close',sdt,edt,days = 'ad')
    update_margin_short(sdt,edt)
    

