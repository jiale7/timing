# -*- coding: utf-8 -*-
import smtplib
import pandas as pd
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SMTP_SERVER = "smtp.exmail.qq.com"
SMTP_PORT = 465

SENDER_EMAIL = "zyzhang@9m-inv.com"
SENDER_PASSWORD = "Zyzhang0608"
RECEIVER_EMAILS = ["alpha@9m-inv.com","jyli@9m-inv.com","zyzhang@9m-inv.com"]
#RECEIVER_EMAILS = ["zyzhang@9m-inv.com"]
SIGNAL_FOLDER = r"\\jy305_server\LiveTrack\EQTools\zhengyang_tools\timing\signal"

#SENDER_EMAIL = "jyli@9m-inv.com"
#SENDER_PASSWORD = "Ljy121025."
#RECEIVER_EMAILS = ["jyli@9m-inv.com"]
#SIGNAL_FOLDER = r"\\jy305_server\ResearchCenter\jyli\IndextTiming\signal"

DAYS_TO_SHOW = 20

SIGNAL_FILENAMES = [
                    "AIAE_signal.xlsx", "amt_signal.xlsx", "basis_signal.xlsx", "CDS_signal.xlsx",
                    "growth_signal.xlsx", "highlow_signal.xlsx", "IC_basis_signal.xlsx",
                    "idy_amt_corr_signal.xlsx", "idy_turn_pct_signal.xlsx", "inflation_signal.xlsx",
                    "margin_signal.xlsx", "monetarycondition_signal.xlsx", "monetarypolicy_signal.xlsx",
                    "pb_signal.xlsx", "pcr_signal.xlsx", "pe_signal.xlsx", "sc_amt_pct_signal.xlsx",
                    "skew_signal.xlsx", "spread_signal.xlsx", "std_signal.xlsx", "trend_signal.xlsx",
                    "usdcnh_signal.xlsx", "vol_signal.xlsx", "ht_signal.xlsx",
                    "macro_cat_signal.xlsx", "value_cat_signal.xlsx", "option_cat_signal.xlsx", "moneyflow_cat_signal.xlsx", "price_cat_signal.xlsx", "cmb_signal.xlsx"
                    ]

CATEGORIES = {
              "宏观": ["growth", "inflation", "monetarycondition", "monetarypolicy", "CDS", "usdcnh", "spread"],
              "估值": ["pe", "pb", "AIAE"],
              "期权": ["basis", "vol", "skew", "pcr"],
              "资金": ["margin", "amt", "sc_amt_pct", "idy_amt_corr", "idy_turn_pct"],
              "价格": ["trend", "highlow", "std", "IC_basis", "ht"],
              "大类打分": ["macro_cat", "value_cat", "option_cat", "moneyflow_cat", "price_cat"],
              "最终权重": ["cmb"],
              }

def load_signals_from_folder(folder_path):
    merged_df = None
    for fn in SIGNAL_FILENAMES:
        fp = os.path.join(folder_path, fn)
        if not os.path.exists(fp): continue
        try:
            df = pd.read_excel(fp, usecols=[0, 1])
            if df.empty: continue
            name = fn.replace('_signal.xlsx', '')
            df.columns = ['date', name]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            merged_df = df if merged_df is None else merged_df.join(df, how='outer')
        except Exception as e:
            print(f"❌ {fn}: {e}")
            continue
    if merged_df is None:
        raise ValueError("无有效信号文件")
    merged_df.reset_index(inplace=True)
    merged_df.sort_values('date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

def get_style(val):
    if pd.isna(val):
        return '#f0f0f0', '#999'
    try:
        v = float(val)
        if v < 0 or v > 1:
            return '#ffffff', '#000'

        if v == 0.5:
            bg = '#ffffff'
            fg = '#000'
        elif v < 0.5:
            ratio = v / 0.5
            r = int(0 + 255 * ratio)
            g = int(128 + 127 * ratio)
            b = int(0 + 255 * ratio)
            bg = f'#{r:02x}{g:02x}{b:02x}'
            fg = '#fff' if v <= 0 else '#000'
        else:
            ratio = (v - 0.5) / 0.5
            r = int(255 - 35 * ratio)
            g = int(255 - 255 * ratio)
            b = int(255 - 255 * ratio)
            bg = f'#{r:02x}{g:02x}{b:02x}'
            fg = '#fff' if v >= 1 else '#000'
        return bg, fg
    except:
        return '#ffffff', '#000'

def generate_heatmap_html(df):
    all_cols = [c for c in df.columns if c != 'date']
    ordered_cols, right_border_cols = [], set()
    cat_cols = ["macro_cat", "value_cat", "option_cat", "moneyflow_cat", "price_cat", "cmb"]
    
    category_header = "<tr style='border-bottom:2px solid black;'><th>Date</th>"
    for cat, factors in CATEGORIES.items():
        valid_factors = [f for f in factors if f in all_cols]
        if valid_factors:
            ordered_cols.extend(valid_factors)
            right_border_cols.add(valid_factors[-1])
            category_header += f"<th colspan='{len(valid_factors)}' style='background:#f8f9fa;border-bottom:2px solid black;text-align:center;font-weight:bold;padding:6px;'>{cat}</th>"
                
    factor_header = "<tr><th>Date</th>"
    for c in ordered_cols:
        # 自定义换行逻辑
        if c == "monetarycondition":
            name = "MC<br>Cond"
        elif c == "monetarypolicy":
            name = "MP<br>Pol"
        elif '_' in c:
            name = '<br>'.join(c.split('_'))
        else:
            name = c  # 如 growth, pe, pb, CDS 等保持原样，不换行
        
        style = "padding:6px;text-align:center;font-size:11px;vertical-align:middle;"
        if c in right_border_cols:
            style += " border-right:2px solid black;"
        factor_header += f"<th style='{style}'>{name}</th>"

    df = df.copy()
    df['date'] = df['date'].dt.strftime('%m/%d')  # 只显示月日
    rows = []
    for i, r in df.iterrows():
        cls = "tr-even" if i % 2 == 0 else "tr-odd"
        row = f'<tr class="{cls}"><td class="date-cell">{r["date"]}</td>'
        for c in ordered_cols:
            val = r[c]
            bg, fg = get_style(val)
            if c in cat_cols:
                disp = "-" if pd.isna(val) else f"{float(val):.2f}" if isinstance(val, (int, float)) else str(val)
            else:
                disp = "-" if pd.isna(val) else f"{float(val):.1f}" if isinstance(val, (int, float)) else str(val)
            cell_style = f"background-color:{bg};color:{fg};font-weight:bold;padding:6px;font-size:11px;"
            if c in right_border_cols:
                cell_style += " border-right:2px solid black;"
            row += f'<td style="{cell_style}">{disp}</td>'
        rows.append(row)
    
    body = "\n".join(rows)
    start, end = df['date'].iloc[0], df['date'].iloc[-1]
    
    return f"""
    <html><head><meta charset="utf-8"><style>
        body {{
            font-family: Segoe UI, Tahoma, sans-serif;
            background: #f4f4f4;
            padding: 20px;
            margin: 0;
        }}
        .container {{
            max-width: 98%;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        h2 {{
            color: #333;
            border-left: 5px solid #0056b3;
            padding-left: 15px;
            margin-top: 0;
        }}
        .info-box {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #555;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            min-width: 100%;
            font-size: 11px;
            table-layout: auto; /* 改为 auto，让列宽自适应内容 */
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: center;
            vertical-align: middle;
            overflow: hidden;
        }}
        th {{
            background: #f8f9fa;
            color: #495057;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: normal;      /* 允许 <br> 生效 */
            word-wrap: break-word;    /* 仅对未手动换行的内容起作用 */
            min-height: 36px;
            min-width: 40px;          /* 防止列太窄导致自动断词 */
        }}
        .tr-even {{ background: #fff; }}
        .tr-odd {{ background: #f9f9f9; }}
        .tr-even:hover, .tr-odd:hover {{ background: #f1f1f1; }}
        .date-cell {{
            font-weight: bold;
            color: #0056b3;
            text-align: center;
            min-width: 50px;
            width: 50px;
        }}
        .score-col, .final-weight-col {{
            min-width: 50px !important;
            width: 50px !important;
            max-width: 50px;
        }}
    </style></head><body>
        <div class="container">
            <h2>📊 指数择时日频信号</h2>
            <div class="info-box">
                <strong>📅 时间范围：</strong> {start} 至 {end}<br>
                <strong>📈 覆盖交易日：</strong> {len(df)} 天<br>
                <strong>🤖 生成时间：</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <table>
                <thead>
                    {category_header}
                    {factor_header}
                </thead>
                <tbody>
                    {body}
                </tbody>
            </table>
        </div>
    </body></html>
    """

def send_latest_signal_email():
    print("📧 正在准备发送【指数择时】信号邮件...")
    try:
        df = load_signals_from_folder(SIGNAL_FOLDER)
        df_latest = df.iloc[-DAYS_TO_SHOW:].copy()
        start_date = df_latest['date'].iloc[0]
        end_date = df_latest['date'].iloc[-1]
        print(f"✅ 成功加载 {len(df.columns)-1} 个信号，数据范围：{start_date.date()} 至 {end_date.date()}")
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        return False

    html_content = generate_heatmap_html(df_latest)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"【指数择时】信号汇总 ({start_date.strftime('%m.%d')}-{end_date.strftime('%m.%d')})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        print("✅ 邮件发送成功！")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败：{e}")
        return False

if __name__ == "__main__":
    send_latest_signal_email()