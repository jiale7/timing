# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =================配置区域=================
SIGNAL_PATH = r"C:\Users\jltao\Desktop\timing\signal"
INDEX_CLOSE_PATH = "//eqserver/data/trade/index_close.pkl"
INDEX_OPEN_PATH = "//eqserver/data/trade/index_open.pkl"
OUTPUT_PATH = r"C:\Users\jltao\Desktop\signal"

BACKTEST_CONFIG = {
    'start_date': '2016-01-01',
    'end_date': '2026-04-03',
    'index_code': 'I000905',
    'exec_price_type': 'close',  # 'open' or 'close0' or 'close'
    'fee_rate': 0,
    'slippage_rate': 0,
    'rebalance_freq': 'weekly',      # 'daily', 'weekly', 'monthly'
    'rebalance_weekday': 4,          # 0=Monday, ..., 4=Friday (仅周频生效)
    'rebalance_monthday': -1         # int: 1=首交易日, -1=末交易日 (仅月频生效)
}

signal_file_names = [
                    "IC_basis_signal",
                    "pcr_signal",
                    "skew_signal",
                    "spread_signal",
                    "usdcnh_signal",
                    "vol_signal",
                    "basis_signal",
                    "highlow_signal",
                    "std_signal",
                    "idy_turn_pct_signal",
                    "trend_signal",
                    "idy_amt_corr_signal",
                    "sc_amt_pct_signal",
                    "amt_signal",
                    "CDS_signal",
                    "margin_signal",
                    "AIAE_signal",
                    "monetarypolicy_signal",
                    "pb_signal",
                    "pe_signal",
                    "monetarycondition_signal",
                    "inflation_signal",
                    "growth_signal",
                    "ht_signal",
                    "macro_cat_signal",
                    "value_cat_signal",
                    "option_cat_signal",
                    "moneyflow_cat_signal",
                    "price_cat_signal",
                    "cmb_signal",
                    ]
# ==========================================


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.csv':
        return pd.read_csv(file_path)
    elif ext.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    elif ext.lower() == '.pkl':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"不支持的格式: {ext}")


def load_signal(signal_name):
    fp = os.path.join(SIGNAL_PATH, signal_name)
    df = load_data(fp)
    if df.shape[1] < 2:
        raise ValueError("信号文件至少需两列：日期 + 信号")
    df = df.iloc[:, :2].copy()
    df.columns = ['date', 'signal']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
    return df['signal'].dropna()


def load_index_price(index_code, price_type="close"):
    fp = INDEX_CLOSE_PATH if price_type == "close" else INDEX_OPEN_PATH
    df = load_data(fp)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    df.columns = df.columns.astype(str).str.strip()
    if index_code not in df.columns:
        raise ValueError(f"指数 {index_code} 不存在。")
    return df[[index_code]].rename(columns={index_code: 'price'})


def generate_rebalance_dates(trading_days, freq, **kwargs):
    if freq == 'daily':
        return trading_days

    trading_days = trading_days.copy()

    if freq == 'weekly':
        weekday = kwargs.get('weekday', 0)
        mask = trading_days.weekday == weekday
        return trading_days[mask]

    elif freq == 'monthly':
        monthday = kwargs.get('monthday', -1)
        if monthday == 0:
            raise ValueError("monthday cannot be 0")

        df_temp = pd.DataFrame(index=trading_days)
        df_temp['year'] = trading_days.year
        df_temp['month'] = trading_days.month
        grouped = df_temp.groupby(['year', 'month'],as_index = False)

        if monthday > 0:
            selected = grouped.nth(monthday - 1)
        else:
            selected = grouped.nth(monthday)

        selected = selected.dropna()
        return selected.index

    else:
        raise ValueError(f"Unsupported freq: {freq}")


def run_backtest_t_plus_1(signal, config):
    open_df = load_index_price(config['index_code'], 'open').rename(columns={'price': 'open_price'})
    close_df = load_index_price(config['index_code'], 'close').rename(columns={'price': 'close_price'})
    
    # 合并信号行情
    df = pd.concat([signal, open_df, close_df], axis=1).sort_index()
    df = df.dropna(subset=['open_price', 'close_price'])
    
    # 生成调仓日
    trading_days = df.index.copy()

    rebalance_dates = generate_rebalance_dates(
        trading_days,
        #判断日频，周频还是月频
        freq=config['rebalance_freq'],
        weekday=config.get('rebalance_weekday', 0),
        monthday=config.get('rebalance_monthday', -1)
    )
    
    # 稀疏信号（只有调仓日才生成）
    signal_sparse = pd.Series(np.nan, index=trading_days, dtype='float64')
    valid_rebalance = rebalance_dates.intersection(trading_days)

    if not valid_rebalance.empty:
        signal_sparse.loc[valid_rebalance] = signal.reindex(valid_rebalance, method='ffill')
    
    # 仓位生成逻辑
    df['position'] = signal_sparse.shift(1).ffill().fillna(0)
    # 换手率
    df['turnover'] = df['position'].diff().abs().fillna(0)
    
    
    df['index_return'] = df['close_price'].pct_change()
    df['c2o'] = df['open_price']/df['close_price'].shift(1)-1
    df['o2c'] = (df['close_price']-df['open_price'])/df['close_price'].shift(1)   
    
    # 交易手续费，滑点
    total_cost_rate = config['fee_rate'] + config['slippage_rate']
    df['trade_cost'] = df['turnover'] * total_cost_rate
    
    
    if config['exec_price_type'] == 'close0':
        df['strategy_return'] = (df['position'] * df['index_return']) - df['trade_cost']
    elif config['exec_price_type'] == 'close':
        df['strategy_return'] = (df['position'].shift(1).fillna(0) * df['index_return']) - df['trade_cost']
    elif config['exec_price_type'] == 'open':
        df['strategy_return'] = (df['position'].shift(1).fillna(0) * df['c2o'] + \
                                 df['position'] * df['o2c']) - df['trade_cost']

    df = df.loc[config['start_date']:config['end_date']].copy()
    df['index_nav'] = (1 + df['index_return'].fillna(0)).cumprod()
    df['strategy_nav'] = (1 + df['strategy_return'].fillna(0)).cumprod()

    return df


def calc_performance_metrics(rets):
    if rets.empty or rets.isna().all():
        return pd.Series({'年化收益': 0, '年化波动': 0, '夏普比': 0, '最大回撤': 0})

    nav = (1 + rets).cumprod()
    total_ret = nav.iloc[-1] - 1
    days = (rets.index[-1] - rets.index[0]).days
    ann_return = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    drawdown = nav / nav.cummax() - 1
    max_dd = drawdown.min()

    return pd.Series({
        '年化收益': ann_return,
        '年化波动': ann_vol,
        '夏普比': sharpe,
        '最大回撤': max_dd
    })


def calc_excess_metrics(excess_rets):
    """计算超额绩效：使用算术年化，返回信息比率"""
    if excess_rets.empty or excess_rets.isna().all():
        return pd.Series({'超额_收益': 0, '超额_波动': 0, '信息比率': 0, '超额_回撤': 0})

    ann_return = excess_rets.mean() * 252
    ann_vol = excess_rets.std() * np.sqrt(252)
    ir = ann_return / ann_vol if ann_vol != 0 else 0

    # 累计超额净值（用于回撤）
    excess_nav = (1 + excess_rets).cumprod()
    drawdown = excess_nav / excess_nav.cummax() - 1
    max_dd = drawdown.min()

    return pd.Series({
        '超额_收益': ann_return,
        '超额_波动': ann_vol,
        '信息比率': ir,
        '超额_回撤': max_dd
    })


def generate_performance_table(result):
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    result_by_year = result.groupby(result.index.year)
    data = []
    for year, group in result_by_year:
        strat_ret = group['strategy_return']
        idx_ret = group['index_return']
        excess_ret = strat_ret - idx_ret

        strat_perf = calc_performance_metrics(strat_ret)
        idx_perf = calc_performance_metrics(idx_ret)
        excess_perf = calc_excess_metrics(excess_ret)

        turnover_rate = group['turnover'].sum()  # 年度总换手量

        row = {
            'year': year,
            '策略_收益': strat_perf['年化收益'],
            '策略_波动': strat_perf['年化波动'],
            '策略_夏普': strat_perf['夏普比'],
            '策略_回撤': strat_perf['最大回撤'],
            '指数_收益': idx_perf['年化收益'],
            '指数_波动': idx_perf['年化波动'],
            '指数_夏普': idx_perf['夏普比'],
            '指数_回撤': idx_perf['最大回撤'],
            '超额_收益': excess_perf['超额_收益'],
            '超额_波动': excess_perf['超额_波动'],
            '信息比率': excess_perf['信息比率'],
            '超额_回撤': excess_perf['超额_回撤'],
            '换手率': turnover_rate
        }
        data.append(row)

    perf_df = pd.DataFrame(data).set_index('year')

    total_strat = calc_performance_metrics(result['strategy_return'])
    total_idx = calc_performance_metrics(result['index_return'])
    total_excess = calc_excess_metrics(result['strategy_return'] - result['index_return'])

    # === 计算年化换手率 ===
    total_turnover = result['turnover'].sum()
    start_date = result.index[0]
    end_date = result.index[-1]
    years = (end_date - start_date).days / 365.25
    annualized_turnover = total_turnover / years if years > 0 else 0

    total_row = {
        'year': 'total',
        '策略_收益': total_strat['年化收益'],
        '策略_波动': total_strat['年化波动'],
        '策略_夏普': total_strat['夏普比'],
        '策略_回撤': total_strat['最大回撤'],
        '指数_收益': total_idx['年化收益'],
        '指数_波动': total_idx['年化波动'],
        '指数_夏普': total_idx['夏普比'],
        '指数_回撤': total_idx['最大回撤'],
        '超额_收益': total_excess['超额_收益'],
        '超额_波动': total_excess['超额_波动'],
        '信息比率': total_excess['信息比率'],
        '超额_回撤': total_excess['超额_回撤'],
        '换手率': annualized_turnover  # ← 使用年化换手率
    }
    total_df = pd.DataFrame([total_row]).set_index('year')
    perf_df = pd.concat([perf_df, total_df])
    return perf_df


def plot_backtest_charts(result, signal_raw, config, output_dir="."):
    plot_data = result.copy()
    plot_data['excess_nav'] = plot_data['strategy_nav'] - plot_data['index_nav']
    signal_plot = signal_raw.loc[config['start_date']:config['end_date']].ffill()

    fig1, ax1_left = plt.subplots(figsize=(12, 6))
    ax1_right = ax1_left.twinx()

    ax1_left.plot(plot_data.index, plot_data['close_price'], color='tab:blue', label='指数收盘价')
    ax1_left.set_ylabel('指数价格', color='black')
    ax1_left.tick_params(axis='y', colors='black')
    ax1_left.tick_params(axis='x', colors='black')

    ax1_right.plot(signal_plot.index, signal_plot, color='tab:orange', alpha=0.7, label='信号权重')
    ax1_right.set_ylabel('信号权重', color='black')
    ax1_right.tick_params(axis='y', colors='black')
    ax1_right.set_ylim(-0.1, 1.1)

    fig1.suptitle(f"{config['index_code']} 指数价格与信号权重", fontsize=14, color='black')

    lines, labels = ax1_left.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1_left.legend(lines + lines2, labels + labels2, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_path = os.path.join(output_dir, "chart_signal_price.png")
    fig1.savefig(fig1_path, dpi=100)
    plt.close(fig1)

    fig2, ax2_left = plt.subplots(figsize=(12, 6))
    ax2_right = ax2_left.twinx()

    ax2_left.plot(plot_data.index, plot_data['index_nav'], color='gray', linestyle='-', label='指数净值')
    ax2_left.plot(plot_data.index, plot_data['strategy_nav'], color='tab:red', linewidth=2, label='策略净值')
    ax2_left.set_ylabel('累计净值', color='black')
    ax2_left.tick_params(axis='y', colors='black')
    ax2_left.tick_params(axis='x', colors='black')

    ax2_right.fill_between(plot_data.index, plot_data['excess_nav'], 0, color='tab:green', alpha=0.3, label='超额收益(右轴)')
    ax2_right.set_ylabel('超额收益', color='black')
    ax2_right.tick_params(axis='y', colors='black')

    fig2.suptitle(f"回测绩效: {config['index_code']} (成交价: {config['exec_price_type']})", fontsize=14, color='black')

    lines, labels = ax2_left.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(output_dir, "chart_performance.png")
    fig2.savefig(fig2_path, dpi=100)
    plt.close(fig2)

    return fig1_path, fig2_path


def save_report_with_charts(result, perf_table, fig_paths, output_file):
    col_map = {
        '策略_收益': '策略\n收益',
        '策略_波动': '策略\n波动',
        '策略_夏普': '策略\n夏普',
        '策略_回撤': '策略\n回撤',
        '指数_收益': '指数\n收益',
        '指数_波动': '指数\n波动',
        '指数_夏普': '指数\n夏普',
        '指数_回撤': '指数\n回撤',
        '超额_收益': '超额\n收益',
        '超额_波动': '超额\n波动',
        '信息比率': '信息\n比率',
        '超额_回撤': '超额\n回撤',
        '换手率': '换手\n率'
    }
    rpt_df = perf_table.rename(columns=col_map).copy()

    # ==================== 转换为百分比（仅收益/波动/回撤）====================
    percent_cols = ['策略\n收益', '策略\n波动', '策略\n回撤',
                    '指数\n收益', '指数\n波动', '指数\n回撤',
                    '超额\n收益', '超额\n波动', '超额\n回撤']
    
    # 收益/波动/回撤 ×100 → 显示为百分比数值（如 0.2 → 20.0）
    for col in percent_cols:
        rpt_df[col] = rpt_df[col].apply(lambda x: '{:.2%}'.format(x))

    # ❌ 不再对换手率 ×100！保留原始值（如 6.3 表示年换手 6.3 倍）
    rpt_df['换手\n率'] = rpt_df['换手\n率'].round(1)  # 保留一位小数即可

    # 夏普、信息比率保留三位小数
    rpt_df['策略\n夏普'] = rpt_df['策略\n夏普'].round(2)
    rpt_df['指数\n夏普'] = rpt_df['指数\n夏普'].round(2)
    rpt_df['信息\n比率'] = rpt_df['信息\n比率'].round(2)

    # ==================== 创建 Excel 写入器 ====================
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Performance')

    # ==================== 设置格式 ====================
    header_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})
    subheader_format = workbook.add_format({'align': 'center', 'border': 1})
    percent_format = workbook.add_format({'num_format': '0.0', 'align': 'center'})      # 百分比列
    decimal_format = workbook.add_format({'num_format': '0.000', 'align': 'center'})    # 夏普等
    float_format = workbook.add_format({'num_format': '0.0', 'align': 'center'})        # 换手率等普通数值
    first_col_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})  # 第一列

    # ==================== 写入列名（主标题 + 子标题）====================
    years = rpt_df.index.tolist()  # 包含 'total'
    strategy_cols = ['策略\n收益', '策略\n波动', '策略\n夏普', '策略\n回撤']
    index_cols = ['指数\n收益', '指数\n波动', '指数\n夏普', '指数\n回撤']
    excess_cols = ['超额\n收益', '超额\n波动', '信息\n比率', '超额\n回撤']
    turnover_col = ['换手\n率']
    all_cols = strategy_cols + index_cols + excess_cols + turnover_col

    # 主标题（策略 / 指数 / 超额 / 换手率）
    col_idx = 1
    for main_title, cols in [('策略', strategy_cols), ('指数', index_cols), ('超额', excess_cols)]:
        worksheet.merge_range(0, col_idx, 0, col_idx + len(cols) - 1, main_title, header_format)
        col_idx += len(cols)
    worksheet.write(0, col_idx, '换手\n率', header_format)

    # 子标题（收益 / 波动 / 夏普 / 回撤）
    row = 1
    col_idx = 1
    for cols in [strategy_cols, index_cols, excess_cols]:
        for sub in ['收益', '波动', '夏普', '回撤']:
            worksheet.write(row, col_idx, sub, subheader_format)
            col_idx += 1
    worksheet.write(row, col_idx, '', subheader_format)

    # “年份”标题
    worksheet.write(0, 0, '年份', header_format)

    # ==================== 写入数据（包括年份列）====================
    data_row_start = 2
    n_rows = len(years)

    for i, year in enumerate(years):
        # ✅ 第一列：年份（全部加粗+边框，无重复）
        worksheet.write(data_row_start + i, 0, year, first_col_format)

        # 数据列
        for j, col_name in enumerate(all_cols):
            val = rpt_df.loc[year, col_name]
            if col_name in percent_cols:
                worksheet.write(data_row_start + i, j + 1, val, percent_format)
            elif col_name in ['策略\n夏普', '指数\n夏普', '信息\n比率']:
                worksheet.write(data_row_start + i, j + 1, val, decimal_format)
            else:  # 包括 换手\n率
                worksheet.write(data_row_start + i, j + 1, val, float_format)

    # ==================== 设置列宽 ====================
    worksheet.set_column(0, 0, 8)   # 年份列
    worksheet.set_column(1, len(all_cols), 10)

    # ==================== 插入图表 ====================
    worksheet.insert_image('O2', fig_paths[0], {'x_scale': 0.8, 'y_scale': 0.8})
    worksheet.insert_image('O30', fig_paths[1], {'x_scale': 0.8, 'y_scale': 0.8})

    result.to_excel(writer,sheet_name = 'backtest')

    writer.close()

    # 删除临时图
    for path in fig_paths:
        if os.path.exists(path):
            os.remove(path)


# =================主程序=================
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 定义信号分类
    SIGNAL_CATEGORIES = {
        "宏观": ["growth", "inflation", "monetarycondition", "monetarypolicy", "CDS", "usdcnh", "spread"],
        "估值": ["pe", "pb", "AIAE"],
        "期权": ["basis", "vol", "skew", "pcr"],
        "量价": ["margin", "amt", "sc_amt_pct", "idy_amt_corr", "idy_turn_pct", "trend", "highlow", "std", "IC_basis", "ht"]
    }

    valid_signal_files = [name + ".xlsx" for name in signal_file_names]
    if not valid_signal_files:
        print("❌ 没有找到任何有效的信号文件！请检查路径或文件名。")
        exit(1)

    print(f"\n🔍 共找到 {len(valid_signal_files)} 个有效信号文件，开始批量回测...\n")

    # === 第一步：逐个回测原始信号，并缓存原始信号数据 ===
    all_signals = {}
    for signal_file_name in valid_signal_files:
        base_name = os.path.splitext(signal_file_name)[0]
        output_report_file = os.path.join(OUTPUT_PATH, f"{base_name}_report.xlsx")

        print(f"\n{'='*60}")
        print(f"▶ 正在处理信号: {signal_file_name}")
        print(f"  → 输出路径: {output_report_file}")
        print(f"{'='*60}")

        try:
            signal_raw = load_signal(signal_file_name)
            all_signals[base_name] = signal_raw

            result = run_backtest_t_plus_1(signal_raw, BACKTEST_CONFIG)
            perf_table = generate_performance_table(result)
            fig_paths = plot_backtest_charts(result, signal_raw, BACKTEST_CONFIG, OUTPUT_PATH)
            save_report_with_charts(result, perf_table, fig_paths, output_report_file)

            print(f"✅ 成功完成: {os.path.basename(output_report_file)}")

        except Exception as e:
            print(f"❌ 处理 {signal_file_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

#    # === 第二步：按类别合成信号并回测 ===
#    category_signals = {}
#    for cat_name, signal_list in SIGNAL_CATEGORIES.items():
#        print(f"\n{'='*60}")
#        print(f"▶ 合成类别信号: {cat_name}")
#        print(f"  包含信号: {signal_list}")
#        print(f"{'='*60}")
#
#        available_signals = []
#        for sig in signal_list:
#            full_name = sig + "_signal"
#            if full_name in all_signals:
#                available_signals.append(all_signals[full_name])
#            else:
#                print(f"  ⚠️ 信号 {full_name} 未成功加载，跳过。")
#
#        if not available_signals:
#            print(f"  ❌ 类别 {cat_name} 无有效信号，跳过。")
#            continue
#
#        common_index = available_signals[0].index
#        for s in available_signals[1:]:
#            common_index = common_index.intersection(s.index)
#        common_index = common_index.sort_values()
#
#        if len(common_index) == 0:
#            print(f"  ❌ 类别 {cat_name} 无公共交易日，跳过。")
#            continue
#
#        aligned_signals = [s.reindex(common_index).fillna(method='ffill').fillna(0) for s in available_signals]
#        combined_signal = pd.concat(aligned_signals, axis=1).mean(axis=1)
#        combined_signal.name = 'signal'
#
#        try:
#            result = run_backtest_t_plus_1(combined_signal, BACKTEST_CONFIG)
#            perf_table = generate_performance_table(result)
#            fig_paths = plot_backtest_charts(result, combined_signal, BACKTEST_CONFIG, OUTPUT_PATH)
#            output_file = os.path.join(OUTPUT_PATH, f"category_{cat_name}_report.xlsx")
#            save_report_with_charts(result, perf_table, fig_paths, output_file)
#            print(f"✅ 完成类别回测: {output_file}")
#
#            category_signals[cat_name] = combined_signal
#
#        except Exception as e:
#            print(f"❌ 类别 {cat_name} 回测失败: {e}")
#            import traceback
#            traceback.print_exc()
#
#    # === 第三步：将四大类信号再次等权合成，进行最终回测 ===
#    if len(category_signals) >= 1:
#        print(f"\n{'='*60}")
#        print("▶ 合成总信号（四大类等权平均）")
#        print(f"{'='*60}")
#
#        cat_signal_list = list(category_signals.values())
#        common_index = cat_signal_list[0].index
#        for s in cat_signal_list[1:]:
#            common_index = common_index.intersection(s.index)
#        common_index = common_index.sort_values()
#
#        if len(common_index) == 0:
#            print("❌ 总合成信号无公共日期，跳过。")
#        else:
#            aligned_cat_signals = [s.reindex(common_index).fillna(method='ffill').fillna(0) for s in cat_signal_list]
#            total_combined_signal = pd.concat(aligned_cat_signals, axis=1).mean(axis=1)
#            total_combined_signal.name = 'signal'
#
#            try:
#                result = run_backtest_t_plus_1(total_combined_signal, BACKTEST_CONFIG)
#                perf_table = generate_performance_table(result)
#                fig_paths = plot_backtest_charts(result, total_combined_signal, BACKTEST_CONFIG, OUTPUT_PATH)
#                output_file = os.path.join(OUTPUT_PATH, "total_combined_report.xlsx")
#                save_report_with_charts(result, perf_table, fig_paths, output_file)
#                print(f"✅ 完成总信号回测: {output_file}")
#            except Exception as e:
#                print(f"❌ 总信号回测失败: {e}")
#                import traceback
#                traceback.print_exc()
#
#    print(f"\n🎉 所有指定信号及分类回测任务已完成！报告已保存至:\n   {OUTPUT_PATH}")
