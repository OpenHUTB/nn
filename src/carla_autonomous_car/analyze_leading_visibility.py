import pandas as pd

def compare_no_leading_ratio(baseline_csv, coop_csv):
    """
    统计每个模式下：
      - 平均每个 episode 中“看不到前车”的比例
      - coop 相比 baseline 下降了多少（绝对值 + 百分比）

    :param baseline_csv: baseline 实验的 monitor.csv 路径
    :param coop_csv:     coop 实验的 monitor.csv 路径
    """
    df_base = pd.read_csv(baseline_csv, comment='#')
    df_coop = pd.read_csv(coop_csv, comment='#')

    # 监控里我们已经写了 no_leading_ratio 列
    base_mean = df_base['no_leading_ratio'].mean()
    coop_mean = df_coop['no_leading_ratio'].mean()

    abs_drop = base_mean - coop_mean
    rel_drop = abs_drop / base_mean * 100 if base_mean > 1e-8 else 0.0

    print('===== 无车可见比例对比（no_leading_ratio）=====')
    print(f'Baseline 平均比例: {base_mean:.4f}')
    print(f'Coop     平均比例: {coop_mean:.4f}')
    print(f'绝对下降: {abs_drop:.4f}')
    print(f'相对下降: {rel_drop:.2f}%')

    return {
        'baseline_mean': base_mean,
        'coop_mean': coop_mean,
        'abs_drop': abs_drop,
        'rel_drop_percent': rel_drop
    }

if __name__ == '__main__':
    baseline_csv = 'logs/agent_1/monitor.csv'
    coop_csv     = 'logs/agent_2/monitor.csv'
    compare_no_leading_ratio(baseline_csv, coop_csv)
