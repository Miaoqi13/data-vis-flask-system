# visualize.py（修复非数值 Y 轴报错，保留所有优化）
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import uuid
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ---------- 修复中文显示 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_chart(df, x, y, chart_type):
    """
    创建图表并保存到 static/charts 目录
    - 支持中文标题 / 轴标签
    - 标题动态显示字段名：图表类型：Y轴字段 vs X轴字段
    - Y轴刻度智能控制（仅当 Y 轴为数值时生效）
    - X轴标签倾斜45°防重叠
    """
    os.makedirs("static/charts", exist_ok=True)

    n_points = len(df)
    x_labels = df[x].astype(str)
    max_label_len = x_labels.str.len().max() if n_points > 0 else 4

    type_name_map = {
        "bar": "柱状图",
        "line": "折线图",
        "pie": "饼图",
        "scatter": "散点图",
        "box": "箱线图"
    }
    chart_cn = type_name_map.get(chart_type, chart_type)

    # 动态画布宽度
    if chart_type in ['bar', 'line', 'scatter']:
        fig_width = max(8, min(22, n_points * 0.4 + max_label_len * 0.08))
        plt.figure(figsize=(fig_width, 5.5))
    elif chart_type == 'box':
        col_count = len(y) if isinstance(y, list) else 1
        fig_width = max(6, col_count * 1.8)
        plt.figure(figsize=(fig_width, 5))
    else:
        plt.figure(figsize=(8, 7))

    # ---------- 绘图 ----------
    if chart_type == "bar":
        plt.bar(
            df[x],
            df[y],
            width=0.7,
            edgecolor='black'
        )
        plt.xticks(rotation=45, ha='right', fontsize=9 if n_points > 20 else 10)
        if n_points > 40:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))

    elif chart_type == "line":
        plt.plot(
            df[x],
            df[y],
            marker='o',
            linewidth=2
        )
        plt.xticks(rotation=45, ha='right', fontsize=9 if n_points > 20 else 10)
        if n_points > 40:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))

    elif chart_type == "scatter":
        plt.scatter(
            df[x],
            df[y],
            s=40,
            alpha=0.6
        )
        plt.xticks(rotation=45, ha='right', fontsize=9 if n_points > 20 else 10)
        if n_points > 40:
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))

    elif chart_type == "pie":
        labels = df[x].astype(str)
        sizes = df[y]
        textprops = {'fontsize': 9} if len(labels) > 10 else {'fontsize': 11}
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                textprops=textprops, pctdistance=0.8, labeldistance=1.1)
        if len(labels) > 5:
            plt.legend(labels, loc='lower right', fontsize=8, ncol=2)

    elif chart_type == "box":
        cols = [y] if isinstance(y, str) else y
        data_to_plot = [df[c].dropna() for c in cols if c in df.columns]
        plt.boxplot(data_to_plot, labels=cols)
        plt.ylabel('Value')
        if max([len(str(c)) for c in cols]) > 8:
            plt.xticks(rotation=30, ha='right')

    else:
        raise ValueError("不支持的图表类型")

    # ---------- 标题和轴标签 ----------
    if chart_type == "pie":
        plt.title(f"{chart_cn}：{y} vs {x}", fontsize=14, pad=15)
    elif chart_type == "box":
        plt.xlabel("字段")
        plt.ylabel("数值")
        plt.title(f"{chart_cn}：{y}", fontsize=14, pad=15)
    else:
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.title(f"{chart_cn}：{y} vs {x}", fontsize=14, pad=15)

    # ---------- Y轴刻度优化（仅数值列）----------
    if chart_type not in ["pie", "box"]:

        ax = plt.gca()

        y_data = df[y].dropna()

        if len(y_data) > 0 and pd.api.types.is_numeric_dtype(y_data):

            try:

                if pd.api.types.is_integer_dtype(y_data):
                    ax.yaxis.set_major_locator(
                        MaxNLocator(integer=True, nbins=5)
                    )

                else:
                    ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=5)
                    )

            except Exception:
                pass
    # 自动裁剪空白区域
    if chart_type not in ['pie', 'box']:

        y_data = df[y]

        if len(y_data) > 0 and np.issubdtype(y_data.dtype, np.number):

            ymin = y_data.min()
            ymax = y_data.max()

            if ymin != ymax:
                margin = (ymax - ymin) * 0.05

                plt.ylim(
                    ymin - margin,
                    ymax + margin
                )
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(bottom=0.25)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("static/charts", filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return f"charts/{filename}"