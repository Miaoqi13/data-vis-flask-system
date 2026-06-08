# 数据清洗模块
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """返回每列的缺失值统计（count, percent）"""
    total = df.isna().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({"missing_count": total, "missing_percent": percent})


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "auto",
    fill_value: Optional[Any] = None,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    处理缺失值。
    strategy:
      - 'drop': 丢弃含缺失值的行（仅在 subset 指定时按 subset 判断，否则任意列）
      - 'mean': 用数值列均值填充（仅数值列）
      - 'median': 用中位数填充（仅数值列）
      - 'mode': 用众数填充（对所有列使用列众数）
      - 'ffill'/'bfill': 前/后向填充
      - 'constant': 使用 `fill_value` 填充
      - 'auto': 数值列用均值、非数值列用众数
    subset: 指定列列表，仅在 'drop' 时有意义
    返回填充后的新 DataFrame
    """
    df = df.copy()
    if strategy == "drop":
        if subset:
            return df.dropna(subset=subset)
        return df.dropna()

    if strategy in {"ffill", "bfill"}:
        return df.fillna(method=strategy)

    if strategy == "constant":
        return df.fillna(fill_value)

    if strategy == "mean":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())
        return df

    if strategy == "median":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        return df

    if strategy == "mode":
        for c in df.columns:
            mode_vals = df[c].mode()
            if not mode_vals.empty:
                df[c] = df[c].fillna(mode_vals.iloc[0])
        return df

    # auto
    num_cols = df.select_dtypes(include=[np.number]).columns
    other_cols = [c for c in df.columns if c not in num_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].mean())
    for c in other_cols:
        mode_vals = df[c].mode()
        if not mode_vals.empty:
            df[c] = df[c].fillna(mode_vals.iloc[0])
    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
) -> Dict[str, Dict[str, Any]]:
    """
    检测指定列（或所有数值列）的异常值。
    返回字典：{col: {"indices": [...], "count": n, "percent": p}}
    支持方法：'zscore'（基于 z 阈值）和 'iqr'（基于 IQR）
    """
    res: Dict[str, Dict[str, Any]] = {}
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    for c in columns:
        if c not in df.columns:
            continue
        series = df[c]
        if not np.issubdtype(series.dtype, np.number):
            continue
        mask = pd.Series(False, index=df.index)
        vals = series.dropna()
        if vals.empty:
            res[c] = {"indices": [], "count": 0, "percent": 0.0}
            continue

        if method == "zscore":
            mean = vals.mean()
            std = vals.std()
            if std == 0 or np.isnan(std):
                mask = pd.Series(False, index=df.index)
            else:
                z = (series - mean) / std
                mask = z.abs() > z_thresh
        else:  # iqr
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            mask = (series < lower) | (series > upper)

        indices = df.index[mask].tolist()
        count = len(indices)
        percent = 100.0 * count / len(df) if len(df) > 0 else 0.0
        res[c] = {"indices": indices, "count": count, "percent": percent}

    return res


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
    inplace: bool = False,
) -> pd.DataFrame:
    """移除在指定列检测到的异常值（按行删除），返回新 DataFrame。"""
    df_work = df if inplace else df.copy()
    outliers = detect_outliers(df_work, columns=columns, method=method, z_thresh=z_thresh, iqr_k=iqr_k)
    # collect all indices to drop
    drop_idx = set()
    for info in outliers.values():
        drop_idx.update(info.get("indices", []))
    if drop_idx:
        df_work = df_work.drop(index=list(drop_idx))
    return df_work.reset_index(drop=True)


def clean_data(
    df: pd.DataFrame,
    missing_strategy: str = "auto",
    missing_fill_value: Optional[Any] = None,
    outlier_method: Optional[str] = "iqr",
    outlier_columns: Optional[List[str]] = None,
    outlier_params: Optional[Dict[str, Any]] = None,
    remove_outliers_flag: bool = False,
) -> Dict[str, Any]:
    """
    综合清洗流程：先处理缺失值，再检测/可选移除异常值。
    返回：{"df": cleaned_df, "missing_summary": DataFrame, "outliers": dict}
    """
    outlier_params = outlier_params or {}
    missing_summary = summarize_missing(df)
    df_clean = handle_missing_values(df, strategy=missing_strategy, fill_value=missing_fill_value)

    outliers = detect_outliers(
        df_clean,
        columns=outlier_columns,
        method=outlier_method or "iqr",
        z_thresh=outlier_params.get("z_thresh", 3.0),
        iqr_k=outlier_params.get("iqr_k", 1.5),
    )

    if remove_outliers_flag and outlier_method is not None:
        df_clean = remove_outliers(
            df_clean,
            columns=outlier_columns,
            method=outlier_method,
            z_thresh=outlier_params.get("z_thresh", 3.0),
            iqr_k=outlier_params.get("iqr_k", 1.5),
        )

    return {"df": df_clean, "missing_summary": missing_summary, "outliers": outliers}
