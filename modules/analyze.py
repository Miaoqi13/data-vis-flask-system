"""
分析模块

实现：
- 聚类比较（KMeans、DBSCAN）
- 降维（PCA、t-SNE）
- 分类比较（LogisticRegression、RandomForestClassifier）
- 回归比较（LinearRegression、RandomForestRegressor）

每个比较函数返回评估指标字典，便于在前端或脚本中展示。
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)


def _get_numeric_df(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    if features:
        df_num = df[features].select_dtypes(include=[np.number])
    else:
        df_num = df.select_dtypes(include=[np.number])
    return df_num.dropna()


def clustering_compare(df: pd.DataFrame, features: Optional[List[str]] = None, k_range: Tuple[int, int] = (2, 6)) -> Dict[str, Any]:
    """比较 KMeans（不同 k）与 DBSCAN，返回指标。"""
    X = _get_numeric_df(df, features)
    if X.shape[0] < 2:
        return {"error": "样本数不足进行聚类"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results: Dict[str, Any] = {"kmeans": {}, "dbscan": {}, "agglo": {}}

    # KMeans 优化 k
    kmin, kmax = k_range
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(Xs)
        # 评估
        try:
            sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else float('nan')
        except Exception:
            sil = float('nan')
        try:
            db = davies_bouldin_score(Xs, labels) if len(set(labels)) > 1 else float('nan')
        except Exception:
            db = float('nan')
        results['kmeans'][k] = {"silhouette": float(np.nan_to_num(sil, nan=np.nan)), "davies_bouldin": float(np.nan_to_num(db, nan=np.nan))}

    # DBSCAN 默认参数扫描
    db_res = {}
    for eps in [0.3, 0.5, 0.8, 1.0]:
        dbs = DBSCAN(eps=eps, min_samples=5)
        labels = dbs.fit_predict(Xs)
        n_clusters = len(set(labels) - {-1})
        try:
            sil = silhouette_score(Xs, labels) if n_clusters > 1 else float('nan')
        except Exception:
            sil = float('nan')
        db_res[eps] = {"n_clusters": int(n_clusters), "silhouette": float(np.nan_to_num(sil, nan=np.nan))}
    results['dbscan'] = db_res

    # Agglomerative（层次聚类）尝试不同簇数
    agg_res = {}
    for k in range(kmin, min(kmax + 1, 11)):
        try:
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(Xs)
            if len(set(labels)) > 1:
                sil = silhouette_score(Xs, labels)
                db = davies_bouldin_score(Xs, labels)
                ch = calinski_harabasz_score(Xs, labels)
            else:
                sil = float('nan')
                db = float('nan')
                ch = float('nan')
        except Exception:
            sil = float('nan')
            db = float('nan')
            ch = float('nan')
        agg_res[k] = {"silhouette": float(np.nan_to_num(sil, nan=np.nan)), "davies_bouldin": float(np.nan_to_num(db, nan=np.nan)), "calinski_harabasz": float(np.nan_to_num(ch, nan=np.nan))}
    results['agglo'] = agg_res

    # 生成中文摘要
    summary_lines = []
    # KMeans 选择最佳 k（基于 silhouette）
    best_k = None
    best_sil = -999
    for k, metrics in results['kmeans'].items():
        sil = metrics.get('silhouette')
        if sil is not None and not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k is not None:
        summary_lines.append(f"KMeans: 建议聚类数 k={best_k}（轮廓系数 silhouette≈{best_sil:.3f}，数值越接近1越好）。")
    else:
        summary_lines.append("KMeans: 无法基于轮廓系数确定最佳 k（可能所有分组数的 silhouette 无效）。")

    # DBSCAN 简要说明
    try:
        best_eps = max(results['dbscan'].items(), key=lambda kv: (kv[1].get('silhouette', -999)))[0]
        best_eps_sil = results['dbscan'][best_eps].get('silhouette')
        n_clusters = results['dbscan'][best_eps].get('n_clusters')
        summary_lines.append(f"DBSCAN: 在 eps={best_eps} 时检测到 {n_clusters} 个簇，轮廓系数≈{best_eps_sil:.3f}（越高越好，-1 表示异常点）。")
    except Exception:
        summary_lines.append("DBSCAN: 未能对不同 eps 的结果做进一步排序。")

    results['summary'] = '\n'.join(summary_lines)

    return results


def dimensionality_reduction(df: pd.DataFrame, features: Optional[List[str]] = None, n_components: int = 2) -> Dict[str, Any]:
    X = _get_numeric_df(df, features)
    if X.shape[0] == 0:
        return {"error": "无数值列可用于降维"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    res: Dict[str, Any] = {}
    # PCA
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(Xs)
    res['pca'] = {"components": Xp.tolist(), "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}

    # t-SNE（只在样本数合理时运行）
    if Xs.shape[0] <= 500:  # 限制规模避免长时间计算
        tsne = TSNE(n_components=n_components, random_state=42, init='pca')
        Xt = tsne.fit_transform(Xs)
        res['tsne'] = {"components": Xt.tolist()}
    else:
        res['tsne'] = {"warning": "样本数过多，跳过 t-SNE（>500）"}

    # 中文摘要
    if 'pca' in res and 'explained_variance_ratio' in res['pca']:
        evr = res['pca']['explained_variance_ratio']
        res['summary'] = f"PCA: 前 {len(evr)} 个主成分的方差贡献比例: {', '.join([f'{v:.3f}' for v in evr])}。数值越高表示该成分解释的方差越多。"
    else:
        res['summary'] = "降维已完成。"

    return res


def classification_compare(df: pd.DataFrame, target: str, features: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """比较 LogisticRegression 与 RandomForestClassifier（用于二分类）。
    若目标为连续型，将自动转换为二分类（profit>0）。
    """
    if target not in df.columns:
        return {"error": f"目标列 {target} 不存在"}

    # Prepare X, y
    Xdf = _get_numeric_df(df, features)
    if target in Xdf.columns:
        # if target included, remove
        Xdf = Xdf.drop(columns=[target], errors='ignore')

    yser = df[target].copy()
    # if numeric continuous, convert to binary using median>0 or >0
    if pd.api.types.is_numeric_dtype(yser):
        y = (yser > 0).astype(int)
    else:
        y = yser.astype('category').cat.codes

    # align X and y by dropping rows with NaNs
    data = pd.concat([Xdf, y], axis=1).dropna()
    if data.shape[0] < 10:
        return {"error": "样本数不足进行分类训练（<10）"}

    X = data[Xdf.columns].values
    y = data[y.name].values if hasattr(y, 'name') else data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
        'SVC': SVC(probability=True, random_state=random_state),
    }

    results: Dict[str, Any] = {}
    for name, mdl in models.items():
        try:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            prec = None
            rec = None
            try:
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(y_test, y_pred, average='macro')
                rec = recall_score(y_test, y_pred, average='macro')
            except Exception:
                prec = None
                rec = None
            cv_acc = None
            cv_f1 = None
            try:
                cv_acc = float(cross_val_score(mdl, X, y, cv=5, scoring='accuracy').mean())
                cv_f1 = float(cross_val_score(mdl, X, y, cv=5, scoring='f1_macro').mean())
            except Exception:
                cv_acc = None
                cv_f1 = None

            # feature importance if available
            feat_imp = None
            try:
                if hasattr(mdl, 'feature_importances_'):
                    feat_imp = mdl.feature_importances_.tolist()
                elif hasattr(mdl, 'coef_'):
                    coef = np.abs(mdl.coef_)
                    feat_imp = coef.mean(axis=0).tolist() if coef.ndim > 1 else coef.tolist()
            except Exception:
                feat_imp = None

            results[name] = {"accuracy": float(acc), "f1_macro": float(f1), "precision_macro": (float(prec) if prec is not None else None), "recall_macro": (float(rec) if rec is not None else None), "cv_accuracy_mean": cv_acc, "cv_f1_macro_mean": cv_f1, "feature_importances": feat_imp}
        except Exception as e:
            results[name] = {"error": str(e)}

    # 中文摘要：比较模型表现并给出建议
    # 排序并挑选最佳模型（以 cv_f1_macro_mean 或 f1_macro 为准）
    best_model = None
    best_score = -999
    for name, stats in results.items():
        if not isinstance(stats, dict):
            continue
        sc = stats.get('cv_f1_macro_mean') if stats.get('cv_f1_macro_mean') is not None else stats.get('f1_macro')
        try:
            if sc is not None and sc > best_score:
                best_score = sc
                best_model = name
        except Exception:
            continue

    if best_model:
        bm = results[best_model]
        summary = f"模型比较完成，推荐: {best_model}（F1_macro≈{bm.get('f1_macro', bm.get('cv_f1_macro_mean', 0)):.3f}, 准确率≈{bm.get('accuracy', 0):.3f}）。"
    else:
        summary = "模型比较完成。"
    results['summary'] = summary
    # 也提供排序的比较表（按 cv_f1 或 f1 排序）
    try:
        comp_list = []
        for name, stats in results.items():
            if not isinstance(stats, dict):
                continue
            comp_list.append((name, stats.get('f1_macro'), stats.get('cv_f1_macro_mean'), stats.get('accuracy')))
        comp_list_sorted = sorted(comp_list, key=lambda x: (x[2] if x[2] is not None else (x[1] if x[1] is not None else -999)), reverse=True)
        results['comparison_table'] = [{'model': r[0], 'f1_macro': r[1], 'cv_f1_macro': r[2], 'accuracy': r[3]} for r in comp_list_sorted]
    except Exception:
        pass
    return results
    return results


def regression_compare(df: pd.DataFrame, target: str, features: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    if target not in df.columns:
        return {"error": f"目标列 {target} 不存在"}

    Xdf = _get_numeric_df(df, features)
    Xdf = Xdf.drop(columns=[target], errors='ignore')
    yser = df[target].copy()

    data = pd.concat([Xdf, yser], axis=1).dropna()
    if data.shape[0] < 10:
        return {"error": "样本数不足进行回归训练（<10）"}

    X = data[Xdf.columns].values
    y = data[yser.name].values if hasattr(yser, 'name') else data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=random_state),
        'SVR': SVR(),
    }

    results: Dict[str, Any] = {}
    for name, mdl in models.items():
        try:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            cv_r2 = None
            try:
                cv_r2 = float(cross_val_score(mdl, X, y, cv=5, scoring='r2').mean())
            except Exception:
                cv_r2 = None
            results[name] = {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2), "cv_r2_mean": cv_r2}
        except Exception as e:
            results[name] = {"error": str(e)}

    # 中文摘要：比较模型表现并给出建议
    # 选择最佳模型（以 cv_r2_mean 或 r2 为准）
    best_model = None
    best_score = -999
    for name, stats in results.items():
        if not isinstance(stats, dict):
            continue
        sc = stats.get('cv_r2_mean') if stats.get('cv_r2_mean') is not None else stats.get('r2')
        try:
            if sc is not None and sc > best_score:
                best_score = sc
                best_model = name
        except Exception:
            continue

    if best_model:
        bm = results[best_model]
        summary = f"模型比较完成，推荐: {best_model}（R2≈{bm.get('r2', 0):.3f}, RMSE≈{bm.get('rmse', 0):.3f}）。"
    else:
        summary = "回归模型比较完成。"
    results['summary'] = summary
    try:
        comp_list = []
        for name, stats in results.items():
            if not isinstance(stats, dict):
                continue
            comp_list.append((name, stats.get('r2'), stats.get('cv_r2_mean'), stats.get('rmse')))
        comp_list_sorted = sorted(comp_list, key=lambda x: (x[2] if x[2] is not None else (x[1] if x[1] is not None else -999)), reverse=True)
        results['comparison_table'] = [{'model': r[0], 'r2': r[1], 'cv_r2': r[2], 'rmse': r[3]} for r in comp_list_sorted]
    except Exception:
        pass
    return results


def run_analysis(df: pd.DataFrame, mode: str = 'auto', **kwargs) -> Dict[str, Any]:
    """统一入口：mode in {'clustering','dim_reduction','classification','regression','auto'}"""
    mode = mode.lower()
    if mode == 'clustering':
        return clustering_compare(df, features=kwargs.get('features'), k_range=kwargs.get('k_range', (2, 6)))
    if mode == 'dim_reduction':
        return dimensionality_reduction(df, features=kwargs.get('features'), n_components=kwargs.get('n_components', 2))
    if mode == 'classification':
        return classification_compare(df, target=kwargs.get('target'), features=kwargs.get('features'))
    if mode == 'regression':
        return regression_compare(df, target=kwargs.get('target'), features=kwargs.get('features'))

    # auto: run clustering, dim reduction, and try both classification/regression if target provided
    res = {}
    res['clustering'] = clustering_compare(df, features=kwargs.get('features'))
    res['dim_reduction'] = dimensionality_reduction(df, features=kwargs.get('features'))
    target = kwargs.get('target')
    if target:
        # if numeric then regression + classification (binary)
        if pd.api.types.is_numeric_dtype(df[target]):
            res['regression'] = regression_compare(df, target=target, features=kwargs.get('features'))
            res['classification'] = classification_compare(df, target=target, features=kwargs.get('features'))
        else:
            res['classification'] = classification_compare(df, target=target, features=kwargs.get('features'))
    return res
