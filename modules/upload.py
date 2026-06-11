# 数据管理模块

import pandas as pd
import os

_current_df = None
_current_filepath = None

def set_current_dataframe(df, filepath=None):
    global _current_df, _current_filepath
    _current_df = df.copy()
    _current_filepath = filepath

def get_current_dataframe():
    global _current_df
    return _current_df

def update_current_dataframe(df,save_to_file=True):
    """清洗/分析模块调用此函数更新数据"""
    global _current_df, _current_filepath
    if df is None:
        return False
    _current_df = df.copy()
    
    if save_to_file and _current_filepath:
        try:
            if _current_filepath.endswith('.csv'):
                df.to_csv(_current_filepath, index=False, encoding='utf-8-sig')
            elif _current_filepath.endswith('.xlsx'):
                df.to_excel(_current_filepath, index=False)
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False
    return True

def load_data(filepath):
    """
    读取 CSV 或 Excel 文件
    """

    if filepath.endswith('.csv'):

        try:
            df = pd.read_csv(filepath,encoding='utf-8')

        except UnicodeDecodeError:

            df = pd.read_csv(filepath,encoding='latin1')

    elif filepath.endswith('.xlsx'):

        df = pd.read_excel(filepath)

    else:
        raise ValueError("不支持的文件格式")

    return df


def save_uploaded_file(file, upload_folder):
    """
    保存上传文件
    """
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder,file.filename)

    # 文件不存在时保存
    file.save(filepath)
    return filepath


def preview_data(df):
    """
    数据预览
    """
    df_show = df.head().reset_index(drop=True)
    df_show.index = df_show.index + 1

    return {
        "head": df_show.to_html(classes='table table-bordered table-striped'),
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }

# modules/upload.py (追加函数)
def load_dataset_by_id(dataset_id):
    """根据历史记录ID加载数据集，并设置为当前数据"""
    from modules.database import get_dataset_by_id
    record = get_dataset_by_id(dataset_id)
    if not record:
        return None, "数据集不存在"
    filepath = record['filepath']
    if not os.path.exists(filepath):
        return None, "文件已丢失"
    try:
        df = load_data(filepath)
        set_current_dataframe(df, filepath)
        return df, "加载成功"
    except Exception as e:
        return None, f"加载失败: {e}"

