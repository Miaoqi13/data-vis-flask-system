from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect,
    url_for,
    flash
)

from modules.upload import (
    save_uploaded_file,
    load_data,
    preview_data,
    set_current_dataframe,
    get_current_dataframe,
    load_dataset_by_id,
    clear_current_dataframe
)
from modules import clean as clean_module
from modules import analyze as analyze_module
from modules.visualize import create_chart
import numpy as np
import pandas as pd
from modules.database import init_db, save_dataset_record, get_all_datasets
from modules.export import export_data
from modules.auth import (
    init_user_table,
    create_user,
    authenticate_user,
    login_required,
    login_user,
    logout_user,
    get_user_by_id
)

app = Flask(__name__)
app.secret_key = 'data-analysis-platform-secret-key-2024-new'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400

UPLOAD_FOLDER = 'datasets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

init_db()
init_user_table()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('main'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('main'))
    
    error = None
    success = None
    
    # 检查是否从注册页跳转过来
    if request.args.get('registered') == '1':
        success = "注册成功！请登录"
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            error = "请输入用户名和密码"
        else:
            user_id, err = authenticate_user(username, password)
            if err:
                error = err
            else:
                # 清空之前用户的数据
                clear_current_dataframe()
                login_user(user_id, username)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('main'))
    
    return render_template('login.html', error=error, success=success)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('main'))
    
    error = None
    success = None
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        
        if not username or not password:
            error = "用户名和密码不能为空"
        elif len(password) < 6:
            error = "密码长度至少6位"
        elif password != password_confirm:
            error = "两次输入的密码不一致"
        else:
            user_id, err = create_user(username, password, email if email else None)
            if err:
                error = err
            else:
                success = "注册成功！请登录"
                return redirect(url_for('login', registered='1'))
    
    return render_template('register.html', error=error, success=success, username=request.form.get('username'), email=request.form.get('email'))

@app.route('/logout')
def logout():
    # 清空当前用户的数据
    clear_current_dataframe()
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_new')
@login_required
def upload_new():
    """更换文件：清空当前数据，跳转到上传页面"""
    clear_current_dataframe()
    session['cleaned'] = False
    return render_template('index.html')

@app.route('/main')
@login_required
def main():
    df = get_current_dataframe()
    if df is None:
        return render_template('index.html')
    data_info = preview_data(df)
    return render_template('main.html', data=data_info, filename=session.get('last_filename', ''))

@app.route('/preview')
@login_required
def preview():
    df = get_current_dataframe()
    if df is None:
        return render_template('index.html')
    data_info = preview_data(df)
    return render_template('preview_only.html', data=data_info, filename=session.get('last_filename', ''))


# 上传文件
@app.route('/upload', methods=['POST'])
@login_required
def upload():

    file = request.files.get('file')
    if not file or file.filename == '':
        return "请选择文件", 400

    # 保存文件
    filepath = save_uploaded_file(file,app.config['UPLOAD_FOLDER'])

    # 读取数据
    try:
        df = load_data(filepath)
    except Exception as e:
        return f"文件解析失败: {e}", 400

    set_current_dataframe(df, filepath)
    session['last_filename'] = file.filename
    session['cleaned'] = False
    data_info = preview_data(df)

    user_id = session.get('user_id')
    save_dataset_record(file.filename, filepath, df.shape[0], df.shape[1], user_id=user_id)

    return render_template('main.html', data=data_info, filename=file.filename)

# 下载文件
@app.route('/download/<filename>')
@login_required
def download(filename):
    filepath = f'datasets/{filename}'
    return send_file(filepath,as_attachment=True)

@app.route('/history')
@login_required
def history():
    user_id = session.get('user_id')
    datasets = get_all_datasets(user_id=user_id)
    return render_template('history.html', datasets=datasets)

@app.route('/load_history/<int:dataset_id>')
@login_required
def load_history(dataset_id):
    df, msg = load_dataset_by_id(dataset_id)
    if df is None:
        return msg, 404
    data_info = preview_data(df)
    from modules.database import get_dataset_by_id
    record = get_dataset_by_id(dataset_id)
    filename = record['filename'] if record else 'history_data'
    session['cleaned'] = False
    return render_template('main.html', data=data_info, filename=filename)


@app.route('/clean', methods=['POST'])
@login_required
def clean_data_route():
    df = get_current_dataframe()
    if df is None:
        return "当前无加载数据", 400

    # 表单字段
    missing_strategy = request.form.get('missing_strategy', 'auto')
    missing_fill_value = request.form.get('missing_fill_value') or None
    outlier_method = request.form.get('outlier_method') or None
    remove_outliers_flag = request.form.get('remove_outliers') == 'on'
    outlier_columns = request.form.getlist('outlier_columns') or None

    outlier_params = {}
    z_thresh = request.form.get('z_thresh')
    iqr_k = request.form.get('iqr_k')
    if z_thresh:
        try:
            outlier_params['z_thresh'] = float(z_thresh)
        except ValueError:
            pass
    if iqr_k:
        try:
            outlier_params['iqr_k'] = float(iqr_k)
        except ValueError:
            pass

    res = clean_module.clean_data(
        df,
        missing_strategy=missing_strategy,
        missing_fill_value=missing_fill_value,
        outlier_method=outlier_method,
        outlier_columns=outlier_columns,
        outlier_params=outlier_params,
        remove_outliers_flag=remove_outliers_flag,
    )

    cleaned_df = res['df']
    # 不覆盖原文件：将清洗后的文件保存为新的文件并提供下载
    original_filename = session.get('last_filename', None)
    if original_filename:
        import os
        name, ext = os.path.splitext(original_filename)
        cleaned_filename = f"{name}_cleaned{ext}"
    else:
        cleaned_filename = 'data_cleaned.csv'

    save_path = f"{app.config['UPLOAD_FOLDER']}/{cleaned_filename}"
    try:
        if cleaned_filename.lower().endswith('.csv'):
            cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        elif cleaned_filename.lower().endswith('.xlsx'):
            cleaned_df.to_excel(save_path, index=False)
        else:
            # 默认保存为 CSV
            cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        return f"保存清洗文件失败: {e}", 500

    # 仅更新内存中的当前 dataframe，不覆盖原文件
    from modules.upload import update_current_dataframe
    update_current_dataframe(cleaned_df, save_to_file=False)

    session['cleaned'] = True
    data_info = preview_data(cleaned_df)
    message = '数据清洗完成（已生成清洗文件）'
    missing_html = None
    try:
        missing_html = res['missing_summary'].to_html(classes='table table-sm table-bordered')
    except Exception:
        missing_html = None

    return render_template(
        'preview_only.html',
        data=data_info,
        filename=original_filename or '',
        message=message,
        cleaned_filename=cleaned_filename,
        missing_summary_html=missing_html,
    )

@app.route('/visualize', methods=['POST'])
@login_required
def visualize():
    if not session.get('cleaned'):
        df = get_current_dataframe()
        if df is None:
            return "当前无数据", 400
        data_info = preview_data(df)
        return render_template(
            'visualization.html',
            data=data_info,
            error="请先完成数据清洗后再进行可视化操作"
        )
    
    df = get_current_dataframe()
    if df is None:
        return "当前无数据", 400

    x = request.form.get("x")
    y = request.form.get("y")
    chart_type = request.form.get("chart_type")

    if x not in df.columns or y not in df.columns:
        error_msg = "选择的字段不存在"
    else:
        x_is_num = pd.api.types.is_numeric_dtype(df[x])
        y_is_num = pd.api.types.is_numeric_dtype(df[y])

        if chart_type == "pie":
            if not y_is_num:
                error_msg = "饼图的'值'字段必须是数值列（例如销售额、数量）"
            elif df[x].nunique() > 50:
                error_msg = "饼图的分类过多（超过50类），请选择类别较少的字段作为'名称'"
            else:
                error_msg = None

        elif chart_type == "box":
            if not y_is_num:
                error_msg = "箱线图的'值'字段必须是数值列"
            else:
                error_msg = None

        elif chart_type in ["bar", "line", "scatter"]:
            if not y_is_num:
                error_msg = f"{chart_type} 图的'值'字段必须是数值列"
            else:
                error_msg = None
        else:
            error_msg = "不支持的图表类型"

    if error_msg:
        data_info = preview_data(df)
        return render_template(
            'visualization.html',
            data=data_info,
            error=error_msg
        )

    df_plot = df.dropna(subset=[x, y])

    if df_plot.empty:
        data_info = preview_data(df)
        return render_template(
            'visualization.html',
            data=data_info,
            error="所选列的有效数据为空，请先清洗缺失值"
        )

    if chart_type in ['bar', 'line', 'pie']:

        if x == y:
            return render_template(
                'visualization.html',
                data=preview_data(df),
                error='X轴和Y轴不能选择同一列'
            )

        df_plot = (
            df_plot.groupby(x, as_index=False)[y]
            .sum()
            .sort_values(y, ascending=False)
        )

        if len(df_plot) > 10:
            df_plot = df_plot.head(10)

    elif chart_type == 'scatter':

        if len(df_plot) > 1000:
            df_plot = df_plot.sample(
                n=1000,
                random_state=42
            )

    try:
        chart_path = create_chart(df_plot, x, y, chart_type)
        data_info = preview_data(df)
        return render_template(
            "visualization.html",
            data=data_info,
            chart_path=chart_path
        )
    except Exception as e:
        data_info = preview_data(df)
        return render_template(
            "visualization.html",
            data=data_info,
            error=f"绘图失败: {e}"
        )

@app.route('/visualization')
@login_required
def visualization():
    if not session.get('cleaned'):
        return "请先进行数据清洗", 400
    
    df = get_current_dataframe()
    if df is None:
        return "当前无加载数据", 400
    data_info = preview_data(df)
    return render_template('visualization.html', data=data_info)

    # ========= 数据预处理 =========

    df_plot = df.dropna(subset=[x, y])

    if df_plot.empty:
        data_info = preview_data(df)
        return render_template(
            'preview.html',
            data=data_info,
            filename=session.get('last_filename', ''),
            error="所选列的有效数据为空，请先清洗缺失值"
        )

    # 柱状图、折线图、饼图自动聚合
    if chart_type in ['bar', 'line', 'pie']:

        if x == y:
            return render_template(
                'preview.html',
                data=preview_data(df),
                filename=session.get('last_filename', ''),
                error='X轴和Y轴不能选择同一列'
            )

        df_plot = (
            df_plot.groupby(x, as_index=False)[y]
            .sum()
            .sort_values(y, ascending=False)
        )

        # 类别太多只显示前10
        if len(df_plot) > 10:
            df_plot = df_plot.head(10)

    # 散点图太多点时抽样
    elif chart_type == 'scatter':

        if len(df_plot) > 1000:
            df_plot = df_plot.sample(
                n=1000,
                random_state=42
            )

    # 箱线图保持原始数据

    try:
        chart_path = create_chart(df_plot, x, y, chart_type)
        data_info = preview_data(df)
        return render_template(
            "preview.html",
            data=data_info,
            filename=session.get('last_filename', ''),
            chart_path=chart_path
        )
    except Exception as e:
        data_info = preview_data(df)
        return render_template(
            "preview.html",
            data=data_info,
            filename=session.get('last_filename', ''),
            error=f"绘图失败: {e}"
        )
@app.route('/export')
@login_required
def export_route():

    df = get_current_dataframe()

    if df is None:
        return "当前无数据", 400

    file_path = export_data(df)

    return send_file(
        file_path,
        as_attachment=True
    )

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze_route():
    if not session.get('cleaned'):
        return "请先进行数据清洗", 400
    
    df = get_current_dataframe()
    # Allow viewing the analyze page even if no data is loaded (GET);
    # require data only for POST analysis execution.
    if df is None:
        numeric_cols = []
        all_cols = []
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

    if request.method == 'GET':
        return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols)

    # POST: 执行分析
    if df is None:
        return "当前无加载数据", 400

    mode = request.form.get('mode', 'auto')
    features = request.form.getlist('features') or []
    target = request.form.get('target') or ''
    # optional k range for clustering
    kmin = request.form.get('kmin')
    kmax = request.form.get('kmax')
    k_range = None
    # 参数校验（不合理时直接返回提示）
    error_msg = None

    # features 如果为空则使用所有数值列
    if not features:
        features = numeric_cols.copy()

    # 校验 features 是否存在且为数值列
    invalid_feats = [f for f in features if f not in numeric_cols]
    if invalid_feats:
        error_msg = f"所选特征列包含非数值或不存在的列：{', '.join(invalid_feats)}。请只选择数值列。"

    # 校验 target 是否存在
    if target:
        if target not in all_cols:
            error_msg = f"目标列 `{target}` 不存在，请选择有效的目标列。"

    # 解析并校验 k 范围
    if kmin or kmax:
        try:
            kmin_i = int(kmin) if kmin else None
            kmax_i = int(kmax) if kmax else None
            if kmin_i is None or kmax_i is None:
                error_msg = "请同时填写 K 范围的最小值和最大值，或都留空。"
            else:
                if kmin_i < 2 or kmax_i < 2:
                    error_msg = "K 值必须 >= 2。"
                elif kmin_i > kmax_i:
                    error_msg = "K 范围不合法：最小值不能大于最大值。"
                elif (kmax_i - kmin_i) > 20:
                    error_msg = "K 范围跨度过大，请缩小范围（<=20）。"
                else:
                    k_range = (kmin_i, kmax_i)
        except ValueError:
            error_msg = "K 范围必须为整数。"

    # 如果有错误，直接显示提示，不执行分析
    if error_msg:
        return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, error=error_msg, mode=mode)

    # 进一步针对 classification/regression 的合理性校验
    if mode in ('classification', 'regression'):
        if not target:
            return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, error='分类/回归模式需要选择目标列（target）。', mode=mode)
        # 若 features 中包含 target，则去重
        features_checked = [f for f in features if f != target]
        # 检查所选特征与目标的数据量
        try:
            sample_df = df[features_checked + [target]].dropna()
        except Exception:
            return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, error='所选特征或目标列不存在于数据中。', mode=mode)

        if sample_df.shape[0] < 10:
            return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, error='有效样本不足（<10），无法进行训练，请选择更多特征或更大数据集。', mode=mode)

        if mode == 'classification':
            # 至少有 2 个类别
            if sample_df[target].nunique() < 2:
                return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, error='目标列类别数少于 2，无法进行分类。', mode=mode)
        # 替换 features 为处理后的特征列表，避免后续重复
        features = features_checked

    analysis_kwargs = {}
    if features:
        analysis_kwargs['features'] = features
    if target:
        analysis_kwargs['target'] = target
    if k_range:
        analysis_kwargs['k_range'] = k_range

    try:
        result = analyze_module.run_analysis(df, mode=mode, **analysis_kwargs)
    except Exception as e:
        return f"分析失败: {e}", 500

    return render_template('analyze.html', numeric_cols=numeric_cols, all_cols=all_cols, result=result, mode=mode)


if __name__ == '__main__':
    print("=" * 50)
    print("数据分析平台启动中...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)