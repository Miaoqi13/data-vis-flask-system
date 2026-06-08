from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session
)

from modules.upload import (
    save_uploaded_file,
    load_data,
    preview_data,
    set_current_dataframe,
    get_current_dataframe,
    load_dataset_by_id
)
from modules import clean as clean_module
from modules.database import init_db, save_dataset_record, get_all_datasets
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
# 上传文件目录
UPLOAD_FOLDER = 'datasets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

init_db()

# 首页
@app.route('/')
def index():
    return render_template('index.html')


# 上传文件
@app.route('/upload', methods=['POST'])
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
    # 记录当前文件名到 session 以便后续操作使用
    session['last_filename'] = file.filename
    # 数据预览
    data_info = preview_data(df)

    session_id = session.get('user_id', 'default')
    save_dataset_record(file.filename, filepath, df.shape[0], df.shape[1], session_id)

    return render_template('preview.html', data=data_info, filename=file.filename)

# 下载文件
@app.route('/download/<filename>')
def download(filename):
    filepath = f'datasets/{filename}'
    return send_file(filepath,as_attachment=True)

@app.route('/history')
def history():
    session_id = session.get('user_id', 'default')
    datasets = get_all_datasets(session_id)
    return render_template('history.html', datasets=datasets)

@app.route('/load_history/<int:dataset_id>')
def load_history(dataset_id):
    df, msg = load_dataset_by_id(dataset_id)
    if df is None:
        return msg, 404
    data_info = preview_data(df)
    # 获取原始文件名（从数据库记录中获取，简化处理：直接调用 get_dataset_by_id）
    from modules.database import get_dataset_by_id
    record = get_dataset_by_id(dataset_id)
    filename = record['filename'] if record else 'history_data'
    return render_template('preview.html', data=data_info, filename=filename)


@app.route('/clean', methods=['POST'])
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

    data_info = preview_data(cleaned_df)
    message = '数据清洗完成（已生成清洗文件）'
    # 将缺失统计转为 HTML 字符串，避免在模板中对 DataFrame 做布尔判断
    missing_html = None
    try:
        missing_html = res['missing_summary'].to_html(classes='table table-sm table-bordered')
    except Exception:
        missing_html = None

    return render_template(
        'preview.html',
        data=data_info,
        filename=original_filename or '',
        message=message,
        cleaned_filename=cleaned_filename,
        missing_summary_html=missing_html,
    )

if __name__ == '__main__':
    app.run(debug=True)