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

if __name__ == '__main__':
    app.run(debug=True)