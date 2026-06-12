# modules/database.py
import sqlite3
import os
from datetime import datetime

DB_PATH = 'data_history.db'

def init_db():
    """初始化数据库表（在 app 启动时调用一次）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 先创建表（如果不存在）
    c.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            row_count INTEGER,
            col_count INTEGER,
            session_id TEXT,
            user_id INTEGER
        )
    ''')
    
    # 检查并添加 user_id 列（迁移逻辑）
    c.execute("PRAGMA table_info(datasets)")
    columns = [col[1] for col in c.fetchall()]
    if 'user_id' not in columns:
        print("正在添加 user_id 列到 datasets 表...")
        c.execute('ALTER TABLE datasets ADD COLUMN user_id INTEGER')
        print("user_id 列添加成功")
    
    conn.commit()
    conn.close()

def save_dataset_record(filename, filepath, row_count, col_count, user_id=None, session_id='default'):
    """上传成功后保存记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    upload_time = datetime.now().isoformat()
    c.execute('''
        INSERT INTO datasets (filename, filepath, upload_time, row_count, col_count, session_id, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (filename, filepath, upload_time, row_count, col_count, session_id, user_id))
    conn.commit()
    last_id = c.lastrowid
    conn.close()
    return last_id

def get_all_datasets(user_id=None, session_id='default'):
    """获取当前用户的所有历史数据集"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if user_id:
        c.execute('SELECT id, filename, upload_time, row_count, col_count FROM datasets WHERE user_id=? ORDER BY upload_time DESC', (user_id,))
    else:
        c.execute('SELECT id, filename, upload_time, row_count, col_count FROM datasets WHERE session_id=? ORDER BY upload_time DESC', (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{'id': r[0], 'filename': r[1], 'upload_time': r[2], 'rows': r[3], 'cols': r[4]} for r in rows]

def get_dataset_by_id(dataset_id):
    """根据ID获取数据集信息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, filename, filepath, row_count, col_count FROM datasets WHERE id=?', (dataset_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'filename': row[1], 'filepath': row[2], 'rows': row[3], 'cols': row[4]}
    return None