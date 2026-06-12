import sqlite3
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, request

DB_PATH = 'data_history.db'

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt + ':' + hashed.hex()

def verify_password(password, stored_hash):
    try:
        salt, hashed = stored_hash.split(':')
        computed = hash_password(password, salt)
        return computed == stored_hash
    except:
        return False

def init_user_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password, email=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        password_hash = hash_password(password)
        created_at = datetime.now().isoformat()
        c.execute('''
            INSERT INTO users (username, password_hash, email, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, email, created_at))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "用户名已存在"
    except Exception as e:
        return None, str(e)

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    
    if row is None:
        return None, "用户名或密码错误"
    
    user_id, stored_hash = row
    if verify_password(password, stored_hash):
        update_last_login(user_id)
        return user_id, None
    else:
        return None, "用户名或密码错误"

def update_last_login(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    last_login = datetime.now().isoformat()
    c.execute('UPDATE users SET last_login=? WHERE id=?', (last_login, user_id))
    conn.commit()
    conn.close()

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, email, created_at, last_login FROM users WHERE id=?', (user_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'username': row[1],
            'email': row[2],
            'created_at': row[3],
            'last_login': row[4]
        }
    return None

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, email, created_at, last_login FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'username': row[1],
            'email': row[2],
            'created_at': row[3],
            'last_login': row[4]
        }
    return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def login_user(user_id, username):
    session['user_id'] = user_id
    session['username'] = username
    session.permanent = True

def logout_user():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('cleaned', None)
    session.pop('last_filename', None)