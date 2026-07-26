import os
import sqlite3
import json
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash


def get_db_connection(db_path='database.db'):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path='database.db'):
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            label TEXT,
            confidence REAL,
            probability REAL,
            keywords TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()

    cursor.execute('SELECT id FROM users WHERE is_admin = 1 LIMIT 1')
    if cursor.fetchone() is None:
        admin_password = generate_password_hash('Admin@123')
        cursor.execute(
            'INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, ?)',
            ('Administrator', 'admin@frauddetector.local', admin_password, 1)
        )
        conn.commit()

    conn.close()


def create_user(name, email, password, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    hashed = generate_password_hash(password)
    try:
        cursor.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (name, email, hashed)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def verify_user(email, password, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    if user and check_password_hash(user['password'], password):
        return dict(user)
    return None


def get_user_by_email(email, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def add_prediction(user_id, text, label, confidence, probability, keywords, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO predictions (user_id, text, label, confidence, probability, keywords) VALUES (?, ?, ?, ?, ?, ?)',
        (user_id, text, label, confidence, probability, json.dumps(keywords))
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id=None, limit=8, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if user_id:
        cursor.execute(
            'SELECT p.*, u.name as user_name FROM predictions p JOIN users u ON p.user_id = u.id WHERE p.user_id = ? ORDER BY p.created_at DESC LIMIT ?',
            (user_id, limit)
        )
    else:
        cursor.execute(
            'SELECT p.*, u.name as user_name FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.created_at DESC LIMIT ?',
            (limit,)
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_history(user_id=None, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if user_id:
        cursor.execute(
            'SELECT p.*, u.name as user_name FROM predictions p JOIN users u ON p.user_id = u.id WHERE p.user_id = ? ORDER BY p.created_at DESC',
            (user_id,)
        )
    else:
        cursor.execute(
            'SELECT p.*, u.name as user_name FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.created_at DESC'
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_users(db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, email, is_admin, created_at FROM users ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_history_entry(prediction_id, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    conn.commit()
    conn.close()


def get_stats(user_id=None, db_path='database.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    where_clause = ' WHERE user_id = ?' if user_id is not None else ''
    parameters = (user_id,) if user_id is not None else ()
    cursor.execute('SELECT COUNT(*) AS total FROM predictions' + where_clause, parameters)
    total = cursor.fetchone()['total']
    cursor.execute(
        'SELECT COUNT(*) AS fake_count FROM predictions' + where_clause + (' AND' if where_clause else ' WHERE') + " label = 'Fake'",
        parameters,
    )
    fake_count = cursor.fetchone()['fake_count']
    cursor.execute(
        'SELECT COUNT(*) AS real_count FROM predictions' + where_clause + (' AND' if where_clause else ' WHERE') + " label = 'Real'",
        parameters,
    )
    real_count = cursor.fetchone()['real_count']
    conn.close()
    return {
        'total': total,
        'fake_count': fake_count,
        'real_count': real_count,
    }


def get_dataset_stats(fake_path='dataset/Fake.csv', true_path='dataset/True.csv'):
    stats = {'fake_rows': 0, 'true_rows': 0, 'total_rows': 0}
    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        stats['fake_rows'] = len(fake_df)
        stats['true_rows'] = len(true_df)
        stats['total_rows'] = stats['fake_rows'] + stats['true_rows']
    except Exception:
        pass
    return stats
