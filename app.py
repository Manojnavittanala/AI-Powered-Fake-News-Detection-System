import os
import json
import logging
import sqlite3
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from utils.database import (
    init_db,
    create_user,
    verify_user,
    add_prediction,
    get_recent_history,
    get_stats,
    get_all_users,
    get_history,
    delete_history_entry,
    get_dataset_stats,
)
from utils.predictor import load_model, predict_news
from utils.preprocessing import validate_news_text

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change_this_secret')
app.config['PERMANENT_SESSION_LIFETIME'] = 1800
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), 'database.db')
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'model.pkl')
app.config['VECTORIZER_PATH'] = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
app.config['MODEL_BACKEND'] = os.environ.get('MODEL_BACKEND', 'classical').lower()
app.config['TRANSFORMER_MODEL'] = os.environ.get('TRANSFORMER_MODEL', '')

init_db(app.config['DATABASE'])


@app.before_request
def ensure_database():
    """Initialize the configured database, including test databases."""
    init_db(app.config['DATABASE'])

try:
    if app.config['MODEL_BACKEND'] == 'transformer':
        from utils.transformer_predictor import load_transformer
        transformer_tokenizer, model = load_transformer(app.config['TRANSFORMER_MODEL'])
        vectorizer = None
        model_metrics = {'selected': {'accuracy': None}, 'model_name': 'transformer'}
    else:
        model, vectorizer, model_metrics = load_model(app.config['MODEL_PATH'], app.config['VECTORIZER_PATH'])
        transformer_tokenizer = None
except Exception as error:
    model = None
    vectorizer = None
    transformer_tokenizer = None
    model_metrics = {}
    logging.exception('Unable to load model: %s', error)


def get_current_user():
    if 'user_id' in session:
        return {
            'id': session['user_id'],
            'name': session.get('user_name'),
            'is_admin': session.get('is_admin', False),
        }
    return None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not get_current_user():
            message = 'Your session has expired. Please login again.'
            if request.path.startswith('/api/'):
                return jsonify({'error': message}), 401
            flash(message, 'error')
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user or not user['is_admin']:
            flash('Administrator access is required.', 'error')
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapper


@app.errorhandler(sqlite3.Error)
def handle_database_error(error):
    logging.exception('Database unavailable: %s', error)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Database unavailable.'}), 503
    flash('Database unavailable. Please try again.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found.'}), 404
    return render_template('index.html', user=get_current_user()), 404


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logging.exception('Unhandled application error: %s', error)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Something went wrong.'}), 500
    return render_template('index.html', user=get_current_user(), error_message='Something went wrong.'), 500


@app.route('/')
def index():
    accuracy = model_metrics.get('selected', {}).get('accuracy')
    accuracy = round(float(accuracy) * 100, 2) if accuracy is not None else None
    return render_template('index.html', user=get_current_user(), model_accuracy=accuracy)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if get_current_user():
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not name or not email or not password:
            flash('Please fill in all registration fields.', 'error')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))

        user_id = create_user(name, email, password, app.config['DATABASE'])
        if user_id:
            flash('Registration Successful. Please log in.', 'success')
            return redirect(url_for('login'))

        flash('An account with that email already exists.', 'error')
        return redirect(url_for('register'))

    return render_template('register.html', user=get_current_user())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if get_current_user():
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = verify_user(email, password, app.config['DATABASE'])

        if user:
            session.clear()
            session.permanent = True
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['is_admin'] = bool(user['is_admin'])
            flash('Login Successful', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid Email or Password', 'error')
        return redirect(url_for('login'))

    return render_template('login.html', user=get_current_user())


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/detect')
@login_required
def detect():
    if model is None or (app.config['MODEL_BACKEND'] == 'classical' and vectorizer is None):
        flash('The prediction model is not available. Run training first.', 'error')
        return redirect(url_for('index'))

    return render_template('detect.html', user=get_current_user())


@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_stats(user_id=session['user_id'], db_path=app.config['DATABASE'])
    recent = get_recent_history(session['user_id'], limit=6, db_path=app.config['DATABASE'])
    selected_metrics = model_metrics.get('selected', {})
    selected_accuracy = selected_metrics.get('accuracy')
    stats['accuracy'] = round(float(selected_accuracy) * 100, 2) if selected_accuracy is not None else 0
    chart_data = {
        'labels': ['Fake', 'Real'],
        'values': [int(stats['fake_count']), int(stats['real_count'])],
    }
    chart_data_json = json.dumps(chart_data)
    return render_template('dashboard.html', user=get_current_user(), stats=stats, recent=recent, chart_data_json=chart_data_json)


@app.route('/history')
@login_required
def history():
    return redirect(url_for('dashboard'))


@app.route('/profile')
@login_required
def profile():
    return redirect(url_for('dashboard'))


@app.route('/admin')
@admin_required
def admin():
    users = get_all_users(app.config['DATABASE'])
    history = get_history(db_path=app.config['DATABASE'])
    dataset_stats = get_dataset_stats()
    summary = get_stats(db_path=app.config['DATABASE'])
    return render_template('admin.html', user=get_current_user(), users=users, history=history, dataset_stats=dataset_stats, summary=summary)


@app.route('/admin/delete/<int:prediction_id>', methods=['POST'])
@admin_required
def admin_delete(prediction_id):
    delete_history_entry(prediction_id, app.config['DATABASE'])
    flash('Prediction history entry deleted.', 'success')
    return redirect(url_for('admin'))


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '')
    user_id = session.get('user_id')

    validation_error = validate_news_text(text)
    if validation_error:
        return jsonify({'error': validation_error}), 400

    if model is None or (app.config['MODEL_BACKEND'] == 'classical' and vectorizer is None):
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500

    try:
        if app.config['MODEL_BACKEND'] == 'transformer':
            from utils.transformer_predictor import predict_transformer
            result = predict_transformer(text.strip(), transformer_tokenizer, model)
        else:
            result = predict_news(text.strip(), model, vectorizer)
    except Exception:
        logging.exception('Prediction failed')
        return jsonify({'error': 'Prediction failed.'}), 500
    if user_id:
        add_prediction(user_id, text.strip(), result['label'], result['confidence'], result['probability'], result['keywords'], app.config['DATABASE'])

    return jsonify(result)


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({'error': 'Missing required fields.'}), 400

    user_id = create_user(name, email, password, app.config['DATABASE'])
    if not user_id:
        return jsonify({'error': 'Email already registered.'}), 409

    return jsonify({'message': 'Registration successful.'}), 201


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = verify_user(email, password, app.config['DATABASE'])

    if not user:
        return jsonify({'error': 'Invalid Email or Password'}), 401

    session.clear()
    session.permanent = True
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    session['is_admin'] = bool(user['is_admin'])
    return jsonify({'message': 'Login successful.'})


@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    records = get_history(user_id=session['user_id'], db_path=app.config['DATABASE'])
    for record in records:
        record['keywords'] = json.loads(record['keywords']) if record.get('keywords') else []
    return jsonify(records)


@app.route('/api/stats', methods=['GET'])
@login_required
def api_stats():
    stats = get_stats(user_id=session['user_id'], db_path=app.config['DATABASE'])
    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=False)
