import os
import tempfile
import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client


def test_register_login_predict_history_endpoints(client):
    file_descriptor, db_path = tempfile.mkstemp()
    os.close(file_descriptor)
    app.config['DATABASE'] = db_path

    # Register new user
    response = client.post('/api/register', json={
        'name': 'Test User',
        'email': 'test.user@example.com',
        'password': 'SecurePass123'
    })
    assert response.status_code == 201

    # Login with registered user
    response = client.post('/api/login', json={
        'email': 'test.user@example.com',
        'password': 'SecurePass123'
    })
    assert response.status_code == 200

    # Predict placeholder text
    article = (
        'Verified officials released a detailed report today explaining the event, '
        'including dates, sources, public records, and supporting evidence for readers '
        'to review before drawing conclusions.'
    )
    response = client.post('/api/predict', json={'text': article})
    assert response.status_code == 200
    data = response.get_json()
    assert 'label' in data
    assert 'confidence' in data

    # History endpoint should require authentication and return list
    response = client.get('/api/history')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

    os.unlink(db_path)


def test_auth_redirect_validation_and_dashboard_serialization(client):
    file_descriptor, db_path = tempfile.mkstemp()
    os.close(file_descriptor)
    app.config['DATABASE'] = db_path

    landing = client.get('/')
    assert b'href="/detect"' not in landing.data
    assert b'>Start Detection<' not in landing.data
    assert b'>Try detection<' not in landing.data
    assert client.get('/dashboard').status_code == 302
    assert client.get('/dashboard').headers['Location'].endswith('/login')
    assert client.get('/detect').headers['Location'].endswith('/login')
    assert client.get('/history').headers['Location'].endswith('/login')
    assert client.get('/profile').headers['Location'].endswith('/login')
    assert client.get('/admin').headers['Location'].endswith('/login')
    assert client.get('/api/history').status_code == 401
    assert client.post('/api/predict', json={'text': 'a' * 120}).status_code == 401
    assert client.post('/api/register', json={
        'name': 'Dashboard User',
        'email': 'dashboard@example.com',
        'password': 'SecurePass123',
    }).status_code == 201
    assert client.post('/api/login', json={
        'email': 'dashboard@example.com',
        'password': 'SecurePass123',
    }).status_code == 200

    short_response = client.post('/api/predict', json={'text': 'qwerty qwerty'})
    assert short_response.status_code == 400

    dashboard = client.get('/dashboard')
    assert dashboard.status_code == 200
    assert b'const chartData = {' in dashboard.data
    assert b'Model accuracy' in dashboard.data

    assert client.get('/logout').status_code == 302
    assert client.get('/dashboard').headers['Location'].endswith('/login')
    os.unlink(db_path)
