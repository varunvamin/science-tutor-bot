import pytest
from app import app

@pytest.fixture
def client():
    """Setup Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page_loads(client):
    """Test that the main interface loads successfully"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Science Tutor Bot" in response.data

def test_chat_empty_message(client):
    """Test the chat endpoint with an empty message"""
    response = client.post('/chat', json={'message': '   '})
    assert response.status_code == 200
    data = response.get_json()
    assert 'Please ask a science question!' in data['response']

def test_chat_too_long(client):
    """Test the chat endpoint with a message exceeding length limits"""
    long_msg = "A" * 1005
    response = client.post('/chat', json={'message': long_msg})
    assert response.status_code == 200
    data = response.get_json()
    assert 'too long' in data['response']
