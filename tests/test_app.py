import pytest
from flask import Flask
from ..app2 import app

# Création d'une application Flask de test en utilisant la fixture de Pytest
@pytest.fixture
def client():
    app.config["TESTING"] = True
    client = app.test_client()
    yield client

# Écriture de fonctions de test pour tester les endpoints de l'API Flask
def test_users_endpoint(client):
    # Requête HTTP GET vers l'endpoint /api/users
    response = client.get('/api/users')
    data = response.get_json()

    # Statut HTTP 200 (OK)
    assert response.status_code == 200

    # Données attendues au format liste
    assert "users" in data
    assert isinstance(data["users"], list)

def test_single_user_endpoint(client):
    # Requête HTTP GET vers l'endpoint /api/user/<user_id>
    user_id = 100005
    response = client.get(f'/api/user/{user_id}')
    data = response.get_json()

    # Statut HTTP 200 (OK)
    assert response.status_code == 200

def test_shap_features_endpoint(client):
    # Requête HTTP GET vers l'endpoint /api/shap_features
    response = client.get('/api/shap_features')
    data = response.get_json()

    # Statut HTTP 200 (OK)
    assert response.status_code == 200