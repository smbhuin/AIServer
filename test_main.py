from fastapi.testclient import TestClient
import api
from settings import (
    ConfigFileSettings,
)

config = ConfigFileSettings()

server_settings = config.server
models_settings = config.models
app = api.create_app(server_settings, models_settings)
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}