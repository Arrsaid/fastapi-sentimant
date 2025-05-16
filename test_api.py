from fastapi.testclient import TestClient
from api import app 

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"text": "I love flying with air paradis!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in ["positif", "négatif"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
