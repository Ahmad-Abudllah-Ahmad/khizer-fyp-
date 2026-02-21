from fastapi.testclient import TestClient
from main import app 

# Use context manager to trigger lifespan events (model loading)
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "ai-service"}

def test_analyze_journal():
    with TestClient(app) as client:
        # Test positive entry
        response = client.post("/analyze/journal", json={"text": "I am feeling very happy today!", "language": "en"})
        assert response.status_code == 200
        data = response.json()
        assert "unified" in data
        assert data["unified"]["emotion"] in ["joy", "neutral", "positive"]
        assert data["unified"]["crisis_risk"] in ["LOW", "SAFE", "SAFE/LOW"]

    with TestClient(app) as client:
        # Test crisis entry
        response = client.post("/analyze/journal", json={"text": "I want to kill myself"})
        assert response.status_code == 200
        data = response.json()
        assert data["unified"]["crisis_risk"] == "CRISIS"
        assert data["unified"]["requires_immediate_action"] is True

