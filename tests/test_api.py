from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_move_returns_legal_move():
    response = client.post(
        "/predict-move",
        json={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "move_uci" in body
    assert "move_san" in body
    assert "predicted_eval_cp" in body


def test_predict_move_rejects_invalid_fen():
    response = client.post("/predict-move", json={"fen": "not-a-real-fen"})
    assert response.status_code == 400
