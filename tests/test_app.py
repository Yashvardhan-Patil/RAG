from starlette.testclient import TestClient

from rag_eval_studio.app import build_app


def test_healthz_endpoint():
    app = build_app()
    client = TestClient(app.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
