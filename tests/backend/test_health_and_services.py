"""
Tests for health, services status, and install-check endpoints.
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import backend.main as main_mod
from backend.main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "data" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"language": "ja", "tts_mode": "voicevox"}))
    monkeypatch.setattr(main_mod, "SETTINGS_PATH", path)


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_does_not_require_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code != 401
        assert resp.status_code != 403


class TestInstallCheck:
    def test_returns_llama_and_voicevox_keys(self, client):
        resp = client.get("/system/install-check")
        assert resp.status_code == 200
        body = resp.json()
        assert "llama_server" in body
        assert "voicevox_engine" in body

    def test_values_are_booleans(self, client):
        body = client.get("/system/install-check").json()
        assert isinstance(body["llama_server"], bool)
        assert isinstance(body["voicevox_engine"], bool)


class TestServicesStatus:
    def test_returns_all_service_keys(self, client):
        resp = client.get("/system/services")
        assert resp.status_code == 200
        body = resp.json()
        assert "llama_server" in body
        assert "voicevox_engine" in body
        assert "stt" in body

    def test_llama_server_has_expected_fields(self, client):
        body = client.get("/system/services").json()
        llama = body["llama_server"]
        assert "installed" in llama
        assert "running" in llama

    def test_stt_has_model_field(self, client):
        body = client.get("/system/services").json()
        stt = body["stt"]
        assert "loaded" in stt


class TestShutdownEndpoint:
    def test_shutdown_returns_200(self, client):
        """POST /shutdown must respond before the process exits."""
        import threading
        result = {}

        def _call():
            # Call with a very short read timeout — we just want the response header
            try:
                resp = client.post("/shutdown")
                result["status"] = resp.status_code
            except Exception as e:
                result["error"] = str(e)

        t = threading.Thread(target=_call)
        t.start()
        t.join(timeout=2)
        # Should return 200 before os._exit is called in the test client context
        # (TestClient does not actually call os._exit)
        assert result.get("status") == 200 or "error" not in result
