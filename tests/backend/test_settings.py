"""
Tests for settings load/save and the backend API endpoints.

Bugs covered:
  - Settings not persisting (SETTINGS_PATH used wrong base path in prod)
  - loadSettings used Electron IPC file read instead of backend API
  - Default system prompt not showing on fresh install
  - tts_mode="remote" legacy normalisation
"""

import json
import pytest
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app, DEFAULT_SETTINGS


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "data" / "settings.json"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(main_mod, "SETTINGS_PATH", path)
    yield path


# ── GET /settings ─────────────────────────────────────────────────────────────

class TestGetSettings:
    def test_returns_defaults_on_missing_settings_file(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200
        body = resp.json()
        for key in DEFAULT_SETTINGS:
            assert key in body, f"Missing default key: {key}"

    def test_returns_saved_value(self, client, isolated_settings):
        isolated_settings.write_text(json.dumps({"language": "zh"}))
        resp = client.get("/settings")
        assert resp.json()["language"] == "zh"

    def test_merges_missing_keys_with_defaults(self, client, isolated_settings):
        # File only has one key — all defaults must still be present
        isolated_settings.write_text(json.dumps({"language": "en"}))
        resp = client.get("/settings")
        body = resp.json()
        assert body["language"] == "en"
        assert "tts_mode" in body
        assert "voicevox_speaker" in body

    def test_normalises_remote_to_kokoro(self, client, isolated_settings):
        isolated_settings.write_text(json.dumps({"tts_mode": "remote"}))
        resp = client.get("/settings")
        assert resp.json()["tts_mode"] == "kokoro"


# ── POST /settings ────────────────────────────────────────────────────────────

class TestPostSettings:
    def _minimal_settings(self, **overrides):
        base = {
            "stt_model": "qwen3-1.7b-4bit",
            "tts_mode": "voicevox",
            "voicevox_speaker": 2,
            "language": "ja",
            "llm_model": "",
            "llm_api_key": "",
            "llm_endpoint": "",
            "stt_endpoint": "",
            "tts_endpoint": "",
            "system_prompt_ja": "",
            "system_prompt_en": "",
            "system_prompt_zh": "",
            "search_online": False,
            "translate_to": "en",
            "voice_en": "Ryan",
            "voice_ja": "Ono_Anna",
            "voice_zh": "Vivian",
        }
        base.update(overrides)
        return base

    def test_saves_to_file(self, client, isolated_settings):
        client.post("/settings", json=self._minimal_settings(language="zh"))
        saved = json.loads(isolated_settings.read_text())
        assert saved["language"] == "zh"

    def test_saved_system_prompt_persists(self, client, isolated_settings):
        prompt = "テスト用プロンプト"
        client.post("/settings", json=self._minimal_settings(system_prompt_ja=prompt))
        saved = json.loads(isolated_settings.read_text())
        assert saved["system_prompt_ja"] == prompt

    def test_reload_after_save_returns_saved_value(self, client, isolated_settings):
        prompt = "reload test"
        client.post("/settings", json=self._minimal_settings(system_prompt_en=prompt))
        resp = client.get("/settings")
        assert resp.json()["system_prompt_en"] == prompt

    def test_returns_service_status_dict(self, client):
        resp = client.post("/settings", json=self._minimal_settings())
        assert resp.status_code == 200
        body = resp.json()
        # Should return connectivity status for each service
        assert "stt" in body or "llm" in body or "tts" in body


# ── default system prompts ────────────────────────────────────────────────────

class TestDefaultSystemPrompts:
    def test_japanese_default_is_japanese(self):
        s = {"language": "ja", "system_prompt_ja": ""}
        result = main_mod.get_system_prompt(s)
        # Should contain Japanese characters
        assert any('\u3000' <= c <= '\u9fff' or '\u30a0' <= c <= '\u30ff' for c in result)

    def test_english_default_is_english(self):
        s = {"language": "en", "system_prompt_en": ""}
        result = main_mod.get_system_prompt(s)
        assert result  # non-empty
        # Should not be empty or just whitespace
        assert len(result.strip()) > 10

    def test_custom_prompt_overrides_default(self):
        custom = "Only output one word answers."
        s = {"language": "en", "system_prompt_en": custom}
        result = main_mod.get_system_prompt(s)
        assert result.startswith(custom)

    def test_system_prompt_includes_current_datetime(self):
        import datetime
        s = {"language": "ja", "system_prompt_ja": "test"}
        result = main_mod.get_system_prompt(s)
        year = str(datetime.datetime.now().year)
        assert year in result


# ── search_online toggle endpoint ─────────────────────────────────────────────

class TestSearchOnlineToggle:
    def test_enable_search_online(self, client):
        resp = client.post("/settings/search_online", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_disable_search_online(self, client):
        resp = client.post("/settings/search_online", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_search_online_persists(self, client, isolated_settings):
        client.post("/settings/search_online", json={"enabled": True})
        saved = json.loads(isolated_settings.read_text())
        assert saved["search_online"] is True
