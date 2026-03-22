"""
Tests for the /tts endpoint VOICEVOX cache.

Cache contract:
  - Same (text, speaker, speed) → returns cached WAV, _voicevox_tts NOT called again
  - Different text             → re-synthesizes, cache overwritten
  - Different speaker          → re-synthesizes, cache overwritten
  - Different speed            → re-synthesizes, cache overwritten
  - Kokoro mode                → cache bypassed entirely (not VOICEVOX)
"""

import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app


FAKE_WAV_A = b"RIFF\x00\x00\x00\x00WAVEfmt A"
FAKE_WAV_B = b"RIFF\x00\x00\x00\x00WAVEfmt B"


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "data" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "tts_mode": "voicevox",
        "voicevox_speaker": 2,
        "voicevox_speed": 1.0,
        "language": "ja",
    }))
    monkeypatch.setattr(main_mod, "SETTINGS_PATH", path)
    yield path


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """Reset the module-level cache before every test."""
    monkeypatch.setattr(main_mod, "_tts_cache", None)


# ── cache hit ─────────────────────────────────────────────────────────────────

class TestCacheHit:
    def test_second_call_returns_cached_wav(self, client):
        call_count = [0]

        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            call_count[0] += 1
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            r1 = client.post("/tts", json={"text": "こんにちは", "language": "ja"})
            r2 = client.post("/tts", json={"text": "こんにちは", "language": "ja"})

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.content == FAKE_WAV_A
        assert r2.content == FAKE_WAV_A
        assert call_count[0] == 1, "Cache hit should not call _voicevox_tts again"

    def test_cache_hit_skips_ensure_voicevox_running(self, client):
        """A cache hit should not even check if VOICEVOX is running."""
        ensure_calls = [0]

        async def counting_ensure(s):
            ensure_calls[0] += 1

        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", side_effect=counting_ensure):
            client.post("/tts", json={"text": "テスト", "language": "ja"})  # miss → synthesize
            client.post("/tts", json={"text": "テスト", "language": "ja"})  # hit  → skip

        assert ensure_calls[0] == 1, "ensure_voicevox_running must not be called on cache hit"


# ── cache invalidation ────────────────────────────────────────────────────────

class TestCacheInvalidation:
    def _make_tts(self, responses: list[bytes]):
        """Returns an async fake_tts that pops from responses on each call."""
        idx = [0]
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            wav = responses[idx[0]]
            idx[0] += 1
            return wav
        return fake_tts

    def test_different_text_invalidates_cache(self, client):
        call_count = [0]
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            call_count[0] += 1
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            client.post("/tts", json={"text": "最初", "language": "ja"})
            client.post("/tts", json={"text": "別のテキスト", "language": "ja"})

        assert call_count[0] == 2, "Different text must trigger re-synthesis"

    def test_different_speaker_invalidates_cache(self, client, isolated_settings):
        call_count = [0]
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            call_count[0] += 1
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            client.post("/tts", json={"text": "テスト", "language": "ja"})
            # Change speaker in settings
            isolated_settings.write_text(json.dumps({
                "tts_mode": "voicevox", "voicevox_speaker": 8,
                "voicevox_speed": 1.0, "language": "ja",
            }))
            client.post("/tts", json={"text": "テスト", "language": "ja"})

        assert call_count[0] == 2, "Different speaker must trigger re-synthesis"

    def test_different_speed_invalidates_cache(self, client, isolated_settings):
        call_count = [0]
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            call_count[0] += 1
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            client.post("/tts", json={"text": "テスト", "language": "ja"})
            # Change speed in settings
            isolated_settings.write_text(json.dumps({
                "tts_mode": "voicevox", "voicevox_speaker": 2,
                "voicevox_speed": 1.5, "language": "ja",
            }))
            client.post("/tts", json={"text": "テスト", "language": "ja"})

        assert call_count[0] == 2, "Different speed must trigger re-synthesis"


# ── cache overwrite ───────────────────────────────────────────────────────────

class TestCacheOverwrite:
    def test_new_text_overwrites_cache(self, client):
        """After a cache miss the slot holds the new audio, not the old one."""
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            return FAKE_WAV_A if text == "最初" else FAKE_WAV_B

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            client.post("/tts", json={"text": "最初", "language": "ja"})
            client.post("/tts", json={"text": "次のターン", "language": "ja"})
            # Replay the second text — must come from cache (FAKE_WAV_B)
            r = client.post("/tts", json={"text": "次のターン", "language": "ja"})

        assert r.content == FAKE_WAV_B

    def test_old_text_after_overwrite_re_synthesizes(self, client):
        """After the cache is overwritten, the previous text is no longer cached."""
        call_count = [0]
        async def fake_tts(c, text, speaker, endpoint, speed=1.0):
            call_count[0] += 1
            return FAKE_WAV_A

        with patch.object(main_mod, "_voicevox_tts", side_effect=fake_tts), \
             patch.object(main_mod, "_ensure_voicevox_running", new=AsyncMock()):
            client.post("/tts", json={"text": "最初", "language": "ja"})   # call 1
            client.post("/tts", json={"text": "次のターン", "language": "ja"})  # call 2 — overwrites
            client.post("/tts", json={"text": "最初", "language": "ja"})   # call 3 — miss, not cached

        assert call_count[0] == 3


# ── kokoro bypass ─────────────────────────────────────────────────────────────

class TestKokoroCacheBypass:
    def test_kokoro_mode_does_not_populate_cache(self, client, isolated_settings, monkeypatch):
        """Cache must remain None after a Kokoro synthesis."""
        isolated_settings.write_text(json.dumps({
            "tts_mode": "kokoro",
            "tts_endpoint": "http://127.0.0.1:9999",
            "language": "ja",
            "voice_ja": "Ono_Anna",
        }))

        mock_post = AsyncMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.content = FAKE_WAV_A

        with patch.object(main_mod.httpx.AsyncClient, "post", mock_post):
            client.post("/tts", json={"text": "テスト", "language": "ja"})

        assert main_mod._tts_cache is None, "Kokoro must not populate the VOICEVOX cache"
