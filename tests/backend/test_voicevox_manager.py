"""
Tests for VoicevoxManager — focus on orphan process detection and cleanup.
"""
import importlib.util
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Load the real VoicevoxManager module directly, bypassing sys.modules so that
# conftest.py's mock of modules.tts.voicevox_manager doesn't shadow the real class.
_vvm_spec = importlib.util.spec_from_file_location(
    "_real_voicevox_manager",
    Path(__file__).parent.parent.parent / "modules" / "tts" / "voicevox_manager.py",
)
_vvm_mod = importlib.util.module_from_spec(_vvm_spec)
_vvm_spec.loader.exec_module(_vvm_mod)

VoicevoxManager = _vvm_mod.VoicevoxManager
VOICEVOX_PORT   = _vvm_mod.VOICEVOX_PORT


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def manager():
    """Fresh VoicevoxManager with no owned process."""
    m = VoicevoxManager()
    m._proc = None
    return m


# ── is_running ────────────────────────────────────────────────────────────────

class TestIsRunning:
    def test_false_when_proc_is_none_and_port_closed(self, manager):
        with patch.object(_vvm_mod.urllib.request, "urlopen", side_effect=OSError("refused")):
            assert manager.is_running() is False

    def test_true_when_port_responds_200(self, manager):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch.object(_vvm_mod.urllib.request, "urlopen", return_value=mock_resp):
            assert manager.is_running() is True

    def test_true_when_proc_alive(self, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        manager._proc = mock_proc
        assert manager.is_running() is True

    def test_false_when_proc_exited(self, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited with code 1
        manager._proc = mock_proc
        with patch.object(_vvm_mod.urllib.request, "urlopen", side_effect=OSError):
            assert manager.is_running() is False


# ── stop: owned process path ──────────────────────────────────────────────────

class TestStopOwnedProcess:
    def test_terminate_called_on_running_proc(self, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process still running
        manager._proc = mock_proc
        # stop() checks: `if self._proc and self._proc.poll() is None`
        manager.stop()
        mock_proc.terminate.assert_called_once()
        assert manager._proc is None

    def test_kill_called_on_timeout(self, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = _vvm_mod.subprocess.TimeoutExpired(cmd="run", timeout=5)
        manager._proc = mock_proc
        manager.stop()
        mock_proc.kill.assert_called_once()
        assert manager._proc is None


# ── stop: orphan path (no owned process) ─────────────────────────────────────

class TestStopOrphan:
    def test_kills_pids_from_lsof_when_proc_is_none(self, manager):
        """When _proc is None, stop() must use lsof to find and SIGKILL orphan."""
        mock_run = MagicMock(return_value=MagicMock(stdout="12345\n67890\n", returncode=0))

        with patch.object(_vvm_mod.subprocess, "run", mock_run), \
             patch.object(_vvm_mod.os, "kill") as mock_kill:
            manager.stop()

        mock_run.assert_called_once_with(
            ["lsof", "-ti", f":{VOICEVOX_PORT}"],
            capture_output=True, text=True,
        )
        assert call(12345, signal.SIGKILL) in mock_kill.call_args_list
        assert call(67890, signal.SIGKILL) in mock_kill.call_args_list

    def test_no_crash_when_lsof_finds_nothing(self, manager):
        """Empty lsof output (no orphan) must not raise."""
        mock_run = MagicMock(return_value=MagicMock(stdout="", returncode=1))
        with patch.object(_vvm_mod.subprocess, "run", mock_run), \
             patch.object(_vvm_mod.os, "kill") as mock_kill:
            manager.stop()  # should not raise
        mock_kill.assert_not_called()

    def test_no_crash_when_lsof_unavailable(self, manager):
        """If lsof itself fails, stop() swallows the error."""
        with patch.object(_vvm_mod.subprocess, "run", side_effect=FileNotFoundError("lsof")):
            manager.stop()  # should not raise

    def test_ignores_invalid_pid_lines(self, manager):
        """Non-numeric lsof output lines are silently skipped."""
        mock_run = MagicMock(return_value=MagicMock(stdout="abc\n99999\n", returncode=0))
        with patch.object(_vvm_mod.subprocess, "run", mock_run), \
             patch.object(_vvm_mod.os, "kill") as mock_kill:
            manager.stop()
        # Only the valid PID 99999 should have been killed
        mock_kill.assert_called_once_with(99999, signal.SIGKILL)


# ── start: orphan restart path ────────────────────────────────────────────────

class TestStartOrphanRestart:
    def test_start_kills_orphan_and_restarts(self, manager, tmp_path):
        """
        If is_running() is True but _proc is None (orphan from previous session),
        start() must call stop() first, then launch a fresh process.
        """
        fake_binary = tmp_path / "run"
        fake_binary.touch()
        fake_binary.chmod(0o755)

        stop_calls = []
        original_stop = manager.stop

        def _recording_stop():
            stop_calls.append(1)
            manager._proc = None  # simulate stop clearing the proc

        manager.stop = _recording_stop

        # Simulate: port is occupied (orphan) but _proc is None
        with patch.object(manager, "is_running", side_effect=[True, False]), \
             patch.object(_vvm_mod, "_find_binary", return_value=fake_binary), \
             patch("subprocess.Popen") as mock_popen, \
             patch("urllib.request.urlopen") as mock_urlopen, \
             patch("time.sleep"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_urlopen.return_value = mock_resp

            manager.start()

        assert len(stop_calls) == 1, "stop() must be called once to clear the orphan"
        mock_popen.assert_called_once()

    def test_start_noop_when_already_owned_and_running(self, manager):
        """If _proc is not None and process is alive, start() is a no-op."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        manager._proc = mock_proc

        with patch.object(manager, "is_running", return_value=True), \
             patch("subprocess.Popen") as mock_popen:
            manager.start()

        mock_popen.assert_not_called()


# ── _proc = None after stop ───────────────────────────────────────────────────

class TestProcNullAfterStop:
    def test_proc_always_none_after_stop(self, manager):
        """_proc must be None after stop() regardless of code path taken."""
        # Path 1: no owned proc, lsof returns nothing
        manager._proc = None
        with patch("subprocess.run", return_value=MagicMock(stdout="")):
            manager.stop()
        assert manager._proc is None

        # Path 2: owned proc
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        manager._proc = mock_proc
        manager.stop()
        assert manager._proc is None
