# SPDX-License-Identifier: LGPL-2.1-or-later

import contextlib
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from mkosi.config import Config
from mkosi.qemu import (
    JOURNAL_REMOTE_TIMEOUT,
    lock_journal_target,
    start_journal_forward_session,
    start_journal_remote,
    start_journal_remote_unix,
)
from mkosi.util import PathString


class FakeConfig:
    def __init__(self, target: Path, journal_remote: Path) -> None:
        self.forward_journal = target
        self._journal_remote = journal_remote

    def find_binary(self, *names: object) -> Path:
        return self._journal_remote

    def sandbox(self, **kwargs: object) -> contextlib.AbstractContextManager[list[PathString]]:
        return contextlib.nullcontext([])


def write_fake_remote(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_lock_journal_target_conflict(tmp_path: Path) -> None:
    target = tmp_path / "foo.journal"
    target.write_bytes(b"old journal data")
    stat = target.stat()

    with lock_journal_target(target):
        with pytest.raises(SystemExit):
            with lock_journal_target(target):
                pass

    # The existing journal must not be touched by the failed session.
    assert target.stat().st_size == stat.st_size
    assert target.stat().st_mtime_ns == stat.st_mtime_ns

    # Once the session exits, the released lock can be acquired again immediately.
    with lock_journal_target(target):
        pass


def test_forward_session_file_target(tmp_path: Path) -> None:
    remote = write_fake_remote(tmp_path / "journal-remote", "exec sleep 60\n")
    target = tmp_path / "sub" / "bar.journal"
    config = cast(Config, FakeConfig(target, remote))

    with start_journal_forward_session(config):
        assert target.parent.is_dir()
        with pytest.raises(SystemExit):
            with start_journal_forward_session(config):
                pass


def test_forward_session_directory_target(tmp_path: Path) -> None:
    remote = write_fake_remote(tmp_path / "journal-remote", "exec sleep 60\n")
    target = tmp_path / "logs"
    config = cast(Config, FakeConfig(target, remote))

    # Directory targets allow multiple concurrent sessions.
    with start_journal_forward_session(config):
        assert target.is_dir()
        with start_journal_forward_session(config):
            with start_journal_forward_session(config):
                pass


def test_journal_remote_unix_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    remote = write_fake_remote(tmp_path / "journal-remote", "exec sleep 60\n")
    config = cast(Config, FakeConfig(tmp_path / "m.journal", remote))

    with start_journal_remote_unix(config) as addr:
        assert addr.is_socket()

    assert not addr.exists()


def test_journal_remote_unix_exception_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    remote = write_fake_remote(tmp_path / "journal-remote", "exec sleep 60\n")
    config = cast(Config, FakeConfig(tmp_path / "m.journal", remote))

    with pytest.raises(RuntimeError, match="boom"):
        with start_journal_remote_unix(config):
            raise RuntimeError("boom")

    assert not list(tmp_path.glob("mkosi-journal-remote-unix-*"))


def test_journal_remote_unix_remote_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    remote = write_fake_remote(tmp_path / "journal-remote", "exit 7\n")
    config = cast(Config, FakeConfig(tmp_path / "m.journal", remote))

    with pytest.raises(SystemExit):
        with start_journal_remote_unix(config):
            pytest.fail("session should not be entered when the journal remote exits immediately")

    # Neither the socket file nor a remote process may be left behind.
    assert not list(tmp_path.glob("mkosi-journal-remote-unix-*"))


@pytest.mark.skipif(os.getuid() != 0, reason="root required to switch user ids")
def test_lock_journal_target_mixed_uid(tmp_path: Path) -> None:
    # Make the per-test directory and all its parents (e.g. pytest-of-root/) traversable by the
    # unprivileged user.
    for d in (tmp_path, *tmp_path.parents):
        if d == Path("/"):
            break
        d.chmod(d.stat().st_mode | 0o005)
        if d == Path("/tmp"):
            break

    target = tmp_path / "m.journal"
    target.write_bytes(b"old journal data")
    stat = target.stat()
    nobody = 65534

    def attempt() -> str:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                # Switch group credentials too: without setgid() the child keeps gid 0 and permission
                # checks match the (empty) root group instead of "other".
                os.setgid(nobody)
                os.setgroups([nobody])
                os.setuid(nobody)
                with lock_journal_target(target):
                    os.write(w, b"OK")
            except SystemExit:
                os.write(w, b"EXIT")
            except PermissionError:
                os.write(w, b"PERM")
            finally:
                os.close(w)
                os._exit(0)

        os.close(w)
        data = os.read(r, 16)
        os.waitpid(pid, 0)
        return data.decode()

    # While root holds the session, the unprivileged user gets a clean die() instead of a crash.
    with lock_journal_target(target):
        assert attempt() == "EXIT"

    # After the session exits, a stale lock file created by root must not block an unprivileged session.
    assert attempt() == "OK"

    assert target.stat().st_size == stat.st_size
    assert target.stat().st_mtime_ns == stat.st_mtime_ns


def test_journal_remote_cancel_during_settle(tmp_path: Path) -> None:
    remote = write_fake_remote(
        tmp_path / "journal-remote",
        "trap '' TERM INT\nexec sleep 30\n",
    )
    config = cast(Config, FakeConfig(tmp_path / "m.journal", remote))

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(os.fspath(tmp_path / "listener.sock"))
    sock.listen()

    def interrupt() -> None:
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt).start()

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        with start_journal_remote(config, sock.fileno()):
            pass
    elapsed = time.monotonic() - start

    # Cancellation in the startup window must stop the remote via the bounded SIGKILL escalation instead
    # of waiting forever for it to honor SIGINT/SIGTERM.
    assert JOURNAL_REMOTE_TIMEOUT - 0.1 < elapsed < JOURNAL_REMOTE_TIMEOUT + 2
