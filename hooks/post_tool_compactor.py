#!/usr/bin/env python3
"""Claude Code ``PostToolUse`` hook: read-only integrity check plus compact turn log.

After every tool call this hook

1. hashes (SHA-256) every file listed in ``<state_dir>/readonly_manifest.json`` and compares it
   with the recorded digest. The files are always hashed in full - there is no size / mtime
   shortcut, because both can be forged. The cost is linear in the manifest (one ``lstat``, one
   ``open`` and one read per file; measured at 0.13-0.27 ms per small file on Windows 11), so
   the 2 ms target holds for small manifests only. Every event records the measured time
   (``integrity.verify_ms``) and whether it exceeded the target (``integrity.over_budget``).
2. on any mismatch or missing file: restores the file from the immutable base commit
   (``git reset HEAD -- <file>``, ``git checkout <base_commit_sha> -- <file>``, ``chmod a-w``),
   posts a ``TAMPER_BLOCKED`` alert to the harness daemon, prints a crimson error to stderr and
   exits 2.
3. appends one compact record to ``<state_dir>/epoch_events.jsonl`` and increments
   ``turn_count`` in ``<state_dir>/turn_state.json``. Output of more than 80 lines is written in
   full to ``<worktree>/.harness/logs/turn_<n>.log``; the record keeps the first and last 40 lines.

Register it for ``PostToolUse`` AND ``PostToolUseFailure``: a Bash command that exits non-zero
fires only the latter, so a mutating command could otherwise skip verification by failing.
Recommended wiring (settings.json lives out of tree)::

    {"matcher": "*", "hooks": [{"type": "command", "command": "python",
      "args": ["-I", "<repo>/hooks/post_tool_compactor.py"], "timeout": 60, "onFailure": "block"}]}

Fail closed: every unexpected exception exits 2, and a watchdog exits 2 after
``HARNESS_HOOK_DEADLINE_MS`` (default 30 s).

Trust boundary: nothing inside the worktree is trusted. The manifest and all state live in
``HARNESS_STATE_DIR`` or ``<git-common-dir>/harness_state/<HARNESS_RUN_ID>/``; in a linked
worktree ``.git`` is a text file, so the common directory is asked from Git and never derived
from ``<worktree>/.git``.

``readonly_manifest.json``::

    {
      "base_commit_sha": "<40 or 64 hex digits>",
      "files": {
        "tests/test_ledger.py": "<sha256>",
        "pyproject.toml": {"sha256": "<sha256>"},
        "C:/harness/run-1/settings.json": {"sha256": "<sha256>", "restore_from": "pristine/settings.json"}
      }
    }

``files`` may also be a list of ``{"path": ..., "sha256": ...}`` objects. Relative keys are
repo-relative and restored from Git. Absolute keys are out-of-tree files (the Claude Code
``settings.json``); they are restored from ``restore_from`` (a pristine copy below the state
directory, verified against the digest) when it is given and are reported as unrestorable
otherwise.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and not sys.flags.safe_path and sys.path:
    # The hook's directory is visible to the worker: a planted hooks/json.py must not be able to
    # shadow the standard library. (Only when run as a script; -I / -P already take care of it.)
    del sys.path[0]

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Final, NamedTuple  # noqa: E402

HOOK: Final = "post_tool_compactor"

DAEMON_HOST: Final = "127.0.0.1"
DAEMON_PORT: Final = 8000
TAMPER_ENDPOINT: Final = "/api/internal/tamper-alert"
TOKEN_HEADER: Final = "X-Harness-Hook-Token"
DAEMON_TIMEOUT_S: Final = 1.0

MANIFEST_FILENAME: Final = "readonly_manifest.json"
EVENTS_FILENAME: Final = "epoch_events.jsonl"
TURN_STATE_FILENAME: Final = "turn_state.json"
LOCK_FILENAME: Final = ".hook_state.lock"
LOG_DIR_PARTS: Final = (".harness", "logs")

MAX_OUTPUT_LINES: Final = 80
HEAD_LINES: Final = 40
TAIL_LINES: Final = 40
MAX_LINE_CHARS: Final = 500
MAX_RECORD_TEXT_CHARS: Final = 12_000
MAX_PAYLOAD_BYTES: Final = 256 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
VERIFY_BUDGET_MS: Final = 2.0
LOCK_TIMEOUT_S: Final = 5.0
GIT_TIMEOUT_S: Final = 15
DEFAULT_DEADLINE_MS: Final = 30_000

CRIMSON: Final = "\x1b[1;38;2;220;20;60m"
RESET: Final = "\x1b[0m"

_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class HookError(Exception):
    """The hook cannot do its job (missing state, malformed payload). Fails closed."""


def _log(message: str) -> None:
    sys.stderr.write(f"[{HOOK}] {message}\n")
    sys.stderr.flush()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------------------
# Fail-closed plumbing
# --------------------------------------------------------------------------------------


def _arm_watchdog() -> None:
    """Exit 2 when the hook overruns its deadline (Claude Code's own timeout does not block)."""
    try:
        deadline_ms = int(os.environ.get("HARNESS_HOOK_DEADLINE_MS", DEFAULT_DEADLINE_MS))
    except ValueError:
        deadline_ms = DEFAULT_DEADLINE_MS
    deadline_ms = min(max(deadline_ms, 500), 300_000)

    def _expire() -> None:
        try:
            _log(f"deadline of {deadline_ms} ms exceeded; failing closed")
        finally:
            os._exit(2)

    timer = threading.Timer(deadline_ms / 1000, _expire)
    timer.daemon = True
    timer.start()


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_PAYLOAD_BYTES + 1)
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise HookError("hook payload is larger than the accepted maximum")
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise HookError("hook payload is not a JSON object")
    return payload


def _run_git(cwd: Path, *args: str) -> tuple[int, str, str]:
    """Run git without hooks, with literal pathspecs and without inherited ``GIT_*`` variables."""
    import subprocess

    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(  # fixed argv, no shell
        ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false", "--literal-pathspecs",
         "-C", str(cwd), *args],  # fmt: skip
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT_S,
        env=env,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git(cwd: Path, *args: str) -> str:
    code, out, err = _run_git(cwd, *args)
    if code != 0:
        raise HookError(f"git {' '.join(args)} failed: {_clip(err, 300)}")
    return out


# --------------------------------------------------------------------------------------
# Out-of-tree state
# --------------------------------------------------------------------------------------


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _worktree_root(payload: dict[str, Any]) -> Path:
    raw = os.environ.get("HARNESS_WORKTREE_ROOT") or os.environ.get("CLAUDE_PROJECT_DIR")
    if raw:
        candidate = Path(raw)
    else:
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            raise HookError("no HARNESS_WORKTREE_ROOT / CLAUDE_PROJECT_DIR and no cwd in the payload")
        candidate = Path(_git(Path(cwd), "rev-parse", "--show-toplevel"))
    if not candidate.is_absolute():
        raise HookError(f"worktree root {str(candidate)!r} is not an absolute path")
    root = candidate.resolve(strict=True)
    # In a linked worktree .git is a FILE: only its presence is checked, it is never read.
    if not root.is_dir() or not os.path.lexists(root / ".git"):
        raise HookError(f"{root} is not the root of a Git worktree")
    return root


def _state_dir(root: Path) -> tuple[Path, str]:
    run_id = os.environ.get("HARNESS_RUN_ID", "").strip()
    raw = os.environ.get("HARNESS_STATE_DIR", "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise HookError("HARNESS_STATE_DIR must be an absolute path")
    else:
        if not run_id:
            raise HookError("neither HARNESS_STATE_DIR nor HARNESS_RUN_ID is set")
        if not _RUN_ID_RE.match(run_id):
            raise HookError("HARNESS_RUN_ID is not a valid run id")
        common = Path(_git(root, "rev-parse", "--git-common-dir"))
        candidate = (common if common.is_absolute() else root / common) / "harness_state" / run_id
    try:
        state = candidate.resolve(strict=True)
    except OSError as exc:
        raise HookError(f"harness state directory {candidate} is not available: {exc}") from exc
    if not state.is_dir():
        raise HookError(f"harness state path {state} is not a directory")
    if state.is_relative_to(root):
        parts = state.relative_to(root).parts
        git_entry = root / ".git"
        if not parts or parts[0].lower() != ".git" or _is_link_or_junction(git_entry) or not git_entry.is_dir():
            raise HookError(f"harness state directory {state} is inside the worktree; refusing to trust it")
    run_id = run_id or state.name
    if not _RUN_ID_RE.match(run_id):
        raise HookError("the run id derived from the state directory is not valid")
    return state, run_id


# --------------------------------------------------------------------------------------
# Read-only manifest
# --------------------------------------------------------------------------------------


class Entry(NamedTuple):
    key: str  # the manifest key, as reported
    path: Path
    rel: str | None  # repo-relative POSIX path, None for an out-of-tree file
    sha256: str
    restore_from: str | None


def _load_manifest(state_dir: Path, root: Path) -> tuple[str, list[Entry]]:
    path = state_dir / MANIFEST_FILENAME
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise HookError(f"cannot read {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST_BYTES:
        raise HookError(f"{path} is not a regular file of acceptable size")
    manifest = json.loads(path.read_bytes().decode("utf-8-sig"))
    if not isinstance(manifest, dict):
        raise HookError(f"{MANIFEST_FILENAME} does not hold a JSON object")
    base = manifest.get("base_commit_sha", manifest.get("base_commit"))
    if not isinstance(base, str) or not _COMMIT_RE.match(base.lower()):
        raise HookError(f"{MANIFEST_FILENAME}: base_commit_sha must be a full commit id")

    items: list[tuple[Any, Any]] = []
    for field in ("files", "external_files"):
        value = manifest.get(field)
        if value is None and field == "external_files":
            continue
        if isinstance(value, dict):
            items.extend(value.items())
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            items.extend((item.get("path"), item) for item in value)
        else:
            raise HookError(f"{MANIFEST_FILENAME}: {field} must be an object or a list of objects")
    if not items:
        raise HookError(f"{MANIFEST_FILENAME} lists no files")

    entries: list[Entry] = []
    for key, value in items:
        digest = value if isinstance(value, str) else value.get("sha256") if isinstance(value, dict) else None
        restore_from = value.get("restore_from") if isinstance(value, dict) else None
        if not isinstance(key, str) or not key.strip() or "\x00" in key:
            raise HookError(f"{MANIFEST_FILENAME}: an entry has no usable path")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest.lower()):
            raise HookError(f"{MANIFEST_FILENAME}: {key!r} has no valid sha256")
        if restore_from is not None and not isinstance(restore_from, str):
            raise HookError(f"{MANIFEST_FILENAME}: {key!r} has an invalid restore_from")
        if Path(key).is_absolute() or key.startswith(("/", "\\")):
            entries.append(Entry(key, Path(key), None, digest.lower(), restore_from))
            continue
        parts = key.replace("\\", "/").split("/")
        if ":" in key or any(part in ("", ".", "..") for part in parts):
            raise HookError(f"{MANIFEST_FILENAME}: {key!r} is not a clean repo-relative path")
        entries.append(Entry(key, root.joinpath(*parts), "/".join(parts), digest.lower(), restore_from))
    return base.lower(), entries


def _sha256(path: Path) -> str:
    with open(path, "rb", buffering=0) as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _inspect(entry: Entry, root: Path, parent_ok_cache: dict[Path, bool]) -> tuple[str, str | None] | None:
    """``(kind, actual digest)`` when the file is not what the manifest says, else None."""
    if entry.rel is not None:
        parent = entry.path.parent
        ok = parent_ok_cache.get(parent)
        if ok is None:
            cur = parent
            root_resolved = root.resolve()
            ok = True
            while cur != root and len(cur.parts) > len(root.parts):
                if _is_link_or_junction(cur):
                    ok = False
                    break
                cur = cur.parent
            if ok:
                try:
                    ok = parent.resolve().is_relative_to(root_resolved)
                except OSError:
                    ok = False
            parent_ok_cache[parent] = ok
        if not ok:
            return "replaced_by_symlink", None

    if _is_link_or_junction(entry.path):
        return "replaced_by_symlink", None

    try:
        info = os.lstat(entry.path)
    except (FileNotFoundError, NotADirectoryError):
        return "missing", None
    if stat.S_ISLNK(info.st_mode):
        return "replaced_by_symlink", None
    if not stat.S_ISREG(info.st_mode):
        return "not_a_regular_file", None
    try:
        actual = _sha256(entry.path)
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})", None
    return ("hash_mismatch", actual) if actual != entry.sha256 else None


def _make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) & ~0o222)  # chmod a-w


def _clear_obstacle(path: Path) -> str | None:
    """Make room for the restored file. Returns a reason when that is not safely possible."""
    if path.is_symlink():
        os.unlink(path)  # removes the link, never its target
        return None
    if getattr(path, "is_junction", lambda: False)():
        os.rmdir(path)
        return None
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if stat.S_ISDIR(info.st_mode):
        return "a directory took the place of the file; it is not removed automatically"
    if stat.S_ISREG(info.st_mode):
        os.chmod(path, stat.S_IMODE(info.st_mode) | 0o200)  # git must be able to replace it
    else:
        os.unlink(path)
    return None


def _clear_git_index_lock(root: Path) -> None:
    code_gd, git_dir_raw, _ = _run_git(root, "rev-parse", "--git-dir")
    if code_gd == 0 and git_dir_raw:
        git_dir = Path(git_dir_raw) if Path(git_dir_raw).is_absolute() else (root / git_dir_raw)
        lock_file = git_dir / "index.lock"
        if lock_file.exists() and not lock_file.is_dir():
            try:
                lock_file.unlink()
            except OSError:
                pass


def _restore_from_git(root: Path, base: str, entry: Entry) -> str | None:
    """Restore one tracked file from the base commit. Returns None on success, else the reason."""
    assert entry.rel is not None
    cur = root
    for part in entry.rel.split("/")[:-1]:
        cur = cur / part
        if cur.is_symlink():
            try:
                os.unlink(cur)
            except OSError:
                pass
        elif getattr(cur, "is_junction", lambda: False)():
            try:
                os.rmdir(cur)
            except OSError:
                pass
    try:
        if not entry.path.parent.resolve().is_relative_to(root.resolve()):
            return "the parent directory resolves outside the worktree (symlink); not restored"
    except OSError:
        return "the parent directory resolves outside the worktree (symlink); not restored"

    obstacle = _clear_obstacle(entry.path)
    if obstacle is not None:
        return obstacle

    reset_code, _, reset_err = _run_git(root, "reset", "-q", "HEAD", "--", entry.rel)
    if reset_code != 0 and "index.lock" in reset_err:
        _clear_git_index_lock(root)
        reset_code, _, reset_err = _run_git(root, "reset", "-q", "HEAD", "--", entry.rel)

    code, _, err = _run_git(root, "-c", "core.autocrlf=false", "checkout", "-f", base, "--", entry.rel)
    if code != 0 and "index.lock" in err:
        _clear_git_index_lock(root)
        code, _, err = _run_git(root, "-c", "core.autocrlf=false", "checkout", "-f", base, "--", entry.rel)

    if code != 0:
        return f"git checkout {base[:12]} failed: {_clip(err, 300)}"

    try:
        if _sha256(entry.path) != entry.sha256:
            _clear_obstacle(entry.path)
            code_crlf, _, err_crlf = _run_git(root, "-c", "core.autocrlf=true", "checkout", "-f", base, "--", entry.rel)
            if code_crlf != 0 and "index.lock" in err_crlf:
                _clear_git_index_lock(root)
                _run_git(root, "-c", "core.autocrlf=true", "checkout", "-f", base, "--", entry.rel)
    except OSError:
        pass

    try:
        _make_readonly(entry.path)
    except OSError as exc:
        return f"failed to make {entry.rel} read-only: {exc}"

    return None


def _restore_from_copy(state_dir: Path, entry: Entry) -> str | None:
    """Restore an out-of-tree file from its pristine copy below the state directory."""
    if not entry.restore_from:
        return "out-of-tree file without restore_from; it must be restored by the harness"
    source = Path(entry.restore_from)
    source = (source if source.is_absolute() else state_dir / source).resolve()
    if not source.is_relative_to(state_dir) or not source.is_file():
        return "restore_from does not name a file below the state directory"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        return "the pristine copy does not match the manifest digest"
    obstacle = _clear_obstacle(entry.path)
    if obstacle is not None:
        return obstacle
    entry.path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(entry.path, data)
    _make_readonly(entry.path)
    return None


def _verify_and_restore(root: Path, state_dir: Path, base: str, entries: list[Entry]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter_ns()
    parent_ok_cache: dict[Path, bool] = {}
    findings = [(entry, found) for entry in entries if (found := _inspect(entry, root, parent_ok_cache)) is not None]
    verify_ms = (time.perf_counter_ns() - started) / 1_000_000

    violations: list[dict[str, Any]] = []
    for entry, (kind, actual) in findings:
        try:
            failure = _restore_from_git(root, base, entry) if entry.rel is not None else _restore_from_copy(state_dir, entry)
            if failure is None and (after := _inspect(entry, root, {})) is not None:
                failure = f"the restored file still differs from the manifest ({after[0]})"
        except Exception as exc:  # one failed restore must not hide the other violations
            failure = f"{type(exc).__name__}: {exc}"
        violations.append(
            {
                "path": entry.key,
                "kind": kind,
                "expected_sha256": entry.sha256,
                "actual_sha256": actual,
                "restored": failure is None,
                "detail": failure,
            }
        )
    return violations, verify_ms


# --------------------------------------------------------------------------------------
# Daemon
# --------------------------------------------------------------------------------------


def _post_json(endpoint: str, body: dict[str, Any], timeout: float = DAEMON_TIMEOUT_S) -> int:
    """POST to the loopback daemon within ``timeout`` seconds overall. Proxies are never used."""
    token = os.environ.get("HARNESS_HOOK_TOKEN", "")
    if not token:
        raise HookError("HARNESS_HOOK_TOKEN is not set")
    import http.client

    deadline = time.monotonic() + timeout
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    connection = http.client.HTTPConnection(DAEMON_HOST, DAEMON_PORT, timeout=timeout)
    try:
        connection.request(
            "POST",
            endpoint,
            body=data,
            headers={"Content-Type": "application/json", TOKEN_HEADER: token, "Connection": "close"},
        )
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"no answer from the harness daemon within {int(timeout * 1000)} ms")
        if connection.sock is not None:
            connection.sock.settimeout(left)
        return connection.getresponse().status
    finally:
        connection.close()


# --------------------------------------------------------------------------------------
# Event log and turn counter
# --------------------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    """Write through a temporary file and ``os.replace`` (which never follows a symlink at ``path``)."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
        for attempt in range(5):  # Windows: a reader may hold the destination open for a moment
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


class _StateLock:
    """Cross-process lock: parallel tool calls run their PostToolUse hooks concurrently."""

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / LOCK_FILENAME
        self._handle: Any = None

    def _try_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __enter__(self) -> _StateLock:
        self._handle = open(self._path, "a+b")
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                self._try_lock()
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    raise HookError(f"could not lock the harness state within {LOCK_TIMEOUT_S:.0f} s") from None
                time.sleep(0.01)

    def __exit__(self, *exc_info: object) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


def _shrink(value: Any, depth: int = 0) -> Any:
    """A small, JSON-safe view of an arbitrary tool input / response."""
    if isinstance(value, str):
        return _clip(value, 200)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(key): _shrink(item, depth + 1) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_shrink(item, depth + 1) for item in value[:5]]
    return f"<{type(value).__name__}>"


def _input_summary(tool_name: str, tool_input: Any) -> Any:
    if not isinstance(tool_input, dict):
        return _shrink(tool_input)
    if tool_name == "Bash":
        return {"command": _clip(str(tool_input.get("command", "")), 2000)}
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if isinstance(path, str):
        summary: dict[str, Any] = {"file_path": path}
        for key in ("content", "old_string", "new_string", "new_source"):
            if isinstance(tool_input.get(key), str):
                summary[f"{key}_chars"] = len(tool_input[key])
        if isinstance(tool_input.get("edits"), list):
            summary["edits"] = len(tool_input["edits"])
        return summary
    return _shrink(tool_input)


def _outputs(payload: dict[str, Any]) -> tuple[str, str, Any]:
    """``(stdout, stderr, compact response)`` of the finished tool call."""
    error = payload.get("error")
    if isinstance(error, str) and error:  # PostToolUseFailure
        return error, "", None
    response = payload.get("tool_response")
    if isinstance(response, str):
        return response, "", None
    if isinstance(response, dict) and isinstance(response.get("stdout"), str):
        stderr = response.get("stderr")
        extra = {key: _shrink(item) for key, item in response.items() if key not in ("stdout", "stderr")}
        return response["stdout"], stderr if isinstance(stderr, str) else "", extra or None
    return "", "", _shrink(response)


def _compact(text: str, log_rel: str) -> tuple[str, int, bool]:
    lines = text.splitlines()
    truncated = len(lines) > MAX_OUTPUT_LINES
    if truncated:
        omitted = len(lines) - HEAD_LINES - TAIL_LINES
        lines = [*lines[:HEAD_LINES], f"... [{omitted} lines omitted; full output in {log_rel}] ...", *lines[-TAIL_LINES:]]
    compact = "\n".join(_clip(line, MAX_LINE_CHARS) for line in lines)
    return _clip(compact, MAX_RECORD_TEXT_CHARS), len(text.splitlines()), truncated


def _write_turn_log(root: Path, turn: int, text: str) -> None:
    directory = root
    for part in LOG_DIR_PARTS:
        directory = directory / part
        if _is_link_or_junction(directory) or (os.path.lexists(directory) and not directory.is_dir()):
            raise HookError(f"{directory} is not a plain directory; refusing to write the turn log through it")
        directory.mkdir(exist_ok=True)
    _atomic_write(directory / f"turn_{turn}.log", text.encode("utf-8", "replace"))


def _record_event(
    payload: dict[str, Any],
    root: Path,
    state_dir: Path,
    run_id: str,
    integrity: dict[str, Any],
    alert: str | None,
) -> int:
    tool_name = str(payload.get("tool_name") or "")
    event = str(payload.get("hook_event_name") or "PostToolUse")
    stdout, stderr, response = _outputs(payload)
    with _StateLock(state_dir):
        state_path = state_dir / TURN_STATE_FILENAME
        try:
            state = json.loads(state_path.read_bytes().decode("utf-8-sig"))
        except FileNotFoundError:
            state = {}
        if not isinstance(state, dict):
            raise HookError(f"{TURN_STATE_FILENAME} does not hold a JSON object")
        previous = state.get("turn_count", 0)
        if isinstance(previous, bool) or not isinstance(previous, int) or previous < 0:
            raise HookError(f"{TURN_STATE_FILENAME}: turn_count is not a non-negative integer")
        turn = previous + 1

        log_rel = "/".join((*LOG_DIR_PARTS, f"turn_{turn}.log"))
        compact, line_count, truncated = _compact(stdout, log_rel)
        if truncated:
            _write_turn_log(root, turn, stdout)
        if integrity["violations"]:
            status = "tamper"
        else:
            status = "error" if event == "PostToolUseFailure" else "ok"
        record = {
            "ts": _now(),
            "turn": turn,
            "run_id": run_id,
            "session_id": payload.get("session_id"),
            "event": event,
            "tool_name": tool_name,
            "tool_use_id": payload.get("tool_use_id"),
            "status": status,
            "input": _input_summary(tool_name, payload.get("tool_input")),
            "stdout": compact,
            "stdout_lines": line_count,
            "stdout_truncated": truncated,
            "log_path": log_rel if truncated else None,
            "stderr": _clip(stderr, 2000),
            "response": response,
            "duration_ms": payload.get("duration_ms"),
            "integrity": integrity,
            "tamper_alert": alert,
        }
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8", "replace")
        with open(state_dir / EVENTS_FILENAME, "ab") as handle:
            handle.write(line)
            handle.flush()

        state.update(
            {"turn_count": turn, "run_id": run_id, "session_id": payload.get("session_id"),
             "last_tool": tool_name, "last_status": status, "updated_at": record["ts"]}
        )  # fmt: skip
        _atomic_write(state_path, json.dumps(state, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    return turn


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _send_alert(payload: dict[str, Any], run_id: str, base: str, violations: list[dict[str, Any]]) -> str:
    body = {
        "event": "TAMPER_BLOCKED",
        "run_id": run_id,
        "session_id": payload.get("session_id"),
        "hook_event": payload.get("hook_event_name"),
        "tool_name": payload.get("tool_name"),
        "tool_use_id": payload.get("tool_use_id"),
        "tool_input": _input_summary(str(payload.get("tool_name") or ""), payload.get("tool_input")),
        "base_commit_sha": base,
        "violations": violations,
        "detected_at": _now(),
    }
    try:
        status = _post_json(TAMPER_ENDPOINT, body)
    except Exception as exc:
        return f"not delivered ({type(exc).__name__}: {exc})"
    return "delivered" if 200 <= status < 300 else f"rejected (HTTP {status})"


def _tamper_message(payload: dict[str, Any], base: str, violations: list[dict[str, Any]], alert: str) -> str:
    lines = [
        f"[{HOOK}] TAMPER_BLOCKED: {len(violations)} read-only file(s) were changed during "
        f"{payload.get('tool_name') or 'the last tool call'}."
    ]
    for violation in violations:
        outcome = (
            f"restored from {base[:12]} and write-protected"
            if violation["restored"]
            else f"NOT RESTORED: {violation['detail']}"
        )
        lines.append(f"  - {violation['path']}: {violation['kind']} -> {outcome}")
    lines.append(f"The harness was alerted ({alert}). Read-only files are verified after every tool call.")
    lines.append("Do not modify them by any means; if the task needs one, request a contract amendment.")
    return CRIMSON + "\n".join(lines) + RESET + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
    _arm_watchdog()
    payload = _read_payload()
    event = payload.get("hook_event_name")
    if event is not None and event not in ("PostToolUse", "PostToolUseFailure"):
        raise HookError(f"this hook handles PostToolUse / PostToolUseFailure, not {event!r}")

    root = _worktree_root(payload)
    state_dir, run_id = _state_dir(root)
    base, entries = _load_manifest(state_dir, root)

    violations, verify_ms = _verify_and_restore(root, state_dir, base, entries)
    alert = _send_alert(payload, run_id, base, violations) if violations else None
    integrity = {
        "files": len(entries),
        "verify_ms": round(verify_ms, 3),
        "over_budget": verify_ms > VERIFY_BUDGET_MS,
        "violations": violations,
    }

    log_failure: Exception | None = None
    try:
        _record_event(payload, root, state_dir, run_id, integrity, alert)
    except Exception as exc:  # the tamper report below must not be lost to a logging problem
        log_failure = exc

    if violations:
        if log_failure is not None:
            _log(f"the event could not be recorded: {type(log_failure).__name__}: {log_failure}")
        sys.stderr.write(_tamper_message(payload, base, violations, alert or "no alert"))
        sys.stderr.flush()
        return 2
    if log_failure is not None:
        raise log_failure
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception as exc:  # fail closed
        _log(f"internal error, failing closed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    except BaseException as exc:  # KeyboardInterrupt and friends must not turn into success
        if isinstance(exc, SystemExit) and exc.code in (0, 2):
            raise
        _log(f"interrupted, failing closed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    sys.exit(_exit_code)
