#!/usr/bin/env python3
"""Claude Code ``PreToolUse`` hook: the write fence around a contract-bound coding worker.

Protocol
--------
* stdin: the Claude Code hook payload (``tool_name``, ``tool_input``, ``session_id``, ``cwd``...).
* exit 0: the tool call may proceed.
* exit 2: the tool call is blocked; the reason is on stderr and is shown to the worker.
* Contract amendment: a well-formed ``Write`` / ``Edit`` of ``.harness/amendment_request.json``
  prints ``{"continue": false, "stopReason": "CONTRACT_AMENDMENT_REQUEST"}`` and exits 0, which
  halts the parent ``claude -p`` epoch. ``continue: false`` takes precedence over the pending tool
  call, so the worktree file may never be written: the validated request is therefore persisted
  to ``<state_dir>/amendment_request.json`` and sent to the daemon by this hook itself.

Fail closed
-----------
Every unexpected exception exits 2. A watchdog thread exits 2 when the hook runs longer than
``HARNESS_HOOK_DEADLINE_MS`` (default 10 s). Claude Code does NOT block a tool call when it kills
a hook on its own timeout, so register the hook with a larger ``timeout`` and, where available,
``"onFailure": "block"``. Recommended wiring (settings.json lives out of tree)::

    {"matcher": "*", "hooks": [{"type": "command", "command": "python",
      "args": ["-I", "<repo>/hooks/pre_tool_guard.py"], "timeout": 30, "onFailure": "block"}]}

``-I`` keeps the hook's own directory and ``PYTHON*`` variables out of the import path; without
it the script removes its directory from ``sys.path`` itself.

Trust boundary
--------------
Nothing inside the worktree is trusted. ``task_contract.json`` is read from ``HARNESS_STATE_DIR``
or, when that is unset, from ``<git-common-dir>/harness_state/<HARNESS_RUN_ID>/``. In a linked
worktree ``.git`` is a text file, so the common directory is asked from Git
(``git rev-parse --git-common-dir``) and never derived from ``<worktree>/.git``. A state directory
inside the worktree is refused unless it lies below a real ``.git`` directory.

Environment: ``HARNESS_STATE_DIR``, ``HARNESS_RUN_ID``, ``HARNESS_HOOK_TOKEN``,
``HARNESS_WORKTREE_ROOT`` (default ``CLAUDE_PROJECT_DIR``, then ``git rev-parse --show-toplevel``),
``HARNESS_EXTRA_ALLOWED_TOOLS`` (comma separated tool names to wave through),
``HARNESS_HOOK_DEADLINE_MS``.

Tier 1 (deterministic, no I/O besides the contract and ``lstat``)
-----------------------------------------------------------------
* ``Edit`` / ``Write`` / ``MultiEdit`` / ``NotebookEdit``: the target is resolved with
  ``Path.resolve()`` and must stay inside the worktree. Blocked: ``.git`` / ``.claude`` at any
  depth, root ``hooks/`` and ``.harness/``, anything matched by ``readonly_files``, existing files
  that are not in ``editable_files``, new files that match no ``allowed_new_file_globs`` entry,
  hard-linked files and names that Windows would alias (trailing dots, 8.3 names, ``:`` streams).
* ``Bash``: ``shlex`` tokenization plus regular expressions. Every word of the command - also the
  words inside quoted arguments, at any nesting depth - is reduced to its basename and compared
  with the block lists, so ``/usr/bin/rm``, ``env rm``, ``nice rm`` and ``watch "rm x"`` are all
  the same thing. Command substitution, process substitution, here-documents, ``$'..'`` quoting,
  non-literal command names and shell-expanded arguments in front of an interpreter's script
  (``F=-c; python $F ...``) are refused outright. Git is allowlisted to read-only subcommands.
  Redirections may only target ``editable_files`` or ``.harness/logs/``.
* Other tools: a fixed set of read-only tools passes; everything else (``PowerShell``,
  ``WebFetch``, MCP tools...) is blocked unless named in ``HARNESS_EXTRA_ALLOWED_TOOLS``.

The Bash rules are a block list and therefore not complete: a worker that may run its own code
(``pytest``, ``python script.py``, package managers) can make that code do anything, including
reading ``HARNESS_HOOK_TOKEN`` from the environment it shares with this hook.
``post_tool_compactor.py`` verifies the read-only manifest after every tool call and is the
backstop for that.

Tier 2 (warm loopback daemon, 1,000 ms)
---------------------------------------
A call that passed tier 1 and touches control flow, assertions, SQL / database calls or security
sinks is sent to ``http://127.0.0.1:8000/api/internal/hook-eval`` for semantic drift evaluation on
``gemini-4-hydrogen`` at ``effort="low"`` (``thinking_level="LOW"``). ``{"allowed": false}`` blocks
with the daemon's reason; an unreachable daemon, a timeout or a malformed answer block as well.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and not sys.flags.safe_path and sys.path:
    # The hook's directory is visible to the worker: a planted hooks/json.py must not be able to
    # shadow the standard library. (Only when run as a script; -I / -P already take care of it.)
    del sys.path[0]

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import stat  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path, PurePosixPath  # noqa: E402
from typing import Any, Final, NamedTuple  # noqa: E402

HOOK: Final = "pre_tool_guard"

DAEMON_HOST: Final = "127.0.0.1"
DAEMON_PORT: Final = 8000
HOOK_EVAL_ENDPOINT: Final = "/api/internal/hook-eval"
TIER2_MODEL: Final = "gemini-4-hydrogen"
TIER2_EFFORT: Final = "low"
AMENDMENT_ENDPOINT: Final = "/api/internal/amendment-request"
TOKEN_HEADER: Final = "X-Harness-Hook-Token"
DAEMON_TIMEOUT_S: Final = 1.0
MAX_DAEMON_RESPONSE_BYTES: Final = 256 * 1024

TASK_CONTRACT_FILENAME: Final = "task_contract.json"
AMENDMENT_STATE_FILENAME: Final = "amendment_request.json"
AMENDMENT_REL: Final = ".harness/amendment_request.json"
AMENDMENT_STOP_REASON: Final = "CONTRACT_AMENDMENT_REQUEST"
LOG_DIR_REL: Final = ".harness/logs"

SNIPPET_LIMIT: Final = 3500
MAX_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
MAX_CONTRACT_BYTES: Final = 16 * 1024 * 1024
MAX_AMENDMENT_BYTES: Final = 64 * 1024
MAX_COMMAND_CHARS: Final = 20_000
MAX_NESTED_QUOTING: Final = 4
MAX_DIFF_BYTES: Final = 200 * 1024
MAX_PATH_CHARS: Final = 512
MAX_PATH_SEGMENTS: Final = 64
DEFAULT_DEADLINE_MS: Final = 10_000

WRITE_TOOLS: Final = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
AMENDMENT_TOOLS: Final = frozenset({"Write", "Edit"})
# Tools that cannot change the worktree or reach the network.
PASSTHROUGH_TOOLS: Final = frozenset(
    {
        "Read", "Glob", "Grep", "LS", "NotebookRead", "TodoWrite", "TodoRead", "Task", "Agent",
        "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskOutput", "TaskStop", "BashOutput",
        "KillShell", "KillBash", "EnterPlanMode", "ExitPlanMode", "AskUserQuestion", "Skill",
        "SlashCommand", "ToolSearch",
    }
)  # fmt: skip

AMEND_HINT: Final = (
    f" If the task really needs this file, write a JSON request to {AMENDMENT_REL} "
    '({"files": ["<path>"], "reason": "..."}) instead of working around the contract.'
)


class Block(Exception):
    """A deliberate denial: the message goes to stderr and the hook exits 2."""

    def __init__(self, tier: str, message: str) -> None:
        super().__init__(message)
        self.tier = tier
        self.message = message


class HookError(Exception):
    """The hook cannot do its job (missing state, malformed payload). Fails closed."""


def _log(message: str) -> None:
    sys.stderr.write(f"[{HOOK}] {message}\n")
    sys.stderr.flush()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


# --------------------------------------------------------------------------------------
# Fail-closed plumbing
# --------------------------------------------------------------------------------------


def _arm_watchdog() -> None:
    """Exit 2 when the hook overruns its deadline (Claude Code's own timeout does not block)."""
    try:
        deadline_ms = int(os.environ.get("HARNESS_HOOK_DEADLINE_MS", DEFAULT_DEADLINE_MS))
    except ValueError:
        deadline_ms = DEFAULT_DEADLINE_MS
    deadline_ms = min(max(deadline_ms, 500), 120_000)

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


def _git(cwd: Path, *args: str) -> str:
    import subprocess

    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(  # fixed argv, no shell
        ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false", "--literal-pathspecs", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        env=env,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    if result.returncode != 0:
        raise HookError(f"git {' '.join(args)} failed: {_clip(result.stderr.strip(), 300)}")
    return result.stdout.strip()


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


# --------------------------------------------------------------------------------------
# Out-of-tree state
# --------------------------------------------------------------------------------------

_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _worktree_root(payload: dict[str, Any]) -> tuple[Path, tuple[Path, ...]]:
    """The resolved worktree root plus the spellings it is known under (for lexical checks)."""
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
    return root, tuple(dict.fromkeys((root, Path(os.path.normpath(candidate)))))


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


def _load_contract(state_dir: Path) -> dict[str, Any]:
    path = state_dir / TASK_CONTRACT_FILENAME
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise HookError(f"cannot read {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONTRACT_BYTES:
        raise HookError(f"{path} is not a regular file of acceptable size")
    contract = json.loads(path.read_bytes().decode("utf-8-sig"))
    if not isinstance(contract, dict):
        raise HookError(f"{path} does not hold a JSON object")
    return contract


def _str_list(contract: dict[str, Any], key: str) -> list[str]:
    value = contract.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HookError(f"{TASK_CONTRACT_FILENAME}: {key} must be a list of strings")
    return value


class Ctx(NamedTuple):
    root: Path
    root_spellings: tuple[Path, ...]
    cwd: Path
    state_dir: Path
    run_id: str
    session_id: str
    tool_use_id: str
    editable: frozenset[str]
    readonly: list[str]
    new_globs: list[str]
    invariants: list[str]
    forbidden_patterns: list[str]


def _load_context(payload: dict[str, Any]) -> Ctx:
    root, spellings = _worktree_root(payload)
    state_dir, run_id = _state_dir(root)
    contract = _load_contract(state_dir)
    readonly = _str_list(contract, "readonly_files")
    if not readonly:  # a validated contract always carries the built-in protections
        raise HookError(f"{TASK_CONTRACT_FILENAME}: readonly_files is empty")
    cwd_raw = payload.get("cwd")
    if isinstance(cwd_raw, str) and cwd_raw:
        native = _native_path(cwd_raw)
        cwd_path = Path(native)
        if not cwd_path.is_absolute():
            raise Block("tier 1", f"payload cwd {cwd_raw!r} is not an absolute path")
        cwd_resolved = cwd_path.resolve()
        if not cwd_resolved.is_relative_to(root.resolve()):
            raise Block("tier 1", f"payload cwd {cwd_raw!r} resolves outside the worktree")
        cur = Path(os.path.normpath(native))
        while True:
            if cur in spellings or cur.resolve() == root.resolve():
                break
            if _is_link_or_junction(cur):
                raise Block("tier 1", f"payload cwd {cwd_raw!r} contains a symbolic link or junction")
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        cwd = cwd_resolved
    else:
        cwd = root
    return Ctx(
        root=root,
        root_spellings=spellings,
        cwd=cwd,
        state_dir=state_dir,
        run_id=run_id,
        session_id=str(payload.get("session_id") or ""),
        tool_use_id=str(payload.get("tool_use_id") or ""),
        editable=frozenset(_str_list(contract, "editable_files")),
        readonly=readonly,
        new_globs=_str_list(contract, "allowed_new_file_globs"),
        invariants=_str_list(contract, "invariants"),
        forbidden_patterns=_str_list(contract, "forbidden_patterns"),
    )


# --------------------------------------------------------------------------------------
# Glob matching (same semantics as backend.contract_compiler, standard library only)
# --------------------------------------------------------------------------------------

_Token = tuple[Any, ...]


def _fold(text: str) -> str:
    return text.lower() if text.isascii() else unicodedata.normalize("NFKC", text).casefold()


def _has_magic(text: str) -> bool:
    return "*" in text or "?" in text or "[" in text


def _compile_segment(segment: str, fold: bool) -> list[_Token]:
    tokens: list[_Token] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            text = "".join(run)
            tokens.extend(("lit", ch) for ch in (_fold(text) if fold else text))
            run.clear()

    index = 0
    while index < len(segment):
        ch = segment[index]
        if ch == "*":
            flush()
            if not tokens or tokens[-1][0] != "star":
                tokens.append(("star",))
            index += 1
        elif ch == "?":
            flush()
            tokens.append(("any",))
            index += 1
        elif ch == "[":
            flush()
            cursor = index + 1
            negate = cursor < len(segment) and segment[cursor] in "!^"
            if negate:
                cursor += 1
            body_start = cursor
            if cursor < len(segment) and segment[cursor] == "]":
                cursor += 1  # a leading ']' is a literal member
            end = segment.find("]", cursor)
            if end == -1:
                raise ValueError("unterminated character class")
            body = segment[body_start:end]
            chars: set[str] = set()
            ranges: list[tuple[str, str]] = []
            position = 0
            while position < len(body):
                if position + 2 < len(body) and body[position + 1] == "-":
                    ranges.append((body[position], body[position + 2]))
                    position += 3
                else:
                    chars.add(body[position])
                    position += 1
            tokens.append(("class", negate, frozenset(chars), tuple(ranges), fold))
            index = end + 1
        else:
            run.append(ch)
            index += 1
    flush()
    return tokens


def _accepts(token: _Token, ch: str) -> bool:
    kind = token[0]
    if kind == "lit":
        return bool(ch == token[1])
    if kind == "any":
        return True
    if kind == "class":
        _, negate, chars, ranges, widen = token
        # Protection matching ignores case in a way that can only widen what is protected.
        candidates = {ch, ch.lower(), ch.upper()} if widen else {ch}
        return any(
            ((variant in chars) or any(low <= variant <= high for low, high in ranges)) != negate
            for variant in candidates
            if len(variant) == 1
        )
    return False


def _match_segment(tokens: list[_Token], text: str) -> bool:
    """Wildcard match with single-star backtracking: O(len(tokens) * len(text))."""
    token_index = text_index = 0
    star_token = -1
    star_text = 0
    while text_index < len(text):
        if token_index < len(tokens) and tokens[token_index][0] == "star":
            star_token = token_index
            star_text = text_index
            token_index += 1
        elif token_index < len(tokens) and _accepts(tokens[token_index], text[text_index]):
            token_index += 1
            text_index += 1
        elif star_token != -1:
            token_index = star_token + 1
            star_text += 1
            text_index = star_text
        else:
            return False
    while token_index < len(tokens) and tokens[token_index][0] == "star":
        token_index += 1
    return token_index == len(tokens)


def _match_glob(pattern: str, segments: list[str], *, fold: bool) -> bool:
    """``**`` spans zero or more directories; ``*`` ``?`` and classes stay inside one segment."""
    compiled: list[list[_Token] | None] = []
    for segment in pattern.replace("\\", "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == "**":
            if not compiled or compiled[-1] is not None:
                compiled.append(None)
        else:
            compiled.append(_compile_segment(segment, fold))
    if not compiled or not segments:
        return False
    count = len(segments)
    row = [False] * count + [True]  # row[j]: the remaining pattern matches segments[j:]
    for part in reversed(compiled):
        current = [False] * (count + 1)
        if part is None:
            current[count] = row[count]
            for j in range(count - 1, -1, -1):
                current[j] = row[j] or current[j + 1]
        else:
            for j in range(count - 1, -1, -1):
                if row[j + 1] and _match_segment(part, segments[j]):
                    current[j] = True
        row = current
    return row[0]


def _readonly_match(rel: str, entries: list[str]) -> str | None:
    """The first ``readonly_files`` entry that protects ``rel`` (malformed entries protect everything)."""
    segments = rel.split("/")
    folded = [_fold(segment) for segment in segments]
    for entry in entries:
        try:
            parts = [part for part in entry.replace("\\", "/").split("/") if part not in ("", ".")]
            if not parts or len(entry) > MAX_PATH_CHARS:
                return entry
            if not _has_magic(entry):  # a literal entry protects itself and everything below it
                literal = [_fold(part) for part in parts]
                if folded[: len(literal)] == literal:
                    return entry
            elif _match_glob(entry, segments, fold=False) or _match_glob(entry, folded, fold=True):
                return entry
        except ValueError:
            return entry
    return None


# --------------------------------------------------------------------------------------
# Path rules
# --------------------------------------------------------------------------------------

_REJECTED_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_SHORT_NAME_RE: Final = re.compile(r"^[^~./\s]{1,6}~[0-9]{1,6}(?:\.[^~./]{0,3})?$")
_DEVICE_NAMES: Final = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{i}" for i in range(10)}
    | {f"lpt{i}" for i in range(10)}
)
_MSYS_DRIVE_RE: Final = re.compile(r"^/([A-Za-z])(/.*)?$")


def _segment_key(segment: str) -> str:
    return unicodedata.normalize("NFKC", segment).casefold().rstrip(". ")


def _hard_protected(rel: str) -> str | None:
    """``.git`` / ``.claude`` at any depth, ``hooks`` / ``.harness`` as the first segment."""
    keys = [_segment_key(segment) for segment in rel.split("/")]
    for key in keys:
        if key in (".git", ".claude"):
            return key
    if keys and keys[0] in ("hooks", ".harness"):
        return keys[0]
    return None


def _path_rule_violation(rel: str) -> str | None:
    """Names that the filesystem would resolve to something else than what was checked."""
    if len(rel) > MAX_PATH_CHARS:
        return f"path is longer than {MAX_PATH_CHARS} characters"
    if ":" in rel:
        return "path contains ':' (drive letter or alternate data stream)"
    if any(unicodedata.category(ch) in _REJECTED_CATEGORIES for ch in rel):
        return "path contains a control or invisible format character"
    segments = rel.split("/")
    if len(segments) > MAX_PATH_SEGMENTS:
        return f"path has more than {MAX_PATH_SEGMENTS} segments"
    for segment in segments:
        compat = unicodedata.normalize("NFKC", segment)
        if segment in ("", ".", "..") or compat in (".", "..") or any(ch in compat for ch in "/\\:"):
            return "path has an empty, relative or look-alike segment"
        if segment != segment.strip() or segment.endswith("."):
            return "a path segment has surrounding whitespace or a trailing dot"
        if _SHORT_NAME_RE.match(segment):
            return "a path segment looks like a Windows 8.3 short-name alias"
        if _segment_key(segment).split(".", 1)[0] in _DEVICE_NAMES:
            return "a path segment is a reserved Windows device name"
    return None


def _native_path(raw: str) -> str:
    """Map Git-Bash drive paths (``/c/Users/x``) to native ones on Windows."""
    if os.name == "nt":
        match = _MSYS_DRIVE_RE.match(raw)
        if match:
            return f"{match.group(1)}:{match.group(2) or '/'}"
        if raw.startswith("/") and not raw.startswith("//"):
            raise Block("tier 1", f"{raw!r} is a POSIX-absolute path outside the worktree")
    return raw


class Target(NamedTuple):
    resolved: Path
    rel: str
    lexical_rel: str | None


def _resolve_target(raw: object, ctx: Ctx) -> Target:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise Block("tier 1", "the tool call names no usable file path")
    worktree_root = ctx.root.resolve()
    path = Path(_native_path(raw))
    if not path.is_absolute():
        path = ctx.cwd / path
    lexical = Path(os.path.normpath(path))
    resolved = path.resolve()  # follows every symlink: this is where the write would really land
    raw_parent_resolved = path.parent.resolve()
    resolved_parent = resolved.parent.resolve()
    if not resolved.is_relative_to(worktree_root):
        raise Block("tier 1", f"{raw!r} resolves to {resolved}, which is outside the worktree")
    if not raw_parent_resolved.is_relative_to(worktree_root) or not resolved_parent.is_relative_to(worktree_root):
        raise Block("tier 1", f"the parent directory of {raw!r} resolves outside the worktree")
    if _is_link_or_junction(lexical):
        raise Block("tier 1", f"{raw!r} is a symbolic link or directory junction")
    for anc in lexical.parents:
        try:
            anc_resolved = anc.resolve()
        except OSError as exc:
            raise Block("tier 1", f"a parent directory of {raw!r} cannot be resolved: {exc}") from exc
        if anc_resolved == worktree_root or anc in ctx.root_spellings:
            break
        if not anc_resolved.is_relative_to(worktree_root):
            raise Block("tier 1", f"a parent directory of {raw!r} resolves outside the worktree")
        if _is_link_or_junction(anc):
            raise Block("tier 1", f"a parent directory of {raw!r} is a symbolic link or junction")
    rel = resolved.relative_to(worktree_root).as_posix()
    if rel in ("", "."):
        raise Block("tier 1", "the target is the worktree root itself")
    lexical_rel: str | None = None
    for spelling in ctx.root_spellings:
        if lexical.is_relative_to(spelling):
            lexical_rel = lexical.relative_to(spelling).as_posix()
            break
    return Target(resolved, rel, lexical_rel)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except NotADirectoryError as exc:
        raise Block("tier 1", f"a parent of {path} is not a directory") from exc


def _check_write_target(ctx: Ctx, target: Target, *, what: str) -> bool:
    """Tier 1 for one file that is about to be written. Returns True when the file exists."""
    worktree_root = ctx.root.resolve()
    for candidate in dict.fromkeys(name for name in (target.rel, target.lexical_rel) if name):
        area = _hard_protected(candidate)
        if area is not None:
            raise Block("tier 1", f"{what} {candidate} touches the protected area {area}; it can never be written")
        problem = _path_rule_violation(candidate)
        if problem is not None:
            raise Block("tier 1", f"{what} {candidate} is refused: {problem}")
        entry = _readonly_match(candidate, ctx.readonly)
        if entry is not None:
            raise Block("tier 1", f"{what} {candidate} is read-only (readonly_files entry {entry!r})." + AMEND_HINT)
    for parent in target.resolved.parents:
        if parent == ctx.root or parent.resolve() == worktree_root:
            break
        if not parent.resolve().is_relative_to(worktree_root):
            raise Block("tier 1", f"a parent directory of {target.rel} resolves outside the worktree")
        if _is_link_or_junction(parent):
            raise Block("tier 1", f"a parent directory of {target.rel} is a symbolic link or junction")
        if os.path.lexists(parent) and not parent.is_dir():
            raise Block("tier 1", f"a parent of {target.rel} is not a directory")
    info = _lstat(target.resolved)
    if info is not None:
        if not stat.S_ISREG(info.st_mode):
            raise Block("tier 1", f"{what} {target.rel} is not a regular file")
        if info.st_nlink > 1:
            raise Block("tier 1", f"{what} {target.rel} has several hard links; writing it could change another file")
        if target.rel not in ctx.editable:
            raise Block("tier 1", f"{what} {target.rel} exists and is not in editable_files." + AMEND_HINT)
        return True
    if target.rel in ctx.editable:  # an editable file that was removed may be written again
        return False
    segments = target.rel.split("/")
    for pattern in ctx.new_globs:
        try:
            if _match_glob(pattern, segments, fold=False):
                return False
        except ValueError:
            continue
    raise Block(
        "tier 1",
        f"{what} {target.rel} does not exist and matches no allowed_new_file_globs pattern." + AMEND_HINT,
    )


# --------------------------------------------------------------------------------------
# Daemon
# --------------------------------------------------------------------------------------


def _post_json(endpoint: str, body: dict[str, Any], timeout: float = DAEMON_TIMEOUT_S) -> tuple[int, Any]:
    """POST to the loopback daemon within ``timeout`` seconds overall. Proxies are never used."""
    token = os.environ.get("HARNESS_HOOK_TOKEN", "")
    if not token:
        raise HookError("HARNESS_HOOK_TOKEN is not set")
    import http.client

    deadline = time.monotonic() + timeout

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"no answer from the harness daemon within {int(timeout * 1000)} ms")
        return left

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    connection = http.client.HTTPConnection(DAEMON_HOST, DAEMON_PORT, timeout=timeout)
    try:
        connection.request(
            "POST",
            endpoint,
            body=data,
            headers={"Content-Type": "application/json", TOKEN_HEADER: token, "Connection": "close"},
        )
        if connection.sock is not None:
            connection.sock.settimeout(remaining())
        response = connection.getresponse()
        if connection.sock is not None:
            connection.sock.settimeout(remaining())
        raw = response.read(MAX_DAEMON_RESPONSE_BYTES + 1)
        status = response.status
    finally:
        connection.close()
    try:
        parsed = json.loads(raw.decode("utf-8")) if len(raw) <= MAX_DAEMON_RESPONSE_BYTES else None
    except ValueError:
        parsed = None
    return status, parsed


# --------------------------------------------------------------------------------------
# Exit-and-resume contract amendment
# --------------------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _amendment_text(tool_name: str, tool_input: dict[str, Any], target: Target) -> str:
    if tool_name == "Write":
        content = tool_input.get("content")
        if not isinstance(content, str):
            raise Block("amendment", "the Write call carries no content")
        return content
    old, new = tool_input.get("old_string"), tool_input.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise Block("amendment", "the Edit call needs a non-empty old_string and a new_string")
    info = _lstat(target.resolved)
    if info is None:
        raise Block("amendment", f"{AMENDMENT_REL} does not exist yet; create it with Write")
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_AMENDMENT_BYTES:
        raise Block("amendment", f"{AMENDMENT_REL} is not a regular file of acceptable size")
    current = target.resolved.read_text(encoding="utf-8")
    occurrences = current.count(old)
    replace_all = tool_input.get("replace_all") is True
    if occurrences == 0 or (occurrences > 1 and not replace_all):
        raise Block("amendment", f"old_string occurs {occurrences} times in {AMENDMENT_REL}; the edit is not applicable")
    return current.replace(old, new) if replace_all else current.replace(old, new, 1)


def _validate_amendment(text: str) -> dict[str, Any]:
    if "\x00" in text:
        raise Block("amendment", "amendment request contains a NUL byte")
    shape = 'Expected a JSON object such as {"files": ["src/x.py"], "reason": "...", "evidence": "..."}.'
    if len(text.encode("utf-8", "surrogatepass")) > MAX_AMENDMENT_BYTES:
        raise Block("amendment", f"the amendment request is larger than {MAX_AMENDMENT_BYTES} bytes")
    try:
        request = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise Block("amendment", f"{AMENDMENT_REL} is not well-formed JSON ({exc}). {shape}") from exc
    if not isinstance(request, dict):
        raise Block("amendment", f"{AMENDMENT_REL} must hold a JSON object. {shape}")
    paths: list[str] = []
    for key in ("file", "path", "files", "requested_files"):
        value = request.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            paths.append(value)
        elif key in ("files", "requested_files") and isinstance(value, list) and all(isinstance(v, str) for v in value):
            paths.extend(value)
        else:
            raise Block("amendment", f"{key!r} must be a path string or a list of path strings. {shape}")
    if not any(path.strip() for path in paths):
        raise Block("amendment", f"the amendment request names no file. {shape}")
    return request


def _atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _handle_amendment(ctx: Ctx, tool_name: str, tool_input: dict[str, Any], target: Target) -> int:
    request = _validate_amendment(_amendment_text(tool_name, tool_input, target))
    envelope = {
        "run_id": ctx.run_id,
        "session_id": ctx.session_id,
        "tool_name": tool_name,
        "tool_use_id": ctx.tool_use_id,
        "path": AMENDMENT_REL,
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request": request,
    }
    delivered = False
    try:  # the trusted copy: the worktree file may never be written once the epoch halts
        payload = json.dumps(envelope, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        _atomic_write(ctx.state_dir / AMENDMENT_STATE_FILENAME, payload)
        delivered = True
    except Exception as exc:
        _log(f"could not persist the amendment request out of tree: {type(exc).__name__}: {exc}")
    try:
        status, _ = _post_json(AMENDMENT_ENDPOINT, envelope)
        if 200 <= status < 300:
            delivered = True
        else:
            _log(f"the harness daemon answered the amendment notification with HTTP {status}")
    except Exception as exc:  # the daemon may be busy halting this very epoch: never wait for it
        _log(f"amendment notification not acknowledged: {type(exc).__name__}: {exc}")
    if not delivered:
        raise Block("amendment", "the amendment request could not be handed to the harness; try again")
    directive = {
        "continue": False,
        "stopReason": AMENDMENT_STOP_REASON,
    }
    sys.stdout.write(json.dumps(directive) + "\n")
    sys.stdout.flush()
    return 0


# --------------------------------------------------------------------------------------
# Tier 1: Bash
# --------------------------------------------------------------------------------------

_PUNCTUATION: Final = "();<>|&"
_OPERATORS: Final = (
    "&>>", "<<<", ";;&", "&>", ">>", ">&", ">|", "<<", "<&", "<>", "&&", "||", "|&", ";;", ";&",
    ">", "<", "|", "&", ";", "(", ")",
)  # fmt: skip
_WRITE_REDIRECTS: Final = frozenset({">", ">>", ">|", "&>", "&>>", ">&", "<>"})
_DEVICE_SINKS: Final = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_FD_RE: Final = re.compile(r"^(?:\d+-?|-)$")
_LITERAL_PATH_RE: Final = re.compile(r"^[A-Za-z0-9_@%+=:,./\\ -]+$")
_LITERAL_COMMAND_RE: Final = re.compile(r"^[A-Za-z0-9_@%+=:,./\\~-]+$")
_SHELL_SYNTAX_WORDS: Final = frozenset({"[", "[[", "]]", "{", "}", "!"})
_ASSIGNMENT_RE: Final = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\+?=")
_NESTED_RE: Final = re.compile(r"[\s;|&()<>]")
_NUMERIC_RE: Final = re.compile(r"^(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?[smhd]?)$")
_SCRIPT_LIKE_RE: Final = re.compile(r"[/\\]|\.[A-Za-z0-9]{1,6}$")
# shlex drops the quotes, so the quote-aware pass marks what the shell would still expand: an
# active "$" (unquoted or inside double quotes) and an unquoted "{" that can start a brace
# expansion. A word that carries a marker is not what the program will receive.
_EXPANSION: Final = chr(0xE000)  # private-use code points
_BRACE: Final = chr(0xE001)
_EXPANSION_START: Final = frozenset("_{@*#?!$-")
_HIDDEN_OPTION_RE: Final = re.compile(rf"^[{_EXPANSION}{_BRACE}]|^-.*[{_EXPANSION}{_BRACE}]")

# Block list 1: forbidden wherever the word appears (after basename normalization).
BANNED_ANYWHERE: Final[dict[str, str]] = {
    **dict.fromkeys(
        ("chmod", "chown", "chgrp", "ln", "unlink", "mount", "umount", "mklink", "chattr", "setfacl", "icacls", "takeown"),
        "permission or link mutation",
    ),
    **dict.fromkeys(
        ("dd", "patch", "tee", "mv", "cp", "rm", "truncate", "shred", "rmdir", "rsync", "scp", "sftp"),
        "file mutation outside the Write/Edit tools",
    ),
    **dict.fromkeys(
        ("curl", "wget", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "ssh"),
        "network access",
    ),
    **dict.fromkeys(
        ("base64", "base32", "basenc", "xxd", "uudecode", "uuencode", "openssl"),
        "encoding or decoding tool that can hide payloads",
    ),
    **dict.fromkeys(("xargs", "sudo", "doas"), "command wrapper that hides what is executed"),
}  # fmt: skip

# Block list 2: forbidden as the command of a simple command (after wrappers are peeled off).
BANNED_COMMANDS: Final[dict[str, str]] = {
    **dict.fromkeys(
        ("eval", "alias", "unalias", "trap", "enable", "script", "expect", "parallel", "su"),
        "indirect command execution",
    ),
    **dict.fromkeys(
        ("touch", "install", "link", "mkfifo", "mknod", "sponge", "split", "csplit"),
        "file creation outside the Write tool",
    ),
    **dict.fromkeys(
        ("tar", "unzip", "zip", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "7z", "7za", "cpio"),
        "archive tool that writes files",
    ),
    **dict.fromkeys(("vi", "vim", "nvim", "nano", "emacs", "ed", "ex"), "interactive editor"),
    **dict.fromkeys(
        ("cmd", "powershell", "pwsh", "wsl", "start", "wscript", "cscript", "mshta", "rundll32", "regsvr32", "msiexec"),
        "foreign shell or script host",
    ),
    **dict.fromkeys(
        ("gh", "hub", "certutil", "bitsadmin", "aria2c", "lynx", "w3m", "http", "https", "xh", "httpie", "grpcurl", "websocat"),
        "network access",
    ),
    **dict.fromkeys(("crontab", "at", "batch", "schtasks"), "deferred execution"),
    **dict.fromkeys(("strace", "ltrace", "gdb", "lldb"), "process tracing"),
}  # fmt: skip

_KEYWORD_PREFIXES: Final = frozenset({"then", "do", "else", "elif", "if", "while", "until", "!", "{", "coproc"})
_WRAPPERS: Final[dict[str, frozenset[str]]] = {  # wrapper -> options that take a separate value
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "command": frozenset(),
    "builtin": frozenset(),
    "exec": frozenset({"-a"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "nohup": frozenset(),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
    "setsid": frozenset(),
    "ionice": frozenset({"-c", "-n", "-p"}),
    "chrt": frozenset(),
    "taskset": frozenset(),
    "winpty": frozenset(),
    "unbuffer": frozenset(),
    "busybox": frozenset(),
    "watch": frozenset({"-n", "--interval"}),
}
_SHELLS: Final = frozenset({"sh", "bash", "zsh", "dash", "ksh", "mksh", "ash", "fish", "csh", "tcsh"})
_INFO_FLAGS: Final = frozenset({"--version", "-V", "-VV", "--help", "-h", "-?"})
_STDIN_OPERANDS: Final = ("/dev/stdin", "/dev/fd/", "/proc/")

_DANGEROUS_ENV_RE: Final = re.compile(
    r"^(?:LD_[A-Z0-9_]+|DYLD_[A-Z0-9_]+|BASH_ENV|PROMPT_COMMAND|PYTHONSTARTUP|PYTHONINSPECT|NODE_OPTIONS"
    r"|PERL5OPT|PERL5DB|RUBYOPT|GIT_[A-Z0-9_]+|IFS|SHELLOPTS|BASHOPTS|PS4|HARNESS_[A-Z0-9_]+|CLAUDE_[A-Z0-9_]+)$"
)

# Git: the product list plus everything else that writes to the repository, the index or a remote.
GIT_MUTATING: Final = frozenset(
    {
        "commit", "merge", "push", "pull", "rebase", "reset", "checkout", "restore", "clean", "add",
        "worktree", "config", "branch", "tag",
        "stash", "apply", "am", "cherry-pick", "revert", "rm", "mv", "switch", "update-index",
        "update-ref", "symbolic-ref", "read-tree", "checkout-index", "filter-branch", "gc", "prune",
        "reflog", "notes", "submodule", "remote", "fetch", "clone", "init", "bisect", "replace",
        "hash-object", "commit-tree", "write-tree", "sparse-checkout", "lfs", "format-patch", "bundle",
        "archive", "difftool", "mergetool", "daemon", "send-email", "fast-import", "pack-refs",
        "repack", "maintenance", "rerere", "credential", "mktag", "mktree", "unpack-objects",
        "update-server-info", "request-pull", "svn", "p4", "gui", "citool", "instaweb",
    }
)  # fmt: skip
GIT_READONLY: Final = frozenset(
    {
        "status", "diff", "log", "show", "blame", "annotate", "grep", "ls-files", "ls-tree",
        "cat-file", "rev-parse", "rev-list", "describe", "shortlog", "show-ref", "for-each-ref",
        "merge-base", "name-rev", "diff-tree", "diff-files", "diff-index", "whatchanged",
        "count-objects", "check-ignore", "check-attr", "var", "version", "help",
    }
)  # fmt: skip
_GIT_VALUE_OPTIONS: Final = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env", "--attr-source"}
)
_GIT_PLAIN_GLOBALS: Final = frozenset(
    {
        "--no-pager", "-P", "-p", "--paginate", "--no-replace-objects", "--no-lazy-fetch",
        "--no-optional-locks", "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
        "--icase-pathspecs", "--version", "-v", "--help", "-h", "--exec-path", "--html-path",
        "--man-path", "--info-path",
    }
)  # fmt: skip
_GIT_BAD_ARGUMENT_RE: Final = re.compile(r"^(?:--output|--ext-diff|--open-files-in-pager|--upload-pack|--exec)")

_FIND_ACTIONS: Final = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}
)
_SEARCH_EXEC_FLAGS: Final = {
    "rg": frozenset({"--pre"}),
    "fd": frozenset({"-x", "--exec", "-X", "--exec-batch"}),
    "fdfind": frozenset({"-x", "--exec", "-X", "--exec-batch"}),
}

_PYTHON_MODULE_DENY: Final = frozenset(
    {
        "timeit", "code", "pdb", "bdb", "runpy", "trace", "cProfile", "profile", "http", "zipfile",
        "tarfile", "zipapp", "json", "base64", "compileall", "py_compile", "idlelib", "webbrowser",
        "ftplib", "smtplib", "telnetlib", "shutil", "urllib", "pydoc", "gzip", "asyncio", "sqlite3",
        "uu", "quopri", "pickle", "pickletools", "xmlrpc", "venv", "ensurepip", "doctest",
    }
)  # fmt: skip
_PHP_INI_RE: Final = re.compile(r"(?i)auto_(?:prepend|append)_file|allow_url_|extension")
_PERL_MODULE_RE: Final = re.compile(r"^-[Mm]-?[\w:]+(?:=[\w:,]+)?$")
_AWK_DANGER_RE: Final = re.compile(r"\bsystem\s*\(|\|\s*getline|\|&|\bprintf?\b[^;}\n]*(?:>|\|)")
_SED_DANGER_RES: Final = (
    # s/../../ with the w (write file) or e (execute) flag
    re.compile(r"s([^\w\s\\])(?:\\.|(?!\1).)*\1(?:\\.|(?!\1).)*\1[A-Za-z0-9]*[we]"),
    # a w / W / e command, optionally behind an address
    re.compile(
        r"(?:^|[;{}\n])\s*(?:(?:\d+|\$|/(?:\\.|[^/\n])*/)(?:\s*,\s*(?:[+~]?\d+|\$|/(?:\\.|[^/\n])*/))?\s*!?)?"
        r"\s*[wWe](?:\s+\S|\s*$|\s*[;}\n])"
    ),
)


class _Family(NamedTuple):
    label: str
    names: re.Pattern[str]
    inline: re.Pattern[str]  # options that carry inline code or edit files in place
    value_options: frozenset[str]  # options whose value is the next word
    stdin_program: bool  # without a script operand the program is read from stdin


_FAMILIES: Final = (
    _Family(
        "python",
        re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?w?$|^py$"),
        re.compile(r"^-(?![-WXm])[A-Za-z]*[ci]"),
        frozenset({"-W", "-X", "--check-hash-based-pycs"}),
        True,
    ),
    _Family(
        "node",
        re.compile(r"^(?:node|nodejs|deno|bun|ts-node|tsx)$"),
        re.compile(r"^(?:-(?![-rC])[A-Za-z]*[epi]|--eval|--print|--interactive)"),
        frozenset({"-r", "--require", "--import", "--loader", "--experimental-loader", "-C", "--conditions"}),
        True,
    ),
    _Family(
        "php",
        re.compile(r"^php(?:\d+(?:\.\d+)*)?(?:-cli)?$"),
        re.compile(r"^(?:-(?![-dzctf])[A-Za-z]*[rRBEFaS]|--run|--interactive|--server|--process-)"),
        frozenset({"-d", "--define", "-c", "--php-ini", "-z", "-t", "-f", "--file"}),
        True,
    ),
    _Family(
        "perl",
        re.compile(r"^perl(?:\d+(?:\.\d+)*)?$"),
        re.compile(r"^-(?![IMmxC-])[A-Za-z0-9.]*[eEid]"),
        frozenset({"-I"}),
        True,
    ),
    _Family(
        "ruby",
        re.compile(r"^(?:ruby|jruby)(?:\d+(?:\.\d+)*)?$"),
        re.compile(r"^-(?![IrCEKTW-])[A-Za-z0-9.]*[ei]"),
        frozenset({"-I", "-r", "-C", "-E", "-F"}),
        True,
    ),
    _Family(
        "lua",
        re.compile(r"^(?:lua|luajit)(?:\d+(?:\.\d+)*)?$"),
        re.compile(r"^-(?![-l])[A-Za-z]*[ei]"),
        frozenset({"-l"}),
        True,
    ),
    _Family(
        "script interpreter",
        re.compile(r"^(?:rscript|julia|elixir|groovy|osascript|raku|racket|guile)$"),
        re.compile(r"^(?:-[eEc]|--eval|--expr|--print)"),
        frozenset(),
        True,
    ),
    _Family(
        "awk",
        re.compile(r"^(?:awk|gawk|mawk|nawk|original-awk)$"),
        re.compile(r"^(?:-i|--include|-l|--load)"),
        frozenset({"-f", "--file", "-F", "--field-separator", "-v", "--assign", "-e", "--source", "-E", "--exec"}),
        False,
    ),
    _Family(
        "sed",
        re.compile(r"^(?:sed|gsed)$"),
        re.compile(r"^(?:-[A-Za-z]*[iI]|--in-place)"),
        frozenset({"-e", "--expression", "-f", "--file", "-l", "--line-length"}),
        False,
    ),
)

_RAW_RULES: Final = (
    (re.compile(r"(?<![\w-])--no-verify(?![\w-])"), "--no-verify (skipping verification hooks) is not allowed"),
    (re.compile(r"/dev/(?:tcp|udp)/"), "/dev/tcp and /dev/udp network redirections are not allowed"),
    (
        re.compile(r"(?i)HARNESS_HOOK_TOKEN|HARNESS_STATE_DIR|HARNESS_RUN_ID|harness_state"),
        "worker commands must not touch the harness state or its credentials",
    ),
)
_LOOPBACK_RE: Final = re.compile(
    r"(?i)(?<![\w.])(?:"
    r"127(?:\.\d{1,3}){1,3}"
    r"|(?:[\w-]+\.)*localhost"
    r"|0\.0\.0\.0"
    r"|0x0*7f[0-9a-f]{6}"
    r"|0x0*7f(?:\.(?:0x)?[0-9a-f]+){1,3}"
    r"|0+177(?:\.[0-7]+){1,3}"
    r"|0+177[0-7]{8,10}"
    r"|213070643\d"
    r"|::1"
    r"|\[::\]"
    r"|\[(?:0+:){1,7}0*1\]"
    r"|::ffff:127(?:\.\d{1,3}){1,3}"
    r")(?![\w])"
)
_LOOPBACK_URL_RE: Final = re.compile(
    r"(?i)(?:https?|wss?)://(?:"
    r"127(?:\.\d{1,3}){1,3}"
    r"|(?:[\w-]+\.)*localhost"
    r"|0\.0\.0\.0"
    r"|\[::1\]|::1|\[::\]|\[(?:0+:){1,7}0*1\]"
    r"|::ffff:127(?:\.\d{1,3}){1,3}"
    r"|0x0*7f[0-9a-f]{6}"
    r"|0x0*7f(?:\.(?:0x)?[0-9a-f]+){1,3}"
    r"|0+177(?:\.[0-7]+){1,3}"
    r"|0+177[0-7]{8,10}"
    r"|213070643\d"
    r"|0(?=[:/])"
    r")"
)
_DAEMON_PORT_RE: Final = re.compile(rf"(?<![\d.]){DAEMON_PORT}(?!\d)")


def _command_name(word: str) -> str:
    """``Path(token).name``, lower-cased and without a Windows executable suffix."""
    name = PurePosixPath(word.replace("\\", "/")).name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _family_of(name: str) -> _Family | None:
    for family in _FAMILIES:
        if family.names.match(name):
            return family
    return None


def _strip_line_continuations(command: str) -> str:
    out: list[str] = []
    quote: str | None = None
    index = 0
    length = len(command)
    while index < length:
        ch = command[index]
        if quote == "'":
            if ch == "'":
                quote = None
            out.append(ch)
            index += 1
        elif ch == "\\":
            if index + 1 < length:
                nxt = command[index + 1]
                if nxt == "\n":
                    index += 2  # strip \\\n (active both unquoted and inside double quotes)
                elif nxt == "\r" and index + 2 < length and command[index + 2] == "\n":
                    index += 3  # strip \\\r\n
                else:
                    out.append(command[index : index + 2])
                    index += 2
            else:
                out.append(ch)
                index += 1
        elif quote == '"':
            if ch == '"':
                quote = None
            out.append(ch)
            index += 1
        elif ch in "'\"":
            quote = ch
            out.append(ch)
            index += 1
        else:
            out.append(ch)
            index += 1
    return "".join(out)


def _prepare_shell_text(command: str) -> str:
    """Quote-aware pass: refuse substitutions and turn unquoted newlines into ``;``."""
    out: list[str] = []
    quote: str | None = None
    index = 0
    length = len(command)
    while index < length:
        ch = command[index]
        if quote == "'":
            if ch == "'":
                quote = None
            out.append(ch)
            index += 1
            continue
        if ch == "\\":
            if index + 1 < length and command[index + 1] == "\n":
                index += 2  # line continuation
                continue
            out.append(command[index : index + 2])
            index += 2
            continue
        if ch == "`":
            raise Block("tier 1", "command substitution with backticks is not allowed")
        if ch == "$" and index + 1 < length:
            following = command[index + 1]
            if following == "(":
                raise Block("tier 1", "command substitution $(...) is not allowed")
            if quote is None and following in "'\"":
                raise Block("tier 1", "$'...' / $\"...\" quoting is not allowed (it can hide command names)")
            if following.isalnum() or following in _EXPANSION_START:
                out.append(_EXPANSION)
                index += 1
                continue
        if quote == '"':
            if ch == '"':
                quote = None
            out.append(ch)
            index += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "<>" and index + 1 < length and command[index + 1] == "(":
            raise Block("tier 1", "process substitution <(...) / >(...) is not allowed")
        elif (
            ch == "{"
            and index + 1 < length
            and not command[index + 1].isspace()
            and command[index + 1] != "}"
            and (not out or out[-1] != _EXPANSION)
        ):
            out.append(_BRACE)  # may start a brace expansion: {a,b} / {1..3}
            index += 1
            continue
        elif ch == "\n":
            out.append(" ; ")
            index += 1
            continue
        elif ch == "\r":
            out.append(" ")
            index += 1
            continue
        out.append(ch)
        index += 1
    if quote is not None:
        raise Block("tier 1", "the command has an unbalanced quote")
    return "".join(out)


def _tokenize(text: str) -> list[tuple[bool, str]]:
    """``shlex`` words and shell operators as ``(is_operator, text)``.

    ``shlex`` drops the quotes, so a quoted argument that consists only of operator characters
    (``">"``) is indistinguishable from the operator. It is treated as the operator, which can
    only make the guard stricter.
    """
    lexer = shlex.shlex(text, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = ""  # '#' inside a word is not a comment for the shell: scan everything
    items: list[tuple[bool, str]] = []
    for token in lexer:
        if token and all(ch in _PUNCTUATION for ch in token):
            position = 0
            while position < len(token):
                operator = next(op for op in _OPERATORS if token.startswith(op, position))
                items.append((True, operator))
                position += len(operator)
        else:
            items.append((False, token))
    return items


def _segments(items: list[tuple[bool, str]]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for is_operator, text in items:
        if is_operator:
            segments.append([])
        else:
            segments[-1].append(text)
    return [segment for segment in segments if segment]


class BashGuard:
    """Tier 1 for one Bash command. ``writes`` lists the files its redirections may write."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.has_cd = False
        self.writes: list[str] = []

    # ---- entry point ------------------------------------------------------------------

    def check(self, command: object) -> None:
        try:
            self._check(command)
        except Block as block:  # show the expansion markers the way the worker wrote them
            shown = block.message.replace(_EXPANSION, "$").replace(_BRACE, "{")
            raise Block(block.tier, shown) from None

    def _check(self, command: object) -> None:
        if not isinstance(command, str) or not command.strip():
            raise Block("tier 1", "the Bash call carries no command")
        if len(command) > MAX_COMMAND_CHARS:
            raise Block("tier 1", f"the command is longer than {MAX_COMMAND_CHARS} characters")
        cleaned_command = _strip_line_continuations(command)
        if any(unicodedata.category(ch) in ("Cc", "Cf", "Co") and ch not in "\n\r\t" for ch in cleaned_command):
            raise Block("tier 1", "the command contains control, invisible or private-use characters")
        self._check_raw(cleaned_command)
        try:
            items = _tokenize(_prepare_shell_text(cleaned_command))
        except ValueError as exc:
            raise Block("tier 1", f"the command cannot be parsed deterministically ({exc})") from exc

        self._check_raw(" ".join(text for is_op, text in items if not is_op))

        redirect_targets: list[str] = []
        segments: list[list[str]] = []
        words: list[str] = []
        index = 0
        while index < len(items):
            is_operator, text = items[index]
            if not is_operator:
                words.append(text)
                index += 1
            elif text in ("<<", "<<<"):
                raise Block("tier 1", "here-documents and here-strings are not allowed; use the Write tool for file content")
            elif text in _WRITE_REDIRECTS or text in ("<", "<&"):
                if index + 1 >= len(items) or items[index + 1][0]:
                    raise Block("tier 1", f"redirection {text} has no literal target")
                target = items[index + 1][1]
                if text in (">&", "<&") and _FD_RE.match(target):
                    pass  # duplicating or closing a file descriptor
                elif text in _WRITE_REDIRECTS:
                    redirect_targets.append(target)
                index += 2
            else:  # a control operator ends the simple command
                if words:
                    segments.append(words)
                words = []
                index += 1
        if words:
            segments.append(words)

        for segment in segments:
            self._scan(segment, depth=0)
            self._check_command(segment)
        for target in redirect_targets:
            written = self._check_redirect(target)
            if written is not None:
                self.writes.append(written)

    # ---- raw text ---------------------------------------------------------------------

    def _check_raw(self, command: str) -> None:
        unquoted = re.sub(r"['\"]", "", command)
        for text in (command, unquoted):
            for pattern, reason in _RAW_RULES:
                if pattern.search(text):
                    raise Block("tier 1", reason)
            if _LOOPBACK_URL_RE.search(text):
                raise Block("tier 1", "worker commands must not target loopback network services")
            if _LOOPBACK_RE.search(text) and _DAEMON_PORT_RE.search(text):
                raise Block("tier 1", f"worker commands must not call the harness daemon on port {DAEMON_PORT}")
            state = self.ctx.state_dir.as_posix().lower()
            norm_cmd = text.replace("\\", "/").lower()
            if state in norm_cmd:
                raise Block("tier 1", "worker commands must not touch the harness state directory")
            if len(state) >= 2 and state[1] == ":" and state[0].isalpha():
                msys_state = "/" + state[0] + state[2:]
                if msys_state in norm_cmd:
                    raise Block("tier 1", "worker commands must not touch the harness state directory")

    # ---- words, wherever they are -------------------------------------------------------

    def _scan(self, words: list[str], *, depth: int) -> None:
        names = [_command_name(word) for word in words]
        for word, name in zip(words, names):
            self._ban(name)
            assignment = _ASSIGNMENT_RE.match(word)
            if assignment is not None:
                if _DANGEROUS_ENV_RE.match(assignment.group(1)):
                    raise Block("tier 1", f"setting {assignment.group(1)} is not allowed")
                value = word[assignment.end() :]
                self._ban(_command_name(value))
                self._scan_nested(value, depth)
            else:
                self._scan_nested(word, depth)
        for index, name in enumerate(names):
            if name == "git":
                self._check_git(words, index, strict=False)
            elif name in _SHELLS:
                self._check_shell(words, index, strict=False)
            else:
                family = _family_of(name)
                if family is not None:
                    self._check_interpreter(family, words, index, strict=False)
        if "find" in names:
            action = next((word for word in words if word in _FIND_ACTIONS or _HIDDEN_OPTION_RE.match(word)), None)
            if action is not None:
                raise Block("tier 1", f"find {action} is not allowed (it executes commands or changes files)")
        for tool, flags in _SEARCH_EXEC_FLAGS.items():
            if tool in names:
                flag = next((word for word in words if word.split("=", 1)[0] in flags), None)
                if flag is not None:
                    raise Block("tier 1", f"{tool} {flag} is not allowed (it executes commands)")

    def _ban(self, name: str) -> None:
        reason = BANNED_ANYWHERE.get(name)
        if reason is not None:
            raise Block("tier 1", f"'{name}' is not allowed in worker commands ({reason})")

    @staticmethod
    def _require_visible(tool: str, word: str) -> None:
        """An argument the shell still expands could turn into an option such as ``-c``."""
        if _HIDDEN_OPTION_RE.match(word):
            raise Block(
                "tier 1",
                f"{tool} argument {word!r} is expanded by the shell and could hide an option; write it literally",
            )

    def _scan_nested(self, word: str, depth: int) -> None:
        """A quoted argument that holds several words may be a command for somebody: scan it."""
        if not _NESTED_RE.search(word):
            return
        if depth >= MAX_NESTED_QUOTING:
            raise Block("tier 1", "quoting is nested too deeply to be inspected")
        try:
            groups = _segments(_tokenize(word))
        except ValueError:  # free text (an apostrophe in a search pattern): fall back to words
            groups = [re.findall(r"[^\s;|&()<>'\"`]+", word)]
        for group in groups:
            self._scan(group, depth=depth + 1)

    def _forbid_harness_exec(self, operand: str) -> None:
        if not operand or not isinstance(operand, str):
            return
        norm = operand.replace("\\", "/").strip().lower()
        parts = [p for p in norm.split("/") if p and p != "."]
        if ".harness" in parts:
            raise Block("tier 1", "executing scripts from .harness/ is not allowed")
        try:
            target = _resolve_target(operand, self.ctx)
            if target.rel.startswith(".harness/") or (target.lexical_rel and target.lexical_rel.startswith(".harness/")):
                raise Block("tier 1", "executing scripts from .harness/ is not allowed")
        except Block as b:
            if "executing scripts from .harness/" in b.message:
                raise
        except Exception:
            pass

    # ---- the command of one simple command -------------------------------------------------

    def _command_index(self, words: list[str]) -> int | None:
        index = 0
        while index < len(words):
            word = words[index]
            name = _command_name(word)
            if _ASSIGNMENT_RE.match(word) or word in _KEYWORD_PREFIXES or word.isdigit():
                index += 1
                continue
            value_options = _WRAPPERS.get(name)
            if value_options is None:
                return index
            index += 1
            if name == "command" and index < len(words) and words[index] in ("-v", "-V"):
                return None  # a lookup, nothing is executed
            while index < len(words):
                word = words[index]
                if word == "--":
                    index += 1
                    break
                if word.startswith("-") and len(word) > 1:
                    if name == "env":
                        if word.startswith("-S") or word.startswith("--split-string"):
                            raise Block("tier 1", "env -S (split-string) is not allowed")
                        if word == "-C" or word.startswith("-C") or word == "--chdir" or word.startswith("--chdir="):
                            raise Block("tier 1", "env -C / --chdir is not allowed")
                    index += 2 if word in value_options else 1
                elif _ASSIGNMENT_RE.match(word) or _NUMERIC_RE.match(word):
                    index += 1
                else:
                    break
        return None

    def _check_command(self, words: list[str]) -> None:
        index = self._command_index(words)
        if index is None:
            return
        word = words[index]
        name = _command_name(word)
        if word not in _SHELL_SYNTAX_WORDS and not _LITERAL_COMMAND_RE.match(word):
            raise Block("tier 1", f"the command name {word!r} is not a literal word (variables, globs and braces can hide it)")
        self._forbid_harness_exec(word)
        reason = BANNED_COMMANDS.get(name)
        if reason is not None:
            raise Block("tier 1", f"'{name}' is not allowed in worker commands ({reason})")
        if name in ("cd", "pushd", "popd"):
            self.has_cd = True
        elif name == "source" or word == ".":
            operand = words[index + 1] if index + 1 < len(words) else ""
            if not operand or operand == "-" or operand.startswith(_STDIN_OPERANDS):
                raise Block("tier 1", f"{word} needs a literal script file")
            self._forbid_harness_exec(operand)
        elif name == "git":
            self._check_git(words, index, strict=True)
        elif name in _SHELLS:
            self._check_shell(words, index, strict=True)
        else:
            family = _family_of(name)
            if family is not None:
                self._check_interpreter(family, words, index, strict=True)

    def _check_git(self, words: list[str], index: int, *, strict: bool) -> None:
        cursor = index + 1
        while cursor < len(words) and words[cursor].startswith("-"):
            option = words[cursor]
            if strict and option != "-C" and option not in _GIT_PLAIN_GLOBALS:
                raise Block("tier 1", f"git {option} is not allowed (it redirects or reconfigures git)")
            cursor += 2 if option in _GIT_VALUE_OPTIONS else 1
        if cursor >= len(words):
            return
        subcommand = words[cursor].lower()
        if subcommand in GIT_MUTATING:
            raise Block("tier 1", f"git {subcommand} is not allowed (history, index, hook or configuration mutation)")
        if not strict:
            return
        if subcommand not in GIT_READONLY:
            raise Block("tier 1", f"git {subcommand} is not on the read-only allowlist")
        for argument in words[cursor + 1 :]:
            self._require_visible("git", argument)
            if _GIT_BAD_ARGUMENT_RE.match(argument) or (subcommand == "grep" and argument.startswith("-O")):
                raise Block("tier 1", f"git {subcommand} {argument} is not allowed (it writes files or runs programs)")

    def _check_shell(self, words: list[str], index: int, *, strict: bool) -> None:
        name = _command_name(words[index])
        cursor = index + 1
        operand: str | None = None
        informational = False
        while cursor < len(words):
            word = words[cursor]
            self._require_visible(name, word)
            if word == "--":
                operand = words[cursor + 1] if cursor + 1 < len(words) else None
                break
            if len(word) > 1 and word[0] in "-+":
                if word.startswith("--"):
                    if word.split("=", 1)[0] in ("--rcfile", "--init-file"):
                        raise Block("tier 1", f"{name} {word} is not allowed")
                    informational = informational or word in _INFO_FLAGS
                    cursor += 1
                    continue
                if word[0] == "-" and any(flag in word[1:] for flag in "cis"):
                    raise Block("tier 1", f"{name} {word} is not allowed (inline or stdin shell code)")
                cursor += 2 if word in ("-o", "-O", "+o", "+O") else 1
                continue
            operand = word
            break
        if operand is not None:
            self._forbid_harness_exec(operand)
        if not strict or informational:
            return
        if operand is None or operand == "-" or operand.startswith(_STDIN_OPERANDS):
            raise Block("tier 1", f"{name} without a script file reads commands from stdin; that is not allowed")

    def _check_interpreter(self, family: _Family, words: list[str], index: int, *, strict: bool) -> None:
        label = family.label
        cursor = index + 1
        operand: str | None = None
        informational = False
        program_given = False
        while cursor < len(words):
            word = words[cursor]
            self._require_visible(label, word)
            if word == "--":
                operand = words[cursor + 1] if cursor + 1 < len(words) else None
                if operand is not None:
                    self._forbid_harness_exec(operand)
                break
            if word == "-":
                if family.stdin_program:
                    raise Block("tier 1", f"{label} reading its program from stdin is not allowed")
                cursor += 1
                continue
            if word.startswith("-"):
                if word in _INFO_FLAGS or (word == "-v" and label != "python"):
                    informational = True
                if family.inline.search(word):
                    raise Block("tier 1", f"{label} {word} is not allowed (inline code or in-place editing)")
                if label == "perl" and word[:2] in ("-M", "-m") and not _PERL_MODULE_RE.match(word):
                    raise Block("tier 1", f"{label} {word} is not allowed (inline code)")
                if label == "python":
                    py_mod = re.match(r"^-(?![-WX])[A-Za-z]*m(.*)$", word)
                    if py_mod:
                        module = py_mod.group(1) or (words[cursor + 1] if cursor + 1 < len(words) else "")
                        mod_base = module.split(".", 1)[0].lower()
                        if not module or mod_base in _PYTHON_MODULE_DENY:
                            raise Block("tier 1", f"python -m {module or '<nothing>'} is not allowed")
                        return  # everything after the module belongs to the module
                if label == "node":
                    if word == "--input-type" or word.startswith("--input-type="):
                        raise Block("tier 1", f"node {word} is not allowed")
                    if word.startswith(("-r=", "--require=", "--import=", "--loader=", "--experimental-loader=")):
                        val = word.split("=", 1)[1]
                        if val.startswith("data:"):
                            raise Block("tier 1", f"node {word} with data: URL is not allowed")
                        self._forbid_harness_exec(val)
                if label == "php":
                    if word.startswith("-d") and len(word) > 2:
                        val = word[2:]
                        if _PHP_INI_RE.search(val):
                            raise Block("tier 1", f"php {word} is not allowed")
                    elif word.startswith("--define="):
                        val = word.split("=", 1)[1]
                        if _PHP_INI_RE.search(val):
                            raise Block("tier 1", f"php {word} is not allowed")
                    if word.startswith(("-f=", "--file=")):
                        program_given = True
                        val = word.split("=", 1)[1]
                        operand = val
                        self._forbid_harness_exec(val)
                if word in family.value_options:
                    value = words[cursor + 1] if cursor + 1 < len(words) else ""
                    if label == "node" and word in ("-r", "--require", "--import", "--loader", "--experimental-loader"):
                        if value.startswith("data:"):
                            raise Block("tier 1", f"node {word} {value} with data: URL is not allowed")
                        self._forbid_harness_exec(value)
                    if label == "php" and word in ("-d", "--define") and _PHP_INI_RE.search(value):
                        raise Block("tier 1", f"php {word} {value} is not allowed")
                    if word in ("-e", "--expression", "--source"):
                        self._check_program(family, value)
                        program_given = True
                    if word in ("-f", "--file"):
                        program_given = True
                        operand = value
                        self._forbid_harness_exec(value)
                    cursor += 2
                    continue
                cursor += 1
                continue
            if label in ("awk", "sed"):
                if not program_given:
                    self._check_program(family, word)
                return
            if label == "node" and operand is None and word in ("eval", "repl"):
                raise Block("tier 1", f"{_command_name(words[index])} {word} is not allowed (inline code)")
            if operand is None:
                operand = word
                self._forbid_harness_exec(operand)
            if _SCRIPT_LIKE_RE.search(word):
                break  # a script file: what follows are its own arguments
            cursor += 1  # not obviously a script: keep looking for inline-code options
        if strict and family.stdin_program and not informational:
            if operand is None or operand.startswith(_STDIN_OPERANDS):
                raise Block("tier 1", f"{label} without a script file reads its program from stdin; that is not allowed")

    def _check_program(self, family: _Family, program: str) -> None:
        if family.label == "awk" and _AWK_DANGER_RE.search(program):
            raise Block("tier 1", "the awk program runs commands or writes files")
        if family.label == "sed" and any(pattern.search(program) for pattern in _SED_DANGER_RES):
            raise Block("tier 1", "the sed script uses the w (write file) or e (execute) command")

    # ---- redirections -------------------------------------------------------------------

    def _check_redirect(self, raw: str) -> str | None:
        if raw in _DEVICE_SINKS:
            return None
        if not _LITERAL_PATH_RE.match(raw):
            raise Block("tier 1", f"redirection target {raw!r} is not a literal path (no variables, globs, braces or ~)")
        if self.has_cd and not Path(_native_path(raw)).is_absolute():
            raise Block("tier 1", f"relative redirection target {raw!r} cannot be resolved because the command changes directory")
        target = _resolve_target(raw, self.ctx)
        scope = "redirections may only write files in editable_files or below .harness/logs/"
        worktree_root = self.ctx.root.resolve()
        if target.rel.startswith(LOG_DIR_REL + "/") and target.lexical_rel == target.rel:
            for parent in target.resolved.parents:
                if parent == self.ctx.root or parent.resolve() == worktree_root:
                    break
                if not parent.resolve().is_relative_to(worktree_root):
                    raise Block("tier 1", f"a parent directory of redirection target {target.rel} resolves outside the worktree")
                if _is_link_or_junction(parent):
                    raise Block("tier 1", f"a parent directory of redirection target {target.rel} is a symbolic link or junction")
                if os.path.lexists(parent) and not parent.is_dir():
                    raise Block("tier 1", f"a parent of redirection target {target.rel} is not a directory")
            info = _lstat(target.resolved)
            if info is not None and (not stat.S_ISREG(info.st_mode) or info.st_nlink > 1):
                raise Block("tier 1", f"redirection target {target.rel} is not a plain file")
            problem = _path_rule_violation(target.rel)
            if problem is not None:
                raise Block("tier 1", f"redirection target {target.rel} is refused: {problem}")
            return target.rel
        if target.rel not in self.ctx.editable:
            raise Block("tier 1", f"redirection target {target.rel} is not in editable_files; {scope}")
        _check_write_target(self.ctx, target, what="redirection target")
        return target.rel


# --------------------------------------------------------------------------------------
# Tier 2: semantic drift
# --------------------------------------------------------------------------------------

_CONTROL_FLOW_RE: Final = (
    r"\b(?:if|elif|else|for|while|try|except|finally|catch|switch|case|match|return|raise|throw|break"
    r"|continue|goto|yield|await|unless|until|sleep|retry|retries|timeout)\b|&&|\|\|"
)
_ASSERTION_RE: Final = (
    r"\bassert\w*|\bexpect\s*\(|\bshould\b|\bverify\w*\s*\(|pytest\.(?:raises|skip|xfail|mark|fail|approx)"
    r"|@(?:pytest\.mark\.)?(?:skip|xfail)\w*|\b(?:it|test|describe)\.(?:skip|only|todo)\b"
    r"|\bx(?:it|describe|test)\s*\(|unittest\.skip|\.to(?:Be|Equal|Throw|Match|Have)\w*\s*\(|markTestSkipped"
    r"|@Ignore|@Disabled|\bt\.Skip|\bnoqa\b|type:\s*ignore|eslint-disable|@ts-ignore|@ts-nocheck"
    r"|phpstan-ignore"
)
_DATABASE_RE: Final = (
    r"(?i)\b(?:select\b.+?\bfrom|insert\s+into|update\s+\S+\s+set|delete\s+from|merge\s+into"
    r"|drop\s+(?:table|database|index|view|schema)|alter\s+table|create\s+(?:table|index|database|view)"
    r"|truncate\s+table|grant\s+\w+|revoke\s+\w+)\b|\.execute(?:many|script)?\s*\(|\bcursor\b|\.query\s*\("
    r"|\.raw\s*\(|\bsession\.(?:add|delete|commit|rollback|execute|query|merge|flush)\b"
    r"|\b(?:commit|rollback)\s*\(|\b(?:sqlalchemy|psycopg\d?|sqlite3|pymysql|asyncpg|pymongo|redis|prisma|knex"
    r"|sequelize|typeorm|mongoose|pdo|mysqli|eloquent|activerecord|entitymanager|jdbctemplate|gorm)\b"
    r"|\b(?:psql|mysql|mongosh?|redis-cli|sqlcmd|pg_dump|pg_restore|alembic)\b|\bmigrat\w+"
)
_SECURITY_RE: Final = (
    r"(?i)\b(?:eval|exec|execfile|compile|system|popen|spawn\w*|shell_exec|passthru|proc_open|unserialize"
    r"|deserialize)\s*\(|\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*|chmod|chown|remove|unlink|rmdir"
    r"|rename|replace|environ|putenv)\b|\bshutil\.|\bchild_process\b|shell\s*=\s*True|\bpickle\b|\bmarshal\b"
    r"|\byaml\.(?:load|unsafe_load)\b|__import__|\bimportlib\b|\binnerHTML\b|dangerouslySetInnerHTML"
    r"|document\.write|\bv-html\b|\|\s*safe\b|mark_safe|\bverify\s*=\s*False|rejectUnauthorized"
    r"|InsecureSkipVerify|\b(?:md5|sha1)\b|\b(?:password|passwd|secret|token|api[_-]?key|credential\w*"
    r"|private[_-]?key|jwt|oauth|bearer|session|cookie|csrf|cors|auth\w*|permission\w*|login|sudo|crypt\w*"
    r"|hmac|hashlib|ssl|tls|cert\w*)\b|\brequests\.|\burllib\b|\bhttpx\b|\bsocket\b|\bfetch\s*\(|\bredirect\w*"
    r"|\bopen\s*\(|\b(?:readFile|writeFile)\w*\s*\(|\bsanitiz\w+|\bescape\w*"
)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _touches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _edit_subject(tool_name: str, tool_input: dict[str, Any], rel: str) -> tuple[str, str]:
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits")
        pairs = [
            (_text(edit.get("old_string")), _text(edit.get("new_string")))
            for edit in (edits if isinstance(edits, list) else [])
            if isinstance(edit, dict)
        ]
    else:
        pairs = [(_text(tool_input.get("old_string")), _text(tool_input.get("new_string")))]
    budget = max(300, (SNIPPET_LIMIT - 100) // max(1, len(pairs)))
    parts = [f"# {tool_name} {rel}"]
    for old, new in pairs:
        parts.append(f"--- removed\n{_clip(old, budget // 3)}\n+++ added\n{_clip(new, budget - budget // 3)}")
    return "\n".join(f"{old}\n{new}" for old, new in pairs), "\n".join(parts)


def _write_subject(tool_input: dict[str, Any], target: Target, existed: bool) -> tuple[str, str]:
    content = _text(tool_input.get("content"))
    header = f"# Write {target.rel}"
    if existed and len(content) <= MAX_DIFF_BYTES:
        try:
            if target.resolved.stat().st_size <= MAX_DIFF_BYTES:
                import difflib

                before = target.resolved.read_text(encoding="utf-8", errors="replace")
                diff = list(difflib.unified_diff(before.splitlines(), content.splitlines(), "before", "after", n=1, lineterm=""))
                changed = [line[1:] for line in diff[2:] if line[:1] in "+-"]
                return "\n".join(changed), header + " (diff against the current file)\n" + "\n".join(diff)
        except OSError:
            pass
    return content, header + (" (replaces the file)\n" if existed else " (new file)\n") + content


def _semantic_subject(
    tool_name: str, tool_input: dict[str, Any], target: Target | None, existed: bool, bash_writes: list[str]
) -> str | None:
    """The snippet for the tier-2 check, or None when the call does not need one."""
    if tool_name == "Bash":
        command = _strip_line_continuations(_text(tool_input.get("command")))
        writes_source = any(not written.startswith(LOG_DIR_REL + "/") for written in bash_writes)
        patterns = (_DATABASE_RE,) + ((_CONTROL_FLOW_RE, _ASSERTION_RE, _SECURITY_RE) if writes_source else ())
        return command if _touches(command, patterns) else None
    if target is None:
        return None
    if tool_name == "Write":
        subject, snippet = _write_subject(tool_input, target, existed)
    elif tool_name == "NotebookEdit":
        subject = _text(tool_input.get("new_source"))
        snippet = f"# NotebookEdit {target.rel}\n{subject}"
    else:
        subject, snippet = _edit_subject(tool_name, tool_input, target.rel)
    if _touches(subject, (_CONTROL_FLOW_RE, _ASSERTION_RE, _DATABASE_RE, _SECURITY_RE)):
        return snippet
    return None


def _tier2(ctx: Ctx, tool_name: str, snippet: str) -> None:
    body = {
        "run_id": ctx.run_id,
        "tool_name": tool_name,
        "snippet": snippet[:SNIPPET_LIMIT],
        "invariants": ctx.invariants,
        "forbidden_patterns": ctx.forbidden_patterns,
        "model": TIER2_MODEL,
        "effort": TIER2_EFFORT,
    }
    try:
        status, verdict = _post_json(HOOK_EVAL_ENDPOINT, body)
    except HookError:
        raise
    except Exception as exc:
        raise Block(
            "tier 2",
            f"the semantic drift check is unavailable ({type(exc).__name__}: {exc}); failing closed. "
            "This is a harness problem, not a verdict on the change.",
        ) from exc
    if status != 200 or not isinstance(verdict, dict) or not isinstance(verdict.get("allowed"), bool):
        raise Block("tier 2", f"the semantic drift check gave no usable verdict (HTTP {status}); failing closed")
    if not verdict["allowed"]:
        reason = verdict.get("reason")
        shown = reason.strip() if isinstance(reason, str) and reason.strip() else "no reason given"
        raise Block("tier 2", f"semantic drift: {_clip(shown, 2000)}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _guard(payload: dict[str, Any], tool_name: str, tool_input: dict[str, Any]) -> int:
    extra = {name.strip() for name in os.environ.get("HARNESS_EXTRA_ALLOWED_TOOLS", "").split(",") if name.strip()}
    if tool_name not in WRITE_TOOLS and tool_name != "Bash":
        if tool_name in PASSTHROUGH_TOOLS or tool_name in extra:
            return 0
        raise Block(
            "tier 1",
            f"the tool {tool_name} is not available to contract-bound workers "
            "(use Read / Glob / Grep, Edit / Write and Bash)",
        )

    ctx = _load_context(payload)
    target: Target | None = None
    existed = False
    bash_writes: list[str] = []
    if tool_name == "Bash":
        guard = BashGuard(ctx)
        guard.check(tool_input.get("command"))
        bash_writes = guard.writes
    else:
        if tool_name == "NotebookEdit":
            nb = tool_input.get("notebook_path")
            fp = tool_input.get("file_path")
            if isinstance(nb, str) and isinstance(fp, str) and nb != fp:
                raise Block("tier 1", "conflicting notebook_path and file_path")
            path_arg = nb if nb is not None else fp
            target = _resolve_target(path_arg, ctx)
        elif tool_name == "MultiEdit":
            edits = tool_input.get("edits")
            if not isinstance(edits, list) or not edits or not all(isinstance(e, dict) for e in edits):
                raise Block("tier 1", "MultiEdit requires a non-empty list of edit objects")
            top_fp = tool_input.get("file_path")
            for edit in edits:
                sub_fp = edit.get("file_path") if "file_path" in edit else edit.get("path")
                if sub_fp is not None and sub_fp != top_fp:
                    raise Block("tier 1", "MultiEdit item targets a different file_path")
            target = _resolve_target(top_fp, ctx)
        else:
            target = _resolve_target(tool_input.get("file_path"), ctx)

        if target.resolved == ctx.root / ".harness" / "amendment_request.json" and target.lexical_rel == AMENDMENT_REL:
            if _is_link_or_junction(ctx.root / ".harness"):
                raise Block("amendment", ".harness directory cannot be a symbolic link or junction")
            info = _lstat(target.resolved)
            if info is not None:
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise Block("amendment", f"{AMENDMENT_REL} is not a regular single-link file")
            if tool_name not in AMENDMENT_TOOLS:
                raise Block("amendment", f"{AMENDMENT_REL} may only be written with Write or Edit")
            return _handle_amendment(ctx, tool_name, tool_input, target)
        existed = _check_write_target(ctx, target, what="file")

    snippet = _semantic_subject(tool_name, tool_input, target, existed, bash_writes)
    if snippet is not None:
        _tier2(ctx, tool_name, snippet)
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
    _arm_watchdog()
    payload = _read_payload()
    event = payload.get("hook_event_name")
    if event is not None and event != "PreToolUse":
        raise HookError(f"this hook handles PreToolUse, not {event!r}")
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_name, str) or not tool_name:
        raise HookError("the hook payload names no tool")
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        raise HookError("tool_input is not a JSON object")
    try:
        return _guard(payload, tool_name, tool_input)
    except Block as block:
        _log(f"BLOCKED ({block.tier}): {block.message}")
        return 2


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception as exc:  # fail closed: an unexpected error blocks the tool call
        _log(f"internal error, failing closed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    except BaseException as exc:  # KeyboardInterrupt and friends must not turn into "allow"
        if isinstance(exc, SystemExit) and exc.code in (0, 2):
            raise
        _log(f"interrupted, failing closed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    sys.exit(_exit_code)
