"""Adversarial DevSecOps Test Suite for PreToolGuard and PostToolCompactor.

Covers the 5 Red-Team Vectors:
Vector 1: Symlink & Relative-Path Escape Gate
Vector 2: Same-UID Bash Bypass & Localhost SSRF Gate
Vector 3: Git Staging Poisoning & <2ms SHA-256 Tamper Restoration Gate
Vector 4: Linked Worktree .git Pointer & Out-of-Tree State Gate
Vector 5: Exit-and-Resume JSON Protocol Gate
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

PRE_GUARD_PATH = Path(__file__).parents[1] / "hooks" / "pre_tool_guard.py"
POST_COMPACTOR_PATH = Path(__file__).parents[1] / "hooks" / "post_tool_compactor.py"


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _rmtree(path: Path) -> None:
    def onexc(func: Any, fpath: str, exc: Any) -> None:
        try:
            os.chmod(fpath, stat.S_IWRITE | stat.S_IREAD)
            func(fpath)
        except OSError:
            pass

    shutil.rmtree(path, onexc=onexc)


def _safe_unlink(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    try:
        if path.is_symlink() or not path.is_dir():
            os.unlink(path)
        else:
            os.rmdir(path)
    except OSError:
        pass


def _can_link_file() -> bool:
    test_link = Path(__file__).parent / ".test_symlink_probe.tmp"
    try:
        os.symlink(__file__, test_link)
        test_link.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


def _link_dir(link: Path, target: Path) -> None:
    """Create a directory symlink or NTFS junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        res = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0:
            return
    pytest.skip("directory symlinks / junctions not supported in this environment")


def _link_file(link: Path, target: Path) -> None:
    """Create a file symlink or hardlink fallback."""
    try:
        os.symlink(target, link)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        res = subprocess.run(
            ["cmd", "/c", "mklink", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0:
            return
    pytest.skip("file symlinks not supported in this environment")


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def _init_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    """Initialize a git repo with standard structure and out-of-tree state directory."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)

    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "auditor@security.local")
    _run_git(repo, "config", "user.name", "Security Auditor")
    _run_git(repo, "config", "core.autocrlf", "false")

    # Layout files
    (repo / "src" / "billing").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / ".harness" / "logs").mkdir(parents=True, exist_ok=True)

    ledger_py = repo / "src" / "billing" / "ledger.py"
    ledger_py.write_text("def compute_total() -> int:\n    return 100\n", encoding="utf-8")

    test_ledger_py = repo / "tests" / "test_ledger.py"
    test_ledger_py.write_text("def test_ledger():\n    assert True\n", encoding="utf-8")

    pyproject_toml = repo / "pyproject.toml"
    pyproject_toml.write_text('[project]\nname = "agent-harness"\n', encoding="utf-8")

    unlisted_file = repo / "src" / "billing" / "unlisted.py"
    unlisted_file.write_text("# unlisted existing file\n", encoding="utf-8")

    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-q", "-m", "Initial commit")
    base_commit = _run_git(repo, "rev-parse", "HEAD").stdout.strip()

    # Create task contract
    contract = {
        "summary": "Fix ledger rounding",
        "root_cause_or_design_rationale": "Audit test harness",
        "editable_files": ["src/billing/ledger.py"],
        "allowed_new_file_globs": ["src/billing/*.py", "tests/unit/*.py"],
        "readonly_files": [
            "tests/test_ledger.py",
            "pyproject.toml",
            "hooks/*",
            ".git/*",
            ".claude/*",
            ".harness/*",
        ],
        "invariants": [],
        "forbidden_patterns": [],
    }
    (state / "task_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    # Create readonly manifest
    manifest = {
        "base_commit_sha": base_commit,
        "files": {
            "tests/test_ledger.py": _sha256(test_ledger_py),
            "pyproject.toml": _sha256(pyproject_toml),
        },
    }
    (state / "readonly_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (state / "turn_state.json").write_text(json.dumps({"turn_count": 0}, indent=2), encoding="utf-8")

    return repo, state, base_commit


def run_pre(repo: Path, state: Path, tool_name: str, tool_input: dict[str, Any], **extra_payload: Any) -> subprocess.CompletedProcess:
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "cwd": str(repo),
        "session_id": "test-session",
        "tool_use_id": "tool-1",
        **extra_payload,
    }
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env["HARNESS_WORKTREE_ROOT"] = str(repo)
    env["HARNESS_STATE_DIR"] = str(state)
    env["HARNESS_HOOK_TOKEN"] = "auditor-secret-token"
    env["PYTHONPATH"] = str(repo)
    return subprocess.run(
        [sys.executable, "-I", str(PRE_GUARD_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo),
        env=env,
        check=False,
    )


def run_post(repo: Path, state: Path, tool_name: str, tool_input: dict[str, Any], **extra_payload: Any) -> subprocess.CompletedProcess:
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": {"stdout": "ok\n", "stderr": ""},
        "cwd": str(repo),
        "session_id": "test-session",
        "tool_use_id": "tool-1",
        "duration_ms": 15,
        **extra_payload,
    }
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env["HARNESS_WORKTREE_ROOT"] = str(repo)
    env["HARNESS_STATE_DIR"] = str(state)
    env["HARNESS_HOOK_TOKEN"] = "auditor-secret-token"
    env["PYTHONPATH"] = str(repo)
    return subprocess.run(
        [sys.executable, "-I", str(POST_COMPACTOR_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo),
        env=env,
        check=False,
    )


# ======================================================================================
# Vector 1 — Symlink & Relative-Path Escape Gate
# ======================================================================================


def test_vector1_allowed_and_blocked_writes(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)

    # Allowed Edit / Write on editable_files
    res = run_pre(
        repo,
        state,
        "Edit",
        {"file_path": "src/billing/ledger.py", "old_string": "100", "new_string": "200"},
    )
    assert res.returncode == 0, res.stderr

    # Allowed Write on allowed_new_file_globs
    res = run_pre(
        repo,
        state,
        "Write",
        {"file_path": "src/billing/calculator.py", "content": "# safe calc\n"},
    )
    assert res.returncode == 0, res.stderr

    # Blocked Write on readonly_files
    res = run_pre(repo, state, "Write", {"file_path": "tests/test_ledger.py", "content": "# pwn\n"})
    assert res.returncode == 2
    assert "read-only" in res.stderr

    res = run_pre(repo, state, "Edit", {"file_path": "pyproject.toml", "old_string": "agent-harness", "new_string": "evil"})
    assert res.returncode == 2
    assert "read-only" in res.stderr

    # Blocked Write on unlisted existing file
    res = run_pre(repo, state, "Write", {"file_path": "src/billing/unlisted.py", "content": "# change\n"})
    assert res.returncode == 2
    assert "not in editable_files" in res.stderr

    # Blocked Write on disallowed new file glob
    res = run_pre(repo, state, "Write", {"file_path": "docs/readme.md", "content": "# hello\n"})
    assert res.returncode == 2
    assert "matches no allowed_new_file_globs" in res.stderr

    # Blocked Write on hard-protected paths
    for protected in (".git/config", ".claude/settings.json", "hooks/pre_tool_guard.py", ".harness/foo.json"):
        res = run_pre(repo, state, "Write", {"file_path": protected, "content": "# pwn\n"})
        assert res.returncode == 2
        assert "touches the protected area" in res.stderr or "can never be written" in res.stderr or "read-only" in res.stderr


def test_vector1_relative_and_symlink_escapes(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)
    outside = tmp_path / "outside_dir"
    outside.mkdir(parents=True, exist_ok=True)
    secret_file = outside / "secret.py"
    secret_file.write_text("SECRET=1\n", encoding="utf-8")

    # Relative traversal ../../outside.py
    res = run_pre(repo, state, "Write", {"file_path": "../../outside_dir/secret.py", "content": "pwn"})
    assert res.returncode == 2
    assert "outside the worktree" in res.stderr

    # Direct file symlink inside worktree pointing outside worktree
    if _can_link_file():
        symlink_file = repo / "src" / "billing" / "sym_outside.py"
        _link_file(symlink_file, secret_file)
        res = run_pre(repo, state, "Write", {"file_path": "src/billing/sym_outside.py", "content": "pwn"})
        assert res.returncode == 2
        assert "symbolic link" in res.stderr or "outside the worktree" in res.stderr
    else:
        # In-process verification of direct symlink check
        import hooks.pre_tool_guard as ptg
        from unittest.mock import patch
        with patch.dict(os.environ, {"HARNESS_STATE_DIR": str(state), "HARNESS_WORKTREE_ROOT": str(repo)}):
            ctx = ptg._load_context({
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "cwd": str(repo),
                "session_id": "test",
                "tool_use_id": "1",
            })
            with patch.object(ptg, "_is_link_or_junction", return_value=True):
                with pytest.raises(ptg.Block, match="symbolic link or directory junction"):
                    ptg._resolve_target("src/billing/sym_outside.py", ctx)

    # Parent directory symlink / junction pointing outside worktree
    sym_dir = repo / "src" / "billing" / "escaped_dir"
    _link_dir(sym_dir, outside)
    res = run_pre(repo, state, "Write", {"file_path": "src/billing/escaped_dir/new_file.py", "content": "pwn"})
    assert res.returncode == 2
    assert "outside the worktree" in res.stderr or "symbolic link or junction" in res.stderr

    # Boomerang symlink attack: escaped_dir -> outside_dir, with outside_dir/back_dir -> repo/src/billing
    back_dir = outside / "back_dir"
    _link_dir(back_dir, repo / "src" / "billing")
    res = run_pre(repo, state, "Write", {"file_path": "src/billing/escaped_dir/back_dir/ledger.py", "content": "pwn"})
    assert res.returncode == 2
    assert "outside the worktree" in res.stderr or "symbolic link or junction" in res.stderr

    # Intra-worktree directory junction: tests/link_dir -> src/billing
    intra_link = repo / "tests" / "link_dir"
    _link_dir(intra_link, repo / "src" / "billing")
    res = run_pre(repo, state, "Write", {"file_path": "tests/link_dir/ledger.py", "content": "pwn"})
    assert res.returncode == 2
    assert "symbolic link" in res.stderr or "read-only" in res.stderr

    # Escaped payload["cwd"] outside worktree
    res = run_pre(
        repo,
        state,
        "Write",
        {"file_path": "src/billing/ledger.py", "content": "pwn"},
        cwd=str(outside),
    )
    assert res.returncode == 2
    assert "resolves outside the worktree" in res.stderr


def test_vector1_multiedit_and_notebookedit_validation(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)

    # Conflicting NotebookEdit paths
    res = run_pre(
        repo,
        state,
        "NotebookEdit",
        {
            "notebook_path": "src/billing/nb1.ipynb",
            "file_path": "src/billing/nb2.ipynb",
            "new_source": "# test",
        },
    )
    assert res.returncode == 2
    assert "conflicting notebook_path and file_path" in res.stderr

    # MultiEdit targeting different file_path in edit items
    res = run_pre(
        repo,
        state,
        "MultiEdit",
        {
            "file_path": "src/billing/ledger.py",
            "edits": [
                {"file_path": "tests/test_ledger.py", "old_string": "assert True", "new_string": "assert False"},
            ],
        },
    )
    assert res.returncode == 2
    assert "MultiEdit item targets a different file_path" in res.stderr

    # MultiEdit empty edits list
    res = run_pre(
        repo,
        state,
        "MultiEdit",
        {"file_path": "src/billing/ledger.py", "edits": []},
    )
    assert res.returncode == 2
    assert "non-empty list" in res.stderr


# ======================================================================================
# Vector 2 — Same-UID Bash Bypass & Localhost SSRF Gate
# ======================================================================================


@pytest.mark.parametrize(
    "cmd",
    [
        'python3 -c "print(1)"',
        'python -Im timeit "1+1"',
        "python3 -m base64",
        "python3 -um http.server",
        "python3 -qm base64",
        'php -r "echo 1;"',
        'php -nr "echo 1;"',
        "php -dauto_prepend_file=x.php index.php",
        'node -e "1"',
        'node -e"console.log(1)" app.js',
        'node -p"1"',
        'perl -e "1"',
        'ruby -e "1"',
        'lua -We "print(1)"',
        "node --input-type=module -",
        "node --import=data:text/javascript,console.log(1) app.js",
    ],
)
def test_vector2_inline_interpreters_blocked(tmp_path: Path, cmd: str) -> None:
    repo, state, _ = _init_repo(tmp_path)
    res = run_pre(repo, state, "Bash", {"command": cmd})
    assert res.returncode == 2
    assert "BLOCKED (tier 1)" in res.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        "echo aGVsbG8= | base64 -d",
        "openssl enc -base64 -d",
        "xxd -r",
        "echo $(id)",
        "echo `id`",
        "echo $\\\n(id)",
        'echo "\'" $\\\n(id)',
        "$\\\n'rm' -rf /",
        "cat <(echo 1)",
        "cat <\\\n(echo 1)",
        "env -C /tmp pytest",
        "echo x > .harness/logs/pwn.py && python3 .harness/logs/pwn.py",
    ],
)
def test_vector2_base64_subshell_and_exec_blocked(tmp_path: Path, cmd: str) -> None:
    repo, state, _ = _init_repo(tmp_path)
    res = run_pre(repo, state, "Bash", {"command": cmd})
    assert res.returncode == 2
    assert "BLOCKED (tier 1)" in res.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        "curl http://127.0.0.1:8000/api/internal/hook-eval",
        "wget http://localhost:8000/api",
        "python3 app.py \"http://local'host':80'00'/api\"",
        "python3 app.py http://evil.localhost:8000",
        "python3 app.py http://0x7f000001:8000",
        "python3 app.py http://[::1]:8000",
        "python3 app.py http://127.0.0.1:9999/ssrf",
        "python3 app.py http://0.0.0.0:8000",
    ],
)
def test_vector2_localhost_ssrf_blocked(tmp_path: Path, cmd: str) -> None:
    repo, state, _ = _init_repo(tmp_path)
    res = run_pre(repo, state, "Bash", {"command": cmd})
    assert res.returncode == 2
    assert "BLOCKED (tier 1)" in res.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest -q",
        "git status",
        "git diff",
        "rg localhost src/",
    ],
)
def test_vector2_safe_bash_commands_allowed(tmp_path: Path, cmd: str) -> None:
    repo, state, _ = _init_repo(tmp_path)
    res = run_pre(repo, state, "Bash", {"command": cmd})
    assert res.returncode == 0, res.stderr


# ======================================================================================
# Vector 3 — Git Staging Poisoning & <2ms SHA-256 Tamper Restoration Gate
# ======================================================================================


def test_vector3_clean_manifest_under_2ms_budget(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)

    # Warm-up run to ensure disk cache
    run_post(repo, state, "Bash", {"command": "pytest -q"})

    # Timed run
    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 0, res.stderr

    events_path = state / "epoch_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    record = json.loads(lines[-1])

    assert record["integrity"]["files"] == 2
    assert record["integrity"]["verify_ms"] < 2.0, f"verify_ms {record['integrity']['verify_ms']} exceeded 2.0ms"
    assert not record["integrity"]["over_budget"]
    assert record["status"] == "ok"


def test_vector3_git_staging_poisoning_recovery(tmp_path: Path) -> None:
    repo, state, base_commit = _init_repo(tmp_path)
    test_ledger = repo / "tests" / "test_ledger.py"
    pristine_sha = _sha256(test_ledger)

    # Attack: tamper tests/test_ledger.py and stage it with git add
    test_ledger.write_text("def test_poisoned_index(): pass\n", encoding="utf-8")
    _run_git(repo, "add", "tests/test_ledger.py")

    # Tamper working tree further
    test_ledger.write_text("def test_double_poisoned(): pass\n", encoding="utf-8")

    # Run post_tool_compactor
    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr

    # Verified bit-for-bit restoration to base commit
    assert _sha256(test_ledger) == pristine_sha

    # Verify staging index was unstaged
    diff_cached = _run_git(repo, "diff", "--cached").stdout.strip()
    assert diff_cached == "", f"Staging area was not cleared: {diff_cached}"

    # Verify restored file is write-protected (chmod a-w)
    mode = os.stat(test_ledger).st_mode
    assert not (mode & 0o222), "Restored file was not write-protected"

    # Verify event record
    events = (state / "epoch_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(events[-1])
    assert record["status"] == "tamper"
    assert record["integrity"]["violations"][0]["restored"] is True
    assert record["integrity"]["violations"][0]["path"] == "tests/test_ledger.py"


def test_vector3_missing_file_and_symlink_restoration(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)
    test_ledger = repo / "tests" / "test_ledger.py"
    pristine_sha = _sha256(test_ledger)

    # 1. Missing (deleted) file
    _safe_unlink(test_ledger)
    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr
    assert test_ledger.exists()
    assert _sha256(test_ledger) == pristine_sha

    # 2. File replaced by symlink or junction
    _safe_unlink(test_ledger)
    outside = tmp_path / "outside_tests"
    outside.mkdir(parents=True, exist_ok=True)
    if _can_link_file():
        _link_file(test_ledger, outside / "fake.py")
    else:
        _link_dir(test_ledger, outside)
    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr
    assert not test_ledger.is_symlink()
    assert getattr(test_ledger, "is_junction", lambda: False)() is False
    assert _sha256(test_ledger) == pristine_sha

    # 3. Parent directory replaced by symlink/junction
    (outside / "test_ledger.py").write_text("evil", encoding="utf-8")
    _rmtree(repo / "tests")
    _link_dir(repo / "tests", outside)

    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr
    assert (repo / "tests" / "test_ledger.py").exists()
    assert _sha256(repo / "tests" / "test_ledger.py") == pristine_sha
    events = (state / "epoch_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    parent_record = json.loads(events[-1])
    assert parent_record["integrity"]["violations"][0]["restored"] is True


def test_vector3_autocrlf_lf_and_crlf_restoration(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)
    _run_git(repo, "config", "core.autocrlf", "true")
    test_ledger = repo / "tests" / "test_ledger.py"
    pristine_sha = _sha256(test_ledger)

    # Even with core.autocrlf=true, an LF-hashed manifest file restores bit-for-bit
    _safe_unlink(test_ledger)
    test_ledger.write_bytes(b"def test_ledger():\r\n    assert False\r\n")
    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert _sha256(test_ledger) == pristine_sha
    events = (state / "epoch_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(events[-1])["integrity"]["violations"][0]["restored"] is True


def test_vector3_index_lock_sabotage_recovery(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)
    test_ledger = repo / "tests" / "test_ledger.py"
    pristine_sha = _sha256(test_ledger)

    # Tamper test file
    test_ledger.write_text("# tampered\n", encoding="utf-8")

    # Sabotage git with index.lock
    index_lock = repo / ".git" / "index.lock"
    index_lock.write_bytes(b"locked\n")

    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr
    assert _sha256(test_ledger) == pristine_sha


def test_vector3_out_of_tree_settings_restore(tmp_path: Path) -> None:
    repo, state, base_commit = _init_repo(tmp_path)
    settings_json = tmp_path / "settings.json"
    settings_json.write_text('{"tools": ["safe"]}\n', encoding="utf-8")
    pristine_sha = _sha256(settings_json)

    # Pristine copy in state dir
    (state / "pristine").mkdir(parents=True, exist_ok=True)
    (state / "pristine" / "settings.json").write_text('{"tools": ["safe"]}\n', encoding="utf-8")

    # Update manifest to include out-of-tree file
    manifest = json.loads((state / "readonly_manifest.json").read_text(encoding="utf-8"))
    manifest["files"][str(settings_json)] = {
        "sha256": pristine_sha,
        "restore_from": "pristine/settings.json",
    }
    (state / "readonly_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Tamper settings.json
    settings_json.write_text('{"tools": ["evil"]}\n', encoding="utf-8")

    res = run_post(repo, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr
    assert _sha256(settings_json) == pristine_sha


def test_vector3_post_tool_failure_and_output_compaction(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)

    # 1. PostToolUseFailure still verifies and catches tamper
    test_ledger = repo / "tests" / "test_ledger.py"
    test_ledger.write_text("# tampered\n", encoding="utf-8")
    res = run_post(
        repo,
        state,
        "Bash",
        {"command": "false"},
        hook_event_name="PostToolUseFailure",
        error="exit code 1",
    )
    assert res.returncode == 2
    assert "TAMPER_BLOCKED" in res.stderr

    # 2. Output compaction for >80 lines
    long_output = "\n".join(f"line {i}" for i in range(120))
    res = run_post(
        repo,
        state,
        "Bash",
        {"command": "pytest -v"},
        tool_response={"stdout": long_output, "stderr": ""},
    )
    assert res.returncode == 0

    events = (state / "epoch_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(events[-1])
    assert record["stdout_truncated"] is True
    assert record["stdout_lines"] == 120
    assert record["log_path"] == f".harness/logs/turn_{record['turn']}.log"

    log_file = repo / record["log_path"]
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == long_output


# ======================================================================================
# Vector 4 — Linked Worktree .git Pointer & Out-of-Tree State Gate
# ======================================================================================


def test_vector4_linked_worktree_out_of_tree_state(tmp_path: Path) -> None:
    repo, state, base_commit = _init_repo(tmp_path)
    linked_wt = tmp_path / "linked_wt"

    # Create REAL linked git worktree
    res = _run_git(repo, "worktree", "add", str(linked_wt), "-b", "linked-branch")
    assert res.returncode == 0, res.stderr

    # Verify .git is a FILE
    git_pointer = linked_wt / ".git"
    assert git_pointer.is_file(), ".git in linked worktree should be a text pointer file"

    # 1. Explicit out-of-tree HARNESS_STATE_DIR in linked worktree
    res = run_pre(
        linked_wt,
        state,
        "Edit",
        {"file_path": "src/billing/ledger.py", "old_string": "100", "new_string": "300"},
    )
    assert res.returncode == 0, res.stderr

    res = run_post(linked_wt, state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 0, res.stderr

    # 2. Unset HARNESS_STATE_DIR, resolve via HARNESS_RUN_ID under <git-common-dir>/harness_state/<run_id>
    common_dir_raw = _run_git(linked_wt, "rev-parse", "--git-common-dir").stdout.strip()
    common_dir = Path(common_dir_raw) if Path(common_dir_raw).is_absolute() else (linked_wt / common_dir_raw).resolve()
    run_state = common_dir / "harness_state" / "run-linked"
    run_state.mkdir(parents=True, exist_ok=True)

    # Copy state to <git-common-dir>/harness_state/run-linked
    shutil.copy(state / "task_contract.json", run_state / "task_contract.json")
    shutil.copy(state / "readonly_manifest.json", run_state / "readonly_manifest.json")
    shutil.copy(state / "turn_state.json", run_state / "turn_state.json")

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env["HARNESS_WORKTREE_ROOT"] = str(linked_wt)
    env["HARNESS_RUN_ID"] = "run-linked"
    env.pop("HARNESS_STATE_DIR", None)

    proc = subprocess.run(
        [sys.executable, "-I", str(PRE_GUARD_PATH)],
        input=json.dumps({
            "hook_event_name": "PreToolUse",
            "tool_name": "Edit",
            "tool_input": {"file_path": "src/billing/ledger.py", "old_string": "100", "new_string": "400"},
            "cwd": str(linked_wt),
        }),
        capture_output=True,
        text=True,
        cwd=str(linked_wt),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # 3. In-worktree HARNESS_STATE_DIR refused fail-closed
    fake_state = linked_wt / "fake_state"
    fake_state.mkdir(parents=True, exist_ok=True)
    shutil.copy(state / "task_contract.json", fake_state / "task_contract.json")

    res = run_pre(linked_wt, fake_state, "Edit", {"file_path": "src/billing/ledger.py", "old_string": "100", "new_string": "500"})
    assert res.returncode == 2
    assert "inside the worktree; refusing to trust it" in res.stderr

    res = run_post(linked_wt, fake_state, "Bash", {"command": "pytest -q"})
    assert res.returncode == 2
    assert "inside the worktree; refusing to trust it" in res.stderr


# ======================================================================================
# Vector 5 — Exit-and-Resume JSON Protocol Gate
# ======================================================================================


def test_vector5_amendment_request_protocol(tmp_path: Path) -> None:
    repo, state, _ = _init_repo(tmp_path)

    # 1. Valid Write to .harness/amendment_request.json
    amend_body = {
        "files": ["src/billing/money.py"],
        "reason": "Need parse_decimal helper for currency precision",
    }
    proc = run_pre(
        repo,
        state,
        "Write",
        {"file_path": ".harness/amendment_request.json", "content": json.dumps(amend_body)},
    )
    assert proc.returncode == 0, proc.stderr
    directive = json.loads(proc.stdout)
    assert directive == {"continue": False, "stopReason": "CONTRACT_AMENDMENT_REQUEST"}

    # Verified persisted copy in state_dir
    persisted = json.loads((state / "amendment_request.json").read_text(encoding="utf-8"))
    assert persisted["request"]["files"] == ["src/billing/money.py"]
    assert persisted["request"]["reason"] == amend_body["reason"]

    # 2. Valid Edit to an existing .harness/amendment_request.json
    amend_file = repo / ".harness" / "amendment_request.json"
    amend_file.write_text(json.dumps(amend_body, indent=2), encoding="utf-8")

    proc = run_pre(
        repo,
        state,
        "Edit",
        {
            "file_path": ".harness/amendment_request.json",
            "old_string": "currency precision",
            "new_string": "currency precision and tax rates",
        },
    )
    assert proc.returncode == 0, proc.stderr
    directive = json.loads(proc.stdout)
    assert directive == {"continue": False, "stopReason": "CONTRACT_AMENDMENT_REQUEST"}

    # 3. Negative cases: malformed JSON, empty files list, MultiEdit, symlinked file, NUL byte
    # Malformed JSON
    proc = run_pre(
        repo,
        state,
        "Write",
        {"file_path": ".harness/amendment_request.json", "content": "not-valid-json {"},
    )
    assert proc.returncode == 2
    assert "not well-formed JSON" in proc.stderr

    # Empty files list
    proc = run_pre(
        repo,
        state,
        "Write",
        {"file_path": ".harness/amendment_request.json", "content": json.dumps({"files": [], "reason": "none"})},
    )
    assert proc.returncode == 2
    assert "names no file" in proc.stderr

    # MultiEdit on amendment request
    proc = run_pre(
        repo,
        state,
        "MultiEdit",
        {
            "file_path": ".harness/amendment_request.json",
            "edits": [{"old_string": "a", "new_string": "b"}],
        },
    )
    assert proc.returncode == 2
    assert "may only be written with Write or Edit" in proc.stderr

    # NUL byte in amendment content
    proc = run_pre(
        repo,
        state,
        "Write",
        {"file_path": ".harness/amendment_request.json", "content": '{"files": ["a.py"], "reason": "bad\x00byte"}'},
    )
    assert proc.returncode == 2
    assert "contains a NUL byte" in proc.stderr
